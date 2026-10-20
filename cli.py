import argparse
import json
import sys
from pathlib import Path

from agent.config import load_settings
from agent.llm import LLMUnavailable, OpenAILLM, UsageTracker, make_llm
from agent.pipeline import run_batch
from agent.schemas import Mention

STAGE_LABEL = {
    "prefilter_dropped": "отсеяно правилами",
    "triaged_low": "закрыто дешёвой моделью",
    "planned": "план от сильной модели",
}


def warmup(settings, keep_alive: str = "30m") -> None:
    import time
    import urllib.request

    if not settings.is_local:
        sys.exit("--warmup нужен только для Ollama (LLM_PROVIDER=ollama)")
    root = (settings.base_url or "").rstrip("/").removesuffix("/v1")
    for model in dict.fromkeys([settings.cheap_model, settings.strong_model]):
        t0 = time.perf_counter()
        body = json.dumps({"model": model, "prompt": "", "keep_alive": keep_alive}).encode()
        req = urllib.request.Request(f"{root}/api/generate", body, {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=settings.timeout).read()
        except OSError as e:
            sys.exit(f"Не удалось загрузить {model}: {e}. Запущена ли Ollama?")
        print(f"{model}: загружена на {keep_alive} ({time.perf_counter() - t0:.1f} с)")


def print_progress(mention_id: str, node: str, update: dict) -> None:
    if node == "prefilter":
        msg = f"отсеяно правилами ({update['drop_reason']})"
    elif node == "triage":
        t = update.get("triage")
        msg = f"триаж: {t.topic}, {t.severity}, уверенность {t.confidence:.2f}" if t else "триаж не распознан"
    elif node == "investigate":
        calls = update.get("pending_calls") or []
        msg = "сильная модель вызывает: " + ", ".join(c.name for c in calls) if calls else "сильная модель: факты собраны"
    elif node == "ensure_playbook":
        if not update:
            return
        msg = "плейбук подтянут автоматически"
    elif node == "tools":
        msg = "инструменты выполнены"
    elif node == "plan":
        msg = "план готов" if update.get("plan") else "план не прошёл валидацию"
    elif node == "review":
        reasons = update.get("review_reasons") or []
        msg = "⚠ ручная проверка: " + ", ".join(reasons) if reasons else "проверка пройдена"
    else:
        msg = node
    print(f"[{mention_id}] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/mentions.json")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--warmup", action="store_true",
                    help="загрузить обе модели в память Ollama на 30 минут и выйти")
    ap.add_argument("--quiet", action="store_true", help="не печатать ход обработки")
    args = ap.parse_args()

    if args.warmup:
        warmup(load_settings())
        return

    raw = json.loads(Path(args.file).read_text(encoding="utf-8"))
    mentions = [Mention.model_validate(x) for x in raw]
    settings = load_settings()
    llm = make_llm(settings, UsageTracker())
    if isinstance(llm, OpenAILLM):
        try:
            llm.check()
        except LLMUnavailable as e:
            sys.exit(f"Ошибка: {e}\nДля запуска без моделей: LLM_PROVIDER=fake python cli.py")
    report = run_batch(mentions, settings, llm, on_progress=None if args.quiet else print_progress)

    if args.json:
        json.dump({"metrics": report.metrics, "results": [r.model_dump() for r in report.results]},
                  sys.stdout, ensure_ascii=False, indent=2)
        return

    texts = {m.id: m.text for m in mentions}
    for r in report.results:
        flag = "  ⚠ РУЧНАЯ ПРОВЕРКА: " + ", ".join(r.review_reasons) if r.needs_human_review else ""
        print(f"\n[{r.mention_id}] {STAGE_LABEL[r.stage]}{flag}")
        print(f"   «{texts[r.mention_id][:90]}»")
        if r.drop_reason:
            print(f"   причина: {r.drop_reason}")
        if r.triage:
            t = r.triage
            print(f"   триаж: {t.topic}, {t.sentiment}, важность {t.severity}, уверенность {t.confidence:.2f}")
        for call in r.tool_calls:
            print(f"   🔧 {call}")
        if r.plan:
            for s in r.plan.steps:
                print(f"   → [{s.owner}, {s.deadline_hours}ч] {s.action}")

    m = report.metrics
    print("\n" + "=" * 70)
    print(f"Режим: {m['mode']}")
    print(f"Упоминаний: {m['mentions_total']} | отсеяно правилами: {m['prefilter_dropped']} "
          f"| закрыто дешёвой: {m['closed_by_cheap_model']} | эскалировано: {m['escalated_to_strong_model']} "
          f"| на ручную проверку: {m['needs_human_review']}")
    print(f"Дешёвая модель: {m['cheap']['calls']} вызовов, {m['cheap']['tokens']} токенов, ${m['cheap']['cost_usd']}")
    print(f"Сильная модель: {m['strong']['calls']} вызовов, {m['strong']['tokens']} токенов, ${m['strong']['cost_usd']}")
    if m["cost_without_cascade_usd"] > 0:
        print(f"Стоимость: ${m['cost_usd']}  vs  без каскада ≈ ${m['cost_without_cascade_usd']}  → экономия {m['savings_pct']}%")
    else:
        print("Стоимость: $0 (" + ("локальные модели" if settings.is_local else "офлайн-режим") + ")")
    print(f"Время моделей: {m['time_ms'] / 1000:.1f} с  vs  без каскада ≈ {m['time_without_cascade_ms'] / 1000:.1f} с"
          f"  → экономия {m['time_savings_pct']}%")


if __name__ == "__main__":
    main()
