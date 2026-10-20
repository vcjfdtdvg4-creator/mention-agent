from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from .config import Settings, load_settings
from .graph import build_graph
from .llm import BaseLLM, UsageTracker, make_llm
from .prefilter import prefilter
from .schemas import Mention, MentionResult
from .tools import ToolContext, load_json


@dataclass
class BatchReport:
    results: list[MentionResult]
    metrics: dict


ProgressFn = Callable[[str, str, dict], None]


def run_batch(mentions: list[Mention], settings: Settings | None = None,
              llm: BaseLLM | None = None, on_progress: ProgressFn | None = None) -> BatchReport:
    settings = settings or load_settings()
    tracker = llm.tracker if llm else UsageTracker()
    llm = llm or make_llm(settings, tracker)
    ctx = ToolContext(mentions=mentions, company=load_json("company.json"), playbook=load_json("playbook.json"))
    graph = build_graph(llm, ctx, settings)

    kept, dropped = prefilter(mentions, ctx.company["aliases"])
    results = [MentionResult(mention_id=mid, stage="prefilter_dropped", drop_reason=reason)
               for mid, reason in dropped.items()]

    if on_progress:
        for mid, reason in dropped.items():
            on_progress(mid, "prefilter", {"drop_reason": reason})

    for m in kept:
        state: dict = {}
        for mode, chunk in graph.stream({"mention": m}, stream_mode=["updates", "values"]):
            if mode == "values":
                state = chunk
            elif on_progress:
                for node, update in chunk.items():
                    on_progress(m.id, node, update or {})
        planned = "review_reasons" in state
        reasons = state.get("review_reasons") or []
        results.append(MentionResult(
            mention_id=m.id,
            stage="planned" if planned else "triaged_low",
            triage=state.get("triage"),
            plan=state.get("plan"),
            needs_human_review=bool(reasons),
            review_reasons=reasons,
            tool_calls=state.get("tool_log") or [],
        ))

    order = {m.id: i for i, m in enumerate(mentions)}
    results.sort(key=lambda r: order[r.mention_id])
    return BatchReport(results, _metrics(results, tracker, settings, len(mentions)))


def _metrics(results: list[MentionResult], tracker: UsageTracker, s: Settings, total: int) -> dict:
    stages = Counter(r.stage for r in results)
    cheap, strong = tracker.stats["cheap"], tracker.stats["strong"]
    cost_cheap = tracker.cost(s, "cheap")
    cost_strong = tracker.cost(s, "strong")
    planned = stages["planned"]

    strong_per_planned = cost_strong / planned if planned else 0.0
    baseline = tracker.cost(s, "cheap", as_tier="strong") * total / max(cheap.calls, 1) + strong_per_planned * total
    actual = cost_cheap + cost_strong

    strong_call_ms = strong.latency_ms / strong.calls if strong.calls else 0.0
    strong_ms_per_planned = strong.latency_ms / planned if planned else 0.0
    time_baseline = (strong_call_ms + strong_ms_per_planned) * total
    time_actual = cheap.latency_ms + strong.latency_ms

    return {
        "mode": "fake_llm" if s.fake_llm else f"{s.provider}: {s.cheap_model} → {s.strong_model}",
        "mentions_total": total,
        "prefilter_dropped": stages["prefilter_dropped"],
        "drop_reasons": dict(Counter(r.drop_reason for r in results if r.drop_reason)),
        "closed_by_cheap_model": stages["triaged_low"],
        "escalated_to_strong_model": planned,
        "needs_human_review": sum(r.needs_human_review for r in results),
        "cheap": {"calls": cheap.calls, "tokens": cheap.prompt_tokens + cheap.completion_tokens,
                  "latency_ms": round(cheap.latency_ms), "cost_usd": round(cost_cheap, 5)},
        "strong": {"calls": strong.calls, "tokens": strong.prompt_tokens + strong.completion_tokens,
                   "latency_ms": round(strong.latency_ms), "cost_usd": round(cost_strong, 5)},
        "cost_usd": round(actual, 5),
        "cost_without_cascade_usd": round(baseline, 5),
        "savings_pct": round((1 - actual / baseline) * 100, 1) if baseline else 0.0,
        "time_ms": round(time_actual),
        "time_without_cascade_ms": round(time_baseline),
        "time_savings_pct": round((1 - time_actual / time_baseline) * 100, 1) if time_baseline else 0.0,
    }
