import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from .config import Settings

Tier = Literal["cheap", "strong"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float | None = None


@dataclass
class TierStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class UsageTracker:
    def __init__(self) -> None:
        self.stats: dict[str, TierStats] = {"cheap": TierStats(), "strong": TierStats()}

    def record(self, tier: Tier, reply: LLMReply, latency_ms: float) -> None:
        s = self.stats[tier]
        s.calls += 1
        s.prompt_tokens += reply.prompt_tokens
        s.completion_tokens += reply.completion_tokens
        s.latency_ms += latency_ms

    def cost(self, settings: Settings, tier: Tier, as_tier: Tier | None = None) -> float:
        s = self.stats[tier]
        price = as_tier or tier
        p_in = settings.cheap_price_in if price == "cheap" else settings.strong_price_in
        p_out = settings.cheap_price_out if price == "cheap" else settings.strong_price_out
        return (s.prompt_tokens * p_in + s.completion_tokens * p_out) / 1_000_000


def assistant_message(reply: LLMReply) -> dict:
    msg: dict = {"role": "assistant", "content": reply.content}
    if reply.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
            for tc in reply.tool_calls
        ]
    return msg


class BaseLLM:
    def __init__(self, settings: Settings, tracker: UsageTracker) -> None:
        self.settings = settings
        self.tracker = tracker

    def chat(self, tier: Tier, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> LLMReply:
        t0 = time.perf_counter()
        reply = self._call(tier, messages, tools, json_mode)
        latency = reply.latency_ms if reply.latency_ms is not None else (time.perf_counter() - t0) * 1000
        self.tracker.record(tier, reply, latency)
        return reply

    def _call(self, tier, messages, tools, json_mode) -> LLMReply:
        raise NotImplementedError


class LLMUnavailable(RuntimeError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def extract_json(text: str | None) -> str | None:
    if not text:
        return text
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    if t.startswith("{"):
        return t
    start, end = t.find("{"), t.rfind("}")
    return t[start:end + 1] if 0 <= start < end else text


def unwrap_schema_echo(text: str | None) -> str | None:
    try:
        data = json.loads(text or "")
    except json.JSONDecodeError:
        return text
    props = data.get("properties") if isinstance(data, dict) else None
    if (isinstance(props, dict) and props and data.get("type") == "object"
            and not any(isinstance(v, dict) and ("type" in v or "enum" in v or "$ref" in v) for v in props.values())):
        return json.dumps(props, ensure_ascii=False)
    return text


def _tool_calls_from_content(content: str | None, tools: list[dict] | None) -> list[ToolCall]:
    if not content or not tools:
        return []
    known = {t["function"]["name"] for t in tools}
    raw = extract_json(content.replace("<tool_call>", "").replace("</tool_call>", ""))
    try:
        data = json.loads(raw or "")
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else [data]
    calls = []
    for item in items:
        if not isinstance(item, dict) or item.get("name") not in known:
            return []
        args = item.get("arguments") or item.get("parameters") or {}
        calls.append(ToolCall(f"call_{uuid.uuid4().hex[:12]}", item["name"], args if isinstance(args, dict) else {"_raw": args}))
    return calls


class OpenAILLM(BaseLLM):
    def __init__(self, settings: Settings, tracker: UsageTracker) -> None:
        super().__init__(settings, tracker)
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                             timeout=settings.timeout, max_retries=2)

    def check(self) -> None:
        import openai

        try:
            available = {m.id for m in self.client.models.list()}
        except openai.APIConnectionError as e:
            hint = " Запустите Ollama: `ollama serve`." if self.settings.is_local else ""
            raise LLMUnavailable(f"Нет связи с {self.settings.base_url}.{hint}") from e
        if not self.settings.is_local:
            return
        missing = [m for m in dict.fromkeys([self.settings.cheap_model, self.settings.strong_model])
                   if m not in available and f"{m}:latest" not in available]
        if missing:
            cmds = "; ".join(f"ollama pull {m}" for m in missing)
            raise LLMUnavailable(f"В Ollama нет моделей: {', '.join(missing)}. Скачайте: {cmds}")

    def _call(self, tier, messages, tools, json_mode) -> LLMReply:
        model = self.settings.cheap_model if tier == "cheap" else self.settings.strong_model
        kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
        if self.settings.is_local:
            kwargs["extra_body"] = {"keep_alive": "30m"}
        if tools:
            kwargs["tools"] = tools
            kwargs["max_tokens"] = 256
        if json_mode and self.settings.is_local:
            kwargs["max_tokens"] = 700
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        content = msg.content
        if not calls:
            calls = _tool_calls_from_content(content, tools)
            if calls:
                content = None
        if json_mode:
            content = unwrap_schema_echo(extract_json(content))
        usage = resp.usage
        return LLMReply(
            content=content,
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


_TOPIC_RULES = [
    ("data_leak", ("утечк",), "negative", "high", 0.9),
    ("legal", ("суд", "роспотребнадзор"), "negative", "high", 0.9),
    ("payment", ("списал", "оплат", "платеж"), "negative", "high", 0.88),
    ("delivery", ("опозда", "ждём", "ждем"), "negative", "medium", 0.85),
    ("quality", ("запах", "сухой", "невкусн"), "negative", "medium", 0.8),
    ("praise", ("спасибо", "герой", "неплох", "отлично"), "positive", "low", 0.9),
]

_SEARCH_QUERY = {
    "delivery": "опоздала опоздание ждём",
    "payment": "списала оплату платежа",
    "quality": "запах сухой",
    "data_leak": "утечка база",
    "legal": "суд Роспотребнадзор",
}

_PLAN_STEPS = {
    "delivery": [("Публично извиниться и запросить номер заказа в личных сообщениях", "support", 2),
                 ("Проверить заказ и начислить компенсацию по политике возвратов", "support", 24),
                 ("Передать в логистику данные о системных опозданиях по району", "product", 24)],
    "payment": [("Связаться с клиентом лично и запросить детали платежа", "support", 2),
                ("Оформить возврат двойного списания и подтвердить клиенту", "support", 24),
                ("Проверить платёжный шлюз на повторяющиеся ошибки", "product", 24)],
    "quality": [("Извиниться и предложить повторную доставку или возврат", "support", 2),
                ("Запросить у ресторана-партнёра объяснение по качеству", "product", 48)],
    "data_leak": [("Уведомить службу безопасности о возможной утечке", "legal", 1),
                  ("Подготовить нейтральное заявление «проверяем информацию»", "pr", 4)],
    "legal": [("Передать претензию юристу для оценки рисков", "legal", 4),
              ("Перевести общение с клиентом в личный канал", "support", 2)],
    "praise": [("Поблагодарить автора и передать отзыв команде", "marketing", 24)],
    "other": [("Ответить автору по ситуации", "support", 24)],
}


def _tokens(text: str | None) -> int:
    return max(1, len(text or "") // 4)


def _field(text: str, name: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"{name}:"):
            return line.split(":", 1)[1].strip()
    return ""


class FakeLLM(BaseLLM):
    def __init__(self, settings: Settings, tracker: UsageTracker, fail_plan_times: int = 0) -> None:
        super().__init__(settings, tracker)
        self.fail_plan_times = fail_plan_times
        self._ids = 0

    def _call(self, tier, messages, tools, json_mode) -> LLMReply:
        system = messages[0]["content"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        prompt_tokens = sum(_tokens(m.get("content")) for m in messages)
        latency = 300.0 if tier == "cheap" else 1500.0

        if "[TASK:TRIAGE]" in system:
            content = json.dumps(self._triage(_field(user, "Текст")), ensure_ascii=False)
            return LLMReply(content, [], prompt_tokens, _tokens(content), latency)

        if "[TASK:INVESTIGATE]" in system:
            if any(m["role"] == "tool" for m in messages):
                content = "Факты собраны, можно составлять план."
                return LLMReply(content, [], prompt_tokens, _tokens(content), latency)
            topic = _field(user, "Тема")
            calls = [ToolCall(self._id(), "get_playbook", {"section": topic}),
                     ToolCall(self._id(), "get_company_facts", {})]
            if topic in _SEARCH_QUERY:
                calls.insert(0, ToolCall(self._id(), "search_related_mentions", {"query": _SEARCH_QUERY[topic]}))
            return LLMReply(None, calls, prompt_tokens, 40, latency)

        if "[TASK:PLAN]" in system:
            if self.fail_plan_times > 0:
                self.fail_plan_times -= 1
                content = '{"summary": "коротко", "steps": []}'
            else:
                content = json.dumps(self._plan(user), ensure_ascii=False)
            return LLMReply(content, [], prompt_tokens, _tokens(content), latency)

        raise ValueError("FakeLLM: неизвестная задача")

    def _id(self) -> str:
        self._ids += 1
        return f"call_{self._ids}"

    @staticmethod
    def _triage(text: str) -> dict:
        low = text.lower()
        for topic, keys, sentiment, severity, conf in _TOPIC_RULES:
            if any(k in low for k in keys):
                return {"relevant": True, "sentiment": sentiment, "severity": severity,
                        "topic": topic, "confidence": conf}
        return {"relevant": True, "sentiment": "neutral", "severity": "low",
                "topic": "other", "confidence": 0.55}

    @staticmethod
    def _plan(user: str) -> dict:
        topic = _field(user, "Тема") or "other"
        evidence = []
        found = _field(user, "Похожих упоминаний")
        if found:
            evidence.append(f"Похожих упоминаний в выборке: {found}")
        evidence.append(f"Правила раздела плейбука '{topic}' учтены")
        risky = topic in ("data_leak", "legal")
        return {
            "summary": f"Негативное упоминание по теме «{topic}», требуется реакция по плейбуку.",
            "playbook_section": topic,
            "steps": [{"action": a, "owner": o, "deadline_hours": h} for a, o, h in _PLAN_STEPS.get(topic, _PLAN_STEPS["other"])],
            "reply_draft": None if risky else "Здравствуйте! Нам очень жаль, разберёмся и напишем вам в личные сообщения.",
            "evidence": evidence,
            "confidence": 0.62 if risky else 0.86,
        }


def make_llm(settings: Settings, tracker: UsageTracker) -> BaseLLM:
    return FakeLLM(settings, tracker) if settings.fake_llm else OpenAILLM(settings, tracker)
