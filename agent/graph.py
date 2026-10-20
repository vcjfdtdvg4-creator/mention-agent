"""Граф агента на LangGraph.

    START → triage (дешёвая модель)
              ├─ нерелевантно / низкая важность и модель уверена → END
              └─ иначе → investigate (сильная модель + tool-calling)
                            ⇄ tools (цикл, не больше MAX_TOOL_STEPS шагов)
                            → ensure_playbook (правило: без плейбука план не строим)
                            → plan (сильная модель, structured output + валидация + 1 ретрай)
                            → review (правила: нужна ли ручная проверка) → END
"""
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .config import Settings
from .llm import BaseLLM, assistant_message
from .schemas import ActionPlan, Mention, Triage
from .tools import ToolContext, execute_tool, openai_tools

ALWAYS_REVIEW_TOPICS = {"data_leak", "legal"}

TRIAGE_SYSTEM = """[TASK:TRIAGE]
Ты сортируешь упоминания бренда в соцсетях. Верни ТОЛЬКО один JSON-объект с данными (не схему!) ровно с этими полями:
{{"relevant": true, "sentiment": "negative", "severity": "medium", "topic": "delivery", "confidence": 0.8}}
relevant — true/false, касается ли текст бренда; sentiment — positive, neutral или negative.
topic — одно из: delivery (опоздания, курьер, остывшая еда), payment (списания, оплата, возвраты),
quality (вкус, свежесть, запах самой еды), data_leak (утечка данных), legal (суд, госорганы, отравление),
praise (благодарность, похвала), other (всё остальное).
severity=high — деньги клиента, безопасность, юридические риски; medium — жалоба на сервис; low — позитив и нейтральное.
confidence — насколько ты уверен в оценке (0..1). Не выдумывай: если неясно, снижай confidence."""

INVESTIGATE_SYSTEM = """[TASK:INVESTIGATE]
Ты аналитик репутации. Прежде чем предлагать план, собери факты инструментами.
Первым ответом ОБЯЗАТЕЛЬНО вызови сразу два инструмента: get_playbook с section, равным теме упоминания,
и search_related_mentions с коротким текстовым query (2–4 ключевых слова), чтобы проверить, единичный ли это случай.
При необходимости добавь get_company_facts. Когда фактов достаточно — ответь коротко текстом без вызова инструментов."""

PLAN_SYSTEM = """[TASK:PLAN]
Составь план реакции на упоминание. Опирайся ТОЛЬКО на собранные факты и правила плейбука, ничего не выдумывай.
Если правила требуют ручной проверки или фактов мало — ставь низкий confidence.
Верни ТОЛЬКО один JSON-объект с данными (не схему!) ровно с этими полями:
{{"summary": "суть проблемы в 1–2 предложениях",
 "playbook_section": "ключ раздела плейбука из списка допустимых",
 "steps": [{{"action": "что сделать", "owner": "support", "deadline_hours": 2}}],
 "reply_draft": "черновик публичного ответа или null",
 "evidence": ["факт из собранных данных"],
 "confidence": 0.8}}
owner — одно из: support, pr, marketing, product, legal. deadline_hours — целое от 1 до 168. steps — от 1 до 6 шагов."""


class AgentState(TypedDict, total=False):
    mention: Mention
    triage: Triage | None
    messages: list[dict]
    pending_calls: list
    tool_steps: int
    tool_log: list[str]
    tool_results: list[dict]
    plan: ActionPlan | None
    review_reasons: list[str]


def build_graph(llm: BaseLLM, ctx: ToolContext, settings: Settings):
    playbook_keys = set(ctx.playbook)

    # ---------- узлы ----------

    def triage(state: AgentState) -> dict:
        m = state["mention"]
        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM.format()},
            {"role": "user", "content": f"Источник: {m.source}\nТекст: {m.text}"},
        ]
        reply = llm.chat("cheap", messages, json_mode=True)
        try:
            return {"triage": Triage.model_validate_json(reply.content or "")}
        except (ValidationError, ValueError):
            return {"triage": None}

    def investigate(state: AgentState) -> dict:
        m, t = state["mention"], state.get("triage")
        messages = list(state.get("messages") or [
            {"role": "system", "content": INVESTIGATE_SYSTEM},
            {"role": "user", "content": (
                f"Упоминание: {m.text}\nИсточник: {m.source}\n"
                f"Тема: {t.topic if t else 'other'}\nСерьёзность: {t.severity if t else 'unknown'}"
            )},
        ])
        reply = llm.chat("strong", messages, tools=openai_tools())
        messages.append(assistant_message(reply))
        return {"messages": messages, "pending_calls": reply.tool_calls}

    def tools(state: AgentState) -> dict:
        ctx_messages = list(state["messages"])
        log = list(state.get("tool_log") or [])
        results = list(state.get("tool_results") or [])
        for call in state["pending_calls"]:
            result = execute_tool(ctx, call.name, call.arguments)
            ctx_messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
            log.append(f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
            results.append({"tool": call.name, "result": result})
        return {"messages": ctx_messages, "pending_calls": [], "tool_log": log,
                "tool_results": results, "tool_steps": state.get("tool_steps", 0) + 1}

    def ensure_playbook(state: AgentState) -> dict:
        """План всегда опирается на плейбук: если модель не вызвала get_playbook, вызываем его сами по теме триажа."""
        results = state.get("tool_results") or []
        if any(r["tool"] == "get_playbook" for r in results):
            return {}
        t = state.get("triage")
        section = t.topic if t and t.topic in playbook_keys else "other"
        args = {"section": section}
        return {"tool_results": results + [{"tool": "get_playbook", "result": execute_tool(ctx, "get_playbook", args)}],
                "tool_log": list(state.get("tool_log") or []) + [f"get_playbook({json.dumps(args, ensure_ascii=False)}) [авто]"]}

    def plan(state: AgentState) -> dict:
        m, t = state["mention"], state.get("triage")
        results = state.get("tool_results") or []
        related = next((r["result"].get("total_found") for r in results
                        if r["tool"] == "search_related_mentions"), None)
        user = (
            f"Упоминание: {m.text}\n"
            f"Тема: {t.topic if t else 'other'}\n"
            f"Серьёзность: {t.severity if t else 'unknown'}\n"
            + (f"Похожих упоминаний: {related}\n" if related is not None else "")
            + f"Допустимые значения playbook_section: {', '.join(sorted(playbook_keys))}\n"
            f"Собранные факты:\n{json.dumps(results, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": PLAN_SYSTEM.format()},
            {"role": "user", "content": user},
        ]
        # Первая попытка + один ретрай с текстом ошибки валидации.
        for attempt in range(2):
            reply = llm.chat("strong", messages, json_mode=True)
            try:
                p = ActionPlan.model_validate_json(reply.content or "")
                if p.playbook_section not in playbook_keys:
                    raise ValueError(f"playbook_section '{p.playbook_section}' нет в плейбуке")
                return {"plan": p}
            except (ValidationError, ValueError) as e:
                messages += [
                    {"role": "assistant", "content": reply.content},
                    {"role": "user", "content": f"Ответ не прошёл валидацию: {e}. Исправь и верни только JSON."},
                ]
        return {"plan": None}

    def review(state: AgentState) -> dict:
        t, p = state.get("triage"), state.get("plan")
        reasons = []
        if t is None:
            reasons.append("triage_invalid")
        if p is None:
            reasons.append("plan_invalid")
        elif p.confidence < settings.review_threshold:
            reasons.append(f"low_confidence({p.confidence:.2f})")
        topic = (p.playbook_section if p else None) or (t.topic if t else None)
        if topic in ALWAYS_REVIEW_TOPICS:
            reasons.append("policy_requires_review")
        if state.get("pending_calls"):  # модель ещё хотела инструменты, но лимит шагов исчерпан
            reasons.append("tool_limit_reached")
        return {"review_reasons": reasons}

    # ---------- маршрутизация ----------

    def after_triage(state: AgentState) -> str:
        t = state.get("triage")
        if t is None:
            return "investigate"  # не смогли разобрать — безопаснее эскалировать
        if not t.relevant:
            return "done"
        if t.severity == "low" and t.confidence >= settings.review_threshold:
            return "done"  # дешёвая модель уверена, что реагировать не нужно
        return "investigate"  # важно или модель не уверена → сильная модель

    def after_investigate(state: AgentState) -> str:
        if not state.get("pending_calls"):
            return "plan"
        if state.get("tool_steps", 0) >= settings.max_tool_steps:
            return "plan"  # защита от зацикливания; review отметит это
        return "tools"

    g = StateGraph(AgentState)
    g.add_node("triage", triage)
    g.add_node("investigate", investigate)
    g.add_node("tools", tools)
    g.add_node("ensure_playbook", ensure_playbook)
    g.add_node("plan", plan)
    g.add_node("review", review)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", after_triage, {"investigate": "investigate", "done": END})
    g.add_conditional_edges("investigate", after_investigate, {"tools": "tools", "plan": "ensure_playbook"})
    g.add_edge("tools", "investigate")
    g.add_edge("ensure_playbook", "plan")
    g.add_edge("plan", "review")
    g.add_edge("review", END)
    return g.compile()
