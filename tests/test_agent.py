import json
from dataclasses import replace

from agent.config import load_settings
from agent.llm import FakeLLM, LLMReply, ToolCall, UsageTracker
from agent.pipeline import run_batch
from agent.prefilter import prefilter
from agent.schemas import Mention
from agent.tools import ToolContext, execute_tool, load_json


def settings():
    return replace(load_settings(), provider="fake", fake_llm=True, cheap_price_in=0.15, cheap_price_out=0.6,
                   strong_price_in=2.5, strong_price_out=10.0)


def mention(mid, text):
    return Mention(id=mid, source="test", author="a", text=text)


def run(mentions, llm=None):
    s = settings()
    return run_batch(mentions, s, llm or FakeLLM(s, UsageTracker()))


def test_prefilter_drops_duplicates_spam_and_offtopic():
    ms = [mention("1", "ВкусДоставка опоздала на час"),
          mention("2", "ВкусДоставка  опоздала на час!"),
          mention("3", "Подпишись! ВкусДоставка промокод http://x"),
          mention("4", "Где купить шины недорого в Москве?")]
    kept, dropped = prefilter(ms, ["вкусдоставка"])
    assert [m.id for m in kept] == ["1"]
    assert dropped == {"2": "duplicate", "3": "spam", "4": "not_about_brand"}


def test_praise_is_closed_by_cheap_model_without_strong_calls():
    s = settings()
    tracker = UsageTracker()
    report = run_batch([mention("1", "Спасибо ВкусДоставка, курьер герой!")], s, FakeLLM(s, tracker))
    assert report.results[0].stage == "triaged_low"
    assert tracker.stats["strong"].calls == 0


def test_complaint_is_escalated_and_uses_tools():
    r = run([mention("1", "ВкусДоставка списала деньги дважды, верните!")]).results[0]
    assert r.stage == "planned"
    assert r.plan is not None and r.plan.playbook_section == "payment"
    assert any(c.startswith("get_playbook") for c in r.tool_calls)
    assert not r.needs_human_review


def test_risky_topics_always_go_to_human_review():
    r = run([mention("1", "У ВкусДоставка утечка данных клиентов")]).results[0]
    assert r.needs_human_review
    assert "policy_requires_review" in r.review_reasons


def test_invalid_plan_is_retried_once_and_then_succeeds():
    s = settings()
    llm = FakeLLM(s, UsageTracker(), fail_plan_times=1)
    r = run_batch([mention("1", "ВкусДоставка опоздала на два часа")], s, llm).results[0]
    assert r.plan is not None and not r.needs_human_review


def test_plan_invalid_twice_goes_to_human_review():
    s = settings()
    llm = FakeLLM(s, UsageTracker(), fail_plan_times=2)
    r = run_batch([mention("1", "ВкусДоставка опоздала на два часа")], s, llm).results[0]
    assert r.plan is None
    assert "plan_invalid" in r.review_reasons


class LoopingLLM(FakeLLM):
    def _call(self, tier, messages, tools, json_mode):
        if "[TASK:INVESTIGATE]" in messages[0]["content"]:
            return LLMReply(None, [ToolCall(self._id(), "get_company_facts", {})], 10, 10, 1.0)
        return super()._call(tier, messages, tools, json_mode)


def test_tool_loop_is_capped():
    s = settings()
    r = run_batch([mention("1", "ВкусДоставка опоздала на два часа")], s, LoopingLLM(s, UsageTracker())).results[0]
    model_calls = [c for c in r.tool_calls if not c.endswith("[авто]")]
    assert len(model_calls) == s.max_tool_steps
    assert "tool_limit_reached" in r.review_reasons


def test_plan_always_gets_playbook_even_if_model_skips_it():
    s = settings()
    r = run_batch([mention("1", "ВкусДоставка опоздала на два часа")], s, LoopingLLM(s, UsageTracker())).results[0]
    assert r.tool_calls[-1] == 'get_playbook({"section": "delivery"}) [авто]'
    assert r.plan is not None


def test_bad_tool_arguments_return_error_instead_of_crash():
    ctx = ToolContext([], load_json("company.json"), load_json("playbook.json"))
    assert "error" in execute_tool(ctx, "search_related_mentions", {"query": 1, "limit": 100})
    assert "error" in execute_tool(ctx, "drop_database", {})


def test_cascade_is_cheaper_than_no_cascade():
    raw = json.loads(open("data/mentions.json", encoding="utf-8").read())
    m = run([Mention.model_validate(x) for x in raw]).metrics
    assert m["cost_usd"] < m["cost_without_cascade_usd"]
    assert m["time_ms"] < m["time_without_cascade_ms"]
    assert m["prefilter_dropped"] == 3


def test_progress_callback_reports_each_stage():
    events = []
    s = settings()
    run_batch([mention("1", "ВкусДоставка опоздала на два часа"), mention("2", "Купи подписчиков дёшево")],
              s, FakeLLM(s, UsageTracker()), on_progress=lambda mid, node, _: events.append((mid, node)))
    assert ("2", "prefilter") in events
    nodes = [n for mid, n in events if mid == "1"]
    assert nodes[0] == "triage" and nodes[-1] == "review" and "tools" in nodes
