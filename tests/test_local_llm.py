import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.config import load_settings
from agent.llm import LLMUnavailable, OpenAILLM, UsageTracker, extract_json
from agent.pipeline import run_batch
from agent.schemas import Mention

PLAN = {
    "summary": "Клиенту дважды списали оплату за заказ.",
    "playbook_section": "payment",
    "steps": [{"action": "Связаться с клиентом и оформить возврат", "owner": "support", "deadline_hours": 24}],
    "reply_draft": None,
    "evidence": ["Похожие жалобы есть"],
    "confidence": 0.9,
}
TRIAGE = {"relevant": True, "sentiment": "negative", "severity": "high", "topic": "payment", "confidence": 0.9}


class FakeOllama(BaseHTTPRequestHandler):
    models = ["qwen2.5:3b", "qwen2.5:7b"]
    requests: list[dict] = []

    def log_message(self, *args):
        pass

    def _send(self, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": m, "object": "model", "owned_by": "library"} for m in self.models]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(req)
        system = req["messages"][0]["content"]
        if "[TASK:TRIAGE]" in system:
            content = "```json\n" + json.dumps(TRIAGE) + "\n```"
        elif "[TASK:INVESTIGATE]" in system:
            if any(m["role"] == "tool" for m in req["messages"]):
                content = "Фактов достаточно."
            else:
                content = '<tool_call>{"name": "get_playbook", "arguments": {"section": "payment"}}</tool_call>'
        else:
            content = "Вот план:\n" + json.dumps(PLAN, ensure_ascii=False)
        self._send({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": req["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })


@pytest.fixture
def ollama_settings(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeOllama.requests = []
    for var in ("LLM_PROVIDER", "FAKE_LLM", "OPENAI_BASE_URL", "OPENAI_API_KEY", "CHEAP_MODEL", "STRONG_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    yield load_settings()
    server.shutdown()


def test_ollama_is_default_and_free(monkeypatch):
    for var in ("LLM_PROVIDER", "FAKE_LLM", "OPENAI_BASE_URL", "OLLAMA_BASE_URL", "CHEAP_MODEL", "STRONG_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = load_settings()
    assert s.provider == "ollama" and not s.fake_llm
    assert s.base_url == "http://localhost:11434/v1"
    assert (s.cheap_model, s.strong_model) == ("qwen2.5:3b", "qwen2.5:7b")
    assert s.strong_price_in == s.strong_price_out == 0


def test_models_and_url_come_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    monkeypatch.setenv("STRONG_MODEL", "llama3.1:8b")
    s = load_settings()
    assert s.base_url == "http://gpu-box:11434/v1" and s.strong_model == "llama3.1:8b"


def test_extract_json_handles_fences_and_chatter():
    assert json.loads(extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(extract_json('Конечно! {"a": {"b": 2}} Готово.')) == {"a": {"b": 2}}
    assert extract_json("нет json") == "нет json"


def test_full_pipeline_through_ollama_api(ollama_settings):
    m = Mention(id="1", source="vk", author="a", text="ВкусДоставка списала оплату дважды, верните деньги")
    llm = OpenAILLM(ollama_settings, UsageTracker())
    llm.check()
    r = run_batch([m], ollama_settings, llm).results[0]

    assert r.stage == "planned"
    assert r.triage.topic == "payment"
    assert r.tool_calls == ['get_playbook({"section": "payment"})']
    assert r.plan.playbook_section == "payment" and not r.needs_human_review
    used = {req["model"] for req in FakeOllama.requests}
    assert used == {"qwen2.5:3b", "qwen2.5:7b"}


def test_missing_model_gives_pull_hint(ollama_settings):
    llm = OpenAILLM(replace(ollama_settings, strong_model="llama3.1:8b"), UsageTracker())
    with pytest.raises(LLMUnavailable, match="ollama pull llama3.1:8b"):
        llm.check()


def test_ollama_not_running_gives_clear_error(ollama_settings):
    llm = OpenAILLM(replace(ollama_settings, base_url="http://127.0.0.1:9/v1"), UsageTracker())
    llm.client = llm.client.with_options(max_retries=0)
    with pytest.raises(LLMUnavailable, match="ollama serve"):
        llm.check()


def test_schema_echo_with_values_is_unwrapped():
    from agent.llm import unwrap_schema_echo

    echo = '{"title": "Triage", "type": "object", "properties": {"relevant": true, "confidence": 0.9}}'
    assert json.loads(unwrap_schema_echo(echo)) == {"relevant": True, "confidence": 0.9}
    schema = '{"type": "object", "properties": {"relevant": {"type": "boolean"}}}'
    assert unwrap_schema_echo(schema) == schema
