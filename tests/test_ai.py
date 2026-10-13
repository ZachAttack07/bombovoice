import json
import urllib.error
from types import SimpleNamespace

import pytest

from bombovoice import ai


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _director(messages):
    d = ai.AIDirector.__new__(ai.AIDirector)
    d.client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    d.model, d.intensity, d.spicy, d.style, d.effort = "claude-opus-5-5", 2, False, "yaadie", "low"
    d.fallback = ai.PatoisTranslator(intensity=2, seed=0)
    return d


def test_ai_plan_is_used():
    plan = ai.DeliveryPlan(patois="Bwoy, mi a go a di store, seen?", segments=[
        ai.PlanSegment(text="Bwoy,", emphasis=2, contour="high", pause_after_ms=300),
        ai.PlanSegment(text="mi a go a di store, seen?", emphasis=7, contour="rise", pause_after_ms=5000),
    ])
    msgs = FakeMessages(SimpleNamespace(stop_reason="end_turn", parsed_output=plan))
    tr = _director(msgs).translate("I'm going to the store")
    assert tr.source == "ai" and tr.patois.startswith("Bwoy")
    assert tr.segments[1].emphasis == 3 and tr.segments[1].pause_after_ms == 900  # clamped
    kw = msgs.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_format"] is ai.DeliveryPlan
    assert kw["fallbacks"] == "default"
    assert "<transcript>" in kw["messages"][0]["content"]


def test_falls_back_to_rules_on_error():
    tr = _director(FakeMessages(error=RuntimeError("offline"))).translate("I think the thing is good")
    assert tr.source == "rules" and "tink" in tr.patois.lower()


def test_falls_back_on_refusal():
    msgs = FakeMessages(SimpleNamespace(stop_reason="refusal", parsed_output=None))
    assert _director(msgs).translate("hello my friend").source == "rules"


# --------------------------------------------------------------------------- OpenRouter

PLAN_JSON = json.dumps({
    "patois": "Bwoy, mi a go a di store, seen?",
    "segments": [
        {"text": "Bwoy,", "emphasis": 2, "contour": "high", "pause_after_ms": 300},
        {"text": "mi a go a di store, seen?", "emphasis": 1, "contour": "rise", "pause_after_ms": 200},
    ],
})


def _openrouter(monkeypatch, replies):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    d = ai.OpenRouterDirector()
    bodies = []

    def fake_post(body):
        bodies.append(json.loads(json.dumps(body)))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    d._post = fake_post
    return d, bodies


def _chat(content, **message):
    return {"choices": [{"message": {"content": content, **message}, "finish_reason": "stop"}]}


def test_openrouter_defaults_to_haiku_and_parses_plan(monkeypatch):
    d, bodies = _openrouter(monkeypatch, [_chat(PLAN_JSON)])
    tr = d.translate("I'm going to the store")
    assert tr.source == "ai" and tr.segments[0].contour == "high"
    body = bodies[0]
    assert body["model"] == "anthropic/claude-haiku-5.5"
    assert body["response_format"]["json_schema"]["schema"] is not None
    assert body["reasoning"] == {"effort": "low"}
    assert body["messages"][0]["role"] == "system" and "<transcript>" in body["messages"][1]["content"]
    assert "temperature" not in body


def test_openrouter_retries_without_structured_options_on_400(monkeypatch):
    err = urllib.error.HTTPError(ai.OPENROUTER_URL, 400, "bad", {}, None)
    err.read = lambda: b'{"error": "response_format not supported"}'
    d, bodies = _openrouter(monkeypatch, [err, _chat("```json\n" + PLAN_JSON + "\n```")])
    tr = d.translate("I'm going to the store")
    assert tr.source == "ai"
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]


def test_openrouter_errors_fall_back_to_rules(monkeypatch):
    err = urllib.error.HTTPError(ai.OPENROUTER_URL, 401, "unauthorized", {}, None)
    err.read = lambda: b'{"error": "bad key"}'
    d, _ = _openrouter(monkeypatch, [err])
    assert d.translate("I think the thing is good").source == "rules"
    d, _ = _openrouter(monkeypatch, [_chat(None, refusal="no")])
    assert d.translate("hello my friend").source == "rules"


def test_parse_plan_tolerates_chatter():
    plan = ai.parse_plan("Sure! Here you go:\n" + PLAN_JSON + "\nEnjoy")
    assert plan.patois.startswith("Bwoy")
    with pytest.raises(RuntimeError):
        ai.parse_plan("no json here")


def test_provider_selection(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert ai.resolve_provider("auto") == "anthropic"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert ai.resolve_provider("auto") == "openrouter"
    t = ai.make_translator(True, provider="auto", intensity=3)
    assert isinstance(t, ai.OpenRouterDirector) and t.model == "anthropic/claude-haiku-5.5" and t.intensity == 3
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert isinstance(ai.make_translator(True, provider="openrouter"), ai.PatoisTranslator)  # no key -> offline
