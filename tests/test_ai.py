from types import SimpleNamespace

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
