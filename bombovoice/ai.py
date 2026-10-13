"""Claude-powered Patois translator and delivery director.

Where the rule-based translator swaps words, Claude rewrites the *whole line*
the way a hilarious Jamaican uncle would say it, then plans the delivery:
which words to punch, where the pitch swoops up and crashes down, where to
pause for comic timing. The plan comes back as structured segments that the
TTS renderer performs.

Two ways to reach Claude:
    openrouter - set OPENROUTER_API_KEY; defaults to anthropic/claude-haiku-5.5 (fast and cheap)
    anthropic  - set ANTHROPIC_API_KEY (or `ant auth login`); defaults to claude-opus-5-5
With provider "auto", OpenRouter is used whenever OPENROUTER_API_KEY is set.
Falls back to the offline translator on any error.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, Field

from .patois import INTENSITY_NAMES, PatoisTranslator, Segment, Translation

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-haiku-5.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

STYLES = {
    "yaadie": "a warm, loud, endlessly entertaining Kingston uncle who turns every sentence into a performance",
    "dancehall": "a hype dancehall MC on the mic at a sound clash - every line is a shout-out, a boast or a 'pull up!'",
    "rasta": "a calm, philosophical rasta elder who speaks in slow, wise 'I and I' proverbs and overstands everything",
    "auntie": "a dramatic Jamaican auntie who is scandalised by everything and loves to cuss you out lovingly",
    "commentator": "an over-excited Jamaican sports commentator calling the play like it's the Olympic 100m final",
}


class PlanSegment(BaseModel):
    text: str = Field(description="Phonetically respelled Patois for this chunk, as it should be spoken")
    emphasis: int = Field(description="0 = normal, 1 = stressed, 2 = big punch, 3 = SHOUT")
    contour: Literal["flat", "rise", "fall", "rise_fall", "high", "low"]
    pause_after_ms: int = Field(description="Comic-timing pause after this chunk, 0-900 ms")


class DeliveryPlan(BaseModel):
    patois: str = Field(description="The full Patois line")
    segments: list[PlanSegment]


SYSTEM_PROMPT = """You are the voice director for Bombovoice, a comedy voice changer. \
A user is talking into their microphone with friends (games, Discord, streams) and wants to \
sound like an over-the-top, lovable Jamaican man. You receive what they just said (a speech-to-text \
transcript) and rewrite it as Jamaican Patois-flavoured English, then plan how to perform it.

Rules for the rewrite:
- Keep the speaker's meaning and keep it understandable to non-Jamaicans; it is a gag, not a puzzle.
- Spell it the common written-Patois way, sticking to familiar spellings like these (the app converts \
them into sounds the voice can pronounce, so do not invent unusual new spellings): \
"di" (the), "dem" (them/they/plural), "mi" (I/me/my), "yuh" (you), "tink" (think), "ting" (thing), \
"likkle", "pickney", "bredda", "bredren", "nyam", "waan", "haffi", "cyaan", "neva", "nuff", "deh", \
"wah gwaan", "bwoy", "irie", "seen?", "ya mon", "mek", "tek", "fi", "pon", "nutt'n", "sinting", "overstand".
- Use Patois grammar: drop the copula ("mi tired"), base-form past tense ("mi go yesterday"), \
"a" for -ing ("mi a come"), plural "dem" ("di car dem"), "fi" infinitives.
- Add comedic flair sized to the intensity level: openers ("Bwoy,", "Yow!", "Lawd have mercy,"), \
tags ("seen?", "ya mon", "fi real", "yuh zimmi?"), dramatic repetition ("big big", "likkle likkle").
- Never write words in ALL CAPS (the voice would spell them out); show excitement with "!" instead.
- Keep it affectionate. Never add drugs, crime, slurs or anything mean-spirited that the speaker did not say.
- Profanity: only use Jamaican "claat" exclamations (bomboclaat, raasclaat, bloodclaat) when SPICY is on.
- Keep the output roughly the same length as the input, at most about 2x plus a short opener/tag.

Rules for the delivery plan (Jamaican intonation is wide and musical):
- Split into short chunks (2-8 words) at natural breath points.
- Statements rise through the middle and crash down at the end (rise_fall on the final chunk).
- Questions swoop up ("rise"). Exclamations go "high". Asides and grumbles go "low".
- Put emphasis on the punchline word, intensifiers and exclamations - Jamaican stress lands late in the phrase.
- Use pause_after_ms for comic timing (bigger pause before a punchline or after "Bwoy...").
"""


# JSON schema for providers that take a raw schema (OpenRouter). Kept flat and strict-mode friendly.
PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["patois", "segments"],
    "properties": {
        "patois": {"type": "string", "description": "The full Patois line"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "emphasis", "contour", "pause_after_ms"],
                "properties": {
                    "text": {"type": "string", "description": "Phonetically respelled Patois for this chunk"},
                    "emphasis": {"type": "integer", "description": "0 = normal, 1 = stressed, 2 = big punch, 3 = SHOUT"},
                    "contour": {"type": "string", "enum": ["flat", "rise", "fall", "rise_fall", "high", "low"]},
                    "pause_after_ms": {"type": "integer", "description": "Comic-timing pause after this chunk, 0-900"},
                },
            },
        },
    },
}

JSON_INSTRUCTION = (
    "\n\nReply with ONLY a JSON object (no markdown, no commentary) of the form "
    '{"patois": "...", "segments": [{"text": "...", "emphasis": 0, "contour": "rise_fall", "pause_after_ms": 200}]}. '
    "contour is one of flat, rise, fall, rise_fall, high, low."
)


class BaseDirector:
    """Shared prompt building, plan validation and offline fallback."""

    provider = "base"

    def __init__(self, model: str, intensity: int = 2, spicy: bool = False, style: str = "yaadie", effort: str = "low"):
        self.model = model
        self.intensity = intensity
        self.spicy = spicy
        self.style = style if style in STYLES else "yaadie"
        self.effort = effort
        self.fallback = PatoisTranslator(intensity=intensity, spicy=spicy)

    def describe(self) -> str:
        return f"Claude AI director ({self.provider}: {self.model})"

    def translate(self, text: str) -> Translation:
        text = text.strip()
        if not text:
            return Translation(original=text, patois="", segments=[], source="ai")
        try:
            return self._to_translation(text, self._plan(text))
        except Exception as exc:  # network down, no key, refusal... the show must go on
            log.warning("AI director unavailable (%s) - using offline translator", exc)
            self.fallback.intensity, self.fallback.spicy = self.intensity, self.spicy
            return self.fallback.translate(text)

    def _plan(self, text: str) -> DeliveryPlan:
        raise NotImplementedError

    def _user_prompt(self, text: str) -> str:
        style = STYLES.get(self.style, STYLES["yaadie"])
        return (
            f"INTENSITY: {self.intensity} ({INTENSITY_NAMES.get(self.intensity, '?')}; 0 = light touch, 3 = absolutely maximum)\n"
            f"SPICY: {'on' if self.spicy else 'off'}\n"
            f"CHARACTER: {style}\n\n"
            f"The speaker said:\n<transcript>\n{text}\n</transcript>"
        )

    @staticmethod
    def _to_translation(text: str, plan: DeliveryPlan | None) -> Translation:
        if plan is None or not plan.segments:
            raise RuntimeError("no delivery plan returned")
        segments = [
            Segment(
                text=s.text.strip(),
                emphasis=max(0, min(3, s.emphasis)),
                contour=s.contour,
                pause_after_ms=max(0, min(900, s.pause_after_ms)),
            )
            for s in plan.segments
            if s.text.strip()
        ]
        if not segments:
            raise RuntimeError("delivery plan had no text")
        return Translation(original=text, patois=plan.patois.strip(), segments=segments, source="ai")


class AIDirector(BaseDirector):
    """Claude via the Anthropic API (official SDK)."""

    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, intensity: int = 2, spicy: bool = False, style: str = "yaadie",
                 effort: str = "low", timeout: float = 20.0):
        super().__init__(model, intensity, spicy, style, effort)
        import anthropic  # imported lazily so the offline mode never needs it

        self.client = anthropic.Anthropic(timeout=timeout, max_retries=1)

    def _plan(self, text: str) -> DeliveryPlan:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=4000,
            # Latency-sensitive: the user is waiting to hear themselves, so keep thinking light.
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._user_prompt(text)}],
            output_format=DeliveryPlan,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("request declined")
        return response.parsed_output


class OpenRouterDirector(BaseDirector):
    """Claude via OpenRouter's chat-completions API (no extra packages needed)."""

    provider = "openrouter"

    def __init__(self, model: str = DEFAULT_OPENROUTER_MODEL, intensity: int = 2, spicy: bool = False,
                 style: str = "yaadie", effort: str = "low", timeout: float = 20.0, api_key: str | None = None):
        super().__init__(model, intensity, spicy, style, effort)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.timeout = timeout
        self._simple = False  # set when the model rejects structured-output / reasoning options

    def _body(self, text: str) -> dict:
        body = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + JSON_INSTRUCTION},
                {"role": "user", "content": self._user_prompt(text)},
            ],
        }
        if not self._simple:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "delivery_plan", "strict": True, "schema": PLAN_SCHEMA},
            }
            # keep thinking light - the user is waiting to hear the line
            body["reasoning"] = {"effort": self.effort}
        return body

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ZachAttack07/bombovoice",
                "X-Title": "Bombovoice",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _plan(self, text: str) -> DeliveryPlan:
        try:
            data = self._post(self._body(text))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code == 400 and not self._simple:
                # some routes don't accept json_schema / reasoning: retry once with a plain prompt
                log.info("OpenRouter rejected structured options (%s); retrying without them", detail)
                self._simple = True
                data = self._post(self._body(text))
            else:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if message.get("refusal") or choice.get("finish_reason") == "refusal":
            raise RuntimeError("request declined")
        return parse_plan(message.get("content") or "")


def parse_plan(content) -> DeliveryPlan:
    """Pull the JSON plan out of a model reply (tolerates code fences or stray text)."""
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("reply contained no JSON plan")
    return DeliveryPlan.model_validate_json(text[start:end + 1])


def resolve_provider(provider: str = "auto") -> str:
    if provider in ("anthropic", "openrouter"):
        return provider
    return "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "anthropic"


def make_translator(use_ai: bool, provider: str = "auto", model: str | None = None,
                    openrouter_model: str | None = None, **kwargs):
    """Return an AI director when possible, else the offline translator."""
    if use_ai:
        try:
            if resolve_provider(provider) == "openrouter":
                return OpenRouterDirector(model=openrouter_model or DEFAULT_OPENROUTER_MODEL, **kwargs)
            return AIDirector(model=model or DEFAULT_MODEL, **kwargs)
        except Exception as exc:
            log.warning("Could not start AI director (%s); using offline translator", exc)
    return PatoisTranslator(intensity=kwargs.get("intensity", 2), spicy=kwargs.get("spicy", False))


def translator_from_settings(s, use_ai: bool | None = None):
    return make_translator(s.use_ai if use_ai is None else use_ai, provider=s.ai_provider, model=s.model,
                           openrouter_model=s.openrouter_model, intensity=s.intensity, spicy=s.spicy,
                           style=s.style, effort=s.effort)


def describe(translator) -> str:
    return translator.describe() if isinstance(translator, BaseDirector) else "offline Patois rules"
