"""Claude-powered Patois translator and delivery director.

Where the rule-based translator swaps words, Claude rewrites the *whole line*
the way a hilarious Jamaican uncle would say it, then plans the delivery:
which words to punch, where the pitch swoops up and crashes down, where to
pause for comic timing. The plan comes back as structured segments that the
TTS renderer performs.

Needs `pip install anthropic` and credentials (ANTHROPIC_API_KEY or an
`ant auth login` profile). Falls back to the offline translator on any error.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from .patois import INTENSITY_NAMES, PatoisTranslator, Segment, Translation

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"

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
- Spell phonetically for an ENGLISH text-to-speech voice so it *pronounces* the accent: \
"di" (the), "dem" (them/they/plural), "mi" (I/me/my), "yuh" (you), "tink" (think), "ting" (thing), \
"likkle", "pickney", "bredda", "bredren", "nyam", "waan", "haffi", "cyaan", "neva", "nuff", "deh", \
"wah gwaan", "bwoy", "irie", "seen?", "ya mon", "mek", "tek", "fi", "pon", "nutt'n", "sinting", "overstand".
- Use Patois grammar: drop the copula ("mi tired"), base-form past tense ("mi go yesterday"), \
"a" for -ing ("mi a come"), plural "dem" ("di car dem"), "fi" infinitives.
- Add comedic flair sized to the intensity level: openers ("Bwoy,", "Yow!", "Lawd have mercy,"), \
tags ("seen?", "ya mon", "fi real", "yuh zimmi?"), dramatic repetition ("big big", "likkle likkle").
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


class AIDirector:
    """Rewrites lines with Claude. Thread-safe; one instance can be shared."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        intensity: int = 2,
        spicy: bool = False,
        style: str = "yaadie",
        effort: str = "low",
        timeout: float = 20.0,
    ):
        import anthropic  # imported lazily so the offline mode never needs it

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=1)
        self.model = model
        self.intensity = intensity
        self.spicy = spicy
        self.style = style if style in STYLES else "yaadie"
        self.effort = effort
        self.fallback = PatoisTranslator(intensity=intensity, spicy=spicy)

    def translate(self, text: str) -> Translation:
        text = text.strip()
        if not text:
            return Translation(original=text, patois="", segments=[], source="ai")
        try:
            return self._ask_claude(text)
        except Exception as exc:  # network down, no key, refusal... the show must go on
            log.warning("AI director unavailable (%s) - using offline translator", exc)
            self.fallback.intensity, self.fallback.spicy = self.intensity, self.spicy
            return self.fallback.translate(text)

    def _ask_claude(self, text: str) -> Translation:
        user = (
            f"INTENSITY: {self.intensity} ({INTENSITY_NAMES.get(self.intensity, '?')}; 0 = light touch, 3 = absolutely maximum)\n"
            f"SPICY: {'on' if self.spicy else 'off'}\n"
            f"CHARACTER: {STYLES[self.style]}\n\n"
            f"The speaker said:\n<transcript>\n{text}\n</transcript>"
        )
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=4000,
            # Latency-sensitive: the user is waiting to hear themselves, so keep thinking light.
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_format=DeliveryPlan,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("request declined")
        plan: DeliveryPlan | None = response.parsed_output
        if plan is None or not plan.segments:
            raise RuntimeError(f"no plan returned (stop_reason={response.stop_reason})")
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
        return Translation(original=text, patois=plan.patois.strip(), segments=segments, source="ai")


def make_translator(use_ai: bool, **kwargs):
    """Return an AIDirector when possible, else the offline translator."""
    if use_ai:
        try:
            return AIDirector(**kwargs)
        except Exception as exc:
            log.warning("Could not start AI director (%s); using offline translator", exc)
    return PatoisTranslator(intensity=kwargs.get("intensity", 2), spicy=kwargs.get("spicy", False))
