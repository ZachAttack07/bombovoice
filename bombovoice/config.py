"""Settings, persisted as JSON in your user config folder."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

MODES = {
    "voice": "Live voice - your own voice, deeper, with Jamaican sing-song lilt and stress (real-time)",
    "ai": "AI Patois - speech-to-text -> Claude rewrites it in Patois -> Jamaican TTS performs it",
    "hybrid": "Hybrid - live Jamaican voice + an AI hype man who repeats your punchlines in Patois",
    "bypass": "Bypass - plain microphone",
}

DEFAULT_HOTKEYS = {
    "airhorn": "<f1>",
    "siren": "<f2>",
    "rewind": "<f3>",
    "bassdrop": "<f4>",
    "riddim": "<f5>",
    "adlib": "<f6>",
    "echo": "<f7>",
    "mode": "<f8>",
    "mute": "<f9>",
    "intensity": "<f10>",
    "laser": "<f11>",
}


SETTINGS_VERSION = 2
# voice settings whose meaning/defaults changed in v2 (natural PSOLA engine) - reset once on upgrade
_RESET_ON_UPGRADE = {2: {"rasp", "warmth_db", "quality"}}


def config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "bombovoice" / "settings.json"


@dataclass
class Settings:
    # audio routing
    input_device: str | int | None = None  # None = system default microphone
    output_device: str | int | None = None  # None = auto-detect virtual cable
    monitor: bool = False  # also play the result in your headphones
    monitor_device: str | int | None = None
    sample_rate: int = 48000
    block_size: int = 256

    # what you sound like
    mode: str = "voice"
    intensity: int = 2  # 0 Tourist .. 3 MAXIMUM BOMBO
    spicy: bool = False  # allow "bomboclaat"-grade exclamations
    style: str = "yaadie"  # AI character: yaadie, dancehall, rasta, auntie, commentator
    base_pitch: float = -2.5  # semitones; negative = deeper
    formant: float = -1.5  # semitones; negative = bigger chest / bigger man
    lilt: float = 1.0  # sing-song amount: 0 off, 1 authentic, 2 cartoon
    warmth_db: float = 3.0
    rasp: float = 0.1
    echo: bool = False
    quality: str = "natural"  # "natural" PSOLA (~40 ms), "pv" phase vocoder, "fast" delay line (~20 ms)
    mic_gain_db: float = 0.0

    # AI
    use_ai: bool = True
    ai_provider: str = "auto"  # auto (OpenRouter if OPENROUTER_API_KEY is set), openrouter, anthropic
    openrouter_model: str = "anthropic/claude-haiku-5.5"
    model: str = "claude-opus-5-5"  # used with the Anthropic API directly
    effort: str = "low"
    whisper_model: str = "base.en"
    hype_man_chance: float = 0.5  # hybrid mode: chance the hype man reacts to a sentence

    # text to speech
    tts_backend: str = "auto"  # auto, edge, pyttsx3
    tts_voice: str | None = None
    tts_lilt: float = 1.0
    tts_pitch: float = -1.0  # extra semitones on the TTS voice

    # sound system
    riddim_volume: float = 0.3
    riddim_bpm: float = 76.0
    sfx_volume: float = 0.8
    auto_adlib: bool = False
    auto_adlib_seconds: float = 25.0

    hotkeys: dict = field(default_factory=lambda: dict(DEFAULT_HOTKEYS))
    settings_version: int = SETTINGS_VERSION

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or config_path()
        s = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                old_version = int(data.get("settings_version", 1))
                for version, keys in _RESET_ON_UPGRADE.items():
                    if old_version < version:
                        for k in keys:
                            data.pop(k, None)
                data["settings_version"] = SETTINGS_VERSION
                names = {f.name for f in fields(cls)}
                for k, v in data.items():
                    if k in names:
                        setattr(s, k, v)
                s.hotkeys = {**DEFAULT_HOTKEYS, **(s.hotkeys or {})}
            except Exception as exc:
                log.warning("Could not read %s (%s); using defaults", path, exc)
        return s

    def save(self, path: Path | None = None) -> Path:
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path
