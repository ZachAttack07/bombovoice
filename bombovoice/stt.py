"""Speech detection and transcription for the AI modes."""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from .dsp import resample

log = logging.getLogger(__name__)


class UtteranceDetector:
    """Energy-based voice activity detection with an adaptive noise floor.

    Feed it mic blocks; it returns a finished utterance (numpy array) when you
    stop talking, otherwise None.
    """

    def __init__(self, sr: int, threshold_db: float = 12.0, hangover_ms: float = 650,
                 min_ms: float = 350, max_s: float = 14.0, preroll_ms: float = 250):
        self.sr = sr
        self.threshold = 10 ** (threshold_db / 20)  # how far above the noise floor counts as speech
        self.hangover = hangover_ms / 1000
        self.min_len = int(sr * min_ms / 1000)
        self.max_len = int(sr * max_s)
        self.preroll = deque(maxlen=max(1, int(preroll_ms / 10)))
        self.noise = 1e-3
        self.active = False
        self.silence = 0.0
        self.chunks: list[np.ndarray] = []
        self.length = 0
        self.level = 0.0  # for meters

    def feed(self, block: np.ndarray) -> np.ndarray | None:
        rms = float(np.sqrt(np.mean(np.square(block)) + 1e-12))
        self.level = rms
        dt = len(block) / self.sr
        speech = rms > max(self.noise * self.threshold, 0.006)
        if not self.active:
            # track the noise floor only while nobody is talking
            a = math.exp(-dt / 1.5)
            self.noise = max(1e-4, a * self.noise + (1 - a) * rms) if rms < self.noise * 4 else self.noise * 1.001
            self.preroll.append(block.copy())
            if speech:
                self.active = True
                self.silence = 0.0
                self.chunks = list(self.preroll)
                self.length = sum(len(c) for c in self.chunks)
            return None
        self.chunks.append(block.copy())
        self.length += len(block)
        self.silence = 0.0 if speech else self.silence + dt
        if self.silence >= self.hangover or self.length >= self.max_len:
            audio = np.concatenate(self.chunks)
            self.active, self.chunks, self.length = False, [], 0
            self.preroll.clear()
            if len(audio) >= self.min_len:
                return audio
        return None

    @property
    def talking(self) -> bool:
        return self.active


class Transcriber:
    """faster-whisper wrapper (runs locally, CPU is fine with the small models)."""

    def __init__(self, model: str = "base.en", device: str = "auto", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("AI modes need speech recognition: pip install faster-whisper") from exc
        log.info("Loading Whisper model %s (first run downloads it)...", model)
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception as exc:
            raise RuntimeError(f"Could not load Whisper model {model!r} (first run needs internet to download it): {exc}") from exc

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        audio16 = resample(audio, sr, 16000)
        segments, _info = self.model.transcribe(
            audio16, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        # Whisper hallucinates these on silence/noise
        if text.lower().strip(" .!") in {"", "you", "thanks for watching", "thank you for watching"}:
            return ""
        return text
