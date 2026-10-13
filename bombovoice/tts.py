"""Text-to-speech that *performs* a delivery plan.

Each prosody segment is synthesised separately with its own speed, pitch and
loudness (stressed words come out slower, higher and louder), then bent along
its Jamaican intonation contour with the phase-vocoder and stitched together
with comic-timing pauses.

Backends:
    edge     - Microsoft Edge neural voices (online, free, best quality)
    pyttsx3  - the voices installed on your PC (offline: SAPI5 / NSSpeech / eSpeak)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from .dsp import apply_contour, normalize, resample
from .patois import Segment, Translation

log = logging.getLogger(__name__)

# There is no Jamaican neural voice, but because the text is respelled
# phonetically these voices end up sounding the part. Deep West-African and
# Caribbean-adjacent voices work best; try a few with `bombovoice voices`.
EDGE_VOICES = [
    "en-NG-AbeoNeural",
    "en-KE-ChilembaNeural",
    "en-TZ-ElimuNeural",
    "en-ZA-LukeNeural",
    "en-GB-RyanNeural",
    "en-US-GuyNeural",
]
DEFAULT_EDGE_VOICE = EDGE_VOICES[0]

CACHE_DIR = Path(os.environ.get("BOMBOVOICE_CACHE", Path.home() / ".cache" / "bombovoice"))


class TTSError(RuntimeError):
    pass


class BaseTTS:
    name = "base"

    def __init__(self, sample_rate: int = 48000, base_rate: int = 6, base_pitch_hz: int = -4):
        self.sr = sample_rate
        self.base_rate = base_rate  # % faster than normal - Jamaican talk is quick and rhythmic
        self.base_pitch_hz = base_pitch_hz

    # -- to implement -------------------------------------------------------
    def synth(self, text: str, rate: int = 0, pitch_hz: int = 0, volume: int = 0) -> np.ndarray:
        raise NotImplementedError

    def synth_many(self, jobs: list[tuple[str, int, int, int]]) -> list[np.ndarray]:
        return [self.synth(*job) for job in jobs]

    # -- shared -------------------------------------------------------------
    def _params(self, seg: Segment) -> tuple[str, int, int, int]:
        e = seg.emphasis
        rate = self.base_rate - 6 * e  # stretch the punch words
        pitch = self.base_pitch_hz + 7 * e
        volume = min(0, -12 + 6 * e) if e < 3 else 0
        if seg.contour == "low":
            pitch -= 8
            rate += 4
        return seg.text, rate, pitch, volume

    def perform(self, translation: Translation | list[Segment], lilt: float = 1.0,
                formant: float = -1.0, deepen: float = 0.0) -> np.ndarray:
        """Render a whole translation (or list of segments) to one clip at self.sr."""
        segments = translation.segments if isinstance(translation, Translation) else translation
        segments = [s for s in segments if s.text.strip()]
        if not segments:
            return np.zeros(0, dtype=np.float32)
        clips = self.synth_many([self._params(s) for s in segments])
        out = []
        for seg, clip in zip(segments, clips):
            clip = _trim_silence(clip, self.sr)
            if lilt > 0 or deepen or formant:
                depth = lilt * (0.8 + 0.3 * seg.emphasis)
                clip = apply_contour(clip, self.sr, seg.contour, depth=depth, base=deepen, formant=formant)
            out.append(clip)
            out.append(np.zeros(int(self.sr * seg.pause_after_ms / 1000), dtype=np.float32))
        return normalize(np.concatenate(out), -2)

    def say(self, text: str, **kwargs) -> np.ndarray:
        return self.perform([Segment(text=text, emphasis=1, contour="rise_fall")], **kwargs)


def _trim_silence(x: np.ndarray, sr: int, thresh: float = 0.01) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > thresh)
    if len(idx) == 0:
        return x
    pad = int(sr * 0.02)
    return x[max(0, idx[0] - pad): idx[-1] + pad]


def _decode(data: bytes, target_sr: int) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), sr, target_sr)


class EdgeTTS(BaseTTS):
    name = "edge"

    def __init__(self, voice: str = DEFAULT_EDGE_VOICE, cache: bool = True, **kw):
        super().__init__(**kw)
        import edge_tts  # noqa: F401  (fail early if missing)

        self.voice = voice
        self.cache = cache

    def _key(self, text, rate, pitch, volume) -> Path:
        h = hashlib.sha1(f"{self.voice}|{text}|{rate}|{pitch}|{volume}".encode()).hexdigest()[:20]
        return CACHE_DIR / "edge" / f"{h}.mp3"

    async def _fetch(self, text: str, rate: int, pitch: int, volume: int) -> bytes:
        import edge_tts

        path = self._key(text, rate, pitch, volume)
        if self.cache and path.exists():
            return path.read_bytes()
        comm = edge_tts.Communicate(text, self.voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz", volume=f"{volume:+d}%")
        data = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                data.extend(chunk["data"])
        if not data:
            raise TTSError("edge-tts returned no audio")
        if self.cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(data))
        return bytes(data)

    def synth_many(self, jobs):
        async def run():
            return await asyncio.gather(*(self._fetch(*job) for job in jobs))

        try:
            blobs = asyncio.run(run())
        except RuntimeError as exc:
            if "event loop" not in str(exc):
                raise
            # called from inside a running loop (e.g. some GUIs): use a private one
            loop = asyncio.new_event_loop()
            try:
                blobs = loop.run_until_complete(run())
            finally:
                loop.close()
        return [_decode(b, self.sr) for b in blobs]

    def synth(self, text, rate=0, pitch_hz=0, volume=0):
        return self.synth_many([(text, rate, pitch_hz, volume)])[0]


class Pyttsx3TTS(BaseTTS):
    name = "pyttsx3"
    _lock = threading.Lock()

    def __init__(self, voice: str | None = None, **kw):
        super().__init__(**kw)
        import pyttsx3

        self._pyttsx3 = pyttsx3
        self.voice = voice
        with self._lock:
            eng = pyttsx3.init()
            self.default_rate = int(eng.getProperty("rate") or 180)
            self.voices = [(v.id, v.name) for v in eng.getProperty("voices")]
            eng.stop()

    def synth(self, text, rate=0, pitch_hz=0, volume=0):
        import soundfile as sf

        with self._lock:
            eng = self._pyttsx3.init()
            if self.voice:
                eng.setProperty("voice", self.voice)
            eng.setProperty("rate", int(self.default_rate * (1 + rate / 100)))
            eng.setProperty("volume", float(np.clip(1 + volume / 100, 0.1, 1.0)))
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                eng.save_to_file(text, path)
                eng.runAndWait()
                eng.stop()
                audio, sr = sf.read(path, dtype="float32", always_2d=True)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        audio = resample(audio.mean(axis=1), sr, self.sr)
        if pitch_hz:  # pyttsx3 has no pitch control; do it ourselves (~4 Hz per semitone for a male voice)
            from .dsp import SpectralShifter

            sh = SpectralShifter(self.sr, formant_semitones=0)
            padded = np.concatenate([audio, np.zeros(sh.latency)])
            audio = sh.process(padded, pitch_hz / 8.0)[sh.latency:]
        return audio


def make_tts(backend: str = "auto", voice: str | None = None, sample_rate: int = 48000) -> BaseTTS:
    order = ["edge", "pyttsx3"] if backend == "auto" else [backend]
    errors = []
    for name in order:
        try:
            if name == "edge":
                return EdgeTTS(voice=voice or DEFAULT_EDGE_VOICE, sample_rate=sample_rate)
            if name == "pyttsx3":
                return Pyttsx3TTS(voice=voice, sample_rate=sample_rate)
            errors.append(f"unknown backend {name!r}")
        except Exception as exc:  # missing package / no voices
            errors.append(f"{name}: {exc}")
    raise TTSError("No text-to-speech backend available (" + "; ".join(errors) + "). "
                   "Install one with: pip install edge-tts   or   pip install pyttsx3")
