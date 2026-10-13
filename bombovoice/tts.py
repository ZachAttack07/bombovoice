"""Text-to-speech that *performs* a delivery plan.

The whole line is spoken in one take (so it flows like real speech) using
"speakable" spellings that the voice pronounces instead of spelling out, then a
gentle Jamaican intonation curve is laid over it with PSOLA (see prosody.py).

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

from .dsp import normalize, resample
from .patois import Segment, Translation
from .pronounce import speakable
from .prosody import apply_lilt

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

    def __init__(self, sample_rate: int = 48000, base_rate: int = 4, base_pitch_hz: int = -2):
        self.sr = sample_rate
        self.base_rate = base_rate  # % faster than normal - Jamaican talk is quick and rhythmic
        self.base_pitch_hz = base_pitch_hz

    # -- to implement -------------------------------------------------------
    def synth(self, text: str, rate: int = 0, pitch_hz: int = 0, volume: int = 0) -> np.ndarray:
        raise NotImplementedError

    def synth_many(self, jobs: list[tuple[str, int, int, int]]) -> list[np.ndarray]:
        return [self.synth(*job) for job in jobs]

    # -- shared -------------------------------------------------------------
    @staticmethod
    def script(segments: list[Segment]) -> tuple[str, list[float]]:
        """One TTS-ready string for the whole line, plus how much of it each segment takes."""
        parts, weights = [], []
        for seg in segments:
            text = speakable(seg.text)
            if not text:
                continue
            if seg.pause_after_ms >= 350 and text[-1] not in ".!?":
                text += "..."  # a beat for comic timing
            parts.append(text)
            weights.append(len(text) + seg.pause_after_ms / 60.0)
        return " ".join(parts), weights

    def perform(self, translation: Translation | list[Segment], lilt: float = 1.0,
                formant: float = 0.0, deepen: float = 0.0) -> np.ndarray:
        """Render a whole translation (or list of segments) to one clip at self.sr.

        `formant` is accepted for compatibility; PSOLA keeps the voice's natural formants.
        """
        segments = translation.segments if isinstance(translation, Translation) else translation
        segments = [s for s in segments if s.text.strip()]
        text, weights = self.script(segments)
        if not text:
            return np.zeros(0, dtype=np.float32)
        audio = _trim_silence(self.synth(text, self.base_rate, self.base_pitch_hz, 0), self.sr)
        audio = apply_lilt(audio, self.sr, [s for s in segments if speakable(s.text)], weights,
                           lilt=lilt, base=deepen)
        return normalize(audio, -2)

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
        # (pyttsx3 has no pitch control; overall pitch is handled by the PSOLA intonation pass)
        return resample(audio.mean(axis=1), sr, self.sr)


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


# ---------------------------------------------------------------------------
# Out-of-process rendering
# ---------------------------------------------------------------------------
# Speech synthesis + PSOLA are heavy and hold Python's GIL; done in a thread they
# starve the real-time audio callbacks and the live voice crackles. So the engine
# renders speech in a separate worker process instead.

_worker_tts: BaseTTS | None = None


def _worker_init(backend: str, voice: str | None, sample_rate: int):
    global _worker_tts
    logging.basicConfig(level=logging.WARNING)
    _worker_tts = make_tts(backend, voice, sample_rate)


def _worker_name() -> str:
    return _worker_tts.name


def _worker_job(kind: str, payload, kwargs: dict) -> np.ndarray:
    if kind == "perform":
        return _worker_tts.perform(payload, **kwargs)
    return _worker_tts.say(payload, **kwargs)


class TTSProcess:
    """Same interface as a TTS backend (name / perform / say) but runs in its own process."""

    def __init__(self, backend: str = "auto", voice: str | None = None, sample_rate: int = 48000,
                 timeout: float = 120.0):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        self.sr = sample_rate
        self.timeout = timeout
        self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=_worker_init, initargs=(backend, voice, sample_rate))
        try:
            self.name = self._pool.submit(_worker_name).result(timeout=90)
        except Exception as exc:
            self._pool.shutdown(wait=False, cancel_futures=True)
            raise TTSError(f"could not start the speech worker: {exc}") from exc

    def perform(self, translation, **kwargs) -> np.ndarray:
        return self._pool.submit(_worker_job, "perform", translation, kwargs).result(timeout=self.timeout)

    def say(self, text: str, **kwargs) -> np.ndarray:
        return self._pool.submit(_worker_job, "say", text, kwargs).result(timeout=self.timeout)

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


def make_tts_process(backend: str = "auto", voice: str | None = None, sample_rate: int = 48000):
    """Speech in a worker process when possible (keeps the live voice glitch-free)."""
    try:
        return TTSProcess(backend, voice, sample_rate)
    except TTSError as exc:
        log.info("speech worker unavailable (%s); rendering in-process", exc)
        return make_tts(backend, voice, sample_rate)
