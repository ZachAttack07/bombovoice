"""The Bombovoice engine: mic -> effects / AI -> mixer -> virtual microphone.

The audio path is split into `process_input()` and `render_output()` so it can
be driven by real sound devices (`start()`) or by files and tests.
"""

from __future__ import annotations

import logging
import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import synth
from .config import MODES, Settings
from .dsp import VoiceChain, db_to_gain
from .patois import AD_LIBS, INTENSITY_NAMES, SPICY_AD_LIBS, Translation

log = logging.getLogger(__name__)

EventFn = Callable[[str, object], None]


class Fifo:
    """Thread-safe jitter buffer between the mic callback and the output callback.

    The two sound devices never tick in perfect lock-step, so the output keeps a small
    cushion (`prefill`) of audio. If the cushion ever runs dry it fades out cleanly and
    re-fills (one tiny gap instead of constant crackle), and if dropouts keep happening
    it grows the cushion a little. If audio piles up, it trims back so latency can't creep.
    """

    def __init__(self, max_samples: int, prefill: int = 0, max_prefill: int | None = None):
        self.buf = np.zeros(0, dtype=np.float32)
        self.max = max_samples
        self.prefill = prefill
        self.max_prefill = max_prefill or prefill * 3
        self.primed = prefill == 0
        self.lock = threading.Lock()
        self.underflows = 0
        self.overflows = 0
        self._recent: list[float] = []

    def push(self, x: np.ndarray):
        with self.lock:
            self.buf = np.concatenate([self.buf, x.astype(np.float32)])
            if len(self.buf) > self.max:
                keep = max(self.prefill * 2, len(x))
                self.buf = self.buf[-keep:]
                self.overflows += 1

    def pull(self, n: int) -> np.ndarray:
        with self.lock:
            if not self.primed:
                if len(self.buf) < self.prefill + n:
                    return np.zeros(n, dtype=np.float32)
                self.primed = True
            if len(self.buf) >= n:
                out, self.buf = self.buf[:n], self.buf[n:]
                return out
            # ran dry: fade out what we have, then rebuild the cushion
            out = np.zeros(n, dtype=np.float32)
            have = self.buf
            if len(have):
                fade = np.linspace(1.0, 0.0, len(have), dtype=np.float32) if len(have) < 64 else np.concatenate(
                    [np.ones(len(have) - 64, dtype=np.float32), np.linspace(1.0, 0.0, 64, dtype=np.float32)])
                out[:len(have)] = have * fade
            self.buf = np.zeros(0, dtype=np.float32)
            self.underflows += 1
            self.primed = self.prefill == 0
            now = time.monotonic()
            self._recent = [t for t in self._recent if now - t < 10.0] + [now]
            if len(self._recent) >= 3 and self.prefill and self.prefill < self.max_prefill:
                self.prefill = min(self.max_prefill, self.prefill + max(64, self.prefill // 2))
                self._recent = []
            return out


class Mixer:
    """Plays one-shots (TTS lines, sound effects) and the riddim loop on top of the voice."""

    def __init__(self, sr: int):
        self.sr = sr
        self.lock = threading.Lock()
        self.voices: list[list] = []  # [clip, pos, gain, tag]
        self.loop: np.ndarray | None = None
        self.loop_pos = 0
        self.loop_on = False
        self.loop_gain = 0.3
        self.loop_hold = 0  # samples to stay silent (after a rewind) before restarting
        self.duck = 1.0

    def play(self, clip: np.ndarray, gain: float = 1.0, tag: str = "sfx", exclusive: bool = False):
        with self.lock:
            if exclusive:
                self.voices = [v for v in self.voices if v[3] != tag]
            self.voices.append([clip.astype(np.float32), 0, gain, tag])

    def stop(self, tag: str | None = None):
        with self.lock:
            self.voices = [v for v in self.voices if tag is not None and v[3] != tag]

    def busy(self, tag: str) -> bool:
        with self.lock:
            return any(v[3] == tag for v in self.voices)

    def render(self, n: int, voice_level: float = 0.0) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        with self.lock:
            alive = []
            for v in self.voices:
                clip, pos, gain, tag = v
                chunk = clip[pos:pos + n]
                out[:len(chunk)] += chunk * gain
                v[1] = pos + n
                if v[1] < len(clip):
                    alive.append(v)
            self.voices = alive
            tts_talking = any(v[3] in ("tts", "adlib") for v in self.voices)
            if self.loop is not None and self.loop_on:
                # sidechain duck the riddim under the voice so you're always heard
                talking = voice_level > 0.02 or tts_talking
                target = 0.45 if talking else 1.0
                self.duck += (target - self.duck) * (0.35 if target < self.duck else 0.04)
                if self.loop_hold > 0:
                    self.loop_hold -= n
                else:
                    idx = (self.loop_pos + np.arange(n)) % len(self.loop)
                    out += self.loop[idx] * self.loop_gain * self.duck
                    self.loop_pos = (self.loop_pos + n) % len(self.loop)
        return out


class Engine:
    def __init__(self, settings: Settings | None = None, on_event: EventFn | None = None):
        self.s = settings or Settings()
        self.sr = int(self.s.sample_rate)
        self.on_event = on_event or (lambda kind, payload: None)
        self.chain = VoiceChain(self.sr, base_pitch=self.s.base_pitch, lilt=self.s.lilt, warmth_db=self.s.warmth_db,
                                rasp=self.s.rasp, echo=self.s.echo, gain_db=self.s.mic_gain_db,
                                formant=self.s.formant, quality=self.s.quality)
        self.mixer = Mixer(self.sr)
        self.mixer.loop_gain = self.s.riddim_volume
        bs = int(self.s.block_size)
        self.out_fifo = Fifo(self.sr // 5, prefill=2 * bs, max_prefill=8 * bs)
        self.monitor_fifo = Fifo(self.sr // 5, prefill=2 * bs, max_prefill=8 * bs)
        self.history = np.zeros(self.sr * 4, dtype=np.float32)  # last 4 s of voice, for the rewind
        self.muted = False
        self.voice_level = 0.0
        self.sounds: dict[str, np.ndarray] = {}
        self.adlibs: dict[str, np.ndarray] = {}
        self.jobs: queue.Queue = queue.Queue()
        self.running = False
        self._streams = []
        self._threads: list[threading.Thread] = []
        self._detector = None
        self._transcriber = None
        self._translator = None
        self._tts = None
        self._tts_backup = None
        self._rvc_stage = None
        self._rvc_lock = threading.Lock()
        self._rvc_generation = 0
        self._rvc_overruns_seen = 0
        self._last_health_check = time.monotonic()
        self._last_adlib = time.monotonic()
        self._ai_ready = threading.Event()
        self._stt_error: str | None = None
        self._stt_lock = threading.Lock()

    # ------------------------------------------------------------------ events
    def emit(self, kind: str, payload: object = None):
        try:
            self.on_event(kind, payload)
        except Exception:  # never let the UI break the audio
            log.exception("event handler failed")

    def status(self, msg: str):
        log.info(msg)
        self.emit("status", msg)

    # ---------------------------------------------------------------- building
    def prepare(self):
        """Build sounds and load the AI pieces (call once; slow parts run in the background)."""
        self.sounds = synth.build_soundboard(self.sr)
        self.mixer.loop = synth.riddim(self.sr, bpm=self.s.riddim_bpm).astype(np.float32)
        t = threading.Thread(target=self._load_ai, name="bombovoice-ai-loader", daemon=True)
        t.start()
        self._threads.append(t)
        worker = threading.Thread(target=self._worker, name="bombovoice-worker", daemon=True)
        worker.start()
        self._threads.append(worker)

    def _load_ai(self):
        from .ai import describe, translator_from_settings
        from .stt import UtteranceDetector

        self._detector = UtteranceDetector(self.sr)
        self._translator = translator_from_settings(self.s)
        self.status(f"Translator: {describe(self._translator)}")
        try:
            self._build_tts()
        except Exception as exc:
            self.status(f"Text-to-speech unavailable: {exc}")
        if self.s.quality == "rvc":
            self._start_rvc()
        if self.s.mode in ("ai", "hybrid"):
            self._ensure_transcriber()
        self._ai_ready.set()
        self._prerender_adlibs()

    # ------------------------------------------------------------- voices
    def _build_tts(self):
        """(Re)create the speaking voice: normal TTS, or the cloned XTTS voice on the GPU."""
        from .tts import make_tts_process

        backend = self.s.tts_backend
        if backend == "xtts":
            self.status("Loading your cloned voice (XTTS)... the first time downloads ~1.8 GB, hold tight")
            try:
                tts = make_tts_process("xtts", None, self.sr, voice_clip=self.s.voice_clip,
                                       license_accepted=bool(self.s.xtts_license_accepted))
            except Exception as exc:
                self.status(f"Cloned voice unavailable ({str(exc)[:160]}) - using the normal voice")
                tts = make_tts_process("auto", self.s.tts_voice, self.sr)
        else:
            tts = make_tts_process(backend, self.s.tts_voice, self.sr)
        old, self._tts = self._tts, tts
        self.adlibs = {}
        if hasattr(old, "close"):
            old.close()
        self.status(f"Voice synth: {tts.name}")

    def _rebuild_tts(self):
        try:
            self._build_tts()
        except Exception as exc:
            self.status(f"Text-to-speech unavailable: {exc}")
            return
        self._prerender_adlibs()

    def _start_rvc(self):
        """Load the GPU AI voice in the background; the PSOLA voice covers you until it's ready."""
        if not self.s.rvc_model:
            self.status("AI voice: pick a voice model (.pth) first - using the Natural engine meanwhile")
            return
        with self._rvc_lock:
            self._rvc_generation += 1
            gen = self._rvc_generation
        threading.Thread(target=self._load_rvc, args=(gen,), name="bombovoice-rvc-loader", daemon=True).start()

    def _load_rvc(self, gen: int):
        try:
            from .rvc.stream import RVCStage, StreamingRVC, torch_info
        except ImportError as exc:
            self.status(f"AI voice needs the GPU pack - run setup_gpu.bat ({exc})")
            return
        info = torch_info()
        if info["torch"] is None:
            self.status("AI voice needs PyTorch - run setup_gpu.bat")
            return
        if info["error"]:
            self.status(info["error"])
        where = info["gpu"] or "CPU (too slow for live use - check setup_gpu.bat)"
        self.status(f"Loading AI voice on {where}... the first time downloads ~550 MB of base models")
        try:
            rvc = StreamingRVC(self.s.rvc_model, io_sr=self.sr, device=self.s.rvc_device, key=self.s.rvc_key,
                               block_time=self.s.rvc_block_ms / 1000.0, index_path=self.s.rvc_index or None,
                               index_rate=self.s.rvc_index_rate, protect=self.s.rvc_protect,
                               allow_pickle=bool(self.s.rvc_trust_model), hubert_path=self.s.rvc_hubert or None,
                               rmvpe_path=self.s.rvc_rmvpe or None)
            rvc.warmup()
        except Exception as exc:
            log.exception("AI voice failed to load")
            self.status(f"AI voice failed: {str(exc)[:200]} - staying on the Natural engine")
            return
        stage = RVCStage(rvc, lilt_engine=self.chain.lilt_engine)
        with self._rvc_lock:
            if gen != self._rvc_generation or self.s.quality != "rvc":
                stage.close()  # settings changed while loading
                return
            old, self._rvc_stage = self._rvc_stage, stage
            self._rvc_overruns_seen = 0
        if old is not None:
            old.close()
        self.status(f"AI voice LIVE on {rvc.device.upper()}: {Path(self.s.rvc_model).name} "
                    f"(delay about {int(rvc.latency_s * 1000) + 40} ms)")

    def _stop_rvc(self):
        with self._rvc_lock:
            self._rvc_generation += 1
            old, self._rvc_stage = self._rvc_stage, None
        if old is not None:
            old.close()

    def _check_health(self):
        """Every few seconds: warn if the GPU can't keep up with the live AI voice."""
        now = time.monotonic()
        if now - self._last_health_check < 5.0:
            return
        self._last_health_check = now
        stage = self._rvc_stage
        if stage is not None and stage.overruns > self._rvc_overruns_seen:
            self._rvc_overruns_seen = stage.overruns
            self.status(f"AI voice is falling behind ({stage.last_ms:.0f} ms per {self.s.rvc_block_ms} ms block) - "
                        "raise the AI voice block size or close other GPU apps")

    def _rebuild_translator(self):
        from .ai import describe, translator_from_settings

        self._translator = translator_from_settings(self.s)
        self.status(f"Translator: {describe(self._translator)}")

    def _ensure_transcriber(self) -> bool:
        with self._stt_lock:
            return self._load_transcriber()

    def _load_transcriber(self) -> bool:
        if self._transcriber is not None:
            return True
        if self._stt_error:
            return False
        try:
            from .stt import Transcriber

            self.status(f"Loading speech recognition ({self.s.whisper_model})...")
            self._transcriber = Transcriber(self.s.whisper_model)
            self.status("Speech recognition ready - talk and Bombovoice will translate yuh!")
            return True
        except Exception as exc:
            self._stt_error = str(exc)
            self.status(f"Speech recognition unavailable: {exc}")
            return False

    def _prerender_adlibs(self):
        if self._tts is None:
            return
        lines = AD_LIBS + (SPICY_AD_LIBS if self.s.spicy else [])
        time.sleep(3.0)  # let the audio settle first
        for line in lines:
            if line in self.adlibs:
                continue
            time.sleep(0.15)  # background work politely - the live voice comes first
            try:  # primary voice only - never cache the robotic fallback voice
                self.adlibs[line] = self._tts.say(line, **self._voice_kw())
            except Exception as exc:
                log.info("ad-lib pre-render stopped: %s", exc)
                return

    def _voice_kw(self) -> dict:
        return dict(lilt=self.s.tts_lilt, formant=self.s.formant, deepen=self.s.tts_pitch)

    def _backup_tts(self):
        """Edge TTS needs internet - keep an offline PC voice on standby for single failed lines."""
        if self._tts is None or self._tts.name not in ("edge", "xtts"):
            return None
        if self._tts.name == "edge" and self.s.tts_backend != "auto":
            return None
        if self._tts_backup is None:
            from .tts import make_tts_process

            try:
                self._tts_backup = make_tts_process("pyttsx3", None, self.sr)
            except Exception:
                return None
        return self._tts_backup

    def _render(self, kind: str, payload) -> np.ndarray:
        """Speak with the main voice; if it fails, say why and use the offline voice for this line only."""
        def run(tts):
            return tts.perform(payload, **self._voice_kw()) if kind == "perform" else tts.say(payload, **self._voice_kw())

        try:
            return run(self._tts)
        except Exception as exc:
            backup = self._backup_tts()
            if backup is None:
                raise
            self.status(f"Online voice failed ({exc.__class__.__name__}: {str(exc)[:120]}) - "
                        "using the offline PC voice for this line")
            return run(backup)

    # ------------------------------------------------------------------- audio
    def process_input(self, block: np.ndarray) -> np.ndarray:
        """Mic block in -> processed voice block (also feeds the AI detector)."""
        x = np.asarray(block, dtype=np.float32)
        self.voice_level = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        mode = self.s.mode
        if self.muted:
            y = np.zeros_like(x)
        elif mode == "bypass":
            y = x * db_to_gain(self.s.mic_gain_db)
        elif mode == "ai":
            y = np.zeros_like(x)  # your real voice is replaced by the AI performance
        elif self.s.quality == "rvc" and self._rvc_stage is not None:
            self._rvc_stage.feed(x)  # GPU AI voice runs on its own thread
            y = self.chain.post(self._rvc_stage.read(len(x)))
        else:
            y = self.chain.process(x)
        if mode in ("ai", "hybrid") and self._detector is not None and not self.muted:
            # don't listen to ourselves while the TTS is talking in AI mode
            if not (mode == "ai" and self.mixer.busy("tts")):
                utt = self._detector.feed(x)
                if utt is not None:
                    self.jobs.put(("utterance", utt))
        return y

    def render_output(self, voice: np.ndarray) -> np.ndarray:
        mix = np.clip(voice + self.mixer.render(len(voice), self.voice_level), -1.0, 1.0)
        n = len(mix)
        if n >= len(self.history):
            self.history = mix[-len(self.history):].copy()
        else:  # remember what just went out, so "rewind" can spin it back
            self.history[:-n] = self.history[n:]
            self.history[-n:] = mix
        self._maybe_auto_adlib()
        self._check_health()
        return mix

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Offline convenience: mic block in -> final output block."""
        return self.render_output(self.process_input(block))

    # --------------------------------------------------------------- streaming
    def start(self):
        import os

        from .devices import find_device, find_virtual_output, linux_virtual_sink, open_stream

        if not self.sounds:
            self.prepare()
        sys.setswitchinterval(0.001)  # let the audio callbacks grab the GIL quickly
        in_dev = find_device(self.s.input_device, "input")
        out_dev = find_device(self.s.output_device, "output") if self.s.output_device else find_virtual_output()
        pulse_sink = None
        if out_dev is None and self.s.output_device is None:
            pulse_sink = linux_virtual_sink()
            if pulse_sink:
                out_dev = find_device("pulse", "output")
        if out_dev is None:
            self.status("No virtual cable found - playing to your speakers. Run `bombovoice setup` to route into apps.")
        bs = int(self.s.block_size)

        def on_input(indata, frames, time_info, status):
            if status:
                log.debug("input status: %s", status)
            y = self.process_input(indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0].copy())
            self.out_fifo.push(y)

        def on_output(outdata, frames, time_info, status):
            mix = self.render_output(self.out_fifo.pull(frames))
            outdata[:] = mix.reshape(-1, 1).repeat(outdata.shape[1], axis=1)
            if self.s.monitor:
                self.monitor_fifo.push(mix)

        def on_monitor(outdata, frames, time_info, status):
            m = self.monitor_fifo.pull(frames)
            outdata[:] = m.reshape(-1, 1).repeat(outdata.shape[1], axis=1)

        opened = []
        try:
            st, in_dev, in_name = open_stream("input", in_dev, self.sr, bs, on_input)
            opened.append(st)
            if pulse_sink:  # the ALSA pulse plugin reads this when the stream opens
                os.environ["PULSE_SINK"] = pulse_sink
            try:
                st, out_dev, out_name = open_stream("output", out_dev, self.sr, bs, on_output)
            finally:
                if pulse_sink:
                    os.environ.pop("PULSE_SINK", None)
            opened.append(st)
            if self.s.monitor:
                mon = find_device(self.s.monitor_device, "output") if self.s.monitor_device else None
                try:
                    st, _, _ = open_stream("output", mon, self.sr, bs, on_monitor)
                    opened.append(st)
                except Exception as exc:  # monitoring is optional - keep going without it
                    self.status(f"Could not open monitor headphones ({exc.__class__.__name__}); continuing without")
            for st in opened:
                st.start()
        except Exception:
            for st in opened:
                try:
                    st.close()
                except Exception:
                    pass
            raise
        self._streams = opened
        self.running = True
        if pulse_sink:
            out_name = "Bombovoice virtual mic"
        self.status(f"LIVE: {in_name}  ->  {out_name}   [{self.s.mode} mode, {INTENSITY_NAMES[self.s.intensity]}]")

    def stop(self):
        self.running = False
        for st in self._streams:
            try:
                st.stop()
                st.close()
            except Exception:
                pass
        self._streams = []
        self.status("Stopped. Likkle more!")

    def close(self):
        self.stop()
        self._stop_rvc()
        self.jobs.put(("quit", None))
        for tts in (self._tts, self._tts_backup):
            if hasattr(tts, "close"):
                tts.close()

    # ------------------------------------------------------------------ worker
    def _worker(self):
        while True:
            kind, payload = self.jobs.get()
            if kind == "quit":
                return
            try:
                if kind == "utterance":
                    self._handle_utterance(payload)
                elif kind == "text":
                    self._perform_text(payload)
                elif kind == "adlib":
                    self._play_adlib(payload)
            except Exception as exc:
                log.exception("worker job failed")
                self.status(f"Oops: {exc}")

    def _handle_utterance(self, audio: np.ndarray):
        if not self._ai_ready.wait(timeout=60) or not self._ensure_transcriber():
            if self.s.mode == "hybrid":
                self._play_adlib(None)
            return
        text = self._transcriber.transcribe(audio, self.sr)
        if not text:
            return
        self.emit("heard", text)
        if self.s.mode == "hybrid":
            if random.random() > self.s.hype_man_chance:
                return
            text = self._hype_line(text)
        self._perform_text(text)

    @staticmethod
    def _hype_line(text: str) -> str:
        """The hype man repeats the best bit of what you said, third person, with gusto."""
        words = text.strip().rstrip(".!?").split()
        tail = " ".join(words[-6:]) if words else ""
        lead = random.choice(["Him seh", "Yuh hear dat? Him seh", "Bwoy! Him seh", "Lawd, him really seh"])
        return f"{lead} {tail}!"

    def _perform_text(self, text: str):
        if not self._ai_ready.wait(timeout=60):
            return
        self._translator.intensity = self.s.intensity
        self._translator.spicy = self.s.spicy
        if hasattr(self._translator, "style"):
            self._translator.style = self.s.style
        translation: Translation = self._translator.translate(text)
        self.emit("patois", translation)
        if self._tts is None:
            self.status("No TTS voice available - showing text only")
            return
        audio = self._render("perform", translation)
        self.mixer.play(audio, gain=1.0, tag="tts")

    def _play_adlib(self, line: str | None):
        pool = AD_LIBS + (SPICY_AD_LIBS if self.s.spicy else [])
        line = line or random.choice(pool)
        clip = self.adlibs.get(line)
        if clip is None and self._tts is not None:
            clip = self._render("say", line)
        self.emit("adlib", line)
        if clip is not None:
            self.mixer.play(clip, gain=0.9, tag="adlib")

    def _maybe_auto_adlib(self):
        if not self.s.auto_adlib:
            return
        now = time.monotonic()
        if now - self._last_adlib > self.s.auto_adlib_seconds * random.uniform(0.6, 1.4):
            self._last_adlib = now
            self.jobs.put(("adlib", None))

    # ----------------------------------------------------------------- actions
    def action(self, name: str):
        """Soundboard / hotkey actions. Safe to call from any thread."""
        vol = self.s.sfx_volume
        if name in self.sounds:
            self.mixer.play(self.sounds[name], gain=vol, tag=name, exclusive=True)
            self.emit("sfx", name)
        elif name == "rewind":
            self.mixer.stop("tts")
            clip = synth.rewind(self.sr, self.history)
            self.mixer.play(clip, gain=vol, tag="rewind", exclusive=True)
            if self.mixer.loop_on:  # "wheel an' come again" - restart the riddim from the top
                self.mixer.loop_pos = 0
                self.mixer.loop_hold = len(clip)
            self.emit("sfx", "rewind")
        elif name == "riddim":
            self.mixer.loop_on = not self.mixer.loop_on
            self.mixer.loop_pos = 0
            self.status(f"Riddim {'ON - selecta!' if self.mixer.loop_on else 'off'}")
        elif name == "adlib":
            self.jobs.put(("adlib", None))
        elif name == "echo":
            self.set(echo=not self.s.echo)
            self.status(f"Dub echo {'ON' if self.s.echo else 'off'}")
        elif name == "mute":
            self.muted = not self.muted
            self.status("Muted" if self.muted else "Unmuted")
        elif name == "mode":
            modes = [m for m in MODES if m != "bypass"]
            nxt = modes[(modes.index(self.s.mode) + 1) % len(modes)] if self.s.mode in modes else modes[0]
            self.set(mode=nxt)
            self.status(f"Mode: {nxt} - {MODES[nxt]}")
        elif name == "intensity":
            self.set(intensity=(self.s.intensity + 1) % 4)
            self.status(f"Intensity: {INTENSITY_NAMES[self.s.intensity]}")
        elif name == "spicy":
            self.set(spicy=not self.s.spicy)
            self.status(f"Spicy {'ON - bomboclaat!' if self.s.spicy else 'off'}")
        else:
            self.status(f"Unknown action {name!r}")

    def say(self, text: str):
        """Type something and have it performed in Patois."""
        if text.strip():
            self.jobs.put(("text", text))

    def set(self, **changes):
        """Change settings live."""
        for k, v in changes.items():
            setattr(self.s, k, v)
        c = self.chain
        if "base_pitch" in changes:
            c.set_base_pitch(self.s.base_pitch)
        if "lilt" in changes:
            c.set_lilt(self.s.lilt)
        if "warmth_db" in changes:
            c.set_warmth(self.s.warmth_db)
        if "formant" in changes:
            c.set_formant(self.s.formant)
        if "quality" in changes:
            c.set_quality(self.s.quality, self.s.formant)
            if self.s.quality == "rvc":
                self._start_rvc()
            else:
                self._stop_rvc()
        stage = self._rvc_stage
        if stage is not None:  # cheap AI voice tweaks apply instantly
            if "rvc_key" in changes:
                stage.rvc.key = float(self.s.rvc_key)
            if "rvc_index_rate" in changes:
                stage.rvc.index_rate = float(self.s.rvc_index_rate)
            if "rvc_protect" in changes:
                stage.rvc.protect = float(self.s.rvc_protect)
        reload_keys = {"rvc_model", "rvc_index", "rvc_block_ms", "rvc_device", "rvc_trust_model", "rvc_hubert", "rvc_rmvpe"}
        if reload_keys & changes.keys() and self.s.quality == "rvc":
            self._start_rvc()
        if {"tts_backend", "voice_clip", "xtts_license_accepted", "tts_voice"} & changes.keys() and self._ai_ready.is_set():
            threading.Thread(target=self._rebuild_tts, name="bombovoice-tts-loader", daemon=True).start()
        if "rasp" in changes:
            c.rasp = float(self.s.rasp)
        if "echo" in changes:
            c.echo_on = bool(self.s.echo)
            if not c.echo_on:
                c.echo.clear()
        if "mic_gain_db" in changes:
            c.gain = db_to_gain(self.s.mic_gain_db)
        if "riddim_volume" in changes:
            self.mixer.loop_gain = float(self.s.riddim_volume)
        if "spicy" in changes and self.s.spicy and self._tts is not None:
            threading.Thread(target=self._prerender_adlibs, daemon=True).start()
        rebuild = {"use_ai", "ai_provider", "openrouter_model", "model"}
        if rebuild & changes.keys() and self._ai_ready.is_set():
            threading.Thread(target=self._rebuild_translator, daemon=True).start()
        if changes.get("mode") in ("ai", "hybrid"):
            threading.Thread(target=self._ensure_transcriber, daemon=True).start()
