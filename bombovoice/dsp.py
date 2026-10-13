"""Real-time audio effects, all numpy/scipy, all block-based and stateful.

The star of the show is the `LiltEngine`: it listens to the loudness of your
voice and bends your pitch the way Jamaican English moves - a wide, musical,
sing-song range, late stress that jumps *up* on the punch syllable, and phrase
endings that swoop up then crash down. Everything runs on small blocks, so it
works live on a microphone.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def semitones_to_ratio(st):
    return np.power(2.0, np.asarray(st, dtype=np.float64) / 12.0)


# ---------------------------------------------------------------------------
# Pitch shifting
# ---------------------------------------------------------------------------


class PitchShifter:
    """Low-latency delay-line pitch shifter with per-sample variable ratio.

    Two read taps sweep through a short delay window at a rate set by the
    pitch ratio and are cross-faded with complementary sin^2 windows. Cheap,
    glitch-free, ~window/2 latency, and the ratio can change every sample -
    which is exactly what we need for the sing-song lilt.
    """

    def __init__(self, sample_rate: int, window_ms: float = 40.0):
        self.sr = sample_rate
        self.window = max(64, int(sample_rate * window_ms / 1000.0))
        self.history = np.zeros(self.window + 4, dtype=np.float64)
        self.phase = 0.0

    def reset(self):
        self.history[:] = 0.0
        self.phase = 0.0

    def process(self, x: np.ndarray, semitones) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        if n == 0:
            return x.astype(np.float32)
        ratio = semitones_to_ratio(semitones)
        if ratio.ndim == 0:
            ratio = np.full(n, float(ratio))
        if np.allclose(ratio, 1.0, atol=1e-4) and self.phase == 0.0:
            self._push(x)
            return x.astype(np.float32)

        W = self.window
        # delay grows when pitching down (ratio < 1), shrinks when pitching up
        inc = (1.0 - ratio) / W
        phases = (self.phase + np.cumsum(inc)) % 1.0
        self.phase = float(phases[-1])

        buf = np.concatenate([self.history, x])
        H = len(self.history)
        idx = H + np.arange(n, dtype=np.float64)

        p1 = phases
        p2 = (phases + 0.5) % 1.0
        out = self._read(buf, idx - p1 * (W - 1) - 1) * np.sin(np.pi * p1) ** 2
        out += self._read(buf, idx - p2 * (W - 1) - 1) * np.sin(np.pi * p2) ** 2
        self._push(x)
        return out.astype(np.float32)

    @staticmethod
    def _read(buf: np.ndarray, pos: np.ndarray) -> np.ndarray:
        pos = np.clip(pos, 0, len(buf) - 2)
        i = pos.astype(np.int64)
        frac = pos - i
        return buf[i] * (1.0 - frac) + buf[i + 1] * frac

    def _push(self, x: np.ndarray):
        L = len(self.history)
        if len(x) >= L:
            self.history[:] = x[-L:]
        else:
            self.history[:-len(x)] = self.history[len(x):]
            self.history[-len(x):] = x


class SpectralShifter:
    """Phase-vocoder pitch shifter with independent formant control.

    Higher quality than `PitchShifter` (no warble), at the cost of ~one frame
    of latency (2048 samples = 43 ms at 48 kHz). The spectral envelope is
    estimated by cepstral smoothing, so pitch and formants move separately:
    drop the formants a little and you get a bigger, deeper chest without
    sounding like a slowed-down tape.
    """

    def __init__(self, sample_rate: int, frame: int = 2048, overlap: int = 4, formant_semitones: float = -1.5,
                 preserve_formants: bool = True):
        self.sr = sample_rate
        self.N = frame
        self.H = frame // overlap
        self.window = np.hanning(frame + 1)[:-1]
        self.bins = frame // 2 + 1
        self.k = np.arange(self.bins, dtype=np.float64)
        self.expected = 2 * np.pi * self.H / self.N * self.k
        self.formant_semitones = formant_semitones
        self.preserve = preserve_formants
        # output gain for hann analysis+synthesis at this overlap
        self.ola_gain = 1.0 / (np.sum(self.window ** 2) / self.H)
        self.lifter = max(8, int(self.N * 0.012))
        self.reset()

    def reset(self):
        self.in_fifo = np.zeros(self.N, dtype=np.float64)
        self.out_acc = np.zeros(self.N * 2, dtype=np.float64)
        self.pending = np.zeros(0, dtype=np.float64)
        self.ready = np.zeros(0, dtype=np.float64)
        self.last_phase = np.zeros(self.bins)
        self.sum_phase = np.zeros(self.bins)
        # prime the output so callers get a constant latency of exactly `frame` samples
        self.ready = np.zeros(self.H, dtype=np.float64)

    @property
    def latency(self) -> int:
        return self.N

    def _envelope(self, mag: np.ndarray) -> np.ndarray:
        log_mag = np.log(mag + 1e-9)
        cep = np.fft.irfft(log_mag)
        cep[self.lifter:-self.lifter] = 0.0
        return np.exp(np.fft.rfft(cep).real)

    def _frame(self, frame: np.ndarray, ratio: float) -> np.ndarray:
        X = np.fft.rfft(frame * self.window)
        mag, phase = np.abs(X), np.angle(X)
        dphi = phase - self.last_phase - self.expected
        self.last_phase = phase
        dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
        true_bin = self.k + dphi * self.N / (2 * np.pi * self.H)

        if self.preserve or self.formant_semitones:
            env = self._envelope(mag)
            flat = mag / (env + 1e-9)
        else:
            env, flat = None, mag

        target = np.round(self.k * ratio).astype(np.int64)
        valid = (target >= 0) & (target < self.bins)
        new_mag = np.zeros(self.bins)
        new_freq = np.zeros(self.bins)
        np.add.at(new_mag, target[valid], flat[valid])
        new_freq[target[valid]] = true_bin[valid] * ratio

        if env is not None:
            f_ratio = 2.0 ** (self.formant_semitones / 12.0)
            warped = np.interp(self.k / f_ratio, self.k, env, right=env[-1] * 0.1)
            new_mag *= warped
        # keep loudness steady while the pitch swings around
        e_in, e_out = float(np.sum(mag ** 2)), float(np.sum(new_mag ** 2))
        if e_out > 1e-12:
            new_mag *= min(4.0, max(0.25, math.sqrt(e_in / e_out)))

        self.sum_phase += 2 * np.pi * self.H / self.N * new_freq
        self.sum_phase = np.mod(self.sum_phase, 2 * np.pi)
        y = np.fft.irfft(new_mag * np.exp(1j * self.sum_phase), n=self.N)
        return y * self.window * self.ola_gain

    def process(self, x: np.ndarray, semitones) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        st = np.asarray(semitones, dtype=np.float64)
        self.pending = np.concatenate([self.pending, x])
        st_track = np.full(n, float(st)) if st.ndim == 0 else st
        produced = [self.ready]
        consumed = 0
        while len(self.pending) >= self.H:
            hop, self.pending = self.pending[:self.H], self.pending[self.H:]
            consumed += self.H
            self.in_fifo = np.concatenate([self.in_fifo[self.H:], hop])
            ratio = float(2.0 ** (st_track[min(consumed, n) - 1] / 12.0)) if n else 1.0
            self.out_acc[:self.N] += self._frame(self.in_fifo, ratio)
            produced.append(self.out_acc[:self.H].copy())
            self.out_acc = np.concatenate([self.out_acc[self.H:], np.zeros(self.H)])
        stream = np.concatenate(produced)
        out, self.ready = stream[:n], stream[n:]
        if len(out) < n:  # only during the very first calls
            out = np.concatenate([np.zeros(n - len(out)), out])
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# The Jamaican lilt
# ---------------------------------------------------------------------------


class LiltEngine:
    """Generates a per-sample pitch offset (in semitones) from the voice itself.

    - `base`: overall pitch shift (negative = deeper, bigger man).
    - `swing`: syllable-rate sing-song wobble depth (semitones).
    - `stress`: how hard stressed syllables jump up.
    - `phrase`: how much each phrase climbs and then falls at its end.
    """

    def __init__(self, sample_rate: int, base: float = -2.5, swing: float = 1.6,
                 stress: float = 2.8, phrase: float = 2.2, gate_db: float = -45.0, seed: int | None = None):
        self.sr = sample_rate
        self.base, self.swing, self.stress, self.phrase = base, swing, stress, phrase
        self.gate = db_to_gain(gate_db)
        self.rng = np.random.default_rng(seed)
        self.env_fast = 0.0
        self.env_slow = 0.0
        self.lfo_phase = 0.0
        self.lfo_rate = 3.3  # roughly syllable rate (Hz)
        self.stress_level = 0.0  # decaying "punch" in semitones
        self.speaking_time = 0.0  # seconds since the current phrase started
        self.silence_time = 1.0
        self.last_value = base
        self.phrase_target = 1.0  # random per phrase, so it doesn't sound robotic
        self.ending = 0.0  # 0..1 progress of the end-of-phrase swoop

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        dt = n / self.sr
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        # envelope followers (block rate)
        a_fast = math.exp(-dt / 0.012)
        a_slow = math.exp(-dt / 0.180)
        self.env_fast = a_fast * self.env_fast + (1 - a_fast) * rms
        self.env_slow = a_slow * self.env_slow + (1 - a_slow) * rms
        voiced = self.env_fast > self.gate

        if voiced:
            if self.silence_time > 0.25:  # a new phrase begins
                self.speaking_time = 0.0
                self.phrase_target = float(self.rng.uniform(0.6, 1.3))
                self.lfo_rate = float(self.rng.uniform(2.8, 4.2))
            self.speaking_time += dt
            self.silence_time = 0.0
            self.ending = max(0.0, self.ending - dt * 4)
            # onset = fast envelope jumping above the slow one -> a stressed syllable
            if self.env_fast > self.env_slow * 1.6 and self.env_fast > self.gate * 3:
                self.stress_level = max(self.stress_level, self.stress * min(1.0, self.env_fast / (self.env_slow * 3 + 1e-9)))
        else:
            self.silence_time += dt
            if self.silence_time < 0.35:
                self.ending = min(1.0, self.ending + dt * 5)

        # phrase arc: climbs over the first ~1.5 s of a phrase
        climb = self.phrase * self.phrase_target * min(1.0, self.speaking_time / 1.5)
        # end-of-phrase: quick swoop up then crash down below the starting pitch
        e = self.ending
        swoop = self.phrase * (1.2 * math.sin(math.pi * min(1.0, e * 1.6)) - 1.6 * e) if e > 0 else 0.0

        t = np.arange(1, n + 1) / self.sr
        lfo = np.sin(2 * np.pi * (self.lfo_phase + self.lfo_rate * t))
        self.lfo_phase = (self.lfo_phase + self.lfo_rate * dt) % 1.0
        gate = 1.0 if voiced else 0.3

        target = self.base + climb + swoop + self.stress_level + self.swing * gate * lfo
        # glide from the previous value to avoid zipper noise
        ramp = np.linspace(0.0, 1.0, n, endpoint=True)
        curve = self.last_value + (target - self.last_value) * np.minimum(1.0, ramp * 2.0)
        self.last_value = float(curve[-1])
        self.stress_level *= math.exp(-dt / 0.14)
        return curve


# ---------------------------------------------------------------------------
# Filters & effects
# ---------------------------------------------------------------------------


class SOSFilter:
    """Stateful second-order-sections filter."""

    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self.zi = signal.sosfilt_zi(sos) * 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


def peaking_eq(sr: int, freq: float, gain_db: float, q: float = 0.9) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return signal.tf2sos(np.array(b) / den[0], np.array(den) / den[0])


def low_shelf(sr: int, freq: float, gain_db: float) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    cw = np.cos(w0)
    b = [a * ((a + 1) - (a - 1) * cw + 2 * np.sqrt(a) * alpha), 2 * a * ((a - 1) - (a + 1) * cw),
         a * ((a + 1) - (a - 1) * cw - 2 * np.sqrt(a) * alpha)]
    den = [(a + 1) + (a - 1) * cw + 2 * np.sqrt(a) * alpha, -2 * ((a - 1) + (a + 1) * cw),
           (a + 1) + (a - 1) * cw - 2 * np.sqrt(a) * alpha]
    return signal.tf2sos(np.array(b) / den[0], np.array(den) / den[0])


class DubEcho:
    """Tape-style feedback delay with a darkening filter in the loop - the dub sound."""

    def __init__(self, sr: int, time_ms: float = 375.0, feedback: float = 0.55, mix: float = 0.45, tone_hz: float = 2200.0):
        self.sr = sr
        self.delay = max(1, int(sr * time_ms / 1000))
        self.feedback, self.mix = feedback, mix
        self.buf = np.zeros(self.delay * 2 + sr, dtype=np.float64)
        self.pos = 0
        self.lp = SOSFilter(signal.butter(2, tone_hz, "low", fs=sr, output="sos"))
        self.hp = SOSFilter(signal.butter(1, 180, "high", fs=sr, output="sos"))

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        # process in chunks no longer than the delay so feedback stays sample-accurate
        start = 0
        while start < len(x):
            chunk = x[start:start + self.delay]
            m = len(chunk)
            L = len(self.buf)
            read_idx = (self.pos - self.delay + np.arange(m)) % L
            delayed = self.buf[read_idx]
            wet = self.hp.process(self.lp.process(delayed))
            write_idx = (self.pos + np.arange(m)) % L
            self.buf[write_idx] = chunk + np.tanh(wet * self.feedback * 1.1)
            out[start:start + m] = chunk + wet * self.mix
            self.pos = (self.pos + m) % L
            start += m
        return out

    def clear(self):
        self.buf[:] = 0.0


class Limiter:
    """Simple look-ahead-free peak limiter so nothing clips on the virtual mic."""

    def __init__(self, sr: int, ceiling_db: float = -1.0, release_ms: float = 120.0):
        self.ceiling = db_to_gain(ceiling_db)
        self.release = math.exp(-1.0 / (sr * release_ms / 1000))
        self.gain = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        target = min(1.0, self.ceiling / peak) if peak > 0 else 1.0
        if target < self.gain:
            gains = np.linspace(self.gain, target, len(x))
            self.gain = target
        else:
            new = target + (self.gain - target) * self.release ** len(x)
            gains = np.linspace(self.gain, new, len(x))
            self.gain = new
        return np.clip(x * gains, -1.0, 1.0)


# ---------------------------------------------------------------------------
# The full live voice chain
# ---------------------------------------------------------------------------


class VoiceChain:
    """Mic in -> Jamaican man out. All parameters can be changed while running."""

    def __init__(self, sr: int, base_pitch: float = -2.5, lilt: float = 1.0, warmth_db: float = 4.0,
                 rasp: float = 0.25, echo: bool = False, gain_db: float = 0.0, formant: float = -1.5,
                 quality: str = "hq", seed: int | None = None):
        self.sr = sr
        # "hq" = phase vocoder with formant control (~45 ms), "fast" = delay line (~20 ms)
        self.quality = quality
        self.shifter = SpectralShifter(sr, formant_semitones=formant) if quality == "hq" else PitchShifter(sr)
        self.lilt_engine = LiltEngine(sr, base=base_pitch, seed=seed)
        self.echo = DubEcho(sr)
        self.echo_on = echo
        self.limiter = Limiter(sr)
        self.hp = SOSFilter(signal.butter(2, 70, "high", fs=sr, output="sos"))
        self.rasp = rasp
        self.gain = db_to_gain(gain_db)
        self.set_lilt(lilt)
        self.set_warmth(warmth_db)
        self.bypass = False

    def set_lilt(self, amount: float):
        """0 = no sing-song, 1 = authentic, 2 = cartoonishly musical."""
        amount = max(0.0, float(amount))
        self.lilt_amount = amount
        e = self.lilt_engine
        e.swing, e.stress, e.phrase = 1.6 * amount, 2.8 * amount, 2.2 * amount

    def set_base_pitch(self, semitones: float):
        self.lilt_engine.base = float(semitones)

    def set_formant(self, semitones: float):
        if isinstance(self.shifter, SpectralShifter):
            self.shifter.formant_semitones = float(semitones)

    def set_warmth(self, db: float):
        self.warmth_db = db
        sos = np.vstack([low_shelf(self.sr, 220, db), peaking_eq(self.sr, 2800, db * 0.5, q=1.2)])
        self.eq = SOSFilter(sos)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.bypass:
            return x.astype(np.float32)
        y = self.hp.process(x)
        curve = self.lilt_engine.process(y)
        y = self.shifter.process(y, curve).astype(np.float64)
        y = self.eq.process(y)
        if self.rasp > 0:  # a little gravel in the throat
            drive = 1.0 + 6.0 * self.rasp
            y = (np.tanh(y * drive) / np.tanh(drive)) * self.rasp + y * (1 - self.rasp)
        if self.echo_on:
            y = self.echo.process(y)
        y = self.limiter.process(y * self.gain)
        return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Offline helpers (used to perform TTS output with contours)
# ---------------------------------------------------------------------------


CONTOUR_SHAPES = {
    "flat": lambda t: 0.4 * np.sin(np.pi * t),
    "rise": lambda t: -0.5 + 3.5 * t ** 1.6,
    "fall": lambda t: 1.5 - 3.5 * t,
    "rise_fall": lambda t: 3.2 * np.sin(np.pi * np.minimum(1.0, t * 1.25)) - 1.8 * np.maximum(0.0, t - 0.6) / 0.4,
    "high": lambda t: 2.5 + 0.8 * np.sin(np.pi * t),
    "low": lambda t: -2.0 + 0.3 * np.sin(np.pi * t),
}


def contour_curve(n: int, contour: str, depth: float = 1.0, base: float = 0.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(1, n))
    shape = CONTOUR_SHAPES.get(contour, CONTOUR_SHAPES["flat"])
    return base + depth * shape(t)


def apply_contour(audio: np.ndarray, sr: int, contour: str, depth: float = 1.0, base: float = 0.0,
                  formant: float = 0.0) -> np.ndarray:
    """Bend a finished clip (e.g. one TTS phrase) along a Jamaican intonation contour."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) == 0:
        return audio.astype(np.float32)
    shifter = SpectralShifter(sr, frame=1024 if sr <= 24000 else 2048, formant_semitones=formant)
    lat = shifter.latency
    padded = np.concatenate([audio, np.zeros(lat)])
    curve = contour_curve(len(audio), contour, depth, base)
    curve = np.concatenate([np.full(lat, curve[0]), curve])  # align the curve with the shifter's latency
    hop = shifter.H
    out = np.concatenate([shifter.process(padded[i:i + hop], curve[i:i + hop]) for i in range(0, len(padded), hop)])
    return out[lat:lat + len(audio)].astype(np.float32)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def normalize(audio: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 1e-9:
        return audio
    return (audio * (db_to_gain(peak_db) / peak)).astype(np.float32)
