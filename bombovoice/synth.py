"""Procedurally synthesised sound-system effects - no samples, no copyright.

Airhorn, dub siren, bass drop, tape rewind and a full one-drop reggae riddim,
all generated from oscillators and noise at start-up.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp import DubEcho, normalize


def _t(sr: int, seconds: float) -> np.ndarray:
    return np.arange(int(sr * seconds)) / sr


def _saw(phase: np.ndarray) -> np.ndarray:
    return 2.0 * (phase % 1.0) - 1.0


def _square(phase: np.ndarray, duty: float = 0.5) -> np.ndarray:
    return np.where((phase % 1.0) < duty, 1.0, -1.0)


def _env(n: int, sr: int, attack: float = 0.005, release: float = 0.05) -> np.ndarray:
    e = np.ones(n)
    a, r = max(1, int(sr * attack)), max(1, int(sr * release))
    e[:a] = np.linspace(0, 1, min(a, n))[: len(e[:a])]
    if r < n:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def _bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, [lo, min(hi, sr / 2 * 0.95)], "band", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def _phase_from_freq(freq: np.ndarray, sr: int) -> np.ndarray:
    return np.cumsum(freq) / sr


# ---------------------------------------------------------------------------
# One-shots
# ---------------------------------------------------------------------------


def airhorn(sr: int) -> np.ndarray:
    """BRAP BRAP BRAAAAAP - the dancehall air horn."""
    blasts = [(0.16, 0.06), (0.16, 0.06), (0.75, 0.0)]
    out = []
    for dur, gap in blasts:
        t = _t(sr, dur)
        f0 = 523.0 * (1 + 0.04 * np.exp(-t * 18))  # a little scoop at the start of each blast
        x = np.zeros_like(t)
        for detune in (0.0, -6.0, 5.0, 1200.0 * np.log2(1.5)):  # stack + a fifth on top
            ph = _phase_from_freq(f0 * 2 ** (detune / 1200), sr)
            x += _saw(ph) * (0.5 if detune > 100 else 1.0)
        x = _bandpass(x, sr, 400, 4500)
        x = np.tanh(x * 2.5)
        out.append(x * _env(len(x), sr, 0.004, 0.03))
        out.append(np.zeros(int(sr * gap)))
    return normalize(np.concatenate(out), -3)


def dub_siren(sr: int, seconds: float = 2.6) -> np.ndarray:
    """The classic sound-system siren: square wave swept by a fast LFO, rising, into dub echo."""
    t = _t(sr, seconds)
    lfo = _square(t * 7.0, 0.5) * 0.5 + 0.5  # wee-oo wee-oo
    sweep = 1.0 + 0.6 * (t / seconds)  # whole thing rises
    freq = (380 + 320 * lfo) * sweep
    x = _square(_phase_from_freq(freq, sr), 0.4) * 0.35
    x = signal.sosfilt(signal.butter(2, 3000, "low", fs=sr, output="sos"), x)
    x *= _env(len(x), sr, 0.01, 0.15)
    tail = np.zeros(int(sr * 1.6))
    echo = DubEcho(sr, time_ms=330, feedback=0.6, mix=0.7, tone_hz=1800)
    return normalize(echo.process(np.concatenate([x, tail])), -4)


def bass_drop(sr: int) -> np.ndarray:
    """BOOOOM - sub sweep for the big moment."""
    t = _t(sr, 1.8)
    freq = 35 + 160 * np.exp(-t * 3.5)
    x = np.sin(2 * np.pi * _phase_from_freq(freq, sr))
    x = np.tanh(x * 3.0) * np.exp(-t * 1.4)
    click = np.random.default_rng(7).normal(0, 1, len(t)) * np.exp(-t * 90) * 0.3
    return normalize(x + click, -2)


def laser(sr: int) -> np.ndarray:
    """Pew pew pew - for when the crowd goes wild."""
    out = []
    for _ in range(3):
        t = _t(sr, 0.18)
        freq = 2200 * np.exp(-t * 22) + 180
        x = _square(_phase_from_freq(freq, sr), 0.3) * np.exp(-t * 14)
        out += [x, np.zeros(int(sr * 0.05))]
    return normalize(np.concatenate(out), -6)


def rewind(sr: int, recent: np.ndarray | None = None) -> np.ndarray:
    """'PULL UP!' - tape rewind of whatever was just said (or a noise burst)."""
    if recent is None or len(recent) < sr // 4 or np.max(np.abs(recent)) < 1e-3:
        recent = np.random.default_rng(3).normal(0, 0.3, sr * 2)
        recent = _bandpass(recent, sr, 300, 3000)
    src = np.asarray(recent[-sr * 3:], dtype=np.float64)[::-1]  # play backwards
    # spin-up: the read speed accelerates from 1x to 4x
    dur = 1.1
    t = _t(sr, dur)
    speed = 1.0 + 3.0 * (t / dur) ** 1.5
    pos = np.cumsum(speed)
    pos = pos[pos < len(src) - 1]
    i = pos.astype(int)
    frac = pos - i
    x = src[i] * (1 - frac) + src[i + 1] * frac
    x *= _env(len(x), sr, 0.01, 0.08)
    # vinyl scratch on the end
    st = _t(sr, 0.25)
    scratch = _bandpass(np.random.default_rng(5).normal(0, 1, len(st)), sr, 800, 5000)
    scratch *= np.abs(np.sin(2 * np.pi * 9 * st)) * 0.4
    return normalize(np.concatenate([x, scratch]), -3)


# ---------------------------------------------------------------------------
# The riddim
# ---------------------------------------------------------------------------


def _kick(sr: int) -> np.ndarray:
    t = _t(sr, 0.45)
    f = 48 + 90 * np.exp(-t * 30)
    return np.sin(2 * np.pi * _phase_from_freq(f, sr)) * np.exp(-t * 7)


def _rim(sr: int, rng) -> np.ndarray:
    t = _t(sr, 0.12)
    tone = np.sin(2 * np.pi * 1700 * t) * np.exp(-t * 60)
    noise = _bandpass(rng.normal(0, 1, len(t)), sr, 1500, 6000) * np.exp(-t * 45)
    return tone * 0.6 + noise * 0.5


def _hat(sr: int, rng, open_: bool = False) -> np.ndarray:
    t = _t(sr, 0.25 if open_ else 0.06)
    noise = signal.sosfilt(signal.butter(2, 7000, "high", fs=sr, output="sos"), rng.normal(0, 1, len(t)))
    return noise * np.exp(-t * (12 if open_ else 70)) * 0.35


def _chord_skank(sr: int, freqs: list[float]) -> np.ndarray:
    """Short choppy offbeat guitar/organ chord - the reggae skank."""
    t = _t(sr, 0.16)
    x = sum(_saw(f * t) + 0.5 * _square(f * 2 * t) for f in freqs)
    x = _bandpass(x, sr, 500, 3500)
    return x * np.exp(-t * 28) * 0.25


def _bass_note(sr: int, f: float, dur: float) -> np.ndarray:
    t = _t(sr, dur)
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    return np.tanh(x * 1.3) * _env(len(t), sr, 0.01, 0.08)


def _place(buf: np.ndarray, clip: np.ndarray, start: int, gain: float = 1.0):
    end = min(len(buf), start + len(clip))
    if start < len(buf):
        buf[start:end] += clip[: end - start] * gain


def riddim(sr: int, bpm: float = 76.0, bars: int = 4, seed: int = 1) -> np.ndarray:
    """Generate a seamless one-drop loop: kick + rim on beat 3, skank on the offbeats,
    a rolling bassline and swung hi-hats. Am - G - F - G."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(sr * beat * 4 * bars)
    drums = np.zeros(n)
    skank = np.zeros(n)
    bass = np.zeros(n)

    def at(bar: int, beat_pos: float) -> int:
        return int(sr * beat * (bar * 4 + beat_pos))

    note = lambda midi: 440.0 * 2 ** ((midi - 69) / 12)  # noqa: E731
    chords = [[57, 60, 64], [55, 59, 62], [53, 57, 60], [55, 59, 62]]  # Am G F G
    roots = [45, 43, 41, 43]
    kick, rim = _kick(sr), _rim(sr, rng)
    for bar in range(bars):
        c = bar % 4
        # one drop: kick + rim together on 3 (beat index 2)
        _place(drums, kick, at(bar, 2.0), 0.9)
        _place(drums, rim, at(bar, 2.0), 0.6)
        if bar % 2 == 1:
            _place(drums, rim, at(bar, 3.5), 0.35)  # little fill
        for eighth in range(8):
            swing = 0.08 if eighth % 2 else 0.0
            _place(drums, _hat(sr, rng, open_=(eighth == 7)), at(bar, eighth / 2 + swing), 0.8 if eighth % 2 else 0.5)
        chord = _chord_skank(sr, [note(m) for m in chords[c]])
        for b in range(4):
            _place(skank, chord, at(bar, b + 0.5))
        # bassline: root - octave walk with space, classic roots-reggae feel
        r = roots[c]
        pattern = [(0.0, r, 0.7), (1.0, r + 7, 0.4), (1.5, r + 12, 0.35), (2.5, r + 10, 0.4), (3.0, r + 7, 0.6)]
        for pos, midi, dur in pattern:
            _place(bass, _bass_note(sr, note(midi), dur * beat), at(bar, pos), 0.7)

    mix = drums * 0.9 + skank * 0.8 + bass * 0.9
    return normalize(mix, -6)


SOUND_NAMES = {
    "airhorn": "Air horn (BRAP BRAP BRAAAP)",
    "siren": "Dub siren",
    "bassdrop": "Bass drop",
    "rewind": "Rewind / pull up",
    "laser": "Pew pew laser",
}


def build_soundboard(sr: int) -> dict[str, np.ndarray]:
    return {
        "airhorn": airhorn(sr).astype(np.float32),
        "siren": dub_siren(sr).astype(np.float32),
        "bassdrop": bass_drop(sr).astype(np.float32),
        "laser": laser(sr).astype(np.float32),
    }
