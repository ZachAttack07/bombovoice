"""Natural-sounding Jamaican intonation for TTS lines.

The whole line is spoken in ONE text-to-speech call (so it flows like real speech),
then a single smooth pitch curve is laid over it: each phrase rises and falls, stressed
words get a late lift, and nothing jumps more than a couple of semitones. The curve is
applied with Praat's PSOLA (pitch-synchronous overlap-add), which moves pitch without
the metallic "robot" sound of FFT-based pitch shifters. If Praat isn't installed the
voice is left untouched rather than processed badly.
"""

from __future__ import annotations

import logging

import numpy as np

from .patois import Segment

log = logging.getLogger(__name__)

MAX_SEMITONES = 3.0

# gentle phrase shapes, t in 0..1, value in semitones (before the lilt amount)
SHAPES = {
    "flat": lambda t: 0.3 * np.sin(np.pi * t),
    "rise": lambda t: -0.3 + 1.6 * t ** 1.5,
    "fall": lambda t: 0.8 - 1.6 * t,
    "rise_fall": lambda t: 1.4 * np.sin(np.pi * np.minimum(1.0, t * 1.2)) - 0.8 * np.clip((t - 0.7) / 0.3, 0, 1),
    "high": lambda t: 0.9 + 0.4 * np.sin(np.pi * t),
    "low": lambda t: -0.8 + 0.2 * np.sin(np.pi * t),
}


def have_psola() -> bool:
    try:
        import parselmouth  # noqa: F401
    except Exception:
        return False
    return True


def lilt_curve(segments: list[Segment], weights: list[float], duration: float, lilt: float = 1.0,
               base: float = 0.0, step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Semitone offset over time for a line made of `segments`.

    `weights` says roughly how much of the line each segment takes (e.g. its text length).
    Returns (times, semitones).
    """
    times = np.arange(0.0, max(duration, step), step)
    curve = np.zeros_like(times)
    total = float(sum(weights)) or 1.0
    start = 0.0
    for seg, w in zip(segments, weights):
        span = duration * w / total
        mask = (times >= start) & (times < start + span) if span > 0 else np.zeros_like(times, bool)
        if mask.any():
            t = (times[mask] - start) / max(span, 1e-6)
            depth = lilt * (0.8 + 0.15 * seg.emphasis)
            shape = SHAPES.get(seg.contour, SHAPES["flat"])(t) * depth
            if seg.emphasis >= 2:  # Jamaican stress lands late and lifts
                shape += 0.6 * lilt * (seg.emphasis - 1) * np.exp(-((t - 0.72) ** 2) / (2 * 0.12 ** 2))
            curve[mask] = shape
        start += span
    # smooth over ~150 ms so phrases glide into each other instead of jumping
    k = max(1, int(0.15 / step))
    kernel = np.hanning(k + 2)[1:-1]
    curve = np.convolve(np.pad(curve, k, mode="edge"), kernel / kernel.sum(), mode="same")[k:-k]
    return times, np.clip(curve + base, -MAX_SEMITONES + min(base, 0), MAX_SEMITONES + max(base, 0))


def psola_pitch(audio: np.ndarray, sr: int, times: np.ndarray, semitones: np.ndarray) -> np.ndarray:
    """Re-pitch `audio` along a semitone curve with Praat's PSOLA."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
    manip = call(snd, "To Manipulation", 0.01, 60, 450)
    tier = call(manip, "Extract pitch tier")
    n = int(call(tier, "Get number of points"))
    if n == 0:
        return np.asarray(audio, dtype=np.float32)
    pts = [(call(tier, "Get time from index", i), call(tier, "Get value at index", i)) for i in range(1, n + 1)]
    call(tier, "Remove points between", 0.0, snd.duration)
    for t, f in pts:
        call(tier, "Add point", t, f * 2.0 ** (float(np.interp(t, times, semitones)) / 12.0))
    call([tier, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    y = np.asarray(out.values[0], dtype=np.float32)
    return y[: len(audio)] if len(y) >= len(audio) else np.pad(y, (0, len(audio) - len(y)))


def apply_lilt(audio: np.ndarray, sr: int, segments: list[Segment], weights: list[float],
               lilt: float = 1.0, base: float = 0.0) -> np.ndarray:
    if len(audio) == 0 or (lilt <= 0 and abs(base) < 1e-3) or not have_psola():
        return audio
    times, semis = lilt_curve(segments, weights, len(audio) / sr, lilt=lilt, base=base)
    try:
        return psola_pitch(audio, sr, times, semis)
    except Exception as exc:  # never lose the line over an effect
        log.info("intonation skipped: %s", exc)
        return audio
