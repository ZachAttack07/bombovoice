import numpy as np
import pytest

from bombovoice.dsp import DubEcho, Limiter, PitchShifter, SpectralShifter, VoiceChain, apply_contour

SR = 48000


def _peak_hz(x, sr=SR):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)]


def _run(shifter, x, st, block=256):
    return np.concatenate([shifter.process(x[i:i + block], st) for i in range(0, len(x), block)])


@pytest.mark.parametrize("semitones", [-12, -3, 5])
def test_spectral_shifter_hits_target_pitch(semitones):
    t = np.arange(SR * 3) / SR
    x = 0.4 * np.sin(2 * np.pi * 220 * t)
    sh = SpectralShifter(SR, preserve_formants=False, formant_semitones=0)
    y = _run(sh, x, semitones)
    assert len(y) == len(x)
    assert abs(_peak_hz(y[SR:]) - 220 * 2 ** (semitones / 12)) < 2.0


@pytest.mark.parametrize("block", [128, 256, 700, 1024])
def test_spectral_shifter_constant_latency(block):
    x = np.zeros(SR)
    x[9000:9600] = np.hanning(600)
    sh = SpectralShifter(SR, preserve_formants=False, formant_semitones=0)
    y = _run(sh, x, 0.0, block)
    lag = np.argmax(np.correlate(y, x, "full")) - (len(x) - 1)
    assert lag == sh.latency


def test_delay_line_shifter_moves_pitch_down():
    t = np.arange(SR * 2) / SR
    x = 0.4 * np.sin(2 * np.pi * 440 * t)
    y = _run(PitchShifter(SR), x, -12)
    assert _peak_hz(y[SR // 2:]) < 300


def test_voice_chain_is_stable_and_bounded():
    rng = np.random.default_rng(0)
    chain = VoiceChain(SR, echo=True, rasp=0.6, seed=1)
    t = np.arange(SR * 2) / SR
    # a fake "voice": buzzy tone with syllable-like loudness bumps
    x = (0.5 * np.sign(np.sin(2 * np.pi * 120 * t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) ** 2
         + 0.01 * rng.normal(size=len(t))).astype(np.float32)
    y = np.concatenate([chain.process(x[i:i + 256]) for i in range(0, len(x), 256)])
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1.0
    assert np.std(y[SR // 2:]) > 0.01


def test_lilt_actually_moves_pitch():
    chain = VoiceChain(SR, base_pitch=0.0, lilt=1.5, seed=2)
    t = np.arange(SR * 2) / SR
    x = (0.3 * np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    curves = [chain.lilt_engine.process(x[i:i + 256]) for i in range(0, len(x), 256)]
    curve = np.concatenate(curves)
    assert curve.max() - curve.min() > 1.0  # at least a semitone of sing-song swing


def test_echo_and_limiter():
    echo = DubEcho(SR, time_ms=100, feedback=0.5, mix=1.0)
    imp = np.zeros(SR)
    imp[0] = 1.0
    y = np.concatenate([echo.process(imp[i:i + 256]) for i in range(0, len(imp), 256)])
    d = int(SR * 0.1)
    assert np.max(np.abs(y[d - 50:d + 400])) > 0.05  # first repeat arrives
    lim = Limiter(SR, ceiling_db=-1.0)
    assert np.max(np.abs(lim.process(np.ones(512) * 3.0))) <= 1.0


def test_apply_contour_keeps_length():
    x = 0.3 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)
    for contour in ("rise", "fall", "rise_fall", "high", "low", "flat"):
        y = apply_contour(x, SR, contour)
        assert len(y) == len(x) and np.isfinite(y).all()
