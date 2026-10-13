import numpy as np

from bombovoice import synth

SR = 24000


def test_soundboard_clips_are_sane():
    board = synth.build_soundboard(SR)
    assert set(board) == {"airhorn", "siren", "bassdrop", "laser"}
    for clip in board.values():
        assert clip.dtype == np.float32 and len(clip) > SR // 4
        assert np.isfinite(clip).all() and 0.1 < np.max(np.abs(clip)) <= 1.0


def test_riddim_loop_length_matches_tempo():
    bpm, bars = 80.0, 2
    loop = synth.riddim(SR, bpm=bpm, bars=bars)
    assert abs(len(loop) - SR * 60 / bpm * 4 * bars) < 2
    assert np.max(np.abs(loop)) <= 1.0


def test_rewind_uses_recent_audio_or_noise():
    recent = 0.5 * np.sin(2 * np.pi * 300 * np.arange(SR * 2) / SR)
    assert len(synth.rewind(SR, recent)) > SR // 2
    assert len(synth.rewind(SR, None)) > SR // 2
