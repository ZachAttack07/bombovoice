import numpy as np
import soundfile as sf

from bombovoice import dataset as ds

SR = 48000


def _voice(seconds, f0=120.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    env = (np.sin(np.pi * np.clip(t / seconds, 0, 1)) > 0.05) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = 0.3 * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * env
    x = np.concatenate([np.zeros(SR // 4), x, np.zeros(SR // 4)])
    return (x + noise * rng.normal(size=len(x))).astype(np.float32)


def _fake_cv(tmp_path):
    root = tmp_path / "cv-corpus-26.0" / "en"
    clips = root / "clips"
    clips.mkdir(parents=True)
    rows = []
    specs = [  # speaker, accent, seconds, noise, n_clips, gender
        ("jam_a", "Jamaican English", 4.0, 0.0, 6, "male_masculine"),
        ("jam_b", "Caribbean English|Jamaican", 3.0, 0.0, 2, "female_feminine"),
        ("trini", "Trinidad and Tobago English", 3.0, 0.0, 3, "male_masculine"),
        ("usa", "United States English", 3.0, 0.0, 5, "male_masculine"),
        ("jam_noisy", "Jamaican Patois", 3.0, 0.2, 2, "male_masculine"),
    ]
    for spk, accent, sec, noise, count, gender in specs:
        for i in range(count):
            name = f"{spk}_{i}.mp3"
            sf.write(clips / name, _voice(sec, noise=noise, seed=i), SR, format="MP3")
            rows.append([spk, name, f"sentence {i}", "2", "0", "", gender, accent])
    rows.append(["jam_a", "downvoted.mp3", "bad", "0", "3", "", "male_masculine", "Jamaican English"])
    sf.write(clips / "downvoted.mp3", _voice(3.0), SR, format="MP3")
    header = ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents"]
    (root / "validated.tsv").write_text("\n".join("\t".join(r) for r in [header] + rows), encoding="utf-8")
    return tmp_path


def test_scan_keeps_only_jamaican_speakers(tmp_path):
    root = _fake_cv(tmp_path)
    reports = ds.scan_common_voice(root)
    names = [r.speaker for r in reports]
    assert names[0] == "jam_a" and set(names) == {"jam_a", "jam_b", "jam_noisy"}
    assert len(reports[0].clips) == 6  # the down-voted clip is skipped
    wide = {r.speaker for r in ds.scan_common_voice(root, ds.CARIBBEAN)}
    assert "trini" in wide and "usa" not in wide


def test_build_dataset_from_common_voice(tmp_path):
    root = _fake_cv(tmp_path)
    out = tmp_path / "out"
    summary = ds.from_common_voice(root, out, sr=40000)
    assert summary["speakers_used"] == ["jam_a"] and summary["clips"] == 6
    wavs = sorted((out / "dataset").glob("*.wav"))
    audio, sr = sf.read(wavs[0])
    assert sr == 40000 and audio.ndim == 1 and np.max(np.abs(audio)) <= 0.9
    assert (out / "xtts_reference.wav").exists()
    assert "CC0" in (out / "SOURCES.txt").read_text()
    mixed = ds.from_common_voice(root, tmp_path / "mix", mix_speakers=True)
    assert mixed["clips"] == 8 and mixed["rejected"] == 2  # the noisy speaker's clips are thrown out


def test_prepare_long_recording(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    long = np.concatenate([_voice(4.0, seed=i) for i in range(8)])  # ~36 s with pauses
    sf.write(rec / "friend.wav", long, SR)
    summary = ds.prepare_recordings(rec, tmp_path / "out")
    assert summary["clips"] >= 4
    for w in (tmp_path / "out" / "dataset").glob("*.wav"):
        info = sf.info(w)
        assert 1.0 <= info.frames / info.samplerate <= 14.0
