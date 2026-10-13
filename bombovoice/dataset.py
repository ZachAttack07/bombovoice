"""Build voice-training data from *consented, openly licensed* speech.

Two jobs:

1. ``from_common_voice``: Mozilla Common Voice (CC0 - every contributor agreed to public
   release) tags many clips with the speaker's self-described accent. This filters the
   English release down to Jamaican (optionally wider Caribbean) speakers, groups clips by
   speaker, picks the speaker with the most clean audio, and writes training-ready WAVs.

2. ``prepare_recordings``: cleans and slices your own long recordings (e.g. a Jamaican
   friend who agreed to be cloned) into 3-12 s training clips.

Both also write a ~20 s ``xtts_reference.wav`` for the voice-cloning feature, and a
SOURCES.txt that records where every clip came from and under which licence.

RVC trains best on ONE speaker with 10+ minutes of clean speech. Mixing speakers gives
a blurred "average" voice - useful if no single speaker has enough audio.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dsp import resample

log = logging.getLogger(__name__)

JAMAICAN = r"jamaic|patois|patwa"
CARIBBEAN = JAMAICAN + r"|caribbean|west ?indi|trinidad|tobago|barbad|bajan|guyan|bahamian|antigua|grenad|st\.? ?lucia|vincent"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class Clip:
    path: Path
    speaker: str
    sentence: str = ""
    accent: str = ""
    gender: str = ""
    seconds: float = 0.0


@dataclass
class SpeakerReport:
    speaker: str
    accent: str
    gender: str
    clips: list[Clip] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(c.seconds for c in self.clips)


# ---------------------------------------------------------------------------
# audio cleaning
# ---------------------------------------------------------------------------


def load_mono(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf

    audio, in_sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), in_sr, sr)


def _frame_rms(x: np.ndarray, sr: int, ms: float = 20.0) -> np.ndarray:
    n = max(1, int(sr * ms / 1000))
    frames = len(x) // n
    if frames == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)


def trim_silence(x: np.ndarray, sr: int, pad_ms: float = 120.0) -> np.ndarray:
    rms = _frame_rms(x, sr)
    if len(rms) == 0:
        return x
    thresh = max(np.percentile(rms, 95) * 0.05, 1e-4)
    voiced = np.flatnonzero(rms > thresh)
    if len(voiced) == 0:
        return x[:0]
    n = len(x) // len(rms)
    pad = int(sr * pad_ms / 1000)
    return x[max(0, voiced[0] * n - pad): min(len(x), (voiced[-1] + 1) * n + pad)]


def noise_floor(x: np.ndarray, sr: int) -> float:
    """Background level, from the quietest moments (measure on audio that still has its pauses)."""
    rms = _frame_rms(x, sr)
    return float(np.percentile(rms, 5)) if len(rms) else 0.0


def quality(x: np.ndarray, sr: int, floor: float | None = None) -> dict:
    """Cheap checks that throw out clips that would hurt training."""
    rms = _frame_rms(x, sr)
    if len(rms) < 10:
        return {"ok": False, "why": "too short"}
    floor = noise_floor(x, sr) if floor is None else floor
    snr = 20 * np.log10(np.percentile(rms, 95) / max(floor, 1e-6))
    clipped = float(np.mean(np.abs(x) > 0.995))
    if clipped > 0.001:
        return {"ok": False, "why": "clipping/distorted", "snr": snr}
    if snr < 20:
        return {"ok": False, "why": f"noisy (SNR {snr:.0f} dB)", "snr": snr}
    return {"ok": True, "snr": snr}


def normalize_loudness(x: np.ndarray, target_db: float = -20.0, peak_db: float = -1.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    y = x * (10 ** (target_db / 20) / rms)
    peak = float(np.max(np.abs(y))) or 1.0
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        y *= limit / peak
    return y.astype(np.float32)


def slice_on_silence(x: np.ndarray, sr: int, min_s: float = 3.0, max_s: float = 12.0) -> list[np.ndarray]:
    """Cut a long recording into training-sized pieces at natural pauses."""
    rms = _frame_rms(x, sr)
    if len(rms) == 0:
        return []
    n = len(x) // len(rms)
    quiet = rms < max(np.percentile(rms, 95) * 0.05, 1e-4)
    pieces, start = [], 0
    run = 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        length = (i - start) * n / sr
        if (run >= 15 and length >= min_s) or length >= max_s:  # 300 ms pause, or too long
            cut = i - run // 2 if run else i
            pieces.append(x[start * n: cut * n])
            start = cut
            run = 0
    pieces.append(x[start * n:])
    out = []
    for p in pieces:
        p = trim_silence(p, sr)
        if len(p) >= sr * 1.5:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Common Voice
# ---------------------------------------------------------------------------


def find_cv_files(root: Path) -> tuple[Path, Path]:
    """Locate validated.tsv and the clips folder inside an extracted Common Voice English release."""
    root = Path(root)
    tsvs = sorted(root.rglob("validated.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No validated.tsv under {root}. Point this at the extracted Common Voice English "
                                "folder (the one containing validated.tsv and clips/).")
    tsv = tsvs[0]
    clips = tsv.parent / "clips"
    if not clips.is_dir():
        raise FileNotFoundError(f"Found {tsv} but no clips/ folder next to it")
    return tsv, clips


def scan_common_voice(root: Path, pattern: str = JAMAICAN) -> list[SpeakerReport]:
    """Return matching speakers (most audio first). Durations are filled in by measuring the clips."""
    import soundfile as sf

    tsv, clips_dir = find_cv_files(root)
    rx = re.compile(pattern, re.I)
    speakers: dict[str, SpeakerReport] = {}
    with open(tsv, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            accent = row.get("accents") or row.get("accent") or ""
            if not accent or not rx.search(accent):
                continue
            try:
                if int(row.get("down_votes") or 0) > int(row.get("up_votes") or 0):
                    continue
            except ValueError:
                pass
            path = clips_dir / row["path"]
            if not path.exists():
                continue
            spk = row.get("client_id", "unknown")
            rep = speakers.setdefault(spk, SpeakerReport(spk, accent, row.get("gender", "") or ""))
            try:
                info = sf.info(str(path))
                seconds = info.frames / info.samplerate
            except Exception:
                continue
            rep.clips.append(Clip(path, spk, row.get("sentence", ""), accent, rep.gender, seconds))
    return sorted(speakers.values(), key=lambda r: r.seconds, reverse=True)


def from_common_voice(root: Path, out: Path, pattern: str = JAMAICAN, speaker: str | None = None,
                      mix_speakers: bool = False, gender: str | None = None, sr: int = 40000) -> dict:
    reports = scan_common_voice(root, pattern)
    if gender:
        reports = [r for r in reports if gender.lower() in r.gender.lower()]
    if not reports:
        raise RuntimeError("No speakers with a matching accent were found. Try --caribbean to widen the filter.")
    if mix_speakers:
        chosen = reports
    elif speaker:
        chosen = [r for r in reports if r.speaker.startswith(speaker)]
        if not chosen:
            raise RuntimeError(f"No matching speaker starts with {speaker!r}")
    else:
        chosen = reports[:1]
    clips = [c for r in chosen for c in r.clips]
    source = ("Mozilla Common Voice (English), CC0-1.0. Contributors released their recordings into the public "
              "domain. https://commonvoice.mozilla.org")
    return _write_dataset(clips, out, sr, source, reports=reports, chosen=chosen)


# ---------------------------------------------------------------------------
# your own recordings
# ---------------------------------------------------------------------------


def prepare_recordings(folder: Path, out: Path, sr: int = 40000, source_note: str = "") -> dict:
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise RuntimeError(f"No audio files ({', '.join(sorted(AUDIO_EXTS))}) found in {folder}")
    clips = [Clip(p, "speaker", seconds=0.0) for p in files]
    note = source_note or "Your own recordings. Make sure the speaker agreed to having their voice used for AI."
    return _write_dataset(clips, out, sr, note, slice_long=True)


# ---------------------------------------------------------------------------


def _write_dataset(clips: list[Clip], out: Path, sr: int, source: str, slice_long: bool = False,
                   reports: list[SpeakerReport] | None = None, chosen: list[SpeakerReport] | None = None) -> dict:
    import soundfile as sf

    out = Path(out)
    wav_dir = out / "dataset"
    wav_dir.mkdir(parents=True, exist_ok=True)
    kept, rejected, total = [], {}, 0.0
    best_for_xtts: list[tuple[float, np.ndarray]] = []
    n = 0
    for clip in clips:
        try:
            audio = load_mono(clip.path, sr)
        except Exception as exc:
            rejected[str(clip.path.name)] = f"unreadable ({exc.__class__.__name__})"
            continue
        floor = noise_floor(audio, sr)  # measured before trimming, while the pauses are still there
        pieces = slice_on_silence(audio, sr) if slice_long and len(audio) > sr * 14 else [trim_silence(audio, sr)]
        for piece in pieces:
            if len(piece) < sr * 1.0:
                rejected[clip.path.name] = "too short"
                continue
            q = quality(piece, sr, floor)
            if not q["ok"]:
                rejected[clip.path.name] = q["why"]
                continue
            piece = normalize_loudness(piece)
            n += 1
            name = f"{n:05d}.wav"
            sf.write(str(wav_dir / name), piece, sr, subtype="PCM_16")
            kept.append((name, clip))
            total += len(piece) / sr
            best_for_xtts.append((q["snr"], piece))

    if not kept:
        raise RuntimeError("Every clip was rejected (too short, noisy or distorted). Check the input audio.")

    # ~20 s reference for voice cloning: the cleanest clips, joined with short pauses
    best_for_xtts.sort(key=lambda t: t[0], reverse=True)
    ref, length = [], 0.0
    for _, piece in best_for_xtts:
        ref += [piece, np.zeros(int(sr * 0.3), np.float32)]
        length += len(piece) / sr + 0.3
        if length >= 20:
            break
    sf.write(str(out / "xtts_reference.wav"), np.concatenate(ref), sr, subtype="PCM_16")

    lines = [
        "Bombovoice training data",
        "=" * 24,
        f"Source / licence: {source}",
        f"Clips kept: {len(kept)}  ({total / 60:.1f} minutes)   rejected: {len(rejected)}",
        "",
        "Use responsibly: don't present the trained voice as the real person, and keep within the licence.",
        "",
        "file\toriginal\tspeaker\taccent\tsentence",
    ]
    lines += [f"{name}\t{c.path.name}\t{c.speaker[:12]}\t{c.accent}\t{c.sentence}" for name, c in kept]
    (out / "SOURCES.txt").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "out": str(out),
        "dataset_dir": str(wav_dir),
        "clips": len(kept),
        "minutes": total / 60,
        "rejected": len(rejected),
        "reject_reasons": _count(rejected.values()),
        "xtts_reference": str(out / "xtts_reference.wav"),
    }
    if reports is not None:
        summary["speakers_found"] = [(r.speaker[:12], r.accent, r.gender, round(r.seconds / 60, 1), len(r.clips))
                                     for r in reports]
        summary["speakers_used"] = [r.speaker[:12] for r in (chosen or [])]
    return summary


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        key = v.split(" (")[0]
        out[key] = out.get(key, 0) + 1
    return out
