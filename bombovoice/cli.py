"""Command-line entry point.

    bombovoice                       open the GUI
    bombovoice live                  run in the terminal with hotkeys
    bombovoice translate "text"      English -> Patois (add --ai for Claude)
    bombovoice say "text" -o out.wav perform text as a Jamaican voice
    bombovoice file in.wav out.wav   Jamaican-ify a recording
    bombovoice demo                  render a demo track with the riddim
    bombovoice devices               list audio devices
    bombovoice setup                 how to route Bombovoice into Discord/games
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from . import __version__
from .config import MODES, Settings
from .patois import INTENSITY_NAMES

BANNER = r"""
  ____                  _                     _
 | __ )  ___  _ __ ___ | |__   _____   _____ (_) ___ ___
 |  _ \ / _ \| '_ ` _ \| '_ \ / _ \ \ / / _ \| |/ __/ _ \
 | |_) | (_) | | | | | | |_) | (_) \ V / (_) | | (_|  __/
 |____/ \___/|_| |_| |_|_.__/ \___/ \_/ \___/|_|\___\___|
            wah gwaan - di Jamaican voice changer  v{v}
"""


def _apply_common(s: Settings, a: argparse.Namespace):
    if getattr(a, "provider", None):
        s.ai_provider = a.provider
    if getattr(a, "model", None):
        from .ai import resolve_provider

        if resolve_provider(s.ai_provider) == "openrouter":
            s.openrouter_model = a.model
        else:
            s.model = a.model
    for key in ("intensity", "style", "mode", "input_device", "output_device", "lilt", "base_pitch",
                "formant", "tts_backend", "tts_voice", "whisper_model", "quality"):
        val = getattr(a, key, None)
        if val is not None:
            setattr(s, key, val)
    if getattr(a, "rvc_model", None):
        s.rvc_model, s.quality = a.rvc_model, "rvc"
    if getattr(a, "rvc_key", None) is not None:
        s.rvc_key = a.rvc_key
    if getattr(a, "spicy", False):
        s.spicy = True
    if getattr(a, "no_ai", False):
        s.use_ai = False
    if getattr(a, "ai", False):
        s.use_ai = True
    if getattr(a, "monitor", False):
        s.monitor = True
    if getattr(a, "echo", False):
        s.echo = True


def _translator(s: Settings, ai: bool):
    from .ai import translator_from_settings

    return translator_from_settings(s, use_ai=ai)


def cmd_translate(a, s: Settings):
    tr = _translator(s, a.ai).translate(" ".join(a.text))
    print(tr.patois)
    if a.plan:
        print(f"\n[{tr.source}] delivery plan:")
        for seg in tr.segments:
            print(f"  {'!' * seg.emphasis:<3} {seg.contour:<9} {seg.pause_after_ms:>4}ms  {seg.text}")


def _write(path: str, audio: np.ndarray, sr: int):
    import soundfile as sf

    sf.write(path, audio, sr)
    print(f"Saved {path} ({len(audio) / sr:.1f}s)")


def _play(audio: np.ndarray, sr: int):
    from .devices import _sd

    sd = _sd()
    sd.play(audio, sr)
    sd.wait()


def cmd_say(a, s: Settings):
    from .tts import make_tts

    text = " ".join(a.text)
    tts = make_tts(s.tts_backend, s.tts_voice, s.sample_rate)
    if a.raw:
        from .patois import Segment

        tr_segments = [Segment(text=text, emphasis=1, contour="rise_fall")]
        audio = tts.perform(tr_segments, lilt=s.tts_lilt, formant=s.formant, deepen=s.tts_pitch)
    else:
        tr = _translator(s, a.ai).translate(text)
        print(tr.patois)
        audio = tts.perform(tr, lilt=s.tts_lilt, formant=s.formant, deepen=s.tts_pitch)
    if a.riddim:
        audio = _over_riddim(audio, s)
    if a.out:
        _write(a.out, audio, s.sample_rate)
    if a.play or not a.out:
        _play(audio, s.sample_rate)


def _over_riddim(voice: np.ndarray, s: Settings) -> np.ndarray:
    from . import synth

    sr = s.sample_rate
    beat = synth.riddim(sr, bpm=s.riddim_bpm)
    intro = int(sr * 60 / s.riddim_bpm * 4)  # one bar of riddim first
    total = intro + len(voice) + sr
    loop = np.resize(beat, total) * 0.35
    out = loop.copy()
    out[intro:intro + len(voice)] = out[intro:intro + len(voice)] * 0.55 + voice
    fade = int(sr * 0.8)
    out[-fade:] *= np.linspace(1, 0, fade)
    return np.clip(out, -1, 1).astype(np.float32)


def cmd_file(a, s: Settings):
    import soundfile as sf

    from .dsp import normalize, resample
    from .engine import Engine

    audio, sr = sf.read(a.input, dtype="float32", always_2d=True)
    audio = resample(audio.mean(axis=1), sr, s.sample_rate)
    if s.mode == "ai":
        from .stt import Transcriber
        from .tts import make_tts

        text = Transcriber(s.whisper_model).transcribe(audio, s.sample_rate)
        print(f"Heard: {text}")
        tr = _translator(s, s.use_ai).translate(text)
        print(f"Patois: {tr.patois}")
        out = make_tts(s.tts_backend, s.tts_voice, s.sample_rate).perform(tr, lilt=s.tts_lilt, formant=s.formant,
                                                                          deepen=s.tts_pitch)
    else:
        eng = Engine(s)
        bs = s.block_size
        sh = eng.chain.shifter
        lat = getattr(sh, "latency", getattr(sh, "window", 0) // 2)
        padded = np.concatenate([audio, np.zeros(lat + bs, dtype=np.float32)])
        out = np.concatenate([eng.chain.process(padded[i:i + bs]) for i in range(0, len(padded), bs)])
        out = out[lat:lat + len(audio)]
    _write(a.output, normalize(out, -1), s.sample_rate)


def cmd_demo(a, s: Settings):
    from .tts import make_tts

    lines = [
        "Hello everyone, welcome to my stream. How are you all doing today?",
        "I think this game is really crazy, my friend. Look at that!",
        "Oh my god, I cannot believe we won. Thank you so much!",
    ]
    tts = make_tts(s.tts_backend, s.tts_voice, s.sample_rate)
    tr_ = _translator(s, a.ai)
    parts = []
    for line in lines:
        tr = tr_.translate(line)
        print(f"  {line}\n  -> {tr.patois}\n")
        parts += [tts.perform(tr, lilt=s.tts_lilt, formant=s.formant, deepen=s.tts_pitch),
                  np.zeros(int(s.sample_rate * 0.5), dtype=np.float32)]
    audio = _over_riddim(np.concatenate(parts), s)
    _write(a.out, audio, s.sample_rate)


def cmd_devices(a, s: Settings):
    from .devices import find_virtual_output, list_devices, test_devices

    if a.test:
        print("Trying to open every device (this is what START does)...\n")
        for kind, idx, name, result in test_devices(s.sample_rate, s.block_size):
            print(f"{kind:<6} {idx:>3}  {result[:60]:<60}  {name}")
        print("\nPick an 'OK' microphone in the app's Microphone list. If every input FAILED, check")
        print("Settings > Privacy & security > Microphone > 'Let desktop apps access your microphone'.")
        return

    virt = find_virtual_output()
    print(f"{'#':>3}  {'in':>2} {'out':>3}  {'host API':<20} name")
    for d in list_devices():
        mark = "  <-- virtual cable" if d["index"] == virt else ""
        print(f"{d['index']:>3}  {d['inputs']:>2} {d['outputs']:>3}  {d['hostapi']:<20} {d['name']}{mark}")


def cmd_gpu_check(a, s: Settings):
    """Report what the AI voice / voice cloning will run on, and test the GPU for real."""
    try:
        from .rvc.stream import torch_info
    except ImportError as exc:
        print(f"GPU pack missing ({exc}). Run setup_gpu.bat.")
        return
    info = torch_info()
    if info["torch"] is None:
        print("PyTorch is not installed. Run setup_gpu.bat.")
        return
    print(f"PyTorch {info['torch']}")
    print(f"GPU:    {info['gpu'] or 'none (CPU only)'}")
    if info["error"]:
        print(f"PROBLEM: {info['error']}")
        return
    if info["device"] != "cuda":
        print("No CUDA GPU visible. Check: NVIDIA driver up to date (RTX 50-series needs 570+), and that setup_gpu.bat "
              "installed a CUDA build of PyTorch (version ends in +cu128/+cu130, not +cpu).")
        return
    import time

    import torch

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        x = (x @ x).clamp_(-1, 1)
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096 ** 3 / (time.perf_counter() - t) / 1e12
    print(f"GPU test OK: {tflops:.1f} TFLOPS fp16. You're good to go - tick 'Live AI voice' in the app.")
    for mod, what in (("transformers", "AI voice"), ("TTS", "voice cloning"), ("faiss", "AI voice .index files")):
        try:
            __import__(mod) if mod != "TTS" else __import__("bombovoice.tts", fromlist=["_import_xtts"])._import_xtts()
            print(f"  {what}: ready")
        except Exception as exc:
            print(f"  {what}: missing ({exc.__class__.__name__}) - re-run setup_gpu.bat")


def cmd_dataset(a, s: Settings):
    from . import dataset as ds

    pattern = ds.CARIBBEAN if getattr(a, "caribbean", False) else ds.JAMAICAN
    if a.dcmd == "scan":
        reports = ds.scan_common_voice(a.folder, pattern)
        if not reports:
            print("No matching speakers found. Try --caribbean.")
            return
        print(f"{'speaker':<14} {'minutes':>7} {'clips':>5}  gender      accent")
        for r in reports:
            print(f"{r.speaker[:12]:<14} {r.seconds / 60:7.1f} {len(r.clips):5d}  {r.gender[:10]:<10}  {r.accent}")
        total = sum(r.seconds for r in reports) / 60
        print(f"\n{len(reports)} speakers, {total:.1f} minutes. RVC wants ONE speaker with 10+ minutes; "
              "use --mix to blend several if nobody has enough.")
        return
    if a.dcmd == "commonvoice":
        summary = ds.from_common_voice(a.folder, a.out, pattern, speaker=a.speaker, mix_speakers=a.mix,
                                       gender=a.gender, sr=a.sr)
    else:
        summary = ds.prepare_recordings(a.folder, a.out, sr=a.sr)
    print(f"Kept {summary['clips']} clips = {summary['minutes']:.1f} minutes  ->  {summary['dataset_dir']}")
    if summary["rejected"]:
        print(f"Rejected {summary['rejected']}: {summary['reject_reasons']}")
    if summary["minutes"] < 10:
        print("Heads up: under 10 minutes. RVC still trains, but more clean audio = a better voice.")
    print(f"Voice-cloning reference: {summary['xtts_reference']}")
    print("Next: train an RVC model on the dataset folder with Applio (https://github.com/IAHispano/Applio),")
    print("      then load the .pth in Bombovoice's 'AI voice (GPU)' panel.")
    if a.set_clone:
        s.voice_clip = summary["xtts_reference"]
        s.save()
        print("Saved the reference as your cloned-voice clip.")


def cmd_setup(a, s: Settings):
    from .devices import create_linux_virtual_mic, setup_instructions

    print(setup_instructions())
    if a.create:
        print(create_linux_virtual_mic())


def cmd_live(a, s: Settings):
    from .engine import Engine
    from .hotkeys import Hotkeys

    print(BANNER.format(v=__version__))
    eng = Engine(s, on_event=_print_event)
    eng.prepare()
    eng.start()
    hk = Hotkeys(eng, s.hotkeys)
    if hk.start():
        print("Hotkeys: " + hk.describe())
    print("Type a command and press Enter:  airhorn siren rewind bassdrop laser riddim adlib echo mode intensity")
    print("                                 spicy mute  |  say <text>  |  pitch <st>  lilt <0-2>  |  quit\n")
    try:
        for line in sys.stdin:
            cmd, _, rest = line.strip().partition(" ")
            if not cmd:
                continue
            if cmd in ("q", "quit", "exit"):
                break
            if cmd == "say":
                eng.say(rest)
            elif cmd in ("pitch", "lilt", "formant", "rasp"):
                key = {"pitch": "base_pitch"}.get(cmd, cmd)
                try:
                    eng.set(**{key: float(rest)})
                    print(f"{key} = {rest}")
                except ValueError:
                    print(f"usage: {cmd} <number>")
            elif cmd in MODES:
                eng.set(mode=cmd)
                print(f"Mode: {MODES[cmd]}")
            else:
                eng.action(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        hk.stop()
        eng.close()
        time.sleep(0.1)


def _print_event(kind, payload):
    if kind == "heard":
        print(f"  you said : {payload}")
    elif kind == "patois":
        print(f"  yaadie   : {payload.patois}   [{payload.source}]")
    elif kind == "adlib":
        print(f"  ad-lib   : {payload}")
    elif kind == "status":
        print(f"  * {payload}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bombovoice", description="Bombovoice - make yuh voice sound like a real yaadie.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-i", "--intensity", type=int, choices=range(4),
                        help="0 " + ", ".join(f"{k}={v}" for k, v in INTENSITY_NAMES.items()))
    common.add_argument("--spicy", action="store_true", help="allow bomboclaat-grade exclamations")
    common.add_argument("--style", choices=["yaadie", "dancehall", "rasta", "auntie", "commentator"])
    common.add_argument("--provider", choices=["auto", "openrouter", "anthropic"],
                        help="where to reach Claude (auto = OpenRouter if OPENROUTER_API_KEY is set)")
    common.add_argument("--model", help="model for the AI director (OpenRouter default anthropic/claude-haiku-5.5, "
                                        "Anthropic default claude-opus-5-5)")
    common.add_argument("--no-ai", action="store_true", help="never call Claude; offline rules only")
    common.add_argument("--tts-backend", choices=["auto", "edge", "pyttsx3"])
    common.add_argument("--tts-voice")
    common.add_argument("--lilt", type=float, help="sing-song amount 0-2")
    common.add_argument("--base-pitch", type=float, help="semitones (negative = deeper)")
    common.add_argument("--formant", type=float, help="semitones (negative = bigger chest)")

    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("gui", parents=[common], help="open the GUI (default)")
    g.set_defaults(fn=None)

    lv = sub.add_parser("live", parents=[common], help="run live in the terminal")
    lv.add_argument("-m", "--mode", choices=list(MODES))
    lv.add_argument("--input-device", help="mic name or index")
    lv.add_argument("--output-device", help="output name or index (default: auto-detect virtual cable)")
    lv.add_argument("--monitor", action="store_true", help="also hear yourself in your headphones")
    lv.add_argument("--echo", action="store_true", help="start with dub echo on")
    lv.add_argument("--quality", choices=["natural", "rvc", "pv", "fast"])
    lv.add_argument("--rvc-model", help="RVC voice model (.pth) - turns on the GPU AI voice")
    lv.add_argument("--rvc-key", type=float, help="AI voice key in semitones")
    lv.add_argument("--whisper-model", help="faster-whisper model (tiny.en, base.en, small.en...)")
    lv.set_defaults(fn=cmd_live)

    t = sub.add_parser("translate", parents=[common], help="translate English text to Patois")
    t.add_argument("text", nargs="+")
    t.add_argument("--ai", action="store_true", help="use Claude instead of the offline rules")
    t.add_argument("--plan", action="store_true", help="show the delivery plan (emphasis, intonation, pauses)")
    t.set_defaults(fn=cmd_translate)

    sy = sub.add_parser("say", parents=[common], help="speak text as a Jamaican voice")
    sy.add_argument("text", nargs="+")
    sy.add_argument("-o", "--out", help="write a .wav instead of playing")
    sy.add_argument("--play", action="store_true", help="play even when writing a file")
    sy.add_argument("--ai", action="store_true", help="use Claude for the translation")
    sy.add_argument("--raw", action="store_true", help="don't translate, just speak with the accent voice")
    sy.add_argument("--riddim", action="store_true", help="put it over the riddim")
    sy.set_defaults(fn=cmd_say)

    f = sub.add_parser("file", parents=[common], help="process a recording")
    f.add_argument("input")
    f.add_argument("output")
    f.add_argument("-m", "--mode", choices=["voice", "ai"], default="voice")
    f.add_argument("--whisper-model")
    f.set_defaults(fn=cmd_file)

    d = sub.add_parser("demo", parents=[common], help="render a demo track")
    d.add_argument("-o", "--out", default="bombovoice_demo.wav")
    d.add_argument("--ai", action="store_true")
    d.set_defaults(fn=cmd_demo)

    dv = sub.add_parser("devices", help="list audio devices")
    dv.add_argument("--test", action="store_true", help="try opening every device and report which work")
    dv.set_defaults(fn=cmd_devices)

    ds = sub.add_parser("dataset", help="build Jamaican voice-training data from consented, open speech")
    dsub = ds.add_subparsers(dest="dcmd", required=True)
    sc = dsub.add_parser("scan", help="list Jamaican-accent speakers in a Common Voice English release")
    sc.add_argument("folder", help="extracted Common Voice English folder (has validated.tsv + clips/)")
    sc.add_argument("--caribbean", action="store_true", help="include other Caribbean accents")
    cv = dsub.add_parser("commonvoice", help="build a training set from Common Voice (CC0)")
    cv.add_argument("folder")
    cv.add_argument("-o", "--out", default="jamaican_voice_data")
    cv.add_argument("--caribbean", action="store_true", help="include other Caribbean accents")
    cv.add_argument("--speaker", help="speaker id prefix from 'dataset scan' (default: the one with most audio)")
    cv.add_argument("--mix", action="store_true", help="blend all matching speakers into one voice")
    cv.add_argument("--gender", help="only speakers whose gender field contains this (e.g. male)")
    rc = dsub.add_parser("recordings", help="clean + slice your own recordings (with the speaker's permission)")
    rc.add_argument("folder")
    rc.add_argument("-o", "--out", default="jamaican_voice_data")
    for p_ in (cv, rc):
        p_.add_argument("--sr", type=int, default=40000, help="sample rate of the output (RVC v2: 40000 or 48000)")
        p_.add_argument("--set-clone", action="store_true", help="use the 20 s reference as your cloned-voice clip")
    ds.set_defaults(fn=cmd_dataset)

    gc = sub.add_parser("gpu-check", help="check the GPU for the AI voice and voice cloning")
    gc.set_defaults(fn=cmd_gpu_check)

    st = sub.add_parser("setup", help="virtual microphone setup instructions")
    st.add_argument("--create", action="store_true", help="Linux: create the virtual mic with pactl")
    st.set_defaults(fn=cmd_setup)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpx2", "anthropic", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    s = Settings.load()
    _apply_common(s, a)
    fn = getattr(a, "fn", None)
    try:
        if fn is None:
            from .gui import run_gui

            run_gui(s)
        else:
            fn(a, s)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
