"""Audio device discovery and the "override my microphone" setup.

No program can literally replace your microphone driver. The standard trick
(used by every voice changer) is a *virtual audio cable*: Bombovoice plays
into the cable's input, and Discord / your game / OBS uses the cable's
output as its microphone.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

VIRTUAL_HINTS = ["cable input", "vb-audio", "voicemeeter input", "blackhole", "bombovoice", "loopback audio"]

SETUP_TEXT = {
    "Windows": """\
WINDOWS - one-time setup (2 minutes):
  1. Download and install VB-Audio Virtual Cable (free): https://vb-audio.com/Cable/
     (run VBCABLE_Setup_x64.exe as administrator, then reboot)
  2. Start Bombovoice. It auto-detects "CABLE Input" as its output.
  3. In Discord / Zoom / your game / OBS choose  "CABLE Output (VB-Audio Virtual Cable)"  as the MICROPHONE.
  4. Tick "Monitor" in Bombovoice if you want to hear yourself in your headphones.
""",
    "Darwin": """\
MACOS - one-time setup:
  1. Install BlackHole 2ch (free):  brew install blackhole-2ch   (or https://existential.audio/blackhole/)
  2. Start Bombovoice. It auto-detects "BlackHole 2ch" as its output.
  3. In Discord / Zoom / your game / OBS choose  "BlackHole 2ch"  as the MICROPHONE.
  4. Allow microphone access for your terminal/Python when macOS asks.
""",
    "Linux": """\
LINUX (PulseAudio / PipeWire) - one-time setup:
  1. Run:  bombovoice setup --create
     (creates a "Bombovoice" sink and a "Bombovoice Mic" virtual microphone with pactl)
  2. Start Bombovoice. It auto-detects the "bombovoice" sink as its output.
  3. In Discord / your game / OBS choose  "Bombovoice Mic"  as the MICROPHONE.
""",
}


def setup_instructions() -> str:
    return SETUP_TEXT.get(platform.system(), SETUP_TEXT["Linux"])


def _sd():
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing
        raise RuntimeError(
            "PortAudio is not available. On Linux: sudo apt install libportaudio2  (Windows/macOS wheels include it)"
        ) from exc
    return sd


def list_devices() -> list[dict]:
    sd = _sd()
    out = []
    hostapis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        out.append({
            "index": i,
            "name": d["name"],
            "inputs": d["max_input_channels"],
            "outputs": d["max_output_channels"],
            "hostapi": hostapis[d["hostapi"]]["name"],
            "default_sr": d["default_samplerate"],
        })
    return out


def _preferred_hostapi_rank(name: str) -> int:
    # On Windows the same device appears once per host API; prefer the ones with sane latency
    order = ["Windows WASAPI", "Windows DirectSound", "MME", "Core Audio", "PulseAudio", "ALSA", "JACK"]
    return order.index(name) if name in order else len(order)


def find_device(spec, kind: str) -> int | None:
    """Resolve a device given an index, an exact/partial name, or None (default)."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)
    key = "inputs" if kind == "input" else "outputs"
    matches = [d for d in list_devices() if d[key] > 0 and str(spec).lower() in d["name"].lower()]
    if not matches:
        raise RuntimeError(f"No {kind} device matching {spec!r}. Run `bombovoice devices` to see them all.")
    matches.sort(key=lambda d: _preferred_hostapi_rank(d["hostapi"]))
    return matches[0]["index"]


def find_virtual_output() -> int | None:
    devs = [d for d in list_devices() if d["outputs"] > 0]
    for hint in VIRTUAL_HINTS:
        hits = [d for d in devs if hint in d["name"].lower()]
        if hits:
            hits.sort(key=lambda d: _preferred_hostapi_rank(d["hostapi"]))
            return hits[0]["index"]
    return None


def linux_virtual_sink() -> str | None:
    """On Linux PortAudio only sees the generic 'pulse' device, so find our sink via pactl."""
    if platform.system() != "Linux" or not shutil.which("pactl"):
        return None
    try:
        sinks = subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None
    for line in sinks.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[1] == "bombovoice":
            return "bombovoice"
    return None


def create_linux_virtual_mic() -> str:
    pactl = shutil.which("pactl")
    if not pactl:
        raise RuntimeError("pactl not found (install pulseaudio-utils or pipewire-pulse)")
    cmds = [
        [pactl, "load-module", "module-null-sink", "sink_name=bombovoice",
         "sink_properties=device.description=Bombovoice"],
        [pactl, "load-module", "module-remap-source", "master=bombovoice.monitor",
         "source_name=bombovoice_mic", "source_properties=device.description=Bombovoice_Mic"],
    ]
    for c in cmds:
        subprocess.run(c, check=True, capture_output=True)
    return "Created sink 'bombovoice' and microphone 'Bombovoice_Mic'. (Lasts until reboot; re-run to recreate.)"
