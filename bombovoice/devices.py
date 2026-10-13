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


# ---------------------------------------------------------------------------
# Robust stream opening
# ---------------------------------------------------------------------------

MIC_HELP = """\
Bombovoice could not open your microphone.

Fixes that usually work on Windows:
  1. Settings > Privacy & security > Microphone: turn ON "Microphone access"
     AND "Let desktop apps access your microphone".
  2. Close other apps that may be holding the mic (Discord, OBS, games, browser tabs), or in
     Control Panel > Sound > Recording > your mic > Properties > Advanced, untick
     "Allow applications to take exclusive control of this device".
  3. Make sure the mic is plugged in and enabled (Sound > Recording, right-click > Show disabled devices).
  4. Pick your microphone explicitly in the "Microphone" list instead of "(system default)",
     then press START again. `bombovoice devices --test` shows which entries open.
"""

OUTPUT_HELP = """\
Bombovoice could not open the output device.

  1. Check VB-Audio Virtual Cable is installed (reboot after installing).
  2. Pick another entry for the same device in the "Send to" list (each device appears once per Windows audio system).
  3. `bombovoice devices --test` shows which entries open.
"""


class AudioOpenError(RuntimeError):
    def __init__(self, kind: str, attempts: list[str]):
        self.kind = kind
        self.attempts = attempts
        details = "\n".join(f"  - {a}" for a in attempts[:6]) or "  - no usable devices found"
        if len(attempts) > 6:
            details += f"\n  - ...and {len(attempts) - 6} more"
        super().__init__((MIC_HELP if kind == "input" else OUTPUT_HELP) + "\nWhat was tried:\n" + details)


def _same_device(a: str, b: str) -> bool:
    # MME truncates names to 31 characters, so compare on the common prefix
    a, b = a.lower().strip(), b.lower().strip()
    n = min(len(a), len(b), 31)
    return n >= 6 and a[:n] == b[:n]


def stream_candidates(kind: str, preferred: int | None) -> list[int]:
    """Device indices to try, best first: the chosen device, then the same device on other host APIs."""
    sd = _sd()
    devs, apis = sd.query_devices(), sd.query_hostapis()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    api_order = sorted(range(len(apis)), key=lambda i: _preferred_hostapi_rank(apis[i]["name"]))
    out: list[int] = []
    if preferred is not None:
        out.append(preferred)
        name = devs[preferred]["name"]
        for a in api_order:
            out += [j for j in apis[a]["devices"] if devs[j][key] > 0 and _same_device(devs[j]["name"], name)]
    else:
        default_key = "default_input_device" if kind == "input" else "default_output_device"
        for a in api_order:
            d = apis[a].get(default_key, -1)
            if d is not None and d >= 0:
                out.append(d)
        try:
            d = sd.default.device[0 if kind == "input" else 1]
            if d is not None and d >= 0:
                out.append(int(d))
        except Exception:
            pass
    seen, result = set(), []
    for i in out:
        if i not in seen and 0 <= i < len(devs) and devs[i][key] > 0:
            seen.add(i)
            result.append(i)
    return result


def open_stream(kind: str, preferred: int | None, samplerate: int, blocksize: int, callback):
    """Open an input/output stream, falling back across host APIs, channel counts and latencies.

    Returns (stream, device_index, description).
    """
    sd = _sd()
    devs, apis = sd.query_devices(), sd.query_hostapis()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    cls = sd.InputStream if kind == "input" else sd.OutputStream
    attempts: list[str] = []
    for idx in stream_candidates(kind, preferred):
        dev = devs[idx]
        api = apis[dev["hostapi"]]["name"]
        extra = None
        if "WASAPI" in api:
            try:  # let Windows convert the sample rate / channel count for us
                extra = sd.WasapiSettings(auto_convert=True)
            except TypeError:
                extra = None
        channels = sorted({1, min(2, int(dev[key]))})
        last_err = None
        for ch in channels:
            for latency in ("low", "high"):
                try:
                    st = cls(device=idx, channels=ch, samplerate=samplerate, blocksize=blocksize, dtype="float32",
                             callback=callback, latency=latency, extra_settings=extra)
                    return st, idx, f"{dev['name']} [{api}]"
                except Exception as exc:
                    last_err = exc
        attempts.append(f"{dev['name']} [{api}]: {last_err}")
    raise AudioOpenError(kind, attempts)


def test_devices(samplerate: int = 48000, blocksize: int = 256) -> list[tuple[str, int, str, str]]:
    """Try to open every input and output device. Returns (kind, index, name, result)."""
    sd = _sd()
    devs, apis = sd.query_devices(), sd.query_hostapis()
    results = []
    for kind, key in (("input", "max_input_channels"), ("output", "max_output_channels")):
        cls = sd.InputStream if kind == "input" else sd.OutputStream
        for i, d in enumerate(devs):
            if d[key] <= 0:
                continue
            name = f"{d['name']} [{apis[d['hostapi']]['name']}]"
            extra = None
            if "WASAPI" in apis[d["hostapi"]]["name"]:
                try:
                    extra = sd.WasapiSettings(auto_convert=True)
                except TypeError:
                    pass
            try:
                st = cls(device=i, channels=1, samplerate=samplerate, blocksize=blocksize, dtype="float32",
                         extra_settings=extra)
                st.close()
                results.append((kind, i, name, "OK"))
            except Exception as exc:
                results.append((kind, i, name, f"FAILED: {exc}"))
    return results
