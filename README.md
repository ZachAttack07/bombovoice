# Bombovoice 🇯🇲

**Wah gwaan!** Bombovoice is a Jamaican voice changer for your PC microphone. It doesn't just
drop your pitch. It rebuilds how you *talk*: the sing-song Jamaican intonation, stress that
lands late and jumps up on the punch word, phrase endings that swoop up and crash down,
Patois grammar and pronunciation, and a dancehall sound system on hotkeys.

It shows up as a microphone in Discord, games, Zoom, OBS, or anything else that takes a mic.

```
you say : "Hey bro, what's up? I'm going to play some games with my friends tonight."
yaadie  : "Yow bredrin, wah gwaan? Mi a go fi play some games wid mi bredren dem tinite, yuh zimmi?"
```

---

## Features

### Three ways to sound Jamaican

| Mode | What happens | Latency |
|---|---|---|
| **Live voice** | Your own voice, deeper with a bigger chest, through a real-time **PSOLA** pitch engine. PSOLA finds your pitch, cuts your voice into grains one vocal-cord cycle long and re-spaces them, so it sounds like a person rather than a robot. Pitch and formants (chest size) are set separately. The **Lilt Engine** follows your speech: phrases start low and climb, stressed syllables lift and glide, and phrase ends fall. There's no constant wobble. Optional gravel/rasp and warmth. You can switch engines live in the GUI (Natural / Smooth / Low latency). | ~40 ms |
| **AI Patois** | Speech-to-text (local Whisper), then **Claude rewrites what you said in Patois** and plans the delivery (which words to punch, where the pitch rises or falls, comic-timing pauses), then a TTS voice performs that plan line by line. | 2–4 s |
| **Hybrid** | Live Jamaican voice, plus an **AI hype man** who jumps in after your sentences: *"Bwoy! Him seh im a go win dis game!"* | live + async |

### The AI director (Claude)
- Rewrites whole lines into understandable, phonetically spelled Patois ("di", "dem", "tink", "likkle",
  "pickney", "wah gwaan", "seen?"), using real Patois grammar: copula drop, base-form past tense,
  "a" progressive, plural "dem".
- Plans emphasis and intonation for each chunk (`rise`, `fall`, `rise_fall`, `high`, `low`) plus pauses.
  The TTS renderer speaks the whole line in one natural take, then lays one smooth Jamaican intonation
  curve over it (phrases rise and fall, stressed words lift late) using Praat's PSOLA. That moves pitch
  without a robotic sound, and the curve is capped at about ±3 semitones so nothing jumps wildly.
- A pronunciation layer turns written Patois ("seh", "cyaan", "bwoy") into spellings the voice actually
  *says* ("say", "kahn", "buh-woy") instead of spelling them out letter by letter. The screen still shows real Patois.
- **5 characters:** `yaadie` (loud Kingston uncle), `dancehall` (hype MC at a sound clash), `rasta`
  (wise elder who *overstands*), `auntie` (scandalised by everything), `commentator` (calls everything like
  the Olympic 100m final).
- **4 intensity levels:** Tourist → Yaadie → Full Patois → **MAXIMUM BOMBO**.
- **Spicy mode** (off by default) unlocks *bomboclaat / raasclaat* exclamations.
- With no API key or no internet, it falls back to the built-in offline translator, so it never stops working.

### Offline Patois translator
A full rule-based English → Patois engine: 300+ word and phrase swaps, th-stopping (think→tink,
the→di), -er→-a, -ing→-in', -ttle→-kkle, consonant-cluster reduction, h-dropping and h-insertion
at max intensity, pronoun and copula grammar, plural "dem", sentence openers ("Bwoy,", "Yow!",
"Lawd have mercy,") and tags (", seen?", ", ya mon", ", yuh zimmi?").

### Dancehall sound system (all synthesised, no samples)
| Hotkey | Sound |
|---|---|
| F1 | Air horn: *BRAP BRAP BRAAAAP* |
| F2 | Dub siren, swept square wave into tape echo |
| F3 | **Rewind!** Spins back whatever was *just said* (tape rewind plus scratch), and if the riddim is playing, it "wheels an' come again" from the top |
| F4 | Bass drop |
| F5 | **Riddim on/off**: a generated one-drop reggae loop (kick and rim on 3, offbeat skank, rolling bassline), automatically ducked under your voice |
| F6 | Random ad-lib in the Jamaican TTS voice ("Big up yuhself!", "Wheel an' come again!", "Lawd have mercy!"…) |
| F7 | Dub echo on your live voice |
| F8 | Cycle mode · F9 mute · F10 cycle intensity · F11 pew pew |

Hotkeys are global, so they work while your game or Discord has focus. They are configurable in the settings file.

Also included: **type-and-say** (type a line and it's performed in Patois on your mic), **random auto
ad-libs** every ~25 s, and **file mode** to process recordings.

---

## Install (Windows)

1. Install **Python 3.10+** from python.org (tick *"Add Python to PATH"*).
2. Install the free **VB-Audio Virtual Cable**: https://vb-audio.com/Cable/ (run the installer as admin, then reboot).
3. Download this repo and **double-click `run_windows.bat`**. The first run installs everything; later runs just launch.
4. In the app, pick your real microphone and press **START**. "Send to" auto-detects the cable.
5. In Discord, your game or OBS, set your **input device / microphone** to **`CABLE Output (VB-Audio Virtual Cable)`**.
6. Tick **Monitor** if you want to hear yourself in your headphones.

### For the AI features
**Option A: OpenRouter (Claude Haiku 5.5, fast and cheap).** Create a key at https://openrouter.ai/keys and set it once (PowerShell):

```powershell
setx OPENROUTER_API_KEY "sk-or-..."
```

When `OPENROUTER_API_KEY` is set, Bombovoice uses OpenRouter automatically, with model
**`anthropic/claude-haiku-5.5`**. Haiku is the quickest option, which matters because you're waiting to hear your line.

**Option B: Anthropic API directly.** Set `ANTHROPIC_API_KEY` from https://console.anthropic.com instead. The default model there is `claude-opus-5-5`.

To force one or the other, use the **"via"** dropdown in the GUI, `--provider openrouter|anthropic` on the
command line, or `"ai_provider"` in the settings file. Restart Bombovoice after setting a key. The first
time you use AI or Hybrid mode, the Whisper speech model downloads (~150 MB).

### macOS / Linux

```bash
pip install -e ".[all]"
bombovoice setup            # shows the virtual-mic steps for your OS (BlackHole on macOS)
bombovoice setup --create   # Linux: creates a "Bombovoice_Mic" virtual microphone via pactl
bombovoice                  # launch the GUI
```
On Linux you also need `sudo apt install libportaudio2 python3-tk`.

---

## GPU features (NVIDIA): sound like a *real* person

These are optional, but they're the biggest upgrade. They need an NVIDIA GPU. RTX 50-series cards
like the RTX 5060 are supported, but need NVIDIA driver 570 or newer.

### One-time setup
1. Update your NVIDIA driver (GeForce Experience / NVIDIA App, or nvidia.com).
2. Run `run_windows.bat` once if you haven't already.
3. Double-click **`setup_gpu.bat`**. It installs PyTorch built for CUDA 12.8+ (about 3 GB; needed for RTX 50-series)
   plus the AI packages, then runs a GPU check. You want to see `GPU test OK`.
   You can re-check any time with `.venv\Scripts\python -m bombovoice gpu-check`.

### 1. Live AI voice (RVC)
A neural network re-voices your microphone as the person in a **voice model** (`.pth` file), live.
It sounds like a real human instead of an effect. Bombovoice's lilt engine steers the AI voice's pitch,
so you still get the Jamaican sing-song, and the dub echo/warmth/limiter still apply.

- In the app: **AI voice (GPU)** panel → *Model (.pth)...* → tick **Live AI voice**.
  If the model came with an `.index` file, add it with *Index...*; it sharpens the likeness.
- **AI voice key:** `0` for a male model when you're male; `+12`/`-12` to jump an octave between
  male and female voices. Use small steps to taste.
- Delay is about 0.3 s, the normal price for real-time AI voice. Lower `rvc_block_ms` in settings (e.g. 160)
  for less delay if your GPU keeps up. The app warns you if it falls behind.
- The first time, it downloads about 550 MB of base models (HuBERT + RMVPE) once.

**Where do voice models come from?** RVC can't make you sound Jamaican by itself. It makes you sound like
*the voice in the model*. The best result is a model of a Jamaican speaker **who agreed to it**. Train one with
[Applio](https://github.com/IAHispano/Applio) from about 10–20 minutes of their clean recordings. A generic deep
male voice model also works well with the Patois lilt. Don't use models that clone real celebrities or other
real people without their permission.
Models load safely by default. Old `.pth` files that need full pickle loading are refused unless you set
`"rvc_trust_model": true`, and you should only do that for files you trust.

### 2. Cloned voice for typed and AI lines (XTTS)
For *Type it* and the AI modes, XTTS-v2 speaks in the **voice and accent of a short clip**. With a clip of
a Jamaican speaker, every line comes out with a real Jamaican accent, not an approximation.

- **AI voice (GPU)** panel → *Voice clip...* (10–30 s of clean speech, one speaker, no music) →
  tick **Cloned voice for typed/AI lines** → accept the licence prompt.
- XTTS-v2 is under the [Coqui Public Model License](https://coqui.ai/cpml): **non-commercial use only**.
  It downloads about 1.8 GB the first time.
- Again: only clone a voice you have permission to use.

**Best combo:** Live AI voice for talking, cloned voice for typed punchlines and AI mode, plus the riddim and air horn.

## Getting Jamaican voice training data (the legit way)

To train your own Jamaican **RVC** voice, or to get a clip for **XTTS** cloning, use speech from people who
agreed to it. Bombovoice includes a tool that filters it down to only the Jamaican voices and cleans it for training.

| Source | What you get | Licence |
|---|---|---|
| **[Mozilla Common Voice](https://commonvoice.mozilla.org)**, English release via [Mozilla Data Collective](https://datacollective.mozillafoundation.org) (free account) | Thousands of speakers, each tagged with their self-described accent. Jamaican/Caribbean ones can be filtered out. | **CC0**: contributors put their voices in the public domain |
| **[Speech Accent Archive](https://accent.gmu.edu/language/jamaican-creole-english/)** | 4 Jamaican Creole English speakers, about 30 s each. Too short for RVC, but ideal as **XTTS clips**. | CC BY-NC-SA 4.0 (non-commercial, credit GMU) |
| **[IDEA Dialects Archive](https://www.dialectsarchive.com/jamaica-11)** | Many Jamaica recordings | Listening is free; reusing the files needs **written permission** (director@dialectsarchive.com) |
| **A Jamaican friend or a hired voice actor** | The best quality: 10–20 min of clean speech | Get their OK to train an AI voice (in writing, for an actor) |

Don't rip YouTube, TikTok or songs: those voices belong to real people who didn't agree, and the audio is copyrighted.

```bash
# 1. Download + extract the Common Voice ENGLISH release (it has validated.tsv and clips/)
# 2. See who's there:
bombovoice dataset scan "D:/cv-corpus-26.0/en"              # Jamaican accents only
bombovoice dataset scan "D:/cv-corpus-26.0/en" --caribbean  # wider Caribbean net
# 3. Build a training set from the speaker with the most clean audio (or --speaker ID, or --mix to blend):
bombovoice dataset commonvoice "D:/cv-corpus-26.0/en" -o jamaican_voice --gender male --set-clone
# Your own recordings of a consenting speaker (long files are sliced at pauses):
bombovoice dataset recordings "D:/recordings/friend" -o friend_voice --set-clone
```

The tool keeps only matching accents and drops down-voted, noisy or distorted clips. It trims silence,
evens out loudness and writes 40 kHz WAVs to `dataset/`. It also creates a 20-second `xtts_reference.wav`
(`--set-clone` makes it your cloned-voice clip) and a `SOURCES.txt` crediting every clip.
Then train with [Applio](https://github.com/IAHispano/Applio) (point it at `dataset/`; RVC v2, 40k, RMVPE, 200–400 epochs)
and load the `.pth` in Bombovoice.

## Command line

```bash
bombovoice                                   # GUI
bombovoice live                              # terminal mode with hotkeys + typed commands
bombovoice live -m ai --style dancehall -i 3 --spicy --monitor
bombovoice translate "I think this game is really crazy"            # offline
bombovoice translate --ai --plan "I can't believe we won"           # Claude + show the delivery plan
bombovoice translate --ai --provider openrouter "Let's go team"     # force OpenRouter (Haiku 5.5)
bombovoice say "Thank you for watching my stream" --riddim -o outro.wav
bombovoice file my_clip.wav my_clip_jamaican.wav                    # live-voice FX on a recording
bombovoice file -m ai my_clip.wav my_clip_patois.wav                # full AI re-performance
bombovoice demo -o demo.wav                                         # 3 lines over the riddim
bombovoice devices                                                  # list audio devices
bombovoice devices --test                                           # check which devices actually open
bombovoice dataset scan <common-voice-en-folder>                    # find Jamaican-accent speakers
```

In `live` mode you can type commands: `airhorn`, `siren`, `rewind`, `riddim`, `adlib`, `echo`,
`mode`, `intensity`, `spicy`, `mute`, `voice` / `ai` / `hybrid`, `pitch -3`, `lilt 1.5`,
`formant -2`, `say <text>`, `quit`.

Settings (including hotkeys, voice, TTS voice and model) are saved to
`%APPDATA%\bombovoice\settings.json` (Windows) or `~/.config/bombovoice/settings.json`.

### Tuning tips
- **Bigger, older man:** Deepness −4, Big-man chest −3, Rasp 0.4.
- **Maximum cartoon:** Lilt 1.8–2.0, Intensity MAXIMUM, style `dancehall`, Spicy on.
- **Lowest latency for live voice:** set `"quality": "fast"` in settings.
- **TTS voice:** default is `en-NG-AbeoNeural` (Microsoft Edge neural voice, online). Try others with
  `--tts-voice en-KE-ChilembaNeural` etc. Offline it uses your PC's built-in voices via `--tts-backend pyttsx3`.
- **AI model:** OpenRouter defaults to `anthropic/claude-haiku-5.5`. To try a bigger Claude model, pass its
  slug from https://openrouter.ai/models with `--model <slug>` (or set `"openrouter_model"` in settings). Expect
  sharper comedy writing but slower replies.

---

## Troubleshooting

**"Could not start audio … Unanticipated host error [PaErrorCode -9999] … [MME error 1]"**
Windows refused to open the microphone. Bombovoice already retries the same mic through WASAPI, DirectSound
and MME, so if you still see this:
1. **Settings > Privacy & security > Microphone**: turn on **Microphone access** *and*
   **Let desktop apps access your microphone**. This is the usual cause.
2. Close apps that may be holding the mic, or in *Control Panel > Sound > Recording > your mic > Properties >
   Advanced*, untick **Allow applications to take exclusive control of this device**.
3. Run `bombovoice devices --test` (or `.venv\Scripts\python -m bombovoice devices --test` from the repo folder).
   It tries every device and marks which ones open with **OK**. Pick one of those in the app's Microphone list.

**The typed/AI voice sounds robotic**
Look at the log panel. It should say `Voice synth: edge`, which is the natural-sounding online neural voice.
If you see `Online voice failed (...)`, that line was spoken by Windows' built-in voice, which is the robotic one.
That usually means no internet, or an outdated `edge-tts`. `run_windows.bat` updates packages on every launch,
so restarting it often fixes this. If it keeps happening, send the full error message from the log.

**Live voice sounds robotic**
Make sure *Voice engine* is **Natural (PSOLA)**. For the most realistic result:
- Keep **Deepness** between −1.5 and −3.5 and **Big-man chest** between −1 and −2.5. Bigger shifts always sound more processed.
- Keep **Sing-song lilt** at 1.0 or lower. Above about 1.2 it adds deliberate cartoon wobble.
- Speak in the rhythm yourself (it helps a lot): stretch the vowels, stress the end of the phrase, and drop the pitch at the end.

## How it works

```
mic ─► VoiceChain ──────────────────────────────┐
       (HPF → Lilt Engine → PSOLA pitch  │
        + formant shift → EQ → rasp → dub echo   ├─► Mixer ─► limiter ─► virtual cable ─► Discord/game
        → limiter)                               │   (+ riddim w/ ducking, soundboard, TTS, ad-libs)
mic ─► VAD ─► Whisper ─► Claude (Haiku 5.5 via OpenRouter) ─► TTS performer (per-segment rate/pitch/volume + contours) ┘
```

| File | Role |
|---|---|
| `bombovoice/dsp.py` | Real-time PSOLA pitch/formant shifter (plus phase-vocoder and delay-line alternatives), **Lilt Engine**, dub echo, EQ, limiter |
| `bombovoice/patois.py` | Offline English → Patois translator plus prosody segmentation |
| `bombovoice/ai.py` | Claude director via OpenRouter or the Anthropic API (structured output: Patois text plus delivery plan) |
| `bombovoice/tts.py` | Edge / pyttsx3 / XTTS voice-cloning TTS that performs the delivery plan (in a worker process) |
| `bombovoice/rvc/` | Streaming GPU AI voice conversion (RVC): HuBERT + RMVPE + voice model, SOLA-stitched blocks |
| `bombovoice/stt.py` | Voice activity detection and faster-whisper transcription |
| `bombovoice/synth.py` | Air horn, siren, bass drop, rewind, laser, one-drop riddim generator |
| `bombovoice/engine.py` | Real-time audio engine, mixer, AI worker, actions |
| `bombovoice/gui.py` / `cli.py` / `hotkeys.py` | The app |

Run the tests with `pip install -e ".[dev]" && pytest`.

## Limitations (honest ones)
- No program can literally replace your mic driver. Like every voice changer, Bombovoice uses a virtual audio cable.
- In **AI mode** the voice is a TTS voice performing your words, not your own timbre. **Live voice** mode keeps your voice.
- AI mode has a 2–4 second delay (speech recognition + Claude + TTS). It works best for punchlines rather than fast back-and-forth.
- There is no Jamaican neural TTS voice. Bombovoice gets the accent through phonetic Patois spelling plus its own intonation contours on top of a West-African English voice.
- It's comedy. Keep it loving: **one love** ❤️💛💚
