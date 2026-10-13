import numpy as np

from bombovoice.config import Settings
from bombovoice.engine import Engine
from bombovoice.patois import PatoisTranslator

SR = 24000


class FakeTTS:
    name = "fake"

    def perform(self, translation, **kw):
        return np.full(SR // 2, 0.1, dtype=np.float32)

    def say(self, text, **kw):
        return np.full(SR // 4, 0.1, dtype=np.float32)


class FakeTranscriber:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, sr):
        self.calls += 1
        return "I am going to the store my friend"


def make_engine(mode="voice"):
    s = Settings(sample_rate=SR, mode=mode, use_ai=False, block_size=256)
    events = []
    eng = Engine(s, on_event=lambda k, p: events.append((k, p)))
    from bombovoice import synth
    from bombovoice.stt import UtteranceDetector

    eng.sounds = synth.build_soundboard(SR)
    eng.mixer.loop = synth.riddim(SR, bars=1)
    eng._detector = UtteranceDetector(SR)
    eng._translator = PatoisTranslator(intensity=2, seed=1)
    eng._tts = FakeTTS()
    eng._transcriber = FakeTranscriber()
    eng._ai_ready.set()
    return eng, events


def _speech_then_silence():
    t = np.arange(int(SR * 1.2)) / SR
    talk = 0.3 * np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return np.concatenate([np.zeros(SR // 2), talk, np.zeros(SR)]).astype(np.float32)


def test_voice_mode_streams_blocks():
    eng, _ = make_engine("voice")
    x = _speech_then_silence()
    out = np.concatenate([eng.process_block(x[i:i + 256]) for i in range(0, len(x) - 256, 256)])
    assert np.isfinite(out).all() and np.max(np.abs(out)) <= 1.0
    assert np.std(out) > 0.005


def test_soundboard_and_riddim_mix_in():
    eng, _ = make_engine("voice")
    eng.action("airhorn")
    out = eng.process_block(np.zeros(256, dtype=np.float32))
    assert np.max(np.abs(out)) > 0.01
    eng.action("riddim")
    assert eng.mixer.loop_on
    eng.action("rewind")
    assert eng.mixer.busy("rewind")
    eng.action("mode")
    assert eng.s.mode == "ai"
    eng.action("intensity")
    assert eng.s.intensity == 3


def test_ai_mode_pipeline_translates_and_speaks():
    eng, events = make_engine("ai")
    x = _speech_then_silence()
    for i in range(0, len(x) - 256, 256):
        out = eng.process_input(x[i:i + 256])
        assert not np.any(out)  # your real voice is replaced in AI mode
    kind, utt = eng.jobs.get_nowait()
    assert kind == "utterance" and len(utt) > SR // 2
    eng._handle_utterance(utt)
    kinds = [k for k, _ in events]
    assert "heard" in kinds and "patois" in kinds
    patois = next(p for k, p in events if k == "patois")
    assert "mi a go" in patois.patois.lower()
    assert eng.mixer.busy("tts")


def test_typed_say_goes_through_translator():
    eng, events = make_engine("voice")
    eng._perform_text("Thank you so much")
    patois = next(p for k, p in events if k == "patois")
    assert "tank yuh" in patois.patois.lower()
