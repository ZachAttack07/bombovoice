"""AI voice (RVC) and voice cloning (XTTS) plumbing, tested with tiny random-weight models.

The output is noise (the weights are random) - these tests check loading, streaming,
chunk stitching and the voice-clip path, not voice quality. Skipped without the GPU pack.
"""

import string

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("librosa")

SR = 48000


class Sneaky:  # an arbitrary pickled object, like a booby-trapped .pth would contain
    pass


@pytest.fixture(scope="module")
def rvc_files(tmp_path_factory):
    from transformers import HubertConfig, HubertModel

    from bombovoice.rvc.models import SynthesizerTrnMs768NSFsid
    from bombovoice.rvc.rmvpe import E2E

    d = tmp_path_factory.mktemp("rvc")
    torch.manual_seed(0)
    HubertModel(HubertConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=12, intermediate_size=256,
                             classifier_proj_size=256)).save_pretrained(d / "hubert")
    torch.save(E2E(4, 1, (2, 2)).state_dict(), d / "rmvpe.pt")
    config = [1025, 32, 64, 64, 128, 2, 2, 3, 0, "1", [3], [[1, 3, 5]], [10, 10, 2, 2], 64, [16, 16, 4, 4], 4, 32, 40000]
    net = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    torch.save({"config": config, "weight": net.state_dict(), "f0": 1, "version": "v2"}, d / "voice.pth")
    return d


def _speech(seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * 120 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_streaming_rvc_blocks(rvc_files):
    from bombovoice.rvc.stream import StreamingRVC

    rvc = StreamingRVC(str(rvc_files / "voice.pth"), io_sr=SR, device="cpu", hubert_path=str(rvc_files / "hubert"),
                       rmvpe_path=str(rvc_files / "rmvpe.pt"), block_time=0.2, extra_time=0.5, key=-2)
    assert rvc.tgt_sr == 40000 and rvc.version == "v2"
    x = _speech()
    outs = [rvc.process_block(x[i:i + rvc.block_io], np.zeros(20)) for i in range(0, len(x) - rvc.block_io, rvc.block_io)]
    assert all(len(o) == rvc.block_io and np.isfinite(o).all() for o in outs)
    assert np.std(np.concatenate(outs)) > 0
    silent = rvc.process_block(np.zeros(rvc.block_io, np.float32))
    assert np.max(np.abs(silent)) < np.max(np.abs(outs[-1])) + 1e-6  # silence is gated, no inference


def test_rvc_rejects_non_models_and_unsafe_pickles(rvc_files, tmp_path):
    from bombovoice.rvc.stream import load_voice_model

    torch.save({"hello": 1}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="not an RVC voice model"):
        load_voice_model(str(tmp_path / "bad.pth"), "cpu", False)

    torch.save({"config": [], "weight": {}, "x": Sneaky()}, tmp_path / "pickle.pth")
    with pytest.raises(RuntimeError, match="can't be loaded safely"):
        load_voice_model(str(tmp_path / "pickle.pth"), "cpu", False)


def test_sola_stitching_is_click_free():
    from bombovoice.rvc.stream import StreamingRVC

    rvc = StreamingRVC.__new__(StreamingRVC)
    rvc.block_io, rvc.cross_io, rvc.sola_io = 4800, 960, 192
    rvc.tail = np.zeros(rvc.cross_io, np.float32)
    t = np.linspace(0, 1, rvc.cross_io, dtype=np.float32)
    rvc.fade_in = np.sin(0.5 * np.pi * t) ** 2
    rvc.fade_out = 1 - rvc.fade_in
    sig = np.sin(2 * np.pi * 220 * np.arange(SR * 2) / SR).astype(np.float32)
    rng = np.random.default_rng(1)
    out, pos = [], 0
    for _ in range(15):  # each chunk starts at a slightly wrong place, like real model output
        jitter = int(rng.integers(0, rvc.sola_io))
        chunk = sig[pos + jitter: pos + jitter + rvc.block_io + rvc.cross_io + rvc.sola_io]
        out.append(rvc._sola(chunk.copy()))
        pos += rvc.block_io
    y = np.concatenate(out[1:])
    assert np.max(np.abs(np.diff(y))) < 0.1  # a clean 220 Hz sine never steps more than ~0.03


def test_rvc_stage_feed_read(rvc_files):
    import time

    from bombovoice.dsp import LiltEngine
    from bombovoice.rvc.stream import RVCStage, StreamingRVC

    rvc = StreamingRVC(str(rvc_files / "voice.pth"), io_sr=SR, device="cpu", hubert_path=str(rvc_files / "hubert"),
                       rmvpe_path=str(rvc_files / "rmvpe.pt"), block_time=0.2, extra_time=0.5)
    stage = RVCStage(rvc, LiltEngine(SR))
    x = _speech(1.5)
    got = []
    for i in range(0, len(x), 512):  # feed and read in small blocks, like the audio callback
        stage.feed(x[i:i + 512])
        got.append(stage.read(512))
        time.sleep(0.004)
    deadline = time.time() + 20
    while time.time() < deadline and stage.out.underflows == 0 and len(stage.out.buf) < SR // 4:
        time.sleep(0.1)  # let the (slow, CPU) converter catch up
    for _ in range(40):
        got.append(stage.read(512))
    stage.close()
    y = np.concatenate(got)
    assert np.isfinite(y).all() and np.std(y) > 0


def test_xtts_clone_from_clip(tmp_path):
    from bombovoice.tts import _import_xtts

    try:  # the app's own import path (plain `import TTS` refuses without torchcodec)
        _import_xtts()
    except ImportError:
        pytest.skip("coqui-tts not installed")
    import soundfile as sf
    from tokenizers import Tokenizer, models, pre_tokenizers

    from bombovoice.patois import translate
    from bombovoice.tts import XTTS, TTSError, _import_xtts

    vocab = {"[STOP]": 0, "[UNK]": 1, "[SPACE]": 2, "[en]": 3}
    for ch in string.ascii_letters + string.digits + ".,!?'-":
        vocab[ch] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=r"\[[a-zA-Z]+\]|.", behavior="isolated")
    tok.save(str(tmp_path / "vocab.json"))
    XttsConfig, Xtts = _import_xtts()
    c = XttsConfig()
    a = c.model_args
    a.gpt_layers, a.gpt_n_model_channels, a.gpt_n_heads, a.decoder_input_dim = 1, 128, 2, 128
    a.gpt_number_text_tokens, a.gpt_start_text_token, a.gpt_stop_text_token = len(vocab), 3, 0
    a.gpt_max_audio_tokens = 20
    a.tokenizer_file = str(tmp_path / "vocab.json")
    m = Xtts.init_from_config(c)
    m.init_models()
    torch.save({"model": m.state_dict()}, tmp_path / "model.pth")
    c.save_json(str(tmp_path / "config.json"))
    sf.write(tmp_path / "clip.wav", _speech(4.0), SR)

    with pytest.raises(TTSError, match="License"):
        XTTS(voice_clip=str(tmp_path / "clip.wav"), model_dir=str(tmp_path), device="cpu")
    tts = XTTS(voice_clip=str(tmp_path / "clip.wav"), model_dir=str(tmp_path), device="cpu",
               license_accepted=True, sample_rate=SR)
    audio = tts.perform(translate("Hello my friend", 2, seed=1))
    assert len(audio) > 0 and np.isfinite(audio).all()
