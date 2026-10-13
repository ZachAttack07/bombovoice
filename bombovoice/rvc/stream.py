"""Real-time streaming RVC.

How it works (same idea as RVC's own real-time GUI):
  * keep the last ~2 s of mic audio as context, process a new block every ~0.25 s
  * HuBERT turns the speech into "what was said" features (speaker-independent)
  * RMVPE tracks the pitch; we shift it (deeper) and bend it with the Jamaican lilt
  * the voice model's synthesizer re-speaks it as the target voice, only for the tail
  * consecutive blocks are stitched with SOLA (find the best-matching offset, then
    crossfade) so there are no clicks between chunks
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path

import numpy as np
from scipy import signal

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("BOMBOVOICE_CACHE", Path.home() / ".cache" / "bombovoice")) / "rvc"
HUBERT_REPO = "r3gm/hubert_base"  # ContentVec/HuBERT base in transformers format
RMVPE_REPO, RMVPE_FILE = "r3gm/sonitranslate_voice_models", "rmvpe.pt"

F0_MIN, F0_MAX = 50.0, 1100.0
_MEL_MIN = 1127 * math.log(1 + F0_MIN / 700)
_MEL_MAX = 1127 * math.log(1 + F0_MAX / 700)


def torch_info() -> dict:
    """What hardware RVC/XTTS will run on (for the UI and diagnostics)."""
    try:
        import torch
    except ImportError:
        return {"torch": None, "device": None, "gpu": None, "error": "PyTorch is not installed"}
    info = {"torch": torch.__version__, "device": "cpu", "gpu": None, "error": None}
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]}{cap[1]}"
            supported = torch.cuda.get_arch_list()
            info.update(device="cuda", gpu=f"{name} ({arch})")
            if supported and arch not in supported and f"compute_{cap[0]}{cap[1]}" not in supported:
                info["error"] = (f"This PyTorch build doesn't support {arch} ({name}). RTX 50-series cards need a "
                                 "PyTorch built for CUDA 12.8 or newer - run setup_gpu.bat.")
        except Exception as exc:  # broken driver etc.
            info["error"] = f"CUDA error: {exc}"
    return info


def pick_device(prefer: str = "auto") -> str:
    import torch

    if prefer in ("cpu", "cuda"):
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_base_models(cache_dir: Path = CACHE_DIR, hubert: str | None = None, rmvpe: str | None = None):
    """Download (once) the speaker-independent models every RVC voice needs."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    if not hubert:
        hubert = snapshot_download(HUBERT_REPO, cache_dir=str(cache_dir / "hf"))
    if not rmvpe:
        rmvpe = hf_hub_download(RMVPE_REPO, RMVPE_FILE, cache_dir=str(cache_dir / "hf"))
    return hubert, rmvpe


def load_hubert(path: str, device: str, half: bool):
    import torch.nn as nn
    from transformers import HubertModel

    class HubertWithProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    try:
        model = HubertWithProj.from_pretrained(path)
    except Exception:
        model = HubertModel.from_pretrained(path)
    model = model.to(device).eval()
    return model.half() if half else model.float()


def load_voice_model(path: str, device: str, half: bool, allow_pickle: bool = False):
    """Load an RVC .pth voice model. Returns (net_g, target_sr, version, has_f0, n_spk)."""
    import torch

    from .models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    try:  # safe loading first: a .pth from the internet can otherwise run code
        cpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_pickle:
            raise RuntimeError(
                f"{Path(path).name} can't be loaded safely ({exc.__class__.__name__}). Only if you trust where this "
                "file came from, set \"rvc_trust_model\": true in settings and try again.") from exc
        log.warning("loading %s with full pickle (trusted by user)", path)
        cpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cpt, dict) or "config" not in cpt or "weight" not in cpt:
        raise RuntimeError(f"{Path(path).name} is not an RVC voice model (no config/weight inside)")
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    has_f0 = int(cpt.get("f0", 1)) == 1
    version = cpt.get("version", "v1")
    if version == "v2":
        net = SynthesizerTrnMs768NSFsid(*config, is_half=half) if has_f0 else SynthesizerTrnMs768NSFsid_nono(*config)
    else:
        net = SynthesizerTrnMs256NSFsid(*config, is_half=half) if has_f0 else SynthesizerTrnMs256NSFsid_nono(*config)
    del net.enc_q
    net.load_state_dict(cpt["weight"], strict=False)
    net = net.eval().to(device)
    net = net.half() if half else net.float()
    return net, int(config[-1]), version, has_f0, int(config[-3])


def load_index(path: str | None):
    if not path:
        return None, None
    try:
        import faiss
    except ImportError:
        log.info("faiss not installed - ignoring the .index file (optional)")
        return None, None
    index = faiss.read_index(path)
    return index, index.reconstruct_n(0, index.ntotal)


class StreamingRVC:
    """Converts a stream of mic audio (at `io_sr`) block by block. Not thread-safe; see RVCStage."""

    def __init__(self, model_path: str, io_sr: int = 48000, device: str = "auto", key: float = 0.0,
                 block_time: float = 0.25, extra_time: float = 2.0, crossfade_time: float = 0.06,
                 sola_time: float = 0.012, index_path: str | None = None, index_rate: float = 0.5,
                 protect: float = 0.33, hubert_path: str | None = None, rmvpe_path: str | None = None,
                 allow_pickle: bool = False, silence_db: float = -50.0):
        import torch

        from .rmvpe import RMVPE

        self.torch = torch
        self.device = pick_device(device)
        self.half = self.device == "cuda"
        self.io_sr = io_sr
        self.key = key
        self.index_rate = index_rate
        self.protect = protect
        self.silence = 10 ** (silence_db / 20)

        hubert_path, rmvpe_path = ensure_base_models(hubert=hubert_path, rmvpe=rmvpe_path)
        self.hubert = load_hubert(hubert_path, self.device, self.half)
        self.rmvpe = RMVPE(rmvpe_path, is_half=self.half, device=self.device)
        self.net, self.tgt_sr, self.version, self.has_f0, _ = load_voice_model(
            model_path, self.device, self.half, allow_pickle)
        self.index, self.big_npy = load_index(index_path)
        self.sid = torch.tensor([0], device=self.device).long()

        # sizes at 16 kHz, all multiples of one 10 ms frame (160 samples)
        z = 160
        f = lambda sec: max(1, round(sec * 16000 / z)) * z  # noqa: E731
        self.block16, self.extra16 = f(block_time), f(extra_time)
        self.cross16, self.sola16 = f(crossfade_time), f(sola_time)
        self.total16 = self.extra16 + self.cross16 + self.sola16 + self.block16
        self.return_frames = (self.block16 + self.cross16 + self.sola16) // z
        # matching sizes at the I/O rate
        r = io_sr / 16000
        self.block_io = int(round(self.block16 * r))
        self.cross_io = int(round(self.cross16 * r))
        self.sola_io = int(round(self.sola16 * r))
        self.ring = np.zeros(int(round(self.total16 * r)), dtype=np.float32)
        self.lilt = np.zeros(self.total16 // z, dtype=np.float32)  # semitone bend per 10 ms frame
        self.tail = np.zeros(self.cross_io, dtype=np.float32)
        t = np.linspace(0.0, 1.0, self.cross_io, dtype=np.float32)
        self.fade_in = np.sin(0.5 * np.pi * t) ** 2
        self.fade_out = 1.0 - self.fade_in
        g = math.gcd(io_sr, 16000)
        self._down = (16000 // g, io_sr // g)
        g = math.gcd(self.tgt_sr, io_sr)
        self._up = (io_sr // g, self.tgt_sr // g)
        log.info("RVC ready on %s (half=%s): model %s %s @ %d Hz, block %.0f ms",
                 self.device, self.half, Path(model_path).name, self.version, self.tgt_sr, block_time * 1000)

    def warmup(self):
        """Run one throwaway block so the GPU kernels are compiled before you go live."""
        saved = self.ring.copy()
        self.ring = (np.random.default_rng(0).normal(0, 0.05, len(self.ring))).astype(np.float32)
        try:
            self._infer()
        finally:
            self.ring = saved

    @property
    def latency_s(self) -> float:
        return (self.block_io + self.cross_io + self.sola_io) / self.io_sr

    # ------------------------------------------------------------------ core
    def process_block(self, block: np.ndarray, lilt_semitones: np.ndarray | None = None) -> np.ndarray:
        """`block` must be exactly `block_io` samples. Returns `block_io` converted samples."""
        block = np.asarray(block, dtype=np.float32)
        self.ring = np.concatenate([self.ring[len(block):], block])
        n_new = self.block16 // 160
        new_lilt = np.zeros(n_new, np.float32) if lilt_semitones is None else np.interp(
            np.linspace(0, 1, n_new), np.linspace(0, 1, len(lilt_semitones)), lilt_semitones).astype(np.float32)
        self.lilt = np.concatenate([self.lilt[n_new:], new_lilt])

        if float(np.sqrt(np.mean(block ** 2))) < self.silence:
            out = np.zeros(self.block_io + self.cross_io + self.sola_io, dtype=np.float32)
        else:
            out = self._infer()
        return self._sola(out)

    def _infer(self) -> np.ndarray:
        torch = self.torch
        x16 = signal.resample_poly(self.ring, *self._down).astype(np.float32)[: self.total16]
        x16 = np.pad(x16, (0, self.total16 - len(x16)))
        dtype = torch.float16 if self.half else torch.float32
        with torch.no_grad():
            src = torch.from_numpy(x16).to(self.device, dtype=dtype).unsqueeze(0)
            hs = self.hubert(src, output_hidden_states=True).hidden_states
            if self.version == "v1":
                feats = hs[min(9, len(hs) - 1)]
                if hasattr(self.hubert, "final_proj"):
                    feats = self.hubert.final_proj(feats)
            else:
                feats = hs[min(12, len(hs) - 1)]
            feats0 = feats.clone() if self.protect < 0.5 and self.has_f0 else None
            if self.index is not None and self.index_rate > 0:
                npy = feats[0].float().cpu().numpy()
                score, ix = self.index.search(npy, k=8)
                w = np.square(1 / np.maximum(score, 1e-9))
                w /= w.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(w, axis=2), axis=1)
                feats = torch.from_numpy(npy).unsqueeze(0).to(self.device, dtype=dtype) * self.index_rate \
                    + (1 - self.index_rate) * feats
            feats = torch.nn.functional.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            p_len = min(feats.shape[1], self.total16 // 160)
            feats = feats[:, :p_len]

            if self.has_f0:
                f0 = self.rmvpe.infer_from_audio(x16, thred=0.03)[:p_len]
                f0 = np.pad(f0, (0, p_len - len(f0)))
                bend = self.lilt[-p_len:] if len(self.lilt) >= p_len else np.pad(self.lilt, (p_len - len(self.lilt), 0))
                f0 = f0 * np.power(2.0, (self.key + bend) / 12.0)
                mel = 1127 * np.log(1 + f0 / 700)
                mel[mel > 0] = (mel[mel > 0] - _MEL_MIN) * 254 / (_MEL_MAX - _MEL_MIN) + 1
                coarse = np.rint(np.clip(mel, 1, 255)).astype(np.int64)
                pitch = torch.from_numpy(coarse).to(self.device).unsqueeze(0)
                pitchf = torch.from_numpy(f0.astype(np.float32)).to(self.device, dtype=dtype).unsqueeze(0)
                if feats0 is not None:  # keep breathy/unvoiced sounds closer to the original
                    feats0 = torch.nn.functional.interpolate(
                        feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)[:, :p_len]
                    mask = torch.where(pitchf > 0, torch.ones_like(pitchf), torch.full_like(pitchf, self.protect))
                    feats = feats * mask.unsqueeze(-1) + feats0 * (1 - mask.unsqueeze(-1))
            lengths = torch.tensor([p_len], device=self.device).long()
            rate = min(1.0, self.return_frames / p_len)
            if self.has_f0:
                audio = self.net.infer(feats.to(dtype), lengths, pitch, pitchf, self.sid, rate=rate)[0][0, 0]
            else:
                audio = self.net.infer(feats.to(dtype), lengths, self.sid, rate=rate)[0][0, 0]
            audio = audio.float().cpu().numpy()
        out = signal.resample_poly(audio, *self._up).astype(np.float32)
        need = self.block_io + self.cross_io + self.sola_io
        return out[-need:] if len(out) >= need else np.pad(out, (need - len(out), 0))

    def _sola(self, out: np.ndarray) -> np.ndarray:
        """Find the offset where the new chunk lines up with the previous tail, then crossfade."""
        c, s = self.cross_io, self.sola_io
        head = out[: c + s]
        if np.any(self.tail) and np.any(head):
            corr = np.correlate(head, self.tail, mode="valid")
            energy = np.sqrt(np.convolve(head ** 2, np.ones(c), mode="valid") + 1e-8)
            k = int(np.argmax(corr / energy))
        else:
            k = 0
        out = out[k:]
        out = np.pad(out, (0, max(0, self.block_io + c - len(out))))
        out[:c] = out[:c] * self.fade_in + self.tail * self.fade_out
        result = out[: self.block_io].copy()
        self.tail = out[self.block_io:self.block_io + c].copy()
        return result


class RVCStage:
    """Runs StreamingRVC on its own thread so the audio callback never waits for the GPU.

    feed() mic blocks of any size from the audio callback; read() the same number of
    converted samples back (silence until the first blocks are ready).
    """

    def __init__(self, rvc: StreamingRVC, lilt_engine=None):
        from ..engine import Fifo

        self.rvc = rvc
        self.lilt_engine = lilt_engine
        self.pending = np.zeros(0, dtype=np.float32)
        self.cond = threading.Condition()
        self.out = Fifo(rvc.io_sr, prefill=rvc.block_io // 2, max_prefill=rvc.block_io * 2)
        self.running = True
        self.last_ms = 0.0
        self.overruns = 0
        self.thread = threading.Thread(target=self._loop, name="bombovoice-rvc", daemon=True)
        self.thread.start()

    def feed(self, block: np.ndarray):
        with self.cond:
            self.pending = np.concatenate([self.pending, np.asarray(block, dtype=np.float32)])
            if len(self.pending) > self.rvc.block_io * 6:  # GPU can't keep up: drop the oldest
                self.pending = self.pending[-self.rvc.block_io * 2:]
                self.overruns += 1
            self.cond.notify()

    def read(self, n: int) -> np.ndarray:
        return self.out.pull(n)

    def _loop(self):
        import time

        bio = self.rvc.block_io
        while self.running:
            with self.cond:
                while self.running and len(self.pending) < bio:
                    self.cond.wait(0.5)
                if not self.running:
                    return
                block, self.pending = self.pending[:bio], self.pending[bio:]
            lilt = None
            if self.lilt_engine is not None:  # one bend value per 10 ms of the new block
                step = max(1, self.rvc.io_sr // 100)
                base = self.lilt_engine.base  # the voice model's key sets the pitch; keep only the movement
                lilt = np.array([float(np.mean(self.lilt_engine.process(block[i:i + step]))) - base
                                 for i in range(0, len(block), step)], dtype=np.float32)
            t0 = time.perf_counter()
            try:
                y = self.rvc.process_block(block, lilt)
            except Exception:
                log.exception("RVC block failed")
                y = np.zeros(bio, dtype=np.float32)
            self.last_ms = (time.perf_counter() - t0) * 1000
            self.out.push(y)

    def close(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
