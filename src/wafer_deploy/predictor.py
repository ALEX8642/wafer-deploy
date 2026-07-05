"""predictor.py — the fixed wafer-mixed model as a CPU inference object.

Loads the frozen checkpoint once and exposes the exact calibrated decision path
wafer-mixed itself uses — ``encode_map`` → ``resize_map`` → model →
``scale_probs`` → ``predict_multihot`` — so a served prediction is bit-for-bit
what wafer-mixed's own evaluate path would produce (the Phase 0 parity test
pins this).

It additionally captures the model's **penultimate features** via a forward
pre-hook on the final Linear layer. Those embeddings are the substrate for the
Phase 1 covariate-drift monitor (MMD / KS in embedding space); exposing them
here — once, at the single inference site — is why the hook lives in Phase 0.
"""
from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from wafer_deploy.bridge import load_thresholds, wafer_mixed_modules
from wafer_deploy.config import DeployConfig
from wafer_deploy.labels import LABELS


@dataclasses.dataclass
class PredictionResult:
    """Batch prediction, all arrays in (N, ...) with columns in LABELS order."""
    logits: np.ndarray      # (N, 8) raw, float32
    probs: np.ndarray       # (N, 8) temperature-scaled sigmoid, float64
    preds: np.ndarray       # (N, 8) multi-hot at per-label tau, int64
    embeddings: np.ndarray  # (N, D) penultimate features, float32


class Predictor:
    """Frozen, calibrated wafer-mixed model wrapped for serving + monitoring."""

    def __init__(self, cfg: DeployConfig) -> None:
        self.cfg = cfg
        self.wm = wafer_mixed_modules(cfg.mixed_root)

        # MixedConfig gives the WAFER_DEVICE env var top priority; a training
        # shell leftover could resolve to cuda and split model/inputs across
        # devices. This repo is CPU-only by policy — force it after init.
        mixed_cfg = self.wm.MixedConfig(
            device="cpu", batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        mixed_cfg.device = "cpu"
        self.mixed_cfg = mixed_cfg

        self.model, self.ckpt = self.wm.model.load_checkpoint_model(
            mixed_cfg, cfg.checkpoint_path)
        self.model.eval()
        self.input_size = int(mixed_cfg.input_size)
        self.labels = list(LABELS)

        self.T, self.tau = load_thresholds(cfg.thresholds_path)

        # Penultimate-feature hook: the final Linear's input is the pooled
        # backbone embedding (512-d for resnet18, 2048-d for resnet50).
        # `_captured` is shared model state written by the hook, so the
        # forward + read must be atomic: uvicorn runs sync endpoints in a
        # threadpool, and without this lock concurrent /predict calls race on
        # the buffer (500s, or — worse — one request silently reading another's
        # embedding into the drift monitor). A single CPU model serialises
        # inference here; throughput scales by replicas, not threads.
        self._captured: torch.Tensor | None = None
        self._infer_lock = threading.Lock()
        self.embedding_dim = int(self.model.fc.in_features)
        self.model.fc.register_forward_pre_hook(self._capture_embedding)

    # ---- internals ---------------------------------------------------------

    def _capture_embedding(self, _module, args) -> None:
        # args[0]: (B, in_features) features fed to the final classification head.
        self._captured = args[0].detach()

    def encode(self, wmap: np.ndarray) -> torch.Tensor:
        """One raw wafer map (values 0/1/2) → (3, S, S) model-ready tensor.

        Identical to MixedWaferDataset with augment=False: the same one-hot
        encode + nearest-neighbour resize, so nothing about the input path can
        diverge from training/eval.
        """
        t = self.wm.data.encode_map(wmap)
        return self.wm.data.resize_map(t, self.input_size)

    # ---- public API --------------------------------------------------------

    @torch.no_grad()
    def predict_batch(self, maps: Sequence[np.ndarray]) -> PredictionResult:
        """Calibrated prediction over a batch of raw wafer maps."""
        if len(maps) == 0:
            raise ValueError("predict_batch received an empty batch")
        batch = torch.stack([self.encode(m) for m in maps]).to(self.mixed_cfg.device)
        # Hold the lock across the forward + capture read so the hook's shared
        # buffer belongs to exactly this call (see __init__). Encoding above is
        # pure/thread-safe and stays outside the critical section.
        with self._infer_lock:
            self._captured = None
            logits = self.model(batch).float().cpu().numpy()
            if self._captured is None:  # hook must have fired
                raise RuntimeError("embedding hook did not capture features")
            embeddings = self._captured.float().cpu().numpy()
        probs = self.wm.calibrate.scale_probs(logits, self.T)
        preds = self.wm.metrics.predict_multihot(probs, self.tau)
        return PredictionResult(
            logits=logits.astype(np.float32),
            probs=np.asarray(probs, dtype=np.float64),
            preds=np.asarray(preds, dtype=np.int64),
            embeddings=embeddings.astype(np.float32),
        )

    def predict_one(self, wmap: np.ndarray) -> PredictionResult:
        """Convenience: single map, still returned as (1, ...) arrays."""
        return self.predict_batch([np.asarray(wmap)])

    @property
    def checkpoint_meta(self) -> dict:
        """Small, JSON-safe description of the loaded checkpoint for /healthz."""
        return {
            "epoch": self.ckpt.get("epoch"),
            "val_macro_f1": (None if self.ckpt.get("val_macro_f1") is None
                             else float(self.ckpt.get("val_macro_f1"))),
            "arch": self.mixed_cfg.arch,
            "input_size": self.input_size,
            "embedding_dim": self.embedding_dim,
        }


def build_predictor(cfg: DeployConfig | None = None,
                    yaml_path: Path | None = None) -> Predictor:
    """Load a Predictor from a DeployConfig (or the default resolved config)."""
    if cfg is None:
        cfg = DeployConfig.load(yaml_path)
    return Predictor(cfg)
