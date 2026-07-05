"""snapshot.py — the frozen reference baseline every monitor compares against.

A drift monitor is only as trustworthy as its "known-good" reference. This
module freezes that reference once, from the wafer-mixed **test** split (data
the model never trained on), and persists it as a small, committed artifact so a
fresh clone can bring the monitoring stack up on CPU without re-running the
model or needing the 400 MB dataset.

Frozen quantities (all in LABELS column order):
    - embeddings        (N, D)  penultimate features — the covariate-drift bank
                                (Phase 1 MMD / KS reference), stored float16;
    - probs             (N, 8)  calibrated probabilities — reference for the
                                calibration monitor (Phase 2);
    - preds             (N, 8)  multi-hot decisions — reference prediction-rate
                                and label histogram (Phase 1 prediction drift);
    - y_true            (N, 8)  ground truth — lets Phase 2 score reference ECE;
    - a JSON summary    prediction_rate / label_histogram / per-label reference
                        ECE, plus a content hash for the determinism guarantee.

Determinism: the same split + seed + checkpoint + thresholds yield byte-identical
arrays (inference is deterministic on CPU, no augmentation, no shuffle). The
content_hash in the sidecar meta is the contract the determinism test checks.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from wafer_deploy.labels import LABELS
from wafer_deploy.predictor import Predictor

SCHEMA_VERSION = 1


@dataclasses.dataclass
class ReferenceSnapshot:
    embeddings: np.ndarray   # (N, D) float32 (stored float16)
    probs: np.ndarray        # (N, 8) float32
    preds: np.ndarray        # (N, 8) int8
    y_true: np.ndarray       # (N, 8) int8
    map_ids: np.ndarray      # (N,)   int64 — wafer-mixed test-split indices
    meta: dict

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])


def _content_hash(embeddings16: np.ndarray, probs: np.ndarray, preds: np.ndarray,
                  y_true: np.ndarray, map_ids: np.ndarray) -> str:
    """Stable hash over the persisted array bytes — the determinism contract."""
    h = hashlib.sha256()
    for arr in (embeddings16, probs.astype(np.float32), preds.astype(np.int8),
                y_true.astype(np.int8), map_ids.astype(np.int64)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _summary(predictor: Predictor, probs: np.ndarray, preds: np.ndarray,
             y_true: np.ndarray) -> dict:
    """Reference prediction-rate, label histogram and per-label ECE."""
    binary_ece = predictor.wm.calibrate.binary_ece
    ece = {LABELS[i]: float(binary_ece(probs[:, i], y_true[:, i]))
           for i in range(len(LABELS))}
    return {
        # fraction of maps on which each label fires (the prediction-rate ref)
        "prediction_rate": {LABELS[i]: float(preds[:, i].mean())
                            for i in range(len(LABELS))},
        # raw positive counts per label (chi-square / PSI reference histogram)
        "label_histogram": {LABELS[i]: int(preds[:, i].sum())
                            for i in range(len(LABELS))},
        # fraction of maps carrying at least one predicted defect
        "defect_rate": float((preds.sum(axis=1) > 0).mean()),
        "reference_ece": ece,
        "reference_ece_mean": float(np.mean(list(ece.values()))),
    }


def build_snapshot(predictor: Predictor, map_ids: np.ndarray,
                   batch_size: int = 128, progress: bool = False) -> ReferenceSnapshot:
    """Run the predictor over the given wafer-mixed indices → ReferenceSnapshot.

    ``map_ids`` are indices into the full MixedWM38 array (typically the test
    split, or a small prefix for the determinism test). Order is preserved.
    """
    wm = predictor.wm
    maps, labels = wm.data.load_raw(predictor.mixed_cfg.data_root)
    map_ids = np.asarray(map_ids, dtype=np.int64)

    emb_chunks, prob_chunks, pred_chunks = [], [], []
    rng = range(0, len(map_ids), batch_size)
    if progress:
        from tqdm import tqdm
        rng = tqdm(rng, desc="reference snapshot")
    for start in rng:
        idx = map_ids[start:start + batch_size]
        res = predictor.predict_batch([maps[i] for i in idx])
        emb_chunks.append(res.embeddings)
        prob_chunks.append(res.probs)
        pred_chunks.append(res.preds)

    embeddings = np.vstack(emb_chunks).astype(np.float32)
    probs = np.vstack(prob_chunks).astype(np.float32)
    preds = np.vstack(pred_chunks).astype(np.int8)
    y_true = labels[map_ids].astype(np.int8)

    embeddings16 = embeddings.astype(np.float16)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "n": int(len(map_ids)),
        "embedding_dim": int(embeddings.shape[1]),
        "label_names": list(LABELS),
        "source": "wafer-mixed test split",
        "checkpoint": predictor.checkpoint_meta,
        "content_hash": _content_hash(embeddings16, probs, preds, y_true, map_ids),
        "summary": _summary(predictor, probs, preds, y_true),
    }
    return ReferenceSnapshot(embeddings=embeddings, probs=probs, preds=preds,
                             y_true=y_true, map_ids=map_ids, meta=meta)


def save_snapshot(snapshot: ReferenceSnapshot, path: Path) -> None:
    """Persist arrays (embeddings as float16) + a sidecar meta JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=snapshot.embeddings.astype(np.float16),
        probs=snapshot.probs.astype(np.float32),
        preds=snapshot.preds.astype(np.int8),
        y_true=snapshot.y_true.astype(np.int8),
        map_ids=snapshot.map_ids.astype(np.int64),
    )
    path.with_suffix(".meta.json").write_text(json.dumps(snapshot.meta, indent=2))


def load_snapshot(path: Path) -> ReferenceSnapshot:
    """Load a persisted snapshot (embeddings restored to float32)."""
    path = Path(path)
    d = np.load(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return ReferenceSnapshot(
        embeddings=d["embeddings"].astype(np.float32),
        probs=d["probs"].astype(np.float32),
        preds=d["preds"].astype(np.int8),
        y_true=d["y_true"].astype(np.int8),
        map_ids=d["map_ids"].astype(np.int64),
        meta=meta,
    )
