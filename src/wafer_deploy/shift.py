"""shift.py — the *scored* controlled-shift sources (Phase 3).

Phase 1's harness carried one deliberately crude covariate shift (``rot90`` +
failing-die injection) so a demo could push a monitor into alarm. Phase 3 needs
shifts that are **graded** — an intensity knob you can sweep to trace a detection
curve — and **honest** about what they do to the input. This module is that
library, and it is numpy-only (no scipy) so the same corruption can run in the
lean serving image or an offline sweep with identical results.

Every per-map corruption has the signature ``fn(wmap, intensity, rng) -> wmap``
over a raw wafer map (values 0=off-wafer, 1=passing die, 2=failing die), with:

    - ``intensity == 0`` an **exact identity** (the false-alarm control at the
      foot of every detection curve depends on this — a sweep's zero point must
      be the un-corrupted map, byte-for-byte);
    - ``intensity`` rising in [0, 1] monotonically increasing the distortion.

The three per-map corruptions probe different failure modes:

    - ``rotation``    — the wafer imaged at a rotated orientation (fixture /
      handler drift). Wafers have ~4-fold symmetry, so the meaningful range is
      0–45°; nearest-neighbour sampling keeps pixels in {0,1,2}.
    - ``noise``       — random die-state flips inside the wafer (probe-card
      contamination / measurement noise). Moves the embedding without a clean
      spatial cause.
    - ``resolution``  — the map coarsened then restored (a different die grid /
      binning). Erodes fine spatial detail, the signal edge- and scratch-type
      defects live in.

``class_prior_campaign`` is different in kind: a **stream-level** shift (a
"defect campaign" ramping one label's prevalence over time), so it returns an
ordering of pool indices rather than a corrupted map — the prediction-rate
monitor is the one built to catch it, and it needs no re-inference.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

Corruption = Callable[[np.ndarray, float, Optional[np.random.Generator]], np.ndarray]


# --------------------------------------------------------------------------- #
# Per-map corruptions (graded; intensity 0 is an exact identity).
# --------------------------------------------------------------------------- #

# Corruption strength at intensity 1.0. Chosen so intensity 1 is a clearly
# out-of-distribution map without being pure noise (the point is a *detectable*
# shift, not a destroyed one — heavy-intensity behaviour is reported honestly).
MAX_ROTATION_DEG = 45.0    # wafer ~4-fold symmetric → 45° is the far corner
MAX_FLIP_FRACTION = 0.30   # fraction of on-wafer die whose state is flipped
MAX_COARSEN_FACTOR = 6     # downsample block size at full intensity


def rotate_map(wmap: np.ndarray, intensity: float,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Rotate the wafer about its centre by ``intensity * MAX_ROTATION_DEG``.

    Nearest-neighbour inverse sampling (numpy-only), so output pixels stay in
    {0,1,2}; anything sampled from outside the source is off-wafer (0). Identity
    at intensity 0 (angle 0 maps every pixel to itself).
    """
    wmap = np.asarray(wmap)
    if intensity <= 0:
        return wmap.copy()
    angle = np.deg2rad(intensity * MAX_ROTATION_DEG)
    h, w = wmap.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.indices((h, w))
    # Inverse map: for each output pixel, find the source pixel (rotate by -angle).
    ca, sa = np.cos(-angle), np.sin(-angle)
    src_x = ca * (xs - cx) - sa * (ys - cy) + cx
    src_y = sa * (xs - cx) + ca * (ys - cy) + cy
    src_xi = np.rint(src_x).astype(int)
    src_yi = np.rint(src_y).astype(int)
    inside = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)
    out = np.zeros_like(wmap)
    out[ys[inside], xs[inside]] = wmap[src_yi[inside], src_xi[inside]]
    return out


def noise_map(wmap: np.ndarray, intensity: float,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Flip a fraction ``intensity * MAX_FLIP_FRACTION`` of on-wafer die states.

    On-wafer die (values 1/2) are randomly toggled pass↔fail; off-wafer pixels
    (0) are untouched, so the wafer *shape* is preserved and only the die-state
    pattern degrades. Identity at intensity 0 (no die selected).
    """
    wmap = np.asarray(wmap)
    if intensity <= 0:
        return wmap.copy()
    rng = rng if rng is not None else np.random.default_rng(0)
    out = wmap.copy()
    on_wafer = out > 0
    flip = on_wafer & (rng.random(out.shape) < intensity * MAX_FLIP_FRACTION)
    # 1 (pass) ↔ 2 (fail): 3 - value swaps the two on-wafer states.
    out[flip] = 3 - out[flip]
    return out


def resolution_map(wmap: np.ndarray, intensity: float,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Coarsen the map to a lower die grid, then restore its size (detail loss).

    Downsample by a block factor ``f`` (1 → MAX_COARSEN_FACTOR as intensity
    rises) taking each block's majority die-state, then nearest-neighbour
    upsample back to the original shape. Identity at intensity 0 (factor 1).
    Numpy-only; the majority vote keeps pixels in {0,1,2}.
    """
    wmap = np.asarray(wmap)
    if intensity <= 0:
        return wmap.copy()
    # factor in [1, MAX_COARSEN_FACTOR]; round to an integer block size.
    f = int(round(1 + intensity * (MAX_COARSEN_FACTOR - 1)))
    if f <= 1:
        return wmap.copy()
    h, w = wmap.shape
    out = np.empty_like(wmap)
    for y0 in range(0, h, f):
        for x0 in range(0, w, f):
            block = wmap[y0:y0 + f, x0:x0 + f]
            # majority die-state in the block (0/1/2), ties → lowest value.
            vals, counts = np.unique(block, return_counts=True)
            out[y0:y0 + f, x0:x0 + f] = vals[counts.argmax()]
    return out


CORRUPTIONS: dict[str, Corruption] = {
    "rotation": rotate_map,
    "noise": noise_map,
    "resolution": resolution_map,
}


# --------------------------------------------------------------------------- #
# Stream-level shift: a class-prior "defect campaign".
# --------------------------------------------------------------------------- #

def class_prior_campaign(labels: np.ndarray, target_label_idx: int,
                         n_windows: int, window_size: int, *,
                         onset_window: int, max_share: float = 0.6,
                         seed: int = 0) -> np.ndarray:
    """Time-ordered pool indices whose target-label prevalence ramps up.

    Simulates a "defect campaign": a fab excursion that makes one defect type
    progressively dominate the incoming stream. Windows before ``onset_window``
    are sampled to match the pool's natural prior (the null); from onset on, the
    target label's share ramps linearly to ``max_share`` by the final window.
    The predicted-label PSI monitor is the one built to catch this, and it does
    so **without any re-inference** — the maps already exist in the pool.

    Returns a flat index array of length ``n_windows * window_size`` into
    ``labels`` (multi-hot, one row per pool map), sampled with replacement.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    has_target = labels[:, target_label_idx] > 0
    tgt_pool = np.flatnonzero(has_target)
    other_pool = np.flatnonzero(~has_target)
    if tgt_pool.size == 0 or other_pool.size == 0:
        raise ValueError("target label absent from, or saturates, the pool")
    base_share = float(has_target.mean())

    out: list[np.ndarray] = []
    for k in range(n_windows):
        if k < onset_window:
            share = base_share
        else:
            frac = (k - onset_window + 1) / max(1, n_windows - onset_window)
            share = base_share + (max_share - base_share) * frac
        n_tgt = int(round(share * window_size))
        idx = np.concatenate([
            rng.choice(tgt_pool, size=n_tgt, replace=True),
            rng.choice(other_pool, size=window_size - n_tgt, replace=True),
        ])
        rng.shuffle(idx)
        out.append(idx)
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# WM-811K → MixedWM38 label bridge (the real cross-dataset shift, Phase 3(b)).
# --------------------------------------------------------------------------- #

def wm811k_to_multihot(label_strings: Sequence[str],
                       labels: Sequence[str]) -> np.ndarray:
    """Map WM-811K single-defect class names to MixedWM38 multi-hot rows.

    WM-811K's 8 defect classes share the MixedWM38 taxonomy exactly (same eight
    names); its ninth class ``none`` (defect-free) has no MixedWM38 label and
    maps to an all-zero row. Unknown names raise — a silent miss would corrupt
    the calibration reference for the cross-dataset experiment.
    """
    index = {name: i for i, name in enumerate(labels)}
    out = np.zeros((len(label_strings), len(labels)), dtype=np.int64)
    for r, name in enumerate(label_strings):
        if name == "none":
            continue
        if name not in index:
            raise ValueError(f"WM-811K label {name!r} not in MixedWM38 labels {list(labels)}")
        out[r, index[name]] = 1
    return out
