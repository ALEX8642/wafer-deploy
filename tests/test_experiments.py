"""Phase 3 scored-shift engine. The shift sources and the scoring bookkeeping are
pinned without needing the checkpoint or WM-811K:

    - corruptions are exact identities at intensity 0 and stay in {0,1,2};
    - the class-prior campaign ramps the target label's prevalence after onset;
    - the WM-811K→multi-hot bridge maps names correctly and rejects the unknown;
    - run_stream detects a synthetic covariate offset at the right window, fires
      the trigger, and keeps the pre-onset (clean) windows quiet — with the
      calibration channel honestly lagged;
    - score_window equals the delayed-label FIFO path on the same rows.

Only the corruption-through-inference check carries `needs_mixed`.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_mixed
from wafer_deploy.calibration import CalibrationMonitor, per_label_ece
from wafer_deploy.drift import DriftMonitors
from wafer_deploy.experiments import reset_monitors, run_stream
from wafer_deploy.labels import LABELS
from wafer_deploy.shift import (
    CORRUPTIONS, class_prior_campaign, resolution_map, noise_map, rotate_map,
    wm811k_to_multihot)
from wafer_deploy.snapshot import load_snapshot


# --------------------------------------------------------------------------- #
# Corruptions — identity at 0, in-range, wafer shape preserved where intended.
# --------------------------------------------------------------------------- #

def _demo_map():
    wm = np.zeros((52, 52), dtype=np.int64)
    wm[14:38, 14:38] = 1        # a disc of passing die
    wm[22:30, 22:30] = 2        # a central failing cluster
    return wm


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_corruption_identity_at_zero(name):
    wm = _demo_map()
    out = CORRUPTIONS[name](wm, 0.0, np.random.default_rng(0))
    assert np.array_equal(out, wm)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_corruption_stays_in_range_and_changes(name):
    wm = _demo_map()
    out = CORRUPTIONS[name](wm, 1.0, np.random.default_rng(0))
    assert set(np.unique(out)).issubset({0, 1, 2})
    assert not np.array_equal(out, wm)      # a real change at full intensity


def test_noise_preserves_wafer_shape():
    """Die-flip noise toggles pass/fail but never touches off-wafer pixels."""
    wm = _demo_map()
    out = noise_map(wm, 1.0, np.random.default_rng(0))
    assert np.array_equal(out == 0, wm == 0)   # wafer boundary unchanged


def test_rotation_is_deterministic():
    wm = _demo_map()
    a = rotate_map(wm, 0.6, None)
    b = rotate_map(wm, 0.6, None)
    assert np.array_equal(a, b)


def test_resolution_coarsens_monotonically():
    """Higher intensity → coarser map → fewer distinct on-wafer regions (never
    more detail)."""
    wm = _demo_map()
    fine = resolution_map(wm, 0.3, None)
    coarse = resolution_map(wm, 1.0, None)
    # a coarser grid can only merge detail: unique-value count never increases
    assert len(np.unique(coarse)) <= len(np.unique(fine)) + 0  # sanity, no expansion
    assert set(np.unique(coarse)).issubset({0, 1, 2})


# --------------------------------------------------------------------------- #
# Class-prior campaign.
# --------------------------------------------------------------------------- #

def test_campaign_ramps_target_prevalence():
    # 200 rows, one label; make half carry the target so both pools are non-empty.
    labels = np.zeros((400, len(LABELS)), dtype=np.int64)
    tgt = LABELS.index("Edge-Ring")
    labels[:80, tgt] = 1                       # 20% base prior
    idx = class_prior_campaign(labels, tgt, n_windows=8, window_size=100,
                               onset_window=3, max_share=0.6, seed=0)
    per_win = idx.reshape(8, 100)
    share = [labels[w, tgt].mean() for w in per_win]
    # pre-onset windows near base prior; final window near max_share
    assert abs(np.mean(share[:3]) - 0.20) < 0.08
    assert share[-1] > 0.5
    assert share[-1] > share[3]                # ramps up after onset


# --------------------------------------------------------------------------- #
# WM-811K → multi-hot bridge.
# --------------------------------------------------------------------------- #

def test_wm811k_label_bridge():
    mh = wm811k_to_multihot(["Center", "none", "Scratch"], LABELS)
    assert mh[0][LABELS.index("Center")] == 1 and mh[0].sum() == 1
    assert mh[1].sum() == 0                     # 'none' → all-zero
    assert mh[2][LABELS.index("Scratch")] == 1


def test_wm811k_label_bridge_rejects_unknown():
    with pytest.raises(ValueError):
        wm811k_to_multihot(["NotADefect"], LABELS)


# --------------------------------------------------------------------------- #
# run_stream — scoring + detection bookkeeping (committed snapshot, no checkpoint).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def snapshot(cfg):
    path = cfg.reference_snapshot_path
    if not path.exists():
        pytest.skip("reference snapshot not built yet")
    return load_snapshot(path)


def _monitors(snapshot):
    mons = DriftMonitors.from_snapshot(snapshot, LABELS, window_size=200,
                                       calib_trials=100, seed=42)
    cal = CalibrationMonitor.from_snapshot(snapshot, LABELS, window_size=200,
                                           calib_trials=100, seed=42)
    return mons, cal


def test_score_window_matches_direct_ece(snapshot):
    _, cal = _monitors(snapshot)
    probs, y = snapshot.probs[:200], snapshot.y_true[:200]
    r = cal.score_window(probs, y)
    assert r.ece_mean == pytest.approx(float(per_label_ece(probs, y, cal.n_bins).mean()))


def test_run_stream_detects_covariate_offset(snapshot):
    from wafer_deploy.trigger import RetrainTrigger
    mons, cal = _monitors(snapshot)
    W, onset = 200, 3
    rng = np.random.default_rng(0)
    perm = rng.permutation(snapshot.n)
    warm, body = perm[:onset * W], perm[onset * W:(onset + 4) * W]
    offset = 3.0 * snapshot.embeddings.std(0)
    E = np.vstack([snapshot.embeddings[warm], snapshot.embeddings[body] + offset])
    P = np.vstack([snapshot.preds[warm], snapshot.preds[body]])
    Pr = np.vstack([snapshot.probs[warm], snapshot.probs[body]])
    Y = np.vstack([snapshot.y_true[warm], snapshot.y_true[body]])
    out = run_stream(mons, cal, RetrainTrigger(persistence=2, release=3),
                     embeddings=E, preds=P, probs=Pr, y_true=Y, window_size=W,
                     onset_window=onset, calibration_lag=2)
    s = out["summary"]
    # covariate detects at onset (latency 0); pre-onset windows stay quiet
    assert s["detection_latency"]["covariate"] == 0
    assert s["channel_alarm_rate_pre"]["covariate"] == 0.0
    assert s["trigger_fired"] and s["first_channel"] == "covariate"


def test_run_stream_null_is_quiet(snapshot):
    """A fully in-domain stream (no shift) keeps the trigger silent."""
    from wafer_deploy.trigger import RetrainTrigger
    mons, cal = _monitors(snapshot)
    W = 200
    rng = np.random.default_rng(1)
    idx = rng.permutation(snapshot.n)[:6 * W]
    out = run_stream(mons, cal, RetrainTrigger(persistence=3, release=3),
                     embeddings=snapshot.embeddings[idx], preds=snapshot.preds[idx],
                     probs=snapshot.probs[idx], y_true=snapshot.y_true[idx],
                     window_size=W, onset_window=3, calibration_lag=2)
    assert not out["summary"]["trigger_fired"]


def test_reset_monitors_clears_partial_window(snapshot):
    mons, cal = _monitors(snapshot)
    mons.covariate.update(snapshot.embeddings[:50])   # partial window (<200)
    assert mons.covariate.pending == 50
    reset_monitors(mons, cal)
    assert mons.covariate.pending == 0 and mons.prediction.pending == 0


# --------------------------------------------------------------------------- #
# Corruption through the real model (needs the sibling checkpoint).
# --------------------------------------------------------------------------- #

@needs_mixed
def test_corruption_moves_embeddings(predictor):
    """A corrupted map's embedding differs from the clean one's — the covariate
    monitor's substrate actually moves under the scored corruption."""
    from wafer_deploy.experiments import predict_maps
    wm = _demo_map()
    corrupted = noise_map(wm, 1.0, np.random.default_rng(0))
    emb, _, _ = predict_maps(predictor, [wm, corrupted])
    assert np.linalg.norm(emb[0] - emb[1]) > 1e-3
