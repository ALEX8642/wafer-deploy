"""Phase 2 calibration-decay monitor. The ECE math and the delayed-label
bookkeeping are pinned the same three ways the drift monitors are:

    - binary_ece equals a hand-computed value on a toy array (the arithmetic
      anchor) and — when the sibling checkout is present — equals wafer-mixed's
      own binary_ece, so windowed ECE is the *same measurement* as the reference;
    - a NO-DRIFT stream (held-out reference rows fed with their true labels)
      keeps ECE below the calibrated threshold — the false-alarm control;
    - a confidence-eroded stream drives ECE over the threshold and alarms.

Delayed-label lag bookkeeping (labels FIFO-matched to the right window, a window
scored only once its labels arrive) is pinned directly. Snapshot-backed tests
need neither the checkpoint nor the dataset; only `needs_mixed` drives the live
/predict → /feedback path end to end.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import needs_mixed
from wafer_deploy.calibration import (
    CalibrationMonitor, binary_ece, per_label_ece, reliability_bins)
from wafer_deploy.labels import LABELS, NUM_LABELS
from wafer_deploy.snapshot import load_snapshot


# --------------------------------------------------------------------------- #
# binary_ece — the arithmetic anchor + parity with wafer-mixed.
# --------------------------------------------------------------------------- #

def test_binary_ece_hand_computed_value():
    # 4 points, n_bins=2. bin1 (0,0.5]: p=[0.2,0.4] y=[0,1] → |0.3-0.5|=0.2, w=0.5
    #            bin2 (0.5,1]: p=[0.6,0.9] y=[1,1] → |0.75-1.0|=0.25, w=0.5
    p = np.array([0.2, 0.4, 0.6, 0.9])
    y = np.array([0, 1, 1, 1])
    hand = 0.5 * abs(0.3 - 0.5) + 0.5 * abs(0.75 - 1.0)
    assert binary_ece(p, y, n_bins=2) == pytest.approx(hand)


def test_binary_ece_zero_when_confidence_equals_rate():
    # Every point at p=0.5 with exactly half positive → perfectly calibrated bin.
    p = np.full(100, 0.5)
    y = np.array([0, 1] * 50)
    assert binary_ece(p, y, n_bins=15) == pytest.approx(0.0, abs=1e-12)


@needs_mixed
def test_binary_ece_matches_wafer_mixed(predictor):
    """Our numpy ECE must equal wafer-mixed's own binary_ece bin-for-bin, so the
    reference ECE (built by wafer-mixed) and windowed ECE (here) are comparable."""
    wm_ece = predictor.wm.calibrate.binary_ece
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = rng.random(500)
        y = (rng.random(500) < p).astype(int)  # roughly calibrated
        assert binary_ece(p, y, 15) == pytest.approx(wm_ece(p, y, 15), rel=1e-12)


def test_reliability_bins_breaks_empty_bins():
    p = np.array([0.05, 0.06, 0.95, 0.96])  # only the first and last bins occupied
    y = np.array([0, 0, 1, 1])
    conf, acc, cnt = reliability_bins(p, y, n_bins=10)
    assert cnt[0] == 2 and cnt[-1] == 2 and cnt[1:-1].sum() == 0
    assert np.isnan(conf[5]) and np.isnan(acc[5])       # empty middle → NaN
    assert acc[0] == pytest.approx(0.0) and acc[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Delayed-label bookkeeping — the lag/FIFO contract.
# --------------------------------------------------------------------------- #

def _toy_monitor(window_size=50, seed=0):
    """A monitor with a synthetic, roughly-calibrated reference."""
    rng = np.random.default_rng(seed)
    probs = rng.random((2000, NUM_LABELS))
    y = (rng.random((2000, NUM_LABELS)) < probs).astype(int)
    return CalibrationMonitor(probs, y, LABELS, window_size=window_size,
                              calib_trials=50, seed=seed), rng


def test_no_score_until_a_full_window_of_labels_arrives():
    mon, rng = _toy_monitor(window_size=50)
    mon.buffer_predictions(rng.random((120, NUM_LABELS)))    # 120 served, unlabelled
    assert mon.pending_labels == 120
    y = (rng.random((49, NUM_LABELS)) < 0.5).astype(int)
    assert mon.add_labels(y) == []                            # 49 < window → nothing
    assert mon.pending_window == 49 and mon.pending_labels == 71
    # the 50th label completes exactly one window
    results = mon.add_labels((rng.random((1, NUM_LABELS)) < 0.5).astype(int))
    assert len(results) == 1 and results[0].window_id == 0
    assert mon.pending_window == 0 and mon.pending_labels == 70


def test_labels_are_matched_fifo_to_the_right_window():
    """A window's ECE must be computed from the probs of the *same* served maps
    its labels belong to — i.e. FIFO alignment, not whichever rows are handy."""
    mon, _ = _toy_monitor(window_size=40)
    rng = np.random.default_rng(7)
    probs = rng.random((40, NUM_LABELS))
    labels = (rng.random((40, NUM_LABELS)) < 0.5).astype(int)
    mon.buffer_predictions(probs)
    (result,) = mon.add_labels(labels)
    # scoring those exact 40 (prob,label) pairs directly must reproduce the window
    expected = float(per_label_ece(probs, labels, mon.n_bins).mean())
    assert result.ece_mean == pytest.approx(expected)


def test_label_without_prediction_raises():
    mon, _ = _toy_monitor(window_size=10)
    with pytest.raises(ValueError):
        mon.add_labels(np.zeros((1, NUM_LABELS), dtype=int))


def test_retention_cap_bounds_buffer_and_keeps_alignment():
    """Past max_pending the oldest awaiting predictions are evicted; their labels
    are skipped on arrival so the surviving window still scores its own rows."""
    rng = np.random.default_rng(1)
    probs = rng.random((500, NUM_LABELS))
    y = (rng.random((500, NUM_LABELS)) < probs).astype(int)
    mon = CalibrationMonitor(probs, y, LABELS, window_size=20, max_pending=50,
                             calib_trials=30, seed=1)
    # Buffer 80 predictions into a 50-cap buffer → 30 oldest evicted.
    stream_p = rng.random((80, NUM_LABELS))
    stream_y = (rng.random((80, NUM_LABELS)) < 0.5).astype(int)
    dropped = mon.buffer_predictions(stream_p)
    assert dropped == 30 and mon.dropped_total == 30
    assert mon.pending_labels == 50                       # bounded at the cap
    # Feed all 80 labels: the first 30 are skipped (their preds are gone), the
    # remaining 50 match the surviving predictions → exactly 2 windows of 20.
    results = mon.add_labels(stream_y)
    assert len(results) == 2
    # The first scored window must be built from predictions 30..49 (the oldest
    # survivors) paired with labels 30..49 — verify against a direct computation.
    expected = float(per_label_ece(stream_p[30:50], stream_y[30:50],
                                   mon.n_bins).mean())
    assert results[0].ece_mean == pytest.approx(expected)


def test_max_pending_below_window_rejected():
    rng = np.random.default_rng(0)
    probs = rng.random((100, NUM_LABELS))
    with pytest.raises(ValueError):
        CalibrationMonitor(probs, (probs > 0.5).astype(int), LABELS,
                           window_size=50, max_pending=40, calib_trials=5)


# --------------------------------------------------------------------------- #
# Snapshot-backed: null control + confidence-erosion alarm.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def snapshot(cfg):
    path = cfg.reference_snapshot_path
    if not path.exists():
        pytest.skip("reference snapshot not built yet")
    return load_snapshot(path)


def _erode(p, gamma):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p ** gamma / (p ** gamma + (1 - p) ** gamma)


def test_reference_ece_matches_committed_meta(snapshot):
    """The monitor's reference ECE must reproduce the snapshot's own summary —
    the same calibrated measurement wafer-mixed reported."""
    mon = CalibrationMonitor.from_snapshot(snapshot, LABELS, window_size=200,
                                           calib_trials=50, seed=0)
    committed = snapshot.meta["summary"]["reference_ece_mean"]
    # rel 1e-4, not exact: the committed value accumulated the bin means in the
    # snapshot's float32 probs; the monitor casts to float64. Same measurement,
    # different accumulation dtype — agreement to ~1e-6 relative is the contract.
    assert mon.reference_ece_mean == pytest.approx(committed, rel=1e-4)
    assert mon.threshold > mon.reference_ece_mean  # null quantile sits above ref


def test_no_drift_stream_false_alarm_control(snapshot):
    """Build on one random half, stream the disjoint half's (probs, labels): the
    windowed ECE stays (mostly) below the calibrated threshold."""
    rng = np.random.default_rng(0)
    perm = rng.permutation(snapshot.n)
    half = snapshot.n // 2
    build, stream = perm[:half], perm[half:]
    mon = CalibrationMonitor(snapshot.probs[build], snapshot.y_true[build], LABELS,
                             window_size=200, ece_quantile=0.99, calib_trials=200,
                             seed=0)
    mon.buffer_predictions(snapshot.probs[stream])
    results = mon.add_labels(snapshot.y_true[stream])
    assert len(results) >= 10, "need enough windows to estimate a rate"
    fa_rate = np.mean([r.alarm for r in results])
    assert fa_rate <= 0.25, f"no-drift false-alarm rate too high: {fa_rate}"


def test_confidence_erosion_raises_ece_and_alarms(snapshot):
    """A monotone confidence-erosion warp (γ=0.3) inflates ECE well past the
    threshold and fires on essentially every window — decisions unchanged."""
    mon = CalibrationMonitor.from_snapshot(snapshot, LABELS, window_size=200,
                                           calib_trials=200, seed=0)
    n = (snapshot.n // 200) * 200
    mon.buffer_predictions(_erode(snapshot.probs[:n], 0.3))
    results = mon.add_labels(snapshot.y_true[:n])
    assert results
    assert np.mean([r.alarm for r in results]) == 1.0
    assert all(r.ece_delta > 0 for r in results)


def test_from_snapshot_builds_calibration(snapshot):
    mon = CalibrationMonitor.from_snapshot(snapshot, LABELS, window_size=150,
                                           calib_trials=100, seed=0)
    assert mon.threshold > 0
    assert set(mon.labels) == set(LABELS)
    mon.buffer_predictions(snapshot.probs[:150])
    (r,) = mon.add_labels(snapshot.y_true[:150])
    assert set(r.ece_per_label) == set(LABELS) and r.n == 150


# --------------------------------------------------------------------------- #
# End-to-end through the live service (needs the sibling checkpoint).
# --------------------------------------------------------------------------- #

@needs_mixed
def test_metrics_expose_calibration_gauges(client):
    from prometheus_client.parser import text_string_to_metric_families
    names = {m.name for m in text_string_to_metric_families(client.get("/metrics").text)}
    for g in ("wafer_deploy_calibration_ece", "wafer_deploy_calibration_ece_threshold",
              "wafer_deploy_calibration_reference_ece"):
        assert g in names, f"{g} missing from /metrics"


@needs_mixed
def test_feedback_scores_window_through_service(client, mixed_data):
    """Serve a full window of maps, then POST their labels to /feedback → a
    calibration window is scored and the calibration windows_total advances."""
    from prometheus_client.parser import text_string_to_metric_families
    maps, labels, test_idx = mixed_data
    if client.app.state.calibration is None:
        pytest.skip("calibration monitor not active (no reference snapshot)")

    def cal_windows() -> float:
        for fam in text_string_to_metric_families(client.get("/metrics").text):
            if fam.name == "wafer_deploy_drift_windows":
                for s in fam.samples:
                    if s.labels.get("monitor") == "calibration":
                        return s.value
        return 0.0

    w = client.app.state.cfg.drift_window_size
    before = cal_windows()
    idxs = [int(test_idx[i % len(test_idx)]) for i in range(w)]
    for i in idxs:
        client.post("/predict", json={"wafer_map": maps[i].tolist()})
    # Feedback is rejected before predictions exist and accepted in FIFO order.
    label_batch = [labels[i].astype(int).tolist() for i in idxs]
    resp = client.post("/feedback", json={"labels": label_batch})
    assert resp.status_code == 200
    body = resp.json()
    assert body["windows_scored"] == 1 and body["latest"]["window_id"] >= 0
    assert cal_windows() >= before + 1


@needs_mixed
def test_feedback_rejects_wrong_width(client):
    if client.app.state.calibration is None:
        pytest.skip("calibration monitor not active")
    bad = client.post("/feedback", json={"labels": [[1, 0, 1]]})  # width != 8
    assert bad.status_code == 422
