"""Phase 1 drift monitors. The covariate/PSI math is pinned three ways:

    - PSI equals a hand-computed value on a toy histogram (the arithmetic anchor);
    - a NO-DRIFT stream (reference held-out rows) keeps MMD² below the calibrated
      threshold — the false-alarm control the Accept criteria require;
    - a synthetic SHIFTED stream drives MMD² above the threshold and fires alarms.

These run off the committed reference snapshot, so they need neither the sibling
checkpoint nor the dataset — only `needs_mixed` tests below drive the live
service end-to-end.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import needs_mixed
from wafer_deploy.drift import (
    CovariateDriftMonitor, DriftMonitors, PredictionDriftMonitor,
    ks_per_dim, psi, rbf_mmd2,
)
from wafer_deploy.labels import LABELS
from wafer_deploy.snapshot import load_snapshot


# --------------------------------------------------------------------------- #
# PSI — the arithmetic anchor.
# --------------------------------------------------------------------------- #

def test_psi_matches_hand_computed_value():
    # expected=[0.4,0.6], actual=[0.5,0.5]:
    #   (0.5-0.4)ln(0.5/0.4) + (0.5-0.6)ln(0.5/0.6)
    expected = [0.4, 0.6]
    actual = [0.5, 0.5]
    hand = (0.5 - 0.4) * math.log(0.5 / 0.4) + (0.5 - 0.6) * math.log(0.5 / 0.6)
    assert psi(expected, actual) == pytest.approx(hand, rel=1e-9)


def test_psi_is_zero_on_identical_distributions():
    assert psi([10, 20, 70], [1, 2, 7]) == pytest.approx(0.0, abs=1e-12)


def test_psi_normalizes_counts_or_proportions():
    # counts and their proportions must give the same PSI
    assert psi([40, 60], [50, 50]) == pytest.approx(psi([0.4, 0.6], [0.5, 0.5]))


# --------------------------------------------------------------------------- #
# MMD² — sanity on constructed distributions.
# --------------------------------------------------------------------------- #

def test_mmd2_near_zero_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 16))
    y = rng.normal(size=(200, 16))
    val = rbf_mmd2(x, y, gamma=1.0 / 16)
    assert abs(val) < 0.02  # unbiased estimator ~0 under the null


def test_mmd2_grows_with_mean_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 16))
    near = rng.normal(loc=0.5, size=(200, 16))
    far = rng.normal(loc=3.0, size=(200, 16))
    g = 1.0 / 16
    assert rbf_mmd2(x, far, g) > rbf_mmd2(x, near, g) > 0


def test_ks_flags_shifted_dimension():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(300, 4))
    win = ref.copy()
    win[:, 2] += 2.0  # move one coordinate hard
    _, ks_max = ks_per_dim(ref, win[np.random.default_rng(1).permutation(300)])
    assert ks_max > 0.4


# --------------------------------------------------------------------------- #
# Window buffering — non-overlapping, bounded.
# --------------------------------------------------------------------------- #

def test_monitor_emits_non_overlapping_windows():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(600, 8))
    mon = CovariateDriftMonitor(ref, window_size=100, max_ref=200,
                                calib_trials=50, seed=0)
    # 250 rows in one call → 2 full windows, 50 buffered.
    results = mon.update(rng.normal(size=(250, 8)))
    assert len(results) == 2
    assert mon.pending == 50
    # 50 more completes the third window.
    assert len(mon.update(rng.normal(size=(50, 8)))) == 1
    assert mon.pending == 0


# --------------------------------------------------------------------------- #
# The committed-snapshot fixtures — real embeddings, no checkpoint needed.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def snapshot(cfg):
    path = cfg.reference_snapshot_path
    if not path.exists():
        pytest.skip("reference snapshot not built yet")
    return load_snapshot(path)


def test_no_drift_stream_false_alarm_control(snapshot):
    """A stream of held-out reference embeddings must (mostly) stay below the
    calibrated MMD² threshold — the honest false-alarm control."""
    emb = snapshot.embeddings
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(emb))
    half = len(perm) // 2
    build, stream = emb[perm[:half]], emb[perm[half:]]

    mon = CovariateDriftMonitor(build, window_size=150, max_ref=1024,
                                mmd_quantile=0.99, calib_trials=200, seed=0)
    results = mon.update(stream)
    assert len(results) >= 10, "need enough windows to estimate a rate"
    alarm_rate = np.mean([r.alarm for r in results])
    # Design false-alarm rate is 1% (quantile 0.99); allow generous slack for the
    # finite sample but it must be far from a broken always-fires monitor.
    assert alarm_rate <= 0.20, f"no-drift false-alarm rate too high: {alarm_rate}"


def test_shifted_stream_raises_mmd_and_alarms(snapshot):
    """A constant embedding offset (a clear covariate shift) must push MMD²
    above threshold and fire alarms on essentially every window."""
    emb = snapshot.embeddings
    mon = CovariateDriftMonitor(emb, window_size=150, max_ref=1024,
                                mmd_quantile=0.99, calib_trials=200, seed=0)
    shift = 3.0 * emb.std(axis=0)  # per-dim, scaled to the data
    shifted = emb[:1500] + shift
    results = mon.update(shifted)
    assert results
    assert all(r.mmd2 > r.threshold for r in results)
    assert np.mean([r.alarm for r in results]) == 1.0


def test_prediction_monitor_null_in_domain(snapshot):
    """Reference preds streamed back through the prediction monitor → low PSI,
    no alarm (the in-domain null)."""
    preds = snapshot.preds
    mon = PredictionDriftMonitor(preds, LABELS, window_size=200, psi_threshold=0.25)
    results = mon.update(preds[np.random.default_rng(0).permutation(len(preds))])
    assert results
    assert np.mean([r.alarm for r in results]) == 0.0
    assert max(r.psi for r in results) < 0.25
    # defect_rate close to the reference defect rate on in-domain windows
    assert abs(np.mean([r.defect_rate for r in results]) - mon.ref_defect_rate) < 0.05


def test_prediction_monitor_flags_label_shift(snapshot):
    """A window whose label mix collapses onto one label must raise PSI."""
    preds = snapshot.preds
    mon = PredictionDriftMonitor(preds, LABELS, window_size=200, psi_threshold=0.25)
    # Every map fires only 'Near-full' — a label rare in the reference: strong shift.
    skew = np.zeros((200, len(LABELS)), dtype=np.int64)
    skew[:, LABELS.index("Near-full")] = 1
    results = mon.update(skew)
    assert results[0].psi > 0.25
    assert results[0].alarm


def test_from_snapshot_builds_both_monitors(snapshot):
    mons = DriftMonitors.from_snapshot(snapshot, LABELS, window_size=150,
                                       calib_trials=100, seed=0)
    assert mons.covariate.threshold > 0
    assert mons.covariate.embedding_dim == snapshot.embeddings.shape[1]
    assert mons.prediction.ref_defect_rate > 0
    # observe() feeds both and returns per-monitor result lists
    cov, pred = mons.observe(snapshot.embeddings[:150], snapshot.preds[:150])
    assert len(cov) == 1 and len(pred) == 1


# --------------------------------------------------------------------------- #
# End-to-end through the live service (needs the sibling checkpoint).
# --------------------------------------------------------------------------- #

@needs_mixed
def test_metrics_expose_drift_gauges(client):
    from prometheus_client.parser import text_string_to_metric_families
    names = {m.name for m in text_string_to_metric_families(client.get("/metrics").text)}
    for g in ("wafer_deploy_covariate_mmd2", "wafer_deploy_covariate_mmd2_threshold",
              "wafer_deploy_prediction_psi", "wafer_deploy_prediction_defect_rate"):
        assert g in names, f"{g} missing from /metrics"


@needs_mixed
def test_drift_window_updates_gauge_through_service(client, mixed_data):
    """Stream one full window of real maps through /predict → a covariate window
    completes and the windows_total counter advances."""
    from prometheus_client.parser import text_string_to_metric_families
    maps, _, test_idx = mixed_data

    def windows_total() -> float:
        for fam in text_string_to_metric_families(client.get("/metrics").text):
            if fam.name == "wafer_deploy_drift_windows":
                for s in fam.samples:
                    if s.labels.get("monitor") == "covariate":
                        return s.value
        return 0.0

    cfg = client.app.state.cfg
    if client.app.state.monitors is None:
        pytest.skip("monitors not active (no reference snapshot)")
    before = windows_total()
    w = cfg.drift_window_size
    # stream a full window (cycle through available test maps if fewer than w)
    for i in range(w):
        idx = int(test_idx[i % len(test_idx)])
        client.post("/predict", json={"wafer_map": maps[idx].tolist()})
    assert windows_total() >= before + 1
