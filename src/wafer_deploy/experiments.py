"""experiments.py — the offline scored-shift engine (Phase 3).

The monitors and the retrain trigger are exercised online in the service; this
module scores them **offline**, under controlled shifts, to produce the honest
numbers the phase is judged on: detection latency and false-alarm rate per shift,
and *which monitor fires first*. Nothing here runs on the serving box — per the
hardware policy all drift science runs on the CPU/5090 box, and only the small
results artifact (``experiments/shift_results.json``) and figures are committed.

The engine is deliberately array-in / result-out so the same scoring path covers
every shift source:

    - a **stream** is warmup windows (clean, in-domain) followed by shifted
      windows; ``onset_window`` is where the shift turns on, so detection latency
      is measured from a known ground-truth moment;
    - ``run_stream`` feeds each window through the real monitor objects (so the
      alarm logic under test is exactly the served one) and the retrain trigger,
      applying the calibration **label lag** — the calibration verdict for data
      window *k* only reaches the trigger *lag* windows later, which is why the
      unsupervised channels lead and calibration confirms.

Three shift sources are built on top (all in ``scripts/run_shift_experiments``):
(a) the synthetic corruption sweep (``shift.CORRUPTIONS`` at rising intensity),
(b) the real WM-811K → MixedWM38 cross-dataset shift, and (c) the class-prior
"defect campaign". Sources (a)/(b) need inference on shifted maps; (c) reuses the
committed snapshot with no re-inference.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from wafer_deploy.calibration import CalibrationMonitor
from wafer_deploy.drift import DriftMonitors
from wafer_deploy.shift import Corruption
from wafer_deploy.trigger import CHANNELS, RetrainTrigger


# --------------------------------------------------------------------------- #
# Inference over (possibly corrupted) raw maps.
# --------------------------------------------------------------------------- #

def predict_maps(predictor, maps, batch_size: int = 128,
                 progress: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the frozen predictor over raw maps → (embeddings, probs, preds).

    Batched CPU inference; the same calibrated path the service uses (so the
    embeddings feeding the covariate monitor and the probs feeding calibration
    are the served ones). ``maps`` is any sequence of 2-D arrays (values 0/1/2).
    """
    n = len(maps)
    rng = range(0, n, batch_size)
    if progress:
        from tqdm import tqdm
        rng = tqdm(rng, desc="inference")
    emb, probs, preds = [], [], []
    for start in rng:
        batch = [np.asarray(maps[i]) for i in range(start, min(start + batch_size, n))]
        res = predictor.predict_batch(batch)
        emb.append(res.embeddings)
        probs.append(res.probs)
        preds.append(res.preds)
    return (np.vstack(emb).astype(np.float32),
            np.vstack(probs).astype(np.float64),
            np.vstack(preds).astype(np.int64))


def corrupt_and_predict(predictor, raw_maps, corruption: Corruption,
                        intensity: float, *, seed: int = 0, batch_size: int = 128,
                        progress: bool = False):
    """Apply a graded corruption to each raw map, then run inference.

    Each map gets its own child RNG (seeded from ``seed`` + position) so the
    corruption is deterministic and independent per map. Returns the same
    (embeddings, probs, preds) triple as ``predict_maps``.
    """
    ss = np.random.SeedSequence([seed, int(intensity * 1e6)])
    child_seeds = ss.spawn(len(raw_maps))
    corrupted = [corruption(np.asarray(raw_maps[i]), intensity,
                            np.random.default_rng(child_seeds[i]))
                 for i in range(len(raw_maps))]
    return predict_maps(predictor, corrupted, batch_size=batch_size, progress=progress)


# --------------------------------------------------------------------------- #
# Window-by-window scoring of a stream through the three monitors + trigger.
# --------------------------------------------------------------------------- #

def _channel_detection(alarms: list[bool], onset: int) -> Optional[int]:
    """First window index >= onset with an alarm, expressed as windows-after-onset."""
    for k in range(onset, len(alarms)):
        if alarms[k]:
            return k - onset
    return None


def run_stream(monitors: DriftMonitors, calibration: CalibrationMonitor,
               trigger: RetrainTrigger, *, embeddings: np.ndarray, preds: np.ndarray,
               probs: np.ndarray, y_true: np.ndarray, window_size: int,
               onset_window: int, calibration_lag: int = 2) -> dict:
    """Score a full stream: per-window monitor + trigger records and a summary.

    The three monitor objects must have been built with the same ``window_size``;
    each window feeds exactly ``window_size`` rows so their internal buffers drain
    to empty at every boundary (the monitors are reusable across streams). The
    calibration channel reaches the trigger delayed by ``calibration_lag`` windows
    — the delayed-label regime — so ``onset``-relative latency for calibration is
    honestly larger than the label-free channels'.
    """
    assert monitors.covariate.pending == 0 and monitors.prediction.pending == 0, \
        "monitors carry a partial window from a previous stream"
    if not (monitors.covariate.window_size == monitors.prediction.window_size
            == calibration.window_size == window_size):
        raise ValueError(
            "window_size must match the monitors' construction window_size "
            f"(got {window_size}, monitors "
            f"{monitors.covariate.window_size}/{monitors.prediction.window_size}/"
            f"{calibration.window_size}) — a mismatch would drop or split windows")
    n = min(len(embeddings), len(preds), len(probs), len(y_true))
    n_windows = n // window_size
    if n_windows == 0:
        raise ValueError("stream shorter than one window")

    cov_alarm, pred_alarm, cal_alarm = [], [], []
    records: list[dict] = []
    for k in range(n_windows):
        sl = slice(k * window_size, (k + 1) * window_size)
        cov = monitors.covariate.update(embeddings[sl])[0]
        prd = monitors.prediction.update(preds[sl])[0]
        cal = calibration.score_window(probs[sl], y_true[sl], window_id=k)
        cov_alarm.append(bool(cov.alarm))
        pred_alarm.append(bool(prd.alarm))
        cal_alarm.append(bool(cal.alarm))
        records.append({
            "window": k, "onset": bool(k >= onset_window),
            "mmd2": cov.mmd2, "mmd2_threshold": cov.threshold, "cov_alarm": bool(cov.alarm),
            "ks_max": cov.ks_max,
            "psi": prd.psi, "psi_threshold": prd.threshold, "pred_alarm": bool(prd.alarm),
            "defect_rate": prd.defect_rate,
            "ece": cal.ece_mean, "ece_threshold": cal.threshold, "cal_alarm": bool(cal.alarm),
            "ece_reference": cal.reference_ece_mean,
        })

    # Feed the trigger in wall-clock order, lagging the calibration verdict: the
    # label-dependent alarm for data window k only lands at wall-clock k+lag.
    for k in range(n_windows):
        cal_effective = cal_alarm[k - calibration_lag] if k - calibration_lag >= 0 else False
        tr = trigger.update(covariate=cov_alarm[k], prediction=pred_alarm[k],
                            calibration=cal_effective)
        records[k]["cal_alarm_effective"] = bool(cal_effective)
        records[k]["trigger_fired"] = bool(tr.fired)
        records[k]["trigger_just_fired"] = bool(tr.just_fired)
        records[k]["trigger_reasons"] = tr.active_reasons

    # Per-channel wall-clock detection latency (windows after onset). Covariate
    # and prediction are visible immediately; calibration carries the label lag.
    channel_alarms = {
        "covariate": cov_alarm,
        "prediction": pred_alarm,
        "calibration": [cal_alarm[k - calibration_lag] if k - calibration_lag >= 0 else False
                        for k in range(n_windows)],
    }
    detection = {c: _channel_detection(channel_alarms[c], onset_window) for c in CHANNELS}
    trig_edges = [k for k in range(n_windows) if records[k]["trigger_just_fired"]]
    trig_latency = next((k - onset_window for k in trig_edges if k >= onset_window), None)

    n_pre = max(0, onset_window)
    n_post = n_windows - onset_window
    summary = {
        "n_windows": n_windows, "onset_window": onset_window,
        "calibration_lag": calibration_lag,
        "detection_latency": detection,   # windows-after-onset per channel (or None)
        "channel_alarm_rate_pre": {
            c: float(np.mean(channel_alarms[c][:onset_window])) if n_pre else 0.0
            for c in CHANNELS},
        "channel_alarm_rate_post": {
            c: float(np.mean(channel_alarms[c][onset_window:])) if n_post > 0 else 0.0
            for c in CHANNELS},
        "trigger_fired": bool(trig_edges),
        "trigger_latency": trig_latency,
        "trigger_fire_windows": trig_edges,
        "first_channel": _first_channel(detection),
    }
    return {"records": records, "summary": summary}


def _first_channel(detection: dict) -> Optional[str]:
    """Which channel detected earliest (min windows-after-onset); None if none did."""
    fired = {c: v for c, v in detection.items() if v is not None}
    return min(fired, key=fired.get) if fired else None


def reset_monitors(monitors: DriftMonitors, calibration: CalibrationMonitor) -> None:
    """Drop any buffered partial window so a monitor set is reusable across streams.

    ``run_stream`` feeds whole windows and expects an empty buffer at entry; a
    stream whose length was not a multiple of ``window_size`` would otherwise
    leave a partial window behind. Cheap: the threshold calibration (the costly
    part of construction) is untouched.
    """
    monitors.covariate._buf._rows.clear()
    monitors.prediction._buf._rows.clear()
    calibration._await_probs.clear()
    calibration._win_probs.clear()
    calibration._win_labels.clear()
    calibration._window_id = 0
    calibration._skip = 0
