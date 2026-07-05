"""metrics.py — Prometheus instrument definitions for the service.

Kept in one module so Phase 1's drift monitors register their gauges alongside
the serving counters here, all on the default registry that ``/metrics``
scrapes. Phase 0 exposes serving health + a per-label prediction counter; the
prediction counter is already the raw material for the Phase 1 prediction-rate
drift panel (rate() of predicted labels vs the reference histogram).
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from wafer_deploy.labels import LABELS

# Service liveness / readiness.
UP = Gauge("wafer_deploy_up", "1 when the model is loaded and ready to serve")

# Request accounting.
REQUESTS = Counter(
    "wafer_deploy_requests_total", "HTTP requests handled",
    ["endpoint", "http_status"],
)
PREDICT_LATENCY = Histogram(
    "wafer_deploy_predict_latency_seconds", "Per-request /predict wall time",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Prediction accounting — the seed of the Phase 1 prediction-rate monitor.
MAPS_PREDICTED = Counter(
    "wafer_deploy_maps_predicted_total", "Wafer maps scored by /predict",
)
PREDICTED_LABELS = Counter(
    "wafer_deploy_predicted_labels_total",
    "Times each defect label fired in a served prediction", ["label"],
)


def init_label_counters() -> None:
    """Materialise every label's counter at 0 so panels aren't blank pre-traffic."""
    for name in LABELS:
        PREDICTED_LABELS.labels(label=name).inc(0)


# --------------------------------------------------------------------------- #
# Phase 1 drift monitors — gauges + record helpers.
#
# Gauges (not counters): each holds the latest completed-window value, which is
# what a drift dashboard reads. Counters below track how many windows and alarms
# have fired so no-drift false-alarm rate is queryable straight off /metrics.
# --------------------------------------------------------------------------- #

# Covariate (input/embedding) drift.
COVARIATE_MMD2 = Gauge(
    "wafer_deploy_covariate_mmd2",
    "RBF-MMD^2 between the last embedding window and the reference bank")
COVARIATE_KS_MEAN = Gauge(
    "wafer_deploy_covariate_ks_mean",
    "Mean per-dimension KS statistic, last embedding window vs reference")
COVARIATE_KS_MAX = Gauge(
    "wafer_deploy_covariate_ks_max",
    "Max per-dimension KS statistic, last embedding window vs reference")
COVARIATE_MMD2_THRESHOLD = Gauge(
    "wafer_deploy_covariate_mmd2_threshold",
    "Calibrated MMD^2 alarm threshold (reference null quantile)")
COVARIATE_ALARM = Gauge(
    "wafer_deploy_covariate_drift_alarm",
    "1 when the last window's MMD^2 exceeded the calibrated threshold")

# Prediction-rate / label-distribution drift.
PREDICTION_PSI = Gauge(
    "wafer_deploy_prediction_psi",
    "PSI of the predicted-label distribution, last window vs reference")
PREDICTION_PSI_THRESHOLD = Gauge(
    "wafer_deploy_prediction_psi_threshold", "PSI alarm threshold")
PREDICTION_DEFECT_RATE = Gauge(
    "wafer_deploy_prediction_defect_rate",
    "Windowed fraction of maps with >=1 predicted defect")
PREDICTION_ALARM = Gauge(
    "wafer_deploy_prediction_drift_alarm",
    "1 when the last window's PSI exceeded the threshold")
WINDOWED_LABEL_RATE = Gauge(
    "wafer_deploy_windowed_label_rate",
    "Per-label fire-rate over the last completed window", ["label"])

# Window / alarm accounting (rate() over these = empirical false-alarm rate).
DRIFT_WINDOWS = Counter(
    "wafer_deploy_drift_windows_total", "Drift windows evaluated", ["monitor"])
DRIFT_ALARMS = Counter(
    "wafer_deploy_drift_alarms_total", "Drift-window alarms fired", ["monitor"])


def init_drift_gauges(mmd2_threshold: float, psi_threshold: float) -> None:
    """Publish thresholds and zero the state gauges so panels render pre-traffic."""
    COVARIATE_MMD2_THRESHOLD.set(mmd2_threshold)
    PREDICTION_PSI_THRESHOLD.set(psi_threshold)
    for g in (COVARIATE_MMD2, COVARIATE_KS_MEAN, COVARIATE_KS_MAX,
              COVARIATE_ALARM, PREDICTION_PSI, PREDICTION_DEFECT_RATE,
              PREDICTION_ALARM):
        g.set(0)
    for name in LABELS:
        WINDOWED_LABEL_RATE.labels(label=name).set(0)


def record_covariate(result) -> None:
    """Reflect one CovariateDriftResult onto the gauges + counters."""
    COVARIATE_MMD2.set(result.mmd2)
    COVARIATE_KS_MEAN.set(result.ks_mean)
    COVARIATE_KS_MAX.set(result.ks_max)
    COVARIATE_MMD2_THRESHOLD.set(result.threshold)
    COVARIATE_ALARM.set(1 if result.alarm else 0)
    DRIFT_WINDOWS.labels(monitor="covariate").inc()
    if result.alarm:
        DRIFT_ALARMS.labels(monitor="covariate").inc()


def record_prediction(result) -> None:
    """Reflect one PredictionDriftResult onto the gauges + counters."""
    PREDICTION_PSI.set(result.psi)
    PREDICTION_DEFECT_RATE.set(result.defect_rate)
    PREDICTION_ALARM.set(1 if result.alarm else 0)
    for name, rate in result.per_label_rate.items():
        WINDOWED_LABEL_RATE.labels(label=name).set(rate)
    DRIFT_WINDOWS.labels(monitor="prediction").inc()
    if result.alarm:
        DRIFT_ALARMS.labels(monitor="prediction").inc()


# --------------------------------------------------------------------------- #
# Phase 2 calibration-decay monitor — gauges + record helper.
#
# Labels arrive late, so these gauges only advance when a window's delayed labels
# land (via /feedback), not on every /predict. They share the DRIFT_WINDOWS /
# DRIFT_ALARMS counters under monitor="calibration" so the empirical false-alarm
# rate is queryable the same way the Phase 1 monitors are.
# --------------------------------------------------------------------------- #

CALIBRATION_ECE = Gauge(
    "wafer_deploy_calibration_ece",
    "Mean per-label ECE of the last delayed-label window")
CALIBRATION_ECE_THRESHOLD = Gauge(
    "wafer_deploy_calibration_ece_threshold",
    "Calibrated ECE alarm threshold (reference null quantile)")
CALIBRATION_REFERENCE_ECE = Gauge(
    "wafer_deploy_calibration_reference_ece",
    "Frozen reference mean per-label ECE (the calibration baseline)")
CALIBRATION_ALARM = Gauge(
    "wafer_deploy_calibration_drift_alarm",
    "1 when the last window's mean ECE exceeded the calibrated threshold")
CALIBRATION_ECE_PER_LABEL = Gauge(
    "wafer_deploy_calibration_ece_per_label",
    "Per-label ECE over the last delayed-label window", ["label"])
CALIBRATION_PENDING_LABELS = Gauge(
    "wafer_deploy_calibration_pending_labels",
    "Served predictions still awaiting a delayed label")
CALIBRATION_DROPPED = Counter(
    "wafer_deploy_calibration_dropped_total",
    "Awaiting predictions evicted by the retention cap (labels lagged too far)")


def init_calibration_gauges(ece_threshold: float, reference_ece_mean: float) -> None:
    """Publish the threshold + reference and zero the state gauges pre-labels."""
    CALIBRATION_ECE_THRESHOLD.set(ece_threshold)
    CALIBRATION_REFERENCE_ECE.set(reference_ece_mean)
    for g in (CALIBRATION_ECE, CALIBRATION_ALARM, CALIBRATION_PENDING_LABELS):
        g.set(0)
    for name in LABELS:
        CALIBRATION_ECE_PER_LABEL.labels(label=name).set(0)


def record_calibration(result) -> None:
    """Reflect one CalibrationDriftResult onto the gauges + counters."""
    CALIBRATION_ECE.set(result.ece_mean)
    CALIBRATION_ECE_THRESHOLD.set(result.threshold)
    CALIBRATION_REFERENCE_ECE.set(result.reference_ece_mean)
    CALIBRATION_ALARM.set(1 if result.alarm else 0)
    for name, val in result.ece_per_label.items():
        CALIBRATION_ECE_PER_LABEL.labels(label=name).set(val)
    DRIFT_WINDOWS.labels(monitor="calibration").inc()
    if result.alarm:
        DRIFT_ALARMS.labels(monitor="calibration").inc()


# --------------------------------------------------------------------------- #
# Phase 3 retrain trigger — the combined decision, gauges + record helper.
#
# The trigger ORs the three monitor channels with hysteresis (see trigger.py).
# Its clock is the covariate window (the always-available heartbeat, one per
# window_size maps); the delayed calibration verdict is folded in at its most
# recent value. RETRAIN_TRIGGER is the latched decision Grafana shows firing.
# --------------------------------------------------------------------------- #

RETRAIN_TRIGGER = Gauge(
    "wafer_deploy_retrain_trigger",
    "1 while the hysteresis retrain trigger is latched (a retrain is warranted)")
RETRAIN_TRIGGER_REASON = Gauge(
    "wafer_deploy_retrain_trigger_reason",
    "1 when this monitor channel is in alarm at the latest trigger evaluation",
    ["channel"])
RETRAIN_TRIGGERS_TOTAL = Counter(
    "wafer_deploy_retrain_triggers_total",
    "Rising edges of the retrain trigger (distinct retrain decisions)")


def init_trigger_gauges() -> None:
    """Zero the trigger gauges so the panel renders before any window lands."""
    from wafer_deploy.trigger import CHANNELS
    RETRAIN_TRIGGER.set(0)
    for c in CHANNELS:
        RETRAIN_TRIGGER_REASON.labels(channel=c).set(0)


def record_trigger(result) -> None:
    """Reflect one TriggerResult onto the trigger gauges + edge counter."""
    from wafer_deploy.trigger import CHANNELS
    RETRAIN_TRIGGER.set(1 if result.fired else 0)
    active = set(result.active_reasons)
    for c in CHANNELS:
        RETRAIN_TRIGGER_REASON.labels(channel=c).set(1 if c in active else 0)
    if result.just_fired:
        RETRAIN_TRIGGERS_TOTAL.inc()
