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
