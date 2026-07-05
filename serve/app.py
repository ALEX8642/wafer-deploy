"""app.py — FastAPI inference service for the frozen wafer-mixed model.

Endpoints:
    POST /predict   wafer map (values 0/1/2) → calibrated per-label probabilities
                    and the multi-hot decision at each label's tuned threshold.
                    This is wafer-mixed's own calibrated path (see predictor.py).
    GET  /healthz   liveness + whether the model loaded, with checkpoint meta.
    GET  /metrics   Prometheus exposition text.

Design notes:
    - The checkpoint + thresholds load ONCE at startup (lifespan), on CPU.
    - If the model can't load (e.g. the sibling wafer-mixed artifacts aren't
      mounted), the app still starts so Prometheus + Grafana come up and the
      dashboards render — /predict then returns 503 and /healthz reports it.
      This keeps the `docker-compose up` guardrail (dashboards up on a fresh
      clone) independent of whether the 45 MB checkpoint is present.
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from serve import metrics as M
from wafer_deploy.calibration import CalibrationMonitor
from wafer_deploy.config import DeployConfig
from wafer_deploy.drift import DriftMonitors
from wafer_deploy.labels import LABELS, NUM_LABELS
from wafer_deploy.predictor import Predictor
from wafer_deploy.snapshot import load_snapshot


class PredictRequest(BaseModel):
    wafer_map: list[list[int]] = Field(
        ..., description="2-D wafer map, integer values in {0,1,2} "
                         "(0 outside wafer, 1 passing die, 2 failing die). "
                         "Any H×W — it is one-hot encoded and resized to the "
                         "model's input size, exactly as in training.")


class PredictResponse(BaseModel):
    labels: list[str]
    probabilities: dict[str, float]   # calibrated per-label probability
    prediction: dict[str, int]        # multi-hot at per-label tuned tau
    predicted_labels: list[str]       # active labels (a fab-facing summary)
    is_normal: bool                   # no label fired
    latency_ms: float


class FeedbackRequest(BaseModel):
    labels: list[list[int]] = Field(
        ..., description="Delayed ground-truth multi-hot vectors, each of length "
                         "NUM_LABELS in the canonical label order, FIFO-matched to "
                         "previously served /predict calls. This is the "
                         "delayed-label channel that feeds the calibration monitor.")


class FeedbackResponse(BaseModel):
    windows_scored: int               # calibration windows completed by this batch
    pending_labels: int               # served predictions still awaiting a label
    latest: dict | None               # newest scored window's ECE / alarm, if any


def create_app(cfg: DeployConfig | None = None) -> FastAPI:
    cfg = cfg or DeployConfig.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        M.init_label_counters()
        try:
            app.state.predictor = Predictor(cfg)
            M.UP.set(1)
        except Exception as exc:  # degrade, don't crash — dashboards still come up
            app.state.predictor = None
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            M.UP.set(0)
        # Build the drift monitors from the committed reference snapshot.
        # Independent of the checkpoint: the snapshot alone parameterises them,
        # though they are only *fed* once /predict starts producing embeddings
        # (Phase 1: covariate + prediction) or /feedback delivers delayed labels
        # (Phase 2: calibration).
        try:
            snap = load_snapshot(cfg.reference_snapshot_path)
            app.state.monitors = DriftMonitors.from_snapshot(
                snap, LABELS, window_size=cfg.drift_window_size,
                max_ref=cfg.drift_max_ref, mmd_quantile=cfg.drift_mmd_quantile,
                calib_trials=cfg.drift_calib_trials,
                psi_threshold=cfg.drift_psi_threshold, seed=cfg.seed)
            M.init_drift_gauges(app.state.monitors.covariate.threshold,
                                cfg.drift_psi_threshold)
            app.state.calibration = CalibrationMonitor.from_snapshot(
                snap, LABELS, window_size=cfg.drift_window_size,
                n_bins=cfg.calibration_n_bins,
                ece_quantile=cfg.calibration_ece_quantile,
                calib_trials=cfg.calibration_calib_trials,
                max_pending=cfg.calibration_max_pending, seed=cfg.seed)
            M.init_calibration_gauges(app.state.calibration.threshold,
                                      app.state.calibration.reference_ece_mean)
        except Exception as exc:  # no reference → serving still fine, no monitors
            app.state.monitors = None
            app.state.calibration = None
            app.state.monitor_error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="wafer-deploy", version="0.1.0", lifespan=lifespan)
    app.state.predictor = None
    app.state.load_error = None
    app.state.monitors = None
    app.state.calibration = None
    app.state.monitor_error = None
    app.state.monitor_lock = threading.Lock()
    app.state.cfg = cfg

    def _predictor() -> Predictor:
        p = app.state.predictor
        if p is None:
            raise HTTPException(
                status_code=503,
                detail=f"model not loaded: {app.state.load_error}")
        return p

    @app.get("/healthz")
    def healthz() -> dict:
        p = app.state.predictor
        loaded = p is not None
        M.REQUESTS.labels(endpoint="/healthz", http_status="200").inc()
        body = {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
                "monitors_active": app.state.monitors is not None,
                "calibration_active": app.state.calibration is not None}
        if loaded:
            body["checkpoint"] = p.checkpoint_meta
        else:
            body["error"] = app.state.load_error
        if app.state.monitors is None and app.state.monitor_error:
            body["monitor_error"] = app.state.monitor_error
        return body

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        p = _predictor()
        try:
            wmap = np.asarray(req.wafer_map, dtype=np.int64)
        except ValueError:
            wmap = np.asarray([], dtype=np.int64)  # ragged rows → fail validation below
        if wmap.ndim != 2 or wmap.size == 0:
            M.REQUESTS.labels(endpoint="/predict", http_status="422").inc()
            raise HTTPException(status_code=422,
                                detail="wafer_map must be a non-empty rectangular 2-D array")

        t0 = time.perf_counter()
        res = p.predict_one(wmap)
        dt = time.perf_counter() - t0
        M.PREDICT_LATENCY.observe(dt)

        probs = res.probs[0]
        preds = res.preds[0].astype(int)
        active = [LABELS[i] for i in range(len(LABELS)) if preds[i]]

        M.MAPS_PREDICTED.inc()
        for name in active:
            M.PREDICTED_LABELS.labels(label=name).inc()
        M.REQUESTS.labels(endpoint="/predict", http_status="200").inc()

        # Feed the monitors. Serving runs sync endpoints in a threadpool, so guard
        # the (stateful, bounded) window buffers with a lock; gauges reflect only
        # completed windows, so most requests just buffer. The unsupervised
        # monitors score here; the calibration monitor only *buffers* the served
        # probs — it is scored later, when the delayed labels arrive at /feedback.
        monitors = app.state.monitors
        calibration = app.state.calibration
        cov_results, pred_results = [], []
        if monitors is not None or calibration is not None:
            with app.state.monitor_lock:
                if monitors is not None:
                    cov_results, pred_results = monitors.observe(
                        res.embeddings, res.preds)
                if calibration is not None:
                    dropped = calibration.buffer_predictions(res.probs)
                    M.CALIBRATION_PENDING_LABELS.set(calibration.pending_labels)
                    if dropped:
                        M.CALIBRATION_DROPPED.inc(dropped)
        for r in cov_results:
            M.record_covariate(r)
        for r in pred_results:
            M.record_prediction(r)

        return PredictResponse(
            labels=list(LABELS),
            probabilities={LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
            prediction={LABELS[i]: int(preds[i]) for i in range(len(LABELS))},
            predicted_labels=active,
            is_normal=len(active) == 0,
            latency_ms=dt * 1e3,
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest) -> FeedbackResponse:
        """Deliver delayed ground-truth labels for previously served predictions.

        Labels are FIFO-matched to earlier /predict calls; the calibration
        monitor scores a window (ECE vs the frozen reference) once that window's
        labels have all arrived. This is the delayed-label channel — in
        production labels lag inference, so the calibration signal necessarily
        trails the unsupervised ones.
        """
        cal = app.state.calibration
        if cal is None:
            raise HTTPException(
                status_code=503,
                detail="calibration monitor not active (no reference snapshot)")
        try:
            arr = np.asarray(req.labels, dtype=np.int64)
        except ValueError:
            arr = np.asarray([], dtype=np.int64)  # ragged → fail validation below
        if arr.ndim != 2 or arr.shape[1] != NUM_LABELS:
            M.REQUESTS.labels(endpoint="/feedback", http_status="422").inc()
            raise HTTPException(
                status_code=422,
                detail=f"each label vector must be length {NUM_LABELS} "
                       "(multi-hot, canonical label order)")

        with app.state.monitor_lock:
            try:
                results = cal.add_labels(arr)
            except ValueError as exc:  # label with no matching served prediction
                M.REQUESTS.labels(endpoint="/feedback", http_status="400").inc()
                raise HTTPException(status_code=400, detail=str(exc))
            pending = cal.pending_labels
        for r in results:
            M.record_calibration(r)
        M.CALIBRATION_PENDING_LABELS.set(pending)
        M.REQUESTS.labels(endpoint="/feedback", http_status="200").inc()

        latest = results[-1] if results else None
        return FeedbackResponse(
            windows_scored=len(results),
            pending_labels=pending,
            latest=None if latest is None else {
                "window_id": latest.window_id,
                "ece_mean": latest.ece_mean,
                "reference_ece_mean": latest.reference_ece_mean,
                "ece_delta": latest.ece_delta,
                "threshold": latest.threshold,
                "alarm": latest.alarm,
            },
        )

    return app


app = create_app()
