"""Endpoint contracts: /healthz, /metrics parseability, and /predict input
validation. These don't need the sibling checkout to be *correct*, only to be
loaded — /healthz reports degraded and /predict 503s when the model is absent,
which is itself part of the contract (dashboards must still come up)."""
from __future__ import annotations

from conftest import needs_mixed
from prometheus_client.parser import text_string_to_metric_families


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["model_loaded"], bool)


@needs_mixed
def test_healthz_reports_loaded_checkpoint(client):
    body = client.get("/healthz").json()
    assert body["model_loaded"] is True
    assert body["checkpoint"]["arch"] == "resnet18"
    assert body["checkpoint"]["embedding_dim"] == 512


def test_metrics_parses_as_prometheus(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    names = {m.name for m in text_string_to_metric_families(resp.text)}
    # Core serving instruments must be exposed.
    assert "wafer_deploy_up" in names
    assert "wafer_deploy_requests" in names  # _total suffix stripped by parser
    assert "wafer_deploy_predict_latency_seconds" in names


@needs_mixed
def test_metrics_count_increments_after_predict(client, mixed_data):
    maps, _, test_idx = mixed_data

    def maps_predicted() -> float:
        for fam in text_string_to_metric_families(client.get("/metrics").text):
            if fam.name == "wafer_deploy_maps_predicted":
                return next(s.value for s in fam.samples
                            if s.name == "wafer_deploy_maps_predicted_total")
        return 0.0

    before = maps_predicted()
    client.post("/predict", json={"wafer_map": maps[int(test_idx[1])].tolist()})
    assert maps_predicted() == before + 1


def test_predict_rejects_non_2d(client):
    # 1-D input is invalid — pydantic rejects it before the model is touched,
    # so this holds even without the sibling checkout.
    resp = client.post("/predict", json={"wafer_map": [0, 1, 2]})
    assert resp.status_code == 422


@needs_mixed
def test_predict_rejects_ragged_map(client):
    # Rows of unequal length: np.asarray(dtype=int) raises — must surface as a
    # clean 422, not an uncaught 500.
    resp = client.post("/predict", json={"wafer_map": [[0, 1], [2]]})
    assert resp.status_code == 422
