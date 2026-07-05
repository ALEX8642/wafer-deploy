# wafer-deploy — model serving + drift monitoring

> **Status: Phase 0 (scaffold + serving skeleton + reference snapshot).**
> The full README — architecture diagram, scored detection results, real GB10
> latency table, figure gallery — lands in Phase 4. This is the scaffold.

The fifth repo in the wafer portfolio. The first four **build**
([`wafer-defect-classifier`](../wafer-defect-classifier),
[`wafer-ssl`](../wafer-ssl)), **push the ceiling**
([`wafer-mixed`](../wafer-mixed), macro-F1 0.9846) and **attribute**
([`wafer-rootcause`](../wafer-rootcause)). This one **serves the model and knows
when to stop trusting it**: a real FastAPI inference service plus an
unsupervised-first drift-monitoring stack, scored with honest numbers
(detection latency, false-alarm rate, calibration deltas — nulls included).

**No training happens here.** The model is a fixed, calibrated artifact
(`wafer-mixed/outputs/best.pt` + `thresholds.json` + `calibration.json`), reused
from a sibling checkout through a single guarded bridge
([`src/wafer_deploy/bridge.py`](src/wafer_deploy/bridge.py)). What a served
prediction returns is *definitionally* what wafer-mixed produces — the parity
test pins it.

## What Phase 0 ships

- **FastAPI service** ([`serve/app.py`](serve/app.py)) — `POST /predict`
  (wafer map → calibrated per-label probabilities, multi-hot at each label's
  tuned threshold), `GET /healthz`, `GET /metrics` (Prometheus). Checkpoint
  loads once at startup, CPU.
- **Reference snapshot** ([`reference/`](reference)) — a frozen baseline from
  the wafer-mixed **test** split (embeddings, prediction-rate, label histogram,
  reference calibration), committed so a fresh clone needs neither the 400 MB
  dataset nor the checkpoint to bring the monitors up. Every later monitor
  compares against this.
- **Self-contained observability** — `docker compose up` brings the service +
  Prometheus + a provisioned Grafana (starter dashboard) up on CPU.

## Quickstart

```bash
# 1. Serve (needs a sibling wafer-mixed checkout for the 45 MB checkpoint;
#    it is bind-mounted read-only, never vendored here).
docker compose up --build

# 2. Send a wafer map and see the prediction.
python scripts/send_sample.py                 # synthetic map
python scripts/send_sample.py --from-mixed 0  # real test-split map

# 3. Dashboards:
#    Grafana     http://localhost:3000   (anonymous, no login)
#    Prometheus  http://localhost:9090
#    Service     http://localhost:8000/healthz
```

Run the tests (needs the sibling checkout + `requirements-dev.txt`):

```bash
pip install -e . && pip install -r requirements-dev.txt
pytest
```

## Reuse boundary / data policy

- The checkpoint, thresholds and calibration are **read** from
  `../wafer-mixed` (override with `WAFER_MIXED_ROOT`); nothing is copied in.
- All drift in this repo is **simulated or public-dataset** (Phase 3 uses a
  synthetic corruption sweep + a WM-811K→MixedWM38 cross-domain shift). **Work
  data never enters this repo.**

## Layout

```
src/wafer_deploy/   bridge, config, predictor (+embedding hook), snapshot
serve/              FastAPI app + Prometheus metrics
monitoring/         prometheus.yml, Grafana provisioning + dashboard JSON
reference/          committed reference snapshot (the monitors' baseline)
configs/            deploy.yaml
scripts/            build_reference_snapshot.py, send_sample.py
tests/              parity, health, metrics, snapshot determinism
```

MIT licensed. See [`STATUS.md`](STATUS.md) for the phase-by-phase record.
