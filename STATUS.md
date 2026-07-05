# STATUS — wafer-deploy

The handoff artifact between phase-sessions. One phase per session; this file is
kept current so stop/start is cheap. See `PLAN-wafer-deploy.md` (workspace root)
for the full plan.

---

## Phase 0 — Scaffold + serving skeleton + reference snapshot ✅ (2026-07-04)

### Reuse-path verification (Phase 0 required this before relying on anything)

All paths verified against the sibling `../wafer-mixed` checkout. **No deviations
from the plan** — every reused artifact exists and loads.

| Reuse target | Path | Verified |
|---|---|---|
| Checkpoint | `wafer-mixed/outputs/best.pt` (45 MB) | ✅ loads; epoch 7, val macro-F1 0.9906, resnet18 |
| Tuned thresholds + temperatures | `wafer-mixed/outputs/thresholds.json` | ✅ 8 labels, `_temperatures` + `thresholds` |
| Calibration | `wafer-mixed/outputs/calibration.json` | ✅ `ece_mean_after` 0.004292 |
| Model import + predict path | `wafer_mixed.{model,data,calibrate,metrics,evaluate}` | ✅ imported via guarded bridge |
| Penultimate features | final Linear `model.fc` input | ✅ **512-d**, captured by forward pre-hook (`predictor.py`) |
| Test split (reference window) | `wafer-mixed/data/splits.npz` `test` | ✅ 7,603 maps, seed 42 |
| Raw MixedWM38 | `wafer-mixed/data/raw/MixedWM38.npz` | ✅ (needed only to *build* the snapshot; not at serve time) |
| WM-811K (Phase 3 OOD source) | `wafer-defect-classifier/data/raw/LSWMD.pkl` (2.1 GB) | ✅ present (used in Phase 3, not now) |

Cross-check: the snapshot's recomputed `reference_ece_mean` = **0.00429**, which
matches wafer-mixed's own `calibration.json` `ece_mean_after` (0.004292) — the
reused calibrated path reproduces wafer-mixed's calibration measurement.

### What was built

- **Package `src/wafer_deploy/`**
  - `bridge.py` — the single owner of the cross-repo dependency (ported from
    wafer-rootcause's hardened pattern): imports wafer-mixed's own modules from
    the sibling checkout, guards against a shadowing install and label-order
    drift. `load_thresholds` returns (T, τ) in canonical LABELS order.
  - `config.py` — `DeployConfig`; YAML (`configs/deploy.yaml`) overlaid by
    `WAFER_DEPLOY_*` / `WAFER_MIXED_ROOT` env vars (env wins, so a container is
    repointed at bind-mounted artifacts without a rebuild).
  - `predictor.py` — loads the frozen checkpoint on **CPU**, exposes
    wafer-mixed's exact calibrated path (encode → resize → model → `scale_probs`
    → `predict_multihot`), and captures **penultimate features** via a forward
    pre-hook on `model.fc` (the Phase 1 covariate-drift substrate).
  - `snapshot.py` — build/save/load the reference snapshot; deterministic
    content hash.
  - `labels.py` — the canonical 8-label order.
- **Serving `serve/`** — FastAPI: `POST /predict`, `GET /healthz`,
  `GET /metrics`. Model loads once at startup (lifespan); **degrades instead of
  crashing** if artifacts are absent, so Prometheus + Grafana still come up.
  `metrics.py` defines serving counters/histogram + a per-label prediction
  counter (the seed of the Phase 1 prediction-rate monitor).
- **Reference snapshot** `reference/reference_snapshot.npz` (+ `.meta.json`),
  **committed, 6.4 MB** — 7,603 test maps: embeddings (float16), calibrated
  probs, multi-hot preds, ground truth, plus a summary (prediction-rate, label
  histogram, per-label reference ECE). `defect_rate` 0.973 on the defect-dense
  test split.
- **docker stack** — `Dockerfile` (CPU, torch-cpu wheel; checkpoint bind-mounted
  not vendored), `docker-compose.yml` (service + Prometheus + Grafana),
  `monitoring/` (scrape config, provisioned datasource, dashboards provider,
  starter dashboard JSON).
- **Scripts** — `build_reference_snapshot.py` (built the committed artifact),
  `send_sample.py` (stdlib-only quickstart POST).
- **Tests** — 10, all green (see below).

### Accept-criteria status

| Criterion | Status |
|---|---|
| `/predict` parity test passes (== wafer-mixed `predict_multihot`) | ✅ 24 maps across the split, multi-hot exact + probs `atol=1e-5` |
| Reference snapshot committed | ✅ `reference/reference_snapshot.npz` (6.4 MB) + determinism + integrity tests |
| `docker compose up` brings service + Prometheus + Grafana up on CPU | ✅ **verified end-to-end** — native Docker Engine installed in-distro; see below |
| Repo pushed | ⏳ committed locally; **push is on Alex** (auto-mode blocks pushes) |

### docker stack — verified end-to-end (2026-07-04)

Native Docker Engine (29.6.1, compose v5.3.0) was installed in this Ubuntu 24.04
WSL distro (systemd-managed). `docker compose up --build` was run for real on
CPU and the whole board came up:

- **service** healthy, `model_loaded: true` (checkpoint bind-mounted from
  `../wafer-mixed`); `/predict` returned correct multi-hot for both a synthetic
  edge-ring map (→ Edge-Ring) and real test map #0 (→ Center/Edge-Loc/Scratch).
- **Prometheus** target `wafer-deploy` **health=up**, scraping
  `service:8000/metrics`; `wafer_deploy_up=1`, `maps_predicted_total` tracked
  the two requests.
- **Grafana** healthy, datasource **provisioned** (Prometheus), dashboard
  **provisioned** (`uid=wafer-deploy-overview`); a panel expression queried
  **through** the Grafana datasource proxy returned 8 live label series — the
  `uid` wiring works end-to-end, not just on paper.

Container image `wafer-deploy:latest` = 1.86 GB (CPU torch). Stack torn down
after the check; `docker compose up -d` relaunches it (image cached).

**Bug the real docker run caught (that the local venv masked):** the service
came up **degraded** on first boot — `ModuleNotFoundError: No module named
'matplotlib'`. `wafer_mixed.calibrate` (whose `scale_probs` / `binary_ece` the
serving + snapshot paths reuse) imports matplotlib at module level, so it is a
**transitive** runtime dependency the lean image (and `requirements.txt`) were
missing. Fixed by adding `matplotlib>=3.8` to `requirements.txt` + `Dockerfile`;
rebuilt → healthy. The graceful-degrade design worked exactly as intended
(Prometheus + Grafana still came up; `/healthz` reported the cause). Lesson: the
TestClient path can't catch missing *image* deps because the venv has them — the
docker run is load-bearing, keep it in the loop before Phase 4.

### Honest caveats / deviations

- The service needs the sibling checkpoint at run time; it is **bind-mounted
  read-only** (`../wafer-mixed`, or `WAFER_MIXED_HOST_PATH`), consistent with the
  "no `*.pt` vendored" policy. A true standalone fresh clone (no sibling) starts
  **degraded**: dashboards render, `/predict` returns 503. Phase 4 decides
  whether the quickstart should ship a tiny bundled checkpoint or keep the
  sibling-mount assumption.
- Local **single-map CPU latency** (in-process, host box, 22-core / 11 torch
  threads): **p50 30.6 ms, p99 36.4 ms** (n=60, warm). This is a reference only
  — the *real* p50/p99 numbers come from the GB10 arm64 deploy in Phase 4.

### Tests (10 passed, ~14 s)

```
tests/test_api.py ............ healthz, metrics-parse, count-increments, 422-on-1D
tests/test_predict_parity.py . parity vs wafer-mixed evaluate path (24 maps); predicted_labels field
tests/test_snapshot.py ....... determinism; committed-artifact integrity; subset reproduces committed rows
```

Warnings are third-party only (starlette TestClient→httpx2; matplotlib/pyparsing).

### Environment notes

- Shared workspace venv `/home/waferclassifier/.venv`; Phase 0 added
  `fastapi`, `uvicorn`, `prometheus-client`, `httpx` to it.
- `pip install -e .` done in that venv so `wafer_deploy` + `serve` import.
- **Native Docker Engine installed in this distro** (systemd service). The
  current session uses a temporary `chmod 666 /var/run/docker.sock` shim so a
  non-login shell can reach the daemon; after the next `wsl --shutdown` the
  `docker` group membership takes over and the shim can be dropped.

---

## Next: Phase 1 — Unsupervised monitors (input + prediction-rate) ← core

Open with: *"Read `ROADMAP.md` and Phase 1 of `PLAN-wafer-deploy.md`. Implement
Phase 1 only."*

Ready-made hooks left by Phase 0:
- **Embeddings** already exposed by `Predictor` (512-d, `predict_batch` returns
  them) and frozen in the snapshot → MMD / KS covariate-drift bank.
- **Reference prediction-rate + label histogram** already in the snapshot
  summary → PSI / chi-square prediction-drift baseline.
- **`serve/metrics.py`** is the single registration site — add drift gauges
  there so they land on `/metrics` alongside the serving counters; add matching
  Grafana panels to `monitoring/grafana/dashboards/wafer_deploy.json`.
- Report the **no-drift false-alarm rate** in this file when Phase 1 lands.
