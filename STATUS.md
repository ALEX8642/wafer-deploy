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

## Phase 1 — Unsupervised monitors: input + prediction-rate ✅ (2026-07-05)

Both label-free monitors are live on `/metrics` + the dashboard, calibrated
against the committed reference snapshot, and fed from the `/predict` path.

### What was built

- **`src/wafer_deploy/drift.py`** — numpy-only (no scipy, so the online path
  stays in the lean serving image), bounded state:
  - **Drift math** as pinned pure functions: `rbf_mmd2` (unbiased squared MMD,
    RBF kernel, diagonal excluded), `median_heuristic_gamma`, `ks_per_dim`
    (per-dimension two-sample KS via sorted-CDF `searchsorted`), `psi`
    (normalizes counts *or* proportions, ε-clipped).
  - **`CovariateDriftMonitor`** — embedding-space MMD² (primary) + KS
    (interpretability) vs a **bounded reference bank** (subsampled to
    `max_ref=1024`). Fixed RBF bandwidth from the bank's median heuristic so
    MMD² is comparable across windows. **The MMD² alarm threshold is calibrated
    from the reference itself** — the `mmd_quantile=0.99` of the null
    window-vs-bank distribution — so the false-alarm rate is an explicit design
    quantile, not a guess.
  - **`PredictionDriftMonitor`** — PSI on the predicted-label *share*
    distribution (multi-label, so PSI compares which labels dominate) + windowed
    defect-rate vs reference; conventional `psi_threshold=0.25`.
  - **Non-overlapping windows** (`_WindowBuffer`): state never exceeds one
    window (drained on completion) → sidecar-safe. `update()` takes a batch or a
    single row, returns a result per completed window. `DriftMonitors` pairs both
    + a `from_snapshot` factory + `observe()`.
- **`serve/metrics.py`** — drift gauges (`covariate_mmd2` + threshold, `ks_mean`,
  `ks_max`, `covariate_drift_alarm`; `prediction_psi` + threshold,
  `prediction_defect_rate`, `prediction_drift_alarm`; `windowed_label_rate` per
  label) and window/alarm **counters** (`drift_windows_total`,
  `drift_alarms_total` by `monitor`) so the empirical false-alarm rate is
  queryable straight off `/metrics`. `init_drift_gauges` publishes thresholds
  pre-traffic; `record_covariate` / `record_prediction` are the single update
  site.
- **`serve/app.py`** — monitors built at startup from the reference snapshot
  (independent of the checkpoint; degrades cleanly if the snapshot is absent, and
  `/healthz` now reports `monitors_active`). Fed under a `threading.Lock` from
  `/predict` (sync endpoints run in a threadpool); gauges reflect completed
  windows only, so most requests just buffer.
- **`configs`/`DeployConfig`** — `drift_window_size=200`, `drift_max_ref=1024`,
  `drift_mmd_quantile=0.99`, `drift_calib_trials=200`, `drift_psi_threshold=0.25`
  (YAML-overridable).
- **Grafana** — a "Phase 1 — unsupervised drift monitors" row: covariate /
  prediction alarm stats, a covariate alarm-rate stat, windows-evaluated, MMD²
  vs threshold, PSI vs threshold, windowed defect-rate, windowed per-label rate.
- **`scripts/replay_stream.py`** — the streaming harness: replays the
  wafer-mixed test split through the live service (`--shift` applies a crude
  demo-only covariate corruption — the *scored* sweep is Phase 3), prints each
  completed window's MMD²/KS/PSI/defect-rate off `/metrics`, and a final
  false-alarm/alarm summary.

### Accept-criteria status

| Criterion | Status |
|---|---|
| Both monitors live on `/metrics` + dashboard | ✅ gauges + counters exposed; 9 new panels |
| No-drift false-alarm rate reported (below) | ✅ |
| Tests pass | ✅ 25 passed (14 new in `test_drift.py`), ~90 s |

### No-drift false-alarm rate (the honest number)

Measured by building each monitor on one random half of the 7,603-map reference
and streaming the **disjoint** other half in non-overlapping windows of 200
(~19 windows/half), averaged over 5 seeds:

| Monitor | Design FA | **Empirical no-drift FA** | Signal under null |
|---|---|---|---|
| Covariate (MMD², q=0.99) | 1.0 % | **3.2 %** (per-seed 0.0–5.3 %) | windowed MMD² 0.00022 ≪ threshold 0.00297 |
| Prediction (PSI, τ=0.25) | — | **0.0 %** | windowed PSI 0.018 ≪ 0.25 |

The ~2 pt covariate gap above the 1 % design point is expected and honest: the
threshold is calibrated on windows drawn from the *build* half (which overlap
the reference bank), so truly held-out windows score marginally higher. It is
still an order of magnitude below a broken always-fires monitor, and the mean
null MMD² sits ~13× under the threshold. **Detection** (not the scored Phase-3
job): a per-dimension 3σ embedding offset drives MMD² over threshold on **100 %**
of windows — pinned in `test_shifted_stream_raises_mmd_and_alarms`.

### Tests (14 new, all green)

```
test_drift.py  psi hand-value + zero + counts/proportions equivalence;
               mmd² null≈0 + grows with mean-shift; ks flags shifted dim;
               non-overlapping window bookkeeping;
               no-drift false-alarm control (committed snapshot, disjoint half);
               shifted stream → 100% alarm; prediction null (in-domain) + label-shift alarm;
               from_snapshot builds both; /metrics exposes drift gauges;
               full window through the live service advances windows_total
```

Covariate/PSI tests run off the **committed snapshot** — no checkpoint or dataset
needed; only the two end-to-end service tests carry `needs_mixed`.

### Honest caveats / deviations

- **PSI is on the label *share* distribution**, not a single categorical
  histogram (multi-label labels don't sum to 1). Documented in `drift.py`; it
  reads as "which defects dominate has shifted." Chi-square was folded into this
  rather than added as a second redundant statistic — the plan listed
  "PSI / chi-square" as alternatives.
- The demo `--shift` in the harness is intentionally crude (rot90 + failing-die
  injection); the **scored** corruption sweep + real WM-811K cross-domain shift
  are Phase 3. No detection-latency curve is claimed here.
- `docker compose up` not re-run this phase (no Dockerfile/deps change — drift
  is numpy-only, already in the image); the Phase-0 end-to-end docker check
  still stands. Worth one re-run in Phase 3/4 once panels matter for the demo.

---

## Next: Phase 2 — Calibration decay + delayed-label eval

Open with: *"Read `ROADMAP.md` and Phase 2 of `PLAN-wafer-deploy.md`. Implement
Phase 2 only."*

Ready-made hooks left by Phase 1:
- **Reference calibration** already frozen in the snapshot (`probs`, `y_true`,
  per-label `reference_ece` in the meta summary) → the Phase 2 ECE baseline.
- **`_WindowBuffer`** (in `drift.py`) is the reusable non-overlapping-window
  primitive; the delayed-label harness needs the same windowing with an N-window
  lag before a window is *scored*.
- **`serve/metrics.py`** is the single gauge-registration + `record_*` site; add
  the calibration gauges there and matching Grafana panels.
- Report whether covariate drift surfaces as calibration decay **before or
  after** accuracy drops (Phase 2 tie-in to the wafer-mixed threshold story).
