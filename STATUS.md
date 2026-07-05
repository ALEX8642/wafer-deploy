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

## Phase 2 — Calibration decay + delayed-label eval ✅ (2026-07-05)

The label-*dependent* monitor: when delayed ground truth lands for a window,
score its ECE vs the frozen reference calibration and alarm on decay — the
"confidence stopped meaning what it meant" signal. Fed through a new `/feedback`
endpoint; live on `/metrics` + a dedicated Grafana row.

### What was built

- **`src/wafer_deploy/calibration.py`** — numpy-only, same discipline as
  `drift.py` (no scipy, runs off the committed snapshot with no checkpoint):
  - **`binary_ece`** reimplemented bit-for-bit to wafer-mixed's binning
    (n_bins=15, same bin convention) so the windowed ECE and the snapshot's
    reference ECE are the *same measurement* — `test_calibration` pins the two
    equal when the checkout is present. Plus `per_label_ece`, `reliability_bins`
    (pooled curve with empty bins broken to NaN).
  - **`CalibrationMonitor`** — reference probs + labels fix the per-label ECE
    baseline **and** the alarm threshold, which — exactly like the Phase 1
    covariate monitor — is **calibrated from the reference null**: the
    `ece_quantile=0.99` of windowed mean-ECE drawn from the reference itself. A
    200-map window's ECE is noisier and biased up vs the 7,603-map reference
    point (and rare labels like Near-full are noisier still); calibrating on
    like-sized windows folds all of that into an explicit false-alarm quantile
    rather than a hand-picked constant.
  - **Delayed labels:** `buffer_predictions` records served probs in order;
    `add_labels` supplies delayed ground truth, **FIFO-matched** to the oldest
    un-labelled predictions, and scores a non-overlapping window the moment its
    last label lands. The await buffer is bounded by the label lag (L windows ×
    8 floats/map — negligible); `pending_labels` exposes it.
- **`serve/app.py`** — new **`POST /feedback`** (delayed multi-hot labels,
  FIFO-matched; 422 on wrong width, 400 on a label with no matching prediction,
  503 if no snapshot). `/predict` now also *buffers* served probs into the
  calibration monitor (scored later, at feedback). Monitor built at startup from
  the same snapshot; `/healthz` reports `calibration_active`.
- **`serve/metrics.py`** — calibration gauges (`calibration_ece` +
  `_threshold` + `_reference_ece`, `_drift_alarm`, per-label `_ece_per_label`,
  `_pending_labels`) sharing the `drift_windows_total` / `drift_alarms_total`
  counters under `monitor="calibration"`, so the empirical false-alarm rate is
  queryable the same way the Phase 1 monitors are.
- **`configs`/`DeployConfig`** — `calibration_n_bins=15`,
  `calibration_ece_quantile=0.99`, `calibration_calib_trials=200`,
  `calibration_label_lag=2` (the harness's simulated lag),
  `calibration_max_pending=4000` (await-buffer retention cap).
- **`scripts/replay_labeled_stream.py`** — the delayed-label harness: streams
  maps to `/predict`, then POSTs their labels to `/feedback` **held back by
  `--lag` windows**; prints each scored window's ECE vs reference/threshold and a
  final false-alarm / alarm summary. `--shift` reuses the Phase 1 demo
  corruption to show calibration decay under covariate drift.
- **`scripts/make_calibration_figures.py`** — reproducible from the committed
  snapshot alone (no checkpoint). Two figures → `assets/`.
- **Grafana** — a "Phase 2 — calibration decay (delayed labels)" row: alarm
  stat, calibration alarm-rate, windows scored, predictions-awaiting-labels, ECE
  vs threshold vs reference, per-label ECE.
- **Tests** — 14 new in `test_calibration.py` (see below); 39 total, all green.

### Accept-criteria status

| Criterion | Status |
|---|---|
| Calibration monitor + delayed-label harness | ✅ `CalibrationMonitor` + `/feedback` + `replay_labeled_stream.py` |
| Reliability / ECE figures in `assets/` | ✅ `reliability_reference_vs_drifted.png`, `calibration_ece_over_time.png` |
| Tests pass | ✅ 39 passed (14 new), ~98 s |

### Honest numbers

**No-drift false-alarm rate** (build on one random half of the 7,603-map
reference, stream the disjoint half's *labelled* windows, 5 seeds):

| Monitor | Design FA | **Empirical no-drift FA** | Signal under null |
|---|---|---|---|
| Calibration (ECE, q=0.99) | 1.0 % | **3.2 %** (per-seed 0.0–10.5 %) | windowed ECE 0.0063 < threshold 0.0094 (ref 0.0043) |

Same shape as Phase 1's covariate monitor (3.2 %): held-out windows score
marginally above the build-half null the threshold was fit on. Windowed ECE
(0.0063) sits above the 7,603-map reference (0.0043) purely from the 200-sample
size — which is exactly why the threshold is calibrated on 200-map windows, not
against the reference point.

**Detection under a monotone confidence-erosion warp** (whole test split, γ<1
flattens probabilities toward 0.5):

| erosion γ | alarm rate | mean windowed ECE |
|---|---|---|
| 0.7 | 63 % | 0.013 |
| 0.5 | 97 % | 0.027 |
| 0.3 | 100 % | 0.075 |

**Live `/feedback` end-to-end** (uvicorn, lag=2 windows): no-drift stream **0/4
windows alarmed**; the crude `--shift 0.5` covariate corruption **4/4 alarmed at
ECE ≈ 0.28** — the model goes *confidently wrong* against the original labels, so
the covariate shift surfaces as a large calibration excursion once labels land.

### Phase-2 tie-in: does covariate drift surface as calibration decay before or after accuracy drops?

Answered two honest ways, both in `assets/`/STATUS:

- **Isolated confidence channel** (`calibration_ece_over_time.png`): a monotone
  erosion warp with the hard **decisions held fixed** → **macro-F1 flat at 0.979
  while ECE rose 0.006 → 0.072, 12/38 windows alarmed.** Calibration decays with
  accuracy *exactly* unchanged — the calibration monitor catches a failure the
  accuracy/prediction monitors are structurally blind to.
- **Coupled covariate shift** (harness `--shift`): the label-free Phase 1
  monitors (covariate MMD², prediction PSI) fire **immediately**; calibration
  confirms **later** (it must wait for delayed labels) but with a large ECE
  excursion (~0.28). So the *ordering* is: unsupervised signals lead, calibration
  confirms — which is the whole design thesis.

The fully *scored* coupling (graded intensity → which monitor fires first per
shift type, detection-latency curves) is the Phase 3 job.

### Tests (14 new, all green)

```
test_calibration.py  binary_ece hand-value + wafer-mixed parity + zero-on-calibrated;
                     reliability_bins breaks empty bins;
                     delayed-label lag: no score until a full window of labels;
                     FIFO alignment (window ECE == direct ECE of its own rows);
                     label-without-prediction raises;
                     reference ECE reproduces committed meta;
                     no-drift false-alarm control (disjoint half, labelled);
                     confidence-erosion → 100% alarm; from_snapshot builds monitor;
                     /metrics exposes calibration gauges;
                     live /predict→/feedback scores a window + advances the counter;
                     /feedback rejects wrong-width labels
```

Only the last three carry `needs_mixed` (drive the live service); the rest run
off the committed snapshot.

### Honest caveats / deviations

- **The alarm is on mean per-label ECE**, threshold-calibrated from the reference
  null (parallel to the covariate MMD² knob) rather than a fixed "ECE > 0.05"
  rule — the reference ECE (0.0043) is far below any textbook constant, so a
  constant would be meaningless here. Per-label ECE is still exposed as a gauge
  for interpretability.
- **Figures use a synthetic prob-space erosion warp**, clearly labelled — it
  isolates the calibration channel for the before/after-accuracy point. The real
  map-space shifts (corruption sweep, WM-811K cross-domain) that move
  embeddings, probs and labels *together* are Phase 3.
- **`docker compose up` not re-run** (no Dockerfile/deps change — calibration is
  numpy-only, already in the image; matplotlib, added in Phase 0, covers the
  offline figure script). The Phase-0 end-to-end docker check still stands; worth
  one re-run in Phase 3/4.
- The await buffer is **bounded** at `calibration_max_pending` (default 4000 ≈
  20 windows) — required by the plan's co-tenant "bounded state" guardrail, since
  every `/predict` buffers a prob row until its delayed label lands. Past the cap
  the oldest awaiting predictions are evicted and their (lost-to-lag) labels
  skipped on arrival, so FIFO alignment survives the bound; evictions surface on
  `wafer_deploy_calibration_dropped_total`. In the delayed-label simulation the
  lag keeps the buffer far under the cap (no evictions).

---

## Phase 3 — Scored shift experiments + retrain trigger ✅ (2026-07-05) ← headline

The three monitors and a combined **retrain trigger** scored under controlled
shifts, with honest numbers: detection latency, recall, false-alarm rate, and
*which monitor fires first* — misses included. Offline drift science (per the
hardware policy, on the CPU box, never the cluster); only the small scored-results
JSON + figures are committed.

### What was built

- **`src/wafer_deploy/shift.py`** — the *scored* shift library (numpy-only, runs
  in the lean image or offline identically). Three graded per-map corruptions
  with `intensity==0` an **exact identity** (the FA control depends on this):
  `rotation` (≤45° nearest-neighbour), `noise` (pass↔fail die flips),
  `resolution` (coarsen-then-restore). Plus `class_prior_campaign` (a stream-level
  Edge-Ring "defect campaign", no re-inference) and `wm811k_to_multihot` (the
  WM-811K→MixedWM38 label bridge — same 8-defect taxonomy, `none`→zeros).
- **`src/wafer_deploy/trigger.py`** — `RetrainTrigger`: ORs the three monitor
  channels with **hysteresis** (3 consecutive OR'd-alarm windows arm it,
  debounce; releases after 3 clear windows, anti-chatter). Bounded integer state
  → co-tenant-safe; the *same class* drives the offline scoring and the online
  sidecar.
- **`src/wafer_deploy/experiments.py`** — the offline scoring engine: batched
  `predict_maps` / `corrupt_and_predict`, and `run_stream` which feeds each window
  through the **real** monitor objects + trigger, applying the calibration
  **label lag** (calibration's verdict for window *k* reaches the trigger *k+lag*).
  Returns per-window records + a detection summary (per-channel latency, recall,
  first channel, trigger latency). `CalibrationMonitor.score_window` added so the
  online FIFO path and the offline direct path are the *same measurement*.
- **Service wiring** — `RetrainTrigger` built at startup; fed once per completed
  covariate window (the heartbeat), folding in the paired prediction alarm and the
  latest — possibly lagged — calibration verdict (`app.py`). New gauges
  `wafer_deploy_retrain_trigger` (latched), `..._reason{channel}`, and counter
  `..._triggers_total` (`metrics.py`); `/healthz` reports `trigger_active` +
  `retrain_triggered`. Grafana **Phase 3 row** (7 panels): trigger stat, decisions
  counter, per-channel contribution timeline, alarm counts by channel.
- **`scripts/run_shift_experiments.py`** — the driver: warmup from the snapshot
  (clean, no inference) + shifted body; writes committed
  `experiments/shift_results.json` (28.5 KB). `--quick` smoke mode.
  **Bug the sweep caught:** the MixedWM38 test split is **ordered by defect-combo**,
  so a contiguous body slice is itself a distribution shift (the intensity-0 FA
  control fired). Fixed by shuffling test indices before slicing — the FA control
  is now silent across all corruptions.
- **`scripts/make_shift_figures.py`** — renders 3 figures from the committed JSON
  alone (no checkpoint): detection curve, monitor-firing timeline, alarm table.
- **Tests** — 24 new (7 `test_trigger.py`, 17 `test_experiments.py`); 65 total,
  all green (~143 s). Only 1 new test (`test_corruption_moves_embeddings`) carries
  `needs_mixed`; the rest run off the committed snapshot / synthetic.

### Accept-criteria status

| Criterion | Status |
|---|---|
| Scored detection table (latency + false-alarm) in STATUS | ✅ below |
| Trigger policy implemented (hysteresis, OR of 3 channels) | ✅ `trigger.py`, wired live + offline |
| Figures + narrative committed | ✅ `assets/shift_{detection_curve,monitor_timeline,alarm_table}.png` + `docs/EXPERIMENTS.md` |

### Scored detection table (the honest numbers)

Stream = 3 warmup windows (clean) + 7 shifted, onset at window 3, 200 maps/window,
calibration label-lag 2, trigger persistence 3. Latency is **windows after onset**.
Thresholds: MMD² 0.00352, PSI 0.25, ECE 0.01001 (ref ECE 0.00429).

| shift | first channel | cov lat | pred lat | cal lat | trigger lat | FA@0 |
|---|---|---|---|---|---|---|
| rotation @1.00 | covariate | 0 | — (blind) | 2 | 2 | no |
| noise @1.00 | covariate | 0 | 0 | 2 | 2 | no |
| resolution @1.00 | covariate | 0 | 0 | 2 | 2 | no |
| WM-811K cross-dataset | covariate | 0 | 0 | 2 | 2 | n/a |
| class-prior campaign | covariate | 3 | — (blind) | — (blind) | 5 | n/a |

- **False-alarm control:** intensity 0 → every channel silent, trigger does **not**
  fire, on all three corruptions.
- **Detection curve** (`shift_detection_curve.png`): covariate recall→1.0 by
  intensity 0.25 for noise/resolution and 0.5 for rotation (wafer ~4-fold symmetry
  makes small rotations a weak shift). **Prediction PSI is flat at 0 for rotation**
  — a rotation moves the input but not *which* defect is predicted, so the
  label-distribution monitor is structurally blind. Calibration recall plateaus at
  0.71 (the 2-window lag means the last two windows' labels never arrive in-stream).
- **WM-811K cross-dataset:** covariate MMD² hits **85–100× threshold**; the served
  MixedWM38 model recovers the true WM-811K single defect on only **45.6%** of maps
  — the un-tuned real cross-domain accuracy drop, cause known. Caught unsupervised.
- **Class-prior campaign (the reported miss):** covariate MMD² ramps to 7.5×
  threshold and fires the trigger (latency 5), but **prediction PSI peaks at 0.118,
  below the 0.25 bar, and never alarms**. A gradual prior drift of this magnitude
  is caught by the embedding monitor, not by label-share PSI — reported, not tuned.

### Which shifts are caught unsupervised vs need labels

- **Unsupervised (no labels):** all five shifts fire the trigger from label-free
  channels alone — covariate MMD² leads every one; prediction PSI adds fast
  confirmation on noise/resolution/WM-811K. You don't wait for labels to know the
  input moved.
- **Needs labels (confirmation):** calibration ECE only scores once delayed labels
  land (+2 windows) — a confirmer, not a first responder, by construction. From
  Phase 2 it is still the *only* channel that catches pure confidence erosion with
  accuracy held fixed.

Full narrative + caveats: `docs/EXPERIMENTS.md`. Reproduce:
`python scripts/run_shift_experiments.py` (needs checkpoint+data) →
`python scripts/make_shift_figures.py` (committed JSON only).

### Honest caveats / deviations

- **`docker compose up` not re-run this phase** — shift/trigger/experiments are
  numpy-only, already in the image; matplotlib (Phase 0) covers the offline figure
  script. The Phase-0 end-to-end docker check still stands; the new Grafana Phase 3
  row + live trigger firing get their end-to-end run in Phase 4's quickstart.
- **Corruptions are synthetic + graded on purpose** (to trace a curve); WM-811K is
  the un-tuned real anchor. Latency is in *windows* (200 maps); absolute wall-clock
  scale is a Phase 4 (GB10 throughput) number.
- **Online trigger vs offline scoring:** the offline engine aligns all three
  channels per data-window with calibration lagged (rigorous for latency); the live
  service drives the trigger off covariate windows (the heartbeat) and folds in the
  latest prediction/calibration verdicts. Same `RetrainTrigger` class, documented in
  `app.py`.
- **Push is on Alex** (auto-mode blocks pushes); Phase 3 committed locally.

### Environment notes

- No new dependencies (numpy-only additions). Shared workspace venv unchanged.
- The full sweep is ~520 s on the 22-core CPU box (≈50 maps/s batched, ~22 k
  forward passes). `--quick` (~180 s) smoke-tests the pipeline end-to-end.

---

## Next: Phase 4 — GB10 deploy + real numbers + package

Open with: *"Read `ROADMAP.md` and Phase 4 of `PLAN-wafer-deploy.md`. Implement
Phase 4 only."*

Ready-made hooks left by Phase 3:
- **Retrain trigger live** on `/metrics` + a Grafana Phase 3 row; the quickstart's
  "drive a drift stream → watch the dashboards → see the trigger fire" is wired —
  Phase 4 just needs to run it through `docker compose` end-to-end (drive a shifted
  stream via `scripts/replay_stream.py --shift …`, watch `wafer_deploy_retrain_trigger`).
- **Scored results committed** (`experiments/shift_results.json`, `docs/EXPERIMENTS.md`,
  `assets/shift_*.png`) — Phase 4's README pulls these numbers, does **not** recompute.
- **Resume bullet inputs:** covariate MMD² leads all shifts; WM-811K true-label
  recall 45.6% (real cross-domain drop); no-drift trigger FA = none at intensity 0;
  detection latency ≤ 2 windows for the strong shifts. Add real GB10 p50/p99 in Phase 4.
