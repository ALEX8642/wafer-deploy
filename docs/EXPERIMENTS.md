# Phase 3 — scored shift experiments + retrain trigger

This is the headline experiment of `wafer-deploy`: the three monitors and the
combined retrain trigger, scored under **controlled shifts** with honest numbers
— detection latency, recall, false-alarm rate, and *which monitor fires first* —
including where a monitor is blind.

Everything here is reproducible from committed artifacts:

```bash
# regenerate the scored results (needs the wafer-mixed checkpoint + datasets):
python scripts/run_shift_experiments.py         # → experiments/shift_results.json
# regenerate the figures (from the committed JSON alone, no checkpoint):
python scripts/make_shift_figures.py            # → assets/shift_*.png
```

The scored results live in the small, committed `experiments/shift_results.json`;
the figures render from it. No model or dataset is needed to reproduce the
figures — only to recompute the numbers.

## Setup

Each experiment is a **stream** of non-overlapping 200-map windows: 3 warmup
windows (clean, in-domain, drawn from the frozen reference snapshot) followed by
7 shifted windows. The shift turns on at **window 3** (the *onset*), so detection
latency is *windows after a known ground-truth onset*. The three monitors are the
served ones (`drift.py`, `calibration.py`), built once from the reference
snapshot; the calibration verdict reaches the trigger **2 windows late** — the
delayed-label lag — so its latency is honestly larger than the label-free
channels'.

The **retrain trigger** (`trigger.py`) ORs the three channels with hysteresis:
**3 consecutive** OR'd-alarm windows arm it (debounce), and it releases after 3
clear windows. Persistence is why a single noisy window can't order a retrain.

Thresholds (calibrated from the reference null in earlier phases):
MMD² 0.00352, PSI 0.25, ECE 0.01001 (reference ECE 0.00429).

## Shift sources

1. **Synthetic corruption sweep** — `rotation` (≤45° about centre), `noise`
   (random pass↔fail die flips), `resolution` (coarsen-then-restore), each at
   intensity {0, 0.25, 0.5, 0.75, 1.0}. Intensity 0 is an exact identity — the
   false-alarm control at the foot of every curve.
2. **Real cross-dataset shift** — WM-811K single-defect maps fed through the
   MixedWM38-served model. A genuine covariate shift with a **known cause** that
   was never tuned for: a different fab, a different die grid, the same 8-defect
   taxonomy.
3. **Class-prior "defect campaign"** — a time-ordered stream whose Edge-Ring
   prevalence ramps from its natural prior to 60% after onset (reuses the
   snapshot, no re-inference).

## Results

### Scored detection table

`assets/shift_alarm_table.png` — latency in windows after onset (200 maps/window):

| shift | first channel | cov lat | pred lat | cal lat | trigger lat | trigger FA@0 |
|---|---|---|---|---|---|---|
| rotation @1.00 | covariate | 0 | — | 2 | 2 | no |
| noise @1.00 | covariate | 0 | 0 | 2 | 2 | no |
| resolution @1.00 | covariate | 0 | 0 | 2 | 2 | no |
| WM-811K cross-dataset | covariate | 0 | 0 | 2 | 2 | n/a |
| class-prior campaign | covariate | 3 | — | — | 5 | n/a |

**False-alarm control:** at intensity 0 every channel is silent and the trigger
does **not** fire, on all three corruptions. (The MixedWM38 test split is ordered
by defect-combo, so the clean body is *shuffled* before use — a contiguous slice
would itself be a distribution shift; that fix is in `run_shift_experiments.py`.)

### Detection curve

`assets/shift_detection_curve.png` — post-onset recall (fraction of shifted
windows that alarm) vs intensity, per channel:

- **rotation** — covariate and calibration recall climb with intensity, but
  **prediction PSI stays flat at 0**. A rotation moves the input embedding without
  changing *which* defect the model predicts, so the label-distribution monitor is
  structurally blind to it. The trigger needs intensity ≥ 0.5 to fire (wafer
  ~4-fold symmetry makes small rotations a weak shift).
- **noise** — covariate leads (recall 1.0 by intensity 0.25); prediction lags
  (0.0 → 0.57 → 1.0 across 0.25/0.5/0.75) because die-flips only flip *predicted
  labels* once they are dense enough. Trigger fires from intensity 0.25.
- **resolution** — covariate and prediction rise together from 0.25; coarsening
  destroys fine defect detail, moving both the embedding and the decision.

Calibration recall plateaus at **0.71** across corruptions — an honest artifact of
the 2-window label lag: the last two windows' labels never arrive within the
stream, so calibration can score at most 5 of 7 post-onset windows.

### Which monitor fires first (the diagnostic-value story)

`assets/shift_monitor_timeline.png` — each monitor's signal ÷ its threshold over
time (1.0 = alarm bar), with the retrain trigger's fire marked.

The ordering is consistent and *is* the design thesis: **the unsupervised
channels lead, calibration confirms on delayed labels.**

- **Synthetic noise (0.5):** covariate MMD² jumps to ~30× threshold at onset;
  calibration ECE to ~13×; prediction PSI hovers just above 1×. Trigger fires at
  window 5 (onset + 2).
- **WM-811K cross-dataset:** covariate MMD² to **85–100× threshold** — a huge,
  genuine covariate shift. The served MixedWM38 model recovers the true WM-811K
  single defect on only **45.6%** of maps (the honest cross-domain accuracy drop,
  cause known and untuned). Covariate and prediction both alarm on 100% of
  post-onset windows; calibration on 67%.
- **Class-prior campaign:** the interesting **miss**. Covariate MMD² ramps to 7.5×
  threshold as the Edge-Ring-heavy embedding mix diverges, firing the trigger at
  window 8. But **prediction PSI peaks at 0.118 — below the 0.25 bar — and never
  alarms**. A gradual class-prior campaign of this magnitude is caught by the
  *embedding* monitor, not by label-share PSI. Reported as a miss rather than
  tuned away: PSI on the label-share distribution is not the right instrument for
  a slow prior drift here; the covariate monitor is.

## What the trigger catches unsupervised vs needs labels

- **Caught unsupervised (no labels):** every shift above eventually fires the
  trigger from label-free channels alone — covariate MMD² is the workhorse and
  leads on all five; prediction PSI adds fast confirmation on noise/resolution and
  the WM-811K shift. This is the production-relevant property: **you do not have to
  wait for labels to know the input has moved.**
- **Needs labels (confirmation):** calibration ECE only scores once delayed labels
  land (here, +2 windows) — it *confirms* the shift is hurting the model's
  confidence, and (from Phase 2) it is the *only* channel that catches a pure
  confidence-erosion failure where the hard decisions, and thus accuracy, don't
  move. It is a confirmer, not a first responder, by construction.

## Honest misses / caveats

- **Prediction PSI is blind to rotation and to the gradual class-prior campaign.**
  Both are real shifts the covariate monitor catches; PSI is the wrong instrument
  for a shift that doesn't change the *mix* of predicted labels enough to cross
  0.25. Kept in the table as `—`.
- **Calibration latency is capped by the label lag** (+2 windows here) and its
  recall by how many windows' labels fit inside the stream (0.71 ceiling). This is
  the delayed-label regime, not a monitor weakness.
- **The corruptions are synthetic and graded on purpose** — they trace a detection
  curve. The WM-811K experiment is the un-tuned real shift that anchors the
  synthetic ones; its known cause (different fab/dataset, same taxonomy) is why it
  is the honest headline, and its 45.6% true-label recall is the un-flattering
  number that makes the monitoring case.
- **Latency is in windows** (200 maps each). Absolute wall-clock latency depends on
  throughput; the GB10 numbers in Phase 4 set that scale.
