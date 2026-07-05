"""run_shift_experiments.py — the Phase 3 scored shift sweep.

Runs the three controlled-shift sources through the offline scoring engine
(``wafer_deploy.experiments``) and writes one small, committed results artifact,
``experiments/shift_results.json``, that STATUS and the figure script read. This
is the offline drift science the hardware policy keeps on the CPU/5090 box; it
needs the sibling wafer-mixed checkpoint + dataset (and, for the cross-dataset
shift, WM-811K), none of which ship in this repo.

    python scripts/run_shift_experiments.py               # full sweep
    python scripts/run_shift_experiments.py --quick        # small smoke run

Shift sources:
    (a) synthetic corruption sweep — rotation / noise / resolution at rising
        intensity → a detection curve (intensity vs latency / recall) with the
        false-alarm control at intensity 0;
    (b) real cross-dataset shift — WM-811K single-defect maps fed through the
        MixedWM38-served model (a genuine covariate shift with a known cause);
    (c) class-prior "defect campaign" — one label's prevalence ramps over time
        (reuses the committed snapshot, no re-inference).

Every stream is warmup windows (clean, drawn from the reference snapshot) then
the shifted body, so detection latency is measured from a known onset window.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wafer_deploy.calibration import CalibrationMonitor  # noqa: E402
from wafer_deploy.config import DeployConfig  # noqa: E402
from wafer_deploy.drift import DriftMonitors  # noqa: E402
from wafer_deploy.experiments import (  # noqa: E402
    corrupt_and_predict, predict_maps, reset_monitors, run_stream)
from wafer_deploy.labels import LABELS  # noqa: E402
from wafer_deploy.predictor import Predictor  # noqa: E402
from wafer_deploy.shift import (  # noqa: E402
    CORRUPTIONS, class_prior_campaign, wm811k_to_multihot)
from wafer_deploy.snapshot import load_snapshot  # noqa: E402

WM811K_PKL = Path("../wafer-defect-classifier/data/raw/LSWMD.pkl")


def build_monitors(snap, cfg, seed):
    mons = DriftMonitors.from_snapshot(
        snap, LABELS, window_size=cfg.drift_window_size, max_ref=cfg.drift_max_ref,
        mmd_quantile=cfg.drift_mmd_quantile, calib_trials=cfg.drift_calib_trials,
        psi_threshold=cfg.drift_psi_threshold, seed=seed)
    cal = CalibrationMonitor.from_snapshot(
        snap, LABELS, window_size=cfg.drift_window_size, n_bins=cfg.calibration_n_bins,
        ece_quantile=cfg.calibration_ece_quantile,
        calib_trials=cfg.calibration_calib_trials, seed=seed)
    return mons, cal


def new_trigger(cfg):
    from wafer_deploy.trigger import RetrainTrigger
    return RetrainTrigger(persistence=cfg.trigger_persistence, release=cfg.trigger_release)


def warmup_arrays(snap, warmup_windows, window_size, seed):
    """Clean in-domain warmup drawn from the reference snapshot (no inference)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(snap.n)[: warmup_windows * window_size]
    return (snap.embeddings[idx], snap.preds[idx], snap.probs[idx], snap.y_true[idx])


def score(mons, cal, cfg, warm, body, onset_window):
    """Concatenate warmup + body, reset monitors, and score the stream."""
    reset_monitors(mons, cal)
    E = np.vstack([warm[0], body[0]])
    P = np.vstack([warm[1], body[1]])
    Pr = np.vstack([warm[2], body[2]])
    Y = np.vstack([warm[3], body[3]])
    return run_stream(mons, cal, new_trigger(cfg), embeddings=E, preds=P, probs=Pr,
                      y_true=Y, window_size=cfg.drift_window_size,
                      onset_window=onset_window, calibration_lag=cfg.calibration_label_lag)


# --------------------------------------------------------------------------- #
# WM-811K cross-dataset source.
# --------------------------------------------------------------------------- #

def load_wm811k_single_defect(n: int, seed: int):
    """Balanced sample of WM-811K single-defect maps + their multi-hot labels."""
    import pandas as pd
    df = pd.read_pickle(WM811K_PKL)

    def unwrap(v):
        a = np.asarray(v).ravel()
        return str(a[0]).strip() if a.size else ""

    df = df[df["failureType"].apply(lambda v: unwrap(v) not in ("", "nan", "none", "Nonetype"))].copy()
    df["label"] = df["failureType"].apply(unwrap)
    df = df[df["label"].isin(LABELS)]                        # the 8 defect classes only
    rng = np.random.default_rng(seed)
    per = max(1, n // len(LABELS))
    picks = []
    for name in LABELS:
        pool = df[df["label"] == name]
        if len(pool):
            picks.append(pool.sample(min(per, len(pool)), random_state=seed))
    sample = (pd.concat(picks)
              .sample(frac=1.0, random_state=seed).reset_index(drop=True))
    maps = sample["waferMap"].tolist()
    y = wm811k_to_multihot(sample["label"].tolist(), LABELS)
    return maps, y


# --------------------------------------------------------------------------- #
# Main sweep.
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 scored shift sweep")
    ap.add_argument("--quick", action="store_true",
                    help="tiny run (fewer maps/intensities) to smoke-test the pipeline")
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = DeployConfig.load()
    W = cfg.drift_window_size
    warmup_windows = 2 if args.quick else 3
    body_windows = 3 if args.quick else 7
    intensities = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.25, 0.5, 0.75, 1.0]
    body_maps = body_windows * W
    onset = warmup_windows

    if not cfg.checkpoint_path.exists():
        print(f"checkpoint missing at {cfg.checkpoint_path} — cannot run", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    predictor = Predictor(cfg)
    snap = load_snapshot(cfg.reference_snapshot_path)
    mons, cal = build_monitors(snap, cfg, args.seed)
    warm = warmup_arrays(snap, warmup_windows, W, args.seed)

    # Raw test maps + labels for the corruption bodies. The MixedWM38 test split
    # is ordered by defect-combo, so a contiguous slice is itself a distribution
    # shift — shuffle first so the intensity-0 body is a representative in-domain
    # sample (the false-alarm control must be genuinely un-shifted).
    maps, labels = predictor.wm.data.load_raw(predictor.mixed_cfg.data_root)
    test_idx = np.asarray(predictor.wm.data.load_splits(predictor.mixed_cfg)["test"])
    test_idx = test_idx[np.random.default_rng(args.seed + 1).permutation(len(test_idx))]
    body_idx = test_idx[:body_maps]
    raw_body = [maps[i] for i in body_idx]
    body_y = labels[body_idx].astype(np.int64)

    results: dict = {
        "meta": {
            "window_size": W, "warmup_windows": warmup_windows,
            "body_windows": body_windows, "onset_window": onset,
            "body_maps": body_maps, "intensities": intensities,
            "calibration_lag": cfg.calibration_label_lag,
            "trigger": {"persistence": cfg.trigger_persistence,
                        "release": cfg.trigger_release},
            "seed": args.seed, "labels": LABELS,
            "mmd2_threshold": float(mons.covariate.threshold),
            "psi_threshold": float(mons.prediction.psi_threshold),
            "ece_threshold": float(cal.threshold),
            "reference_ece_mean": float(cal.reference_ece_mean),
            "checkpoint": predictor.checkpoint_meta,
            "quick": bool(args.quick),
        },
        "sweeps": {}, "headline": {},
    }

    # ---- (a) synthetic corruption sweep ------------------------------------
    for cname, cfn in CORRUPTIONS.items():
        print(f"\n== corruption sweep: {cname} ==")
        rows = []
        for inten in intensities:
            emb, probs, preds = corrupt_and_predict(
                predictor, raw_body, cfn, inten, seed=args.seed)
            out = score(mons, cal, cfg, warm, (emb, preds, probs, body_y), onset)
            s = out["summary"]
            rows.append({"intensity": inten, "summary": s})
            print(f"  intensity {inten:.2f}: first={s['first_channel']} "
                  f"latency={s['detection_latency']} "
                  f"trigger_fired={s['trigger_fired']}(lat {s['trigger_latency']})")
            # keep full per-window records for one representative mid-intensity run
            if cname == "noise" and inten == 0.5:
                results["headline"]["noise_0.5"] = out
        results["sweeps"][cname] = rows

    # ---- (b) real cross-dataset shift (WM-811K) ----------------------------
    if WM811K_PKL.exists():
        print("\n== cross-dataset shift: WM-811K single-defect ==")
        wm_maps, wm_y = load_wm811k_single_defect(body_maps, args.seed)
        emb, probs, preds = predict_maps(predictor, wm_maps)
        n = (len(preds) // W) * W
        out = score(mons, cal, cfg, warm,
                    (emb[:n], preds[:n], probs[:n], wm_y[:n]), onset)
        # honest "known cause" read: did the served model recover the true label?
        hit = float(((preds[:n] * wm_y[:n]).sum(axis=1) > 0).mean())
        out["summary"]["wm811k_true_label_recall"] = hit
        results["headline"]["wm811k"] = out
        s = out["summary"]
        print(f"  first={s['first_channel']} latency={s['detection_latency']} "
              f"trigger_fired={s['trigger_fired']}(lat {s['trigger_latency']}) "
              f"true-label recall={hit:.3f}")
    else:
        print(f"\n(WM-811K not found at {WM811K_PKL} — skipping cross-dataset shift)")
        results["headline"]["wm811k"] = None

    # ---- (c) class-prior defect campaign -----------------------------------
    print("\n== class-prior defect campaign (Edge-Ring ramp) ==")
    n_win = warmup_windows + body_windows
    tgt = LABELS.index("Edge-Ring")
    idx = class_prior_campaign(snap.preds, tgt, n_win, W,
                               onset_window=onset, max_share=0.6, seed=args.seed)
    body = (snap.embeddings[idx], snap.preds[idx], snap.probs[idx], snap.y_true[idx])
    reset_monitors(mons, cal)
    out = run_stream(mons, cal, new_trigger(cfg), embeddings=body[0], preds=body[1],
                     probs=body[2], y_true=body[3], window_size=W,
                     onset_window=onset, calibration_lag=cfg.calibration_label_lag)
    results["headline"]["campaign"] = out
    s = out["summary"]
    print(f"  first={s['first_channel']} latency={s['detection_latency']} "
          f"trigger_fired={s['trigger_fired']}(lat {s['trigger_latency']})")

    out_path = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "experiments" / "shift_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)  "
          f"in {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
