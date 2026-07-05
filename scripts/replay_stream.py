"""replay_stream.py — drive a stream of wafer maps through the running service.

The monitors live *inside* the service (they are fed from the /predict path), so
"exercising" them means POSTing a sequence of maps and watching the windowed
drift gauges update on /metrics (and the Grafana panels). This script is the
Phase 1 streaming harness:

    # no-drift baseline: replay the wafer-mixed test split in its own order
    python scripts/replay_stream.py --n 800

    # a crude synthetic covariate shift (demo only — Phase 3 owns the scored
    # corruption sweep); rotates + injects failing-die noise to move the input
    python scripts/replay_stream.py --n 800 --shift 0.4

Needs the sibling wafer-mixed checkout for the raw maps (same dependency as the
reference-snapshot build). Prints each completed-window's drift readout as it
lands, pulled straight from /metrics, plus a final false-alarm / alarm summary.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

import numpy as np

from wafer_deploy.bridge import wafer_mixed_modules
from wafer_deploy.config import DeployConfig


def load_test_maps(cfg: DeployConfig, n: int, seed: int) -> np.ndarray:
    """First ``n`` wafer-mixed test-split maps (shuffled deterministically)."""
    wm = wafer_mixed_modules(cfg.mixed_root)
    mixed_cfg = wm.MixedConfig(device="cpu")
    maps, _ = wm.data.load_raw(mixed_cfg.data_root)
    test_idx = wm.data.load_splits(mixed_cfg)["test"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(test_idx))[:n]
    return maps[test_idx[order]]


def corrupt(wmap: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """A deliberately crude covariate shift for the harness demo.

    Rotates the map 90° and flips a fraction of *passing* die to *failing*
    inside the wafer — enough to move the embedding distribution. This is NOT
    the Phase 3 scored corruption sweep; it exists so the harness can visibly
    push the covariate monitor into alarm without any dataset labels.
    """
    out = np.rot90(wmap).copy()
    on_wafer = out > 0
    passing = on_wafer & (out == 1)
    flip = passing & (rng.random(out.shape) < intensity)
    out[flip] = 2
    return out


def get_metrics(url: str) -> dict[str, float]:
    """Scrape the drift gauges/counters we care about from /metrics."""
    from prometheus_client.parser import text_string_to_metric_families
    with urllib.request.urlopen(f"{url}/metrics") as resp:
        text = resp.read().decode()
    wanted = {
        "wafer_deploy_covariate_mmd2", "wafer_deploy_covariate_mmd2_threshold",
        "wafer_deploy_covariate_ks_max", "wafer_deploy_prediction_psi",
        "wafer_deploy_prediction_defect_rate",
    }
    out: dict[str, float] = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name in wanted:
                out[s.name] = s.value
            elif s.name == "wafer_deploy_drift_windows_total":
                out[f"windows_{s.labels.get('monitor')}"] = s.value
            elif s.name == "wafer_deploy_drift_alarms_total":
                out[f"alarms_{s.labels.get('monitor')}"] = s.value
    return out


def post_map(url: str, wmap: np.ndarray) -> None:
    payload = json.dumps({"wafer_map": wmap.tolist()}).encode()
    req = urllib.request.Request(f"{url}/predict", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a wafer-map stream through the service")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=800, help="maps to stream")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="synthetic covariate-shift intensity in [0,1] (demo only)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = DeployConfig.load()
    maps = load_test_maps(cfg, args.n, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    window = cfg.drift_window_size
    label = f"shift={args.shift:.2f}" if args.shift > 0 else "no-drift"
    print(f"streaming {len(maps)} maps ({label}), window={window} → {args.url}")

    last_windows = 0.0
    for i, wmap in enumerate(maps, 1):
        m = corrupt(wmap, args.shift, rng) if args.shift > 0 else wmap
        post_map(args.url, m)
        if i % window == 0:
            g = get_metrics(args.url)
            cw = g.get("windows_covariate", 0.0)
            if cw > last_windows:  # a fresh covariate window landed
                last_windows = cw
                print(f"  window @ {i:>5}  MMD²={g.get('wafer_deploy_covariate_mmd2', 0):.4f} "
                      f"(thr={g.get('wafer_deploy_covariate_mmd2_threshold', 0):.4f})  "
                      f"KSmax={g.get('wafer_deploy_covariate_ks_max', 0):.3f}  "
                      f"PSI={g.get('wafer_deploy_prediction_psi', 0):.4f}  "
                      f"defect_rate={g.get('wafer_deploy_prediction_defect_rate', 0):.3f}")

    g = get_metrics(args.url)
    for mon in ("covariate", "prediction"):
        w = g.get(f"windows_{mon}", 0.0)
        a = g.get(f"alarms_{mon}", 0.0)
        rate = a / w if w else 0.0
        tag = "false-alarm rate" if args.shift == 0 else "alarm rate"
        print(f"{mon:>10}: {int(a)}/{int(w)} windows alarmed  ({tag} {rate:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
