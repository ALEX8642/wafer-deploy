"""replay_labeled_stream.py — drive the delayed-label calibration monitor.

Phase 1's harness (``replay_stream.py``) only needed maps: its monitors are
label-free. The calibration monitor needs ground truth, and in production ground
truth arrives *late* — so this harness streams maps through ``/predict`` and then
POSTs their labels to ``/feedback`` **held back by ``--lag`` windows**, exactly
the delayed-label regime the monitor is built for:

    # in-domain null — labels lag 2 windows behind inference
    python scripts/replay_labeled_stream.py --n 1600 --lag 2

    # covariate shift (demo corruption, as in replay_stream) → watch calibration
    # decay once the (still-original) labels for the shifted windows land
    python scripts/replay_labeled_stream.py --n 1600 --lag 2 --shift 0.4

Labels are the wafer-mixed test-split ground truth (multi-hot). The service scores
a calibration window the moment its delayed labels arrive; this script prints each
scored window's ECE vs the reference/threshold, and a final false-alarm / alarm
summary pulled off /metrics. Needs the sibling wafer-mixed checkout (same
dependency as the reference-snapshot build).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Make the repo root importable so `scripts` resolves when this file is run
# directly (python scripts/replay_labeled_stream.py) — sys.path[0] is scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wafer_deploy.bridge import wafer_mixed_modules  # noqa: E402
from wafer_deploy.config import DeployConfig  # noqa: E402
from scripts.replay_stream import corrupt  # noqa: E402 — reuse the demo corruption


def load_test(cfg: DeployConfig, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """First ``n`` wafer-mixed test maps + their multi-hot labels (shuffled)."""
    wm = wafer_mixed_modules(cfg.mixed_root)
    mixed_cfg = wm.MixedConfig(device="cpu")
    maps, labels = wm.data.load_raw(mixed_cfg.data_root)
    test_idx = wm.data.load_splits(mixed_cfg)["test"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(test_idx))[:n]
    idx = test_idx[order]
    return maps[idx], labels[idx].astype(int)


def post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(f"{url}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def get_metrics(url: str) -> dict[str, float]:
    from prometheus_client.parser import text_string_to_metric_families
    with urllib.request.urlopen(f"{url}/metrics") as resp:
        text = resp.read().decode()
    out: dict[str, float] = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name in ("wafer_deploy_calibration_ece",
                          "wafer_deploy_calibration_ece_threshold",
                          "wafer_deploy_calibration_reference_ece"):
                out[s.name] = s.value
            elif s.name == "wafer_deploy_drift_windows_total" \
                    and s.labels.get("monitor") == "calibration":
                out["windows"] = s.value
            elif s.name == "wafer_deploy_drift_alarms_total" \
                    and s.labels.get("monitor") == "calibration":
                out["alarms"] = s.value
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Delayed-label calibration harness")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=1600, help="maps to stream")
    ap.add_argument("--lag", type=int, default=2, help="windows to delay labels")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="synthetic covariate-shift intensity in [0,1] (demo only)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = DeployConfig.load()
    maps, labels = load_test(cfg, args.n, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    window = cfg.drift_window_size
    tag = f"shift={args.shift:.2f}" if args.shift > 0 else "no-drift"
    print(f"streaming {len(maps)} maps ({tag}), window={window}, "
          f"label lag={args.lag} windows → {args.url}")

    served = 0
    released_windows = 0
    label_queue: list[np.ndarray] = []
    for wmap, y in zip(maps, labels):
        m = corrupt(wmap, args.shift, rng) if args.shift > 0 else wmap
        post(args.url, "/predict", {"wafer_map": m.tolist()})
        served += 1
        label_queue.append(y)
        # Release a full window of labels once `lag` further windows have been
        # served past it — the delayed-label lag, realized on the wire.
        while (served >= (released_windows + 1 + args.lag) * window
               and len(label_queue) >= window):
            batch = [label_queue.pop(0).tolist() for _ in range(window)]
            resp = post(args.url, "/feedback", {"labels": batch})
            released_windows += 1
            latest = resp.get("latest")
            if latest:
                flag = "  ⚠ ALARM" if latest["alarm"] else ""
                print(f"  window {latest['window_id']:>2}  "
                      f"ECE={latest['ece_mean']:.4f}  "
                      f"Δref={latest['ece_delta']:+.4f}  "
                      f"(thr={latest['threshold']:.4f}){flag}")

    g = get_metrics(args.url)
    w, a = g.get("windows", 0.0), g.get("alarms", 0.0)
    rate = a / w if w else 0.0
    label = "false-alarm rate" if args.shift == 0 else "alarm rate"
    print(f"calibration: {int(a)}/{int(w)} scored windows alarmed  ({label} {rate:.2%})")
    print(f"  reference ECE={g.get('wafer_deploy_calibration_reference_ece', 0):.4f}  "
          f"threshold={g.get('wafer_deploy_calibration_ece_threshold', 0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
