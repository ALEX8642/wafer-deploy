"""send_sample.py — POST one wafer map to the running service (quickstart).

Stdlib only, so it works from a fresh clone with no extra installs:

    python scripts/send_sample.py                 # a synthetic center+edge-ring map
    python scripts/send_sample.py --from-mixed 0  # real test-split map #0 (needs sibling)
    python scripts/send_sample.py --url http://localhost:8000

Prints the service's per-label probabilities and multi-hot decision.
"""
from __future__ import annotations

import argparse
import json
import urllib.request


def synthetic_map(size: int = 52) -> list[list[int]]:
    """A center blob + an edge ring on a circular wafer (values 0/1/2).

    Not tuned to any label — just a plausible defect-y map so the quickstart
    produces a non-trivial prediction without needing the dataset.
    """
    c = (size - 1) / 2.0
    r_out = c - 1.0
    wmap = [[0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
            if d > r_out:
                continue                      # outside wafer boundary → 0
            val = 1                           # passing die
            if d < 0.22 * r_out:
                val = 2                       # center blob → failing
            elif 0.85 * r_out < d <= r_out:
                val = 2                       # edge ring → failing
            wmap[y][x] = val
    return wmap


def from_mixed(index: int) -> list[list[int]]:
    """Pull test-split map #index from the sibling wafer-mixed checkout."""
    import sys
    from pathlib import Path
    from wafer_deploy.config import DeployConfig
    cfg = DeployConfig.load()
    sys.path.insert(0, str(cfg.mixed_root / "src"))
    from wafer_mixed import data as wmdata
    maps, _ = wmdata.load_raw(cfg.mixed_root / "data" / "raw")
    d = __import__("numpy").load(cfg.mixed_root / "data" / "splits.npz")
    return maps[int(d["test"][index])].tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description="POST a wafer map to the service")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--from-mixed", type=int, default=None,
                    help="Use real test-split map at this index (needs sibling checkout)")
    args = ap.parse_args()

    wmap = from_mixed(args.from_mixed) if args.from_mixed is not None else synthetic_map()
    payload = json.dumps({"wafer_map": wmap}).encode()
    req = urllib.request.Request(f"{args.url}/predict", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode())

    print(f"predicted_labels : {body['predicted_labels']}  (is_normal={body['is_normal']})")
    print(f"latency_ms       : {body['latency_ms']:.1f}")
    print("probabilities    :")
    for name in body["labels"]:
        print(f"    {name:<10} {body['probabilities'][name]:.3f}  "
              f"→ {body['prediction'][name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
