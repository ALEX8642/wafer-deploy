"""build_reference_snapshot.py — freeze the committed reference baseline.

Runs the frozen wafer-mixed model over the FULL test split (~7.6k maps, ~3 min
CPU) and writes reference/reference_snapshot.npz + .meta.json. Run once; the
artifact is committed, so serving + monitoring never need the 400 MB dataset or
the checkpoint at runtime.

    python scripts/build_reference_snapshot.py [--limit N] [--out PATH]

--limit is for a quick smoke build over a prefix of the test split; the
committed artifact uses the whole split (no --limit).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wafer_deploy.config import DeployConfig
from wafer_deploy.predictor import build_predictor
from wafer_deploy.snapshot import build_snapshot, save_snapshot


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the reference snapshot")
    ap.add_argument("--config", type=Path, default=None,
                    help="Path to configs/deploy.yaml (default: resolved config)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only the first N test maps (smoke build)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .npz (default: cfg.reference_snapshot_path)")
    args = ap.parse_args()

    cfg = DeployConfig.load(args.config)
    predictor = build_predictor(cfg)

    # The test split is the reference window: data the model never trained on.
    test_idx = predictor.wm.data.load_splits(predictor.mixed_cfg)["test"]
    if args.limit is not None:
        test_idx = test_idx[:args.limit]

    print(f"Building reference snapshot over {len(test_idx):,} test maps "
          f"(embedding_dim {predictor.embedding_dim})...")
    snap = build_snapshot(predictor, test_idx, batch_size=cfg.batch_size,
                          progress=True)

    out = args.out or cfg.reference_snapshot_path
    save_snapshot(snap, out)
    s = snap.meta["summary"]
    print(f"\nWrote {out}  ({snap.n:,} maps, hash {snap.meta['content_hash'][:12]}…)")
    print(f"  defect_rate={s['defect_rate']:.3f}  "
          f"reference_ece_mean={s['reference_ece_mean']:.5f}")
    print("  prediction_rate:", {k: round(v, 3) for k, v in s['prediction_rate'].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
