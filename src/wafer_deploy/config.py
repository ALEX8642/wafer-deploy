"""config.py — DeployConfig: where the reused artifacts live and how to serve.

The model, thresholds and calibration are NOT in this repo; they are read from a
sibling wafer-mixed checkout (default ``../wafer-mixed``). Paths resolve in this
order of precedence, highest first:

    1. environment variables (WAFER_DEPLOY_*), so a container can be pointed at a
       bind-mounted artifact dir without editing files;
    2. an explicit YAML config (configs/deploy.yaml);
    3. the dataclass defaults below.

Everything is anchored to the repo root so the service behaves the same no
matter the working directory it is launched from.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Optional

import yaml

# Invariant: repo root regardless of working directory or symlinks.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _anchor(p: Path, base: Path = REPO_ROOT) -> Path:
    """Make a relative path absolute against ``base`` (default the repo root)."""
    return p if p.is_absolute() else base / p


@dataclasses.dataclass
class DeployConfig:
    # --- reused wafer-mixed artifacts (read-only) ---
    wafer_mixed_root: str = "../wafer-mixed"      # sibling checkout, rel to repo root
    checkpoint: str = "outputs/best.pt"           # rel to wafer_mixed_root
    thresholds: str = "outputs/thresholds.json"   # rel to wafer_mixed_root
    calibration: str = "outputs/calibration.json" # rel to wafer_mixed_root

    # --- repo-local artifacts ---
    reference_path: str = "reference/reference_snapshot.npz"  # rel to repo root
    output_dir: str = "outputs"                               # rel to repo root

    # --- inference ---
    device: str = "cpu"            # this repo is CPU-only by policy (GB10 deploy in Phase 4)
    batch_size: int = 64
    num_workers: int = 0           # single-request serving; 0 avoids worker spin-up cost

    # --- serving ---
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int = 42

    # --- drift monitors (Phase 1) ---
    # Non-overlapping window of maps per drift evaluation.
    drift_window_size: int = 200
    # Bounded reference bank size for the MMD/KS comparison (co-tenant memory cap).
    drift_max_ref: int = 1024
    # MMD² alarm threshold = this quantile of the reference null → expected
    # false-alarm rate under no drift is (1 − quantile).
    drift_mmd_quantile: float = 0.99
    # Null-distribution trials used to calibrate the MMD² threshold at startup.
    drift_calib_trials: int = 200
    # PSI alarm threshold on the predicted-label distribution (conventional 0.25).
    drift_psi_threshold: float = 0.25

    # ---- resolved absolute paths -------------------------------------------

    @property
    def mixed_root(self) -> Path:
        return _anchor(Path(self.wafer_mixed_root))

    @property
    def checkpoint_path(self) -> Path:
        return _anchor(Path(self.checkpoint), self.mixed_root)

    @property
    def thresholds_path(self) -> Path:
        return _anchor(Path(self.thresholds), self.mixed_root)

    @property
    def calibration_path(self) -> Path:
        return _anchor(Path(self.calibration), self.mixed_root)

    @property
    def reference_snapshot_path(self) -> Path:
        return _anchor(Path(self.reference_path))

    @property
    def output_path(self) -> Path:
        return _anchor(Path(self.output_dir))

    # ---- constructors -------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path]) -> "DeployConfig":
        raw: dict = {}
        if yaml_path is not None and Path(yaml_path).exists():
            with open(yaml_path) as f:
                raw = yaml.safe_load(f) or {}
        return cls(**raw)

    @classmethod
    def load(cls, yaml_path: Optional[Path] = None) -> "DeployConfig":
        """YAML (or defaults) overlaid with WAFER_DEPLOY_* environment variables.

        Env wins so a Docker image built once can be repointed at a mounted
        artifact directory purely through compose ``environment:`` entries.
        """
        if yaml_path is None:
            env_cfg = os.environ.get("WAFER_DEPLOY_CONFIG")
            yaml_path = Path(env_cfg) if env_cfg else REPO_ROOT / "configs" / "deploy.yaml"
        cfg = cls.from_yaml(yaml_path)
        env_map = {
            "wafer_mixed_root": "WAFER_MIXED_ROOT",
            "checkpoint": "WAFER_DEPLOY_CHECKPOINT",
            "thresholds": "WAFER_DEPLOY_THRESHOLDS",
            "calibration": "WAFER_DEPLOY_CALIBRATION",
            "reference_path": "WAFER_DEPLOY_REFERENCE",
            "device": "WAFER_DEPLOY_DEVICE",
        }
        for field_name, env_name in env_map.items():
            val = os.environ.get(env_name)
            if val:
                setattr(cfg, field_name, val)
        return cfg
