"""bridge.py — the single owner of the cross-repo dependency on wafer-mixed.

The service does not re-implement the architecture, one-hot encoding, temperature
scaling or threshold rule: it imports wafer-mixed's OWN modules from a sibling
checkout, so what the endpoint returns is *definitionally* what wafer-mixed
produces. Everything downstream goes through the namespace returned here.

Two guards keep the bridge honest (both ported from wafer-rootcause, where this
pattern was hardened): the imported package must actually live under
``mixed_root`` — ``sys.modules`` pins the first import for the whole process, so
an installed ``wafer_mixed`` could otherwise shadow the checkout — and its label
ordering must equal this repo's LABELS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wafer_deploy.labels import LABELS


def wafer_mixed_modules(mixed_root: Path) -> SimpleNamespace:
    """Import wafer-mixed's package from a sibling checkout, guarded."""
    src = Path(mixed_root) / "src"
    if not src.is_dir():
        raise FileNotFoundError(
            f"{src} not found — set wafer_mixed_root (configs/deploy.yaml or the "
            "WAFER_MIXED_ROOT env var) to a wafer-mixed checkout")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from wafer_mixed import calibrate, data, evaluate, metrics, model
    from wafer_mixed.config import MixedConfig
    loaded_from = Path(data.__file__).resolve()
    if not loaded_from.is_relative_to(src.resolve()):
        raise RuntimeError(
            f"wafer_mixed resolved to {loaded_from}, not the configured checkout "
            f"{src} — another import (installed package or earlier config) got "
            "there first")
    if list(data.LABEL_NAMES) != LABELS:
        raise RuntimeError(
            f"label-order drift: wafer_mixed.data.LABEL_NAMES "
            f"{list(data.LABEL_NAMES)} != wafer_deploy LABELS {LABELS}")
    return SimpleNamespace(calibrate=calibrate, data=data, evaluate=evaluate,
                           metrics=metrics, model=model, MixedConfig=MixedConfig)


def load_thresholds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(temperatures, taus) in LABELS order from wafer-mixed thresholds.json.

    The file is self-contained on purpose — wafer-mixed embeds the per-label
    temperatures the taus were tuned on. Refuse label-set drift loudly.
    """
    raw = json.loads(Path(path).read_text())
    temps, taus = raw["_temperatures"], raw["thresholds"]
    for name, d in (("_temperatures", temps), ("thresholds", taus)):
        if set(d) != set(LABELS):
            raise ValueError(f"{path}: {name} labels {sorted(d)} != {sorted(LABELS)}")
    return (np.array([temps[n] for n in LABELS], dtype=np.float64),
            np.array([taus[n] for n in LABELS], dtype=np.float64))
