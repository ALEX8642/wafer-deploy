"""Shared fixtures. Loading the checkpoint is a few seconds, so the predictor
and the API client are session-scoped. Tests that need the sibling wafer-mixed
checkout skip cleanly when its artifacts are absent, so the suite stays honest
on a machine that only has this repo."""
from __future__ import annotations

import numpy as np
import pytest

from wafer_deploy.config import DeployConfig

_cfg = DeployConfig.load()
_have_artifacts = _cfg.checkpoint_path.exists() and _cfg.thresholds_path.exists()
needs_mixed = pytest.mark.skipif(
    not _have_artifacts,
    reason=f"wafer-mixed artifacts not found under {_cfg.mixed_root}")


@pytest.fixture(scope="session")
def cfg() -> DeployConfig:
    return _cfg


@pytest.fixture(scope="session")
def predictor(cfg):
    from wafer_deploy.predictor import Predictor
    return Predictor(cfg)


@pytest.fixture(scope="session")
def mixed_data(predictor):
    """(maps, labels, test_idx) from the sibling checkout."""
    maps, labels = predictor.wm.data.load_raw(predictor.mixed_cfg.data_root)
    test_idx = predictor.wm.data.load_splits(predictor.mixed_cfg)["test"]
    return maps, labels, test_idx


@pytest.fixture(scope="session")
def client(cfg):
    from starlette.testclient import TestClient
    from serve.app import create_app
    with TestClient(create_app(cfg)) as c:
        yield c
