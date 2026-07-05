"""The reference snapshot is the baseline every monitor trusts, so two things
must hold: it is deterministic (same split + seed + checkpoint → identical
bytes), and the committed artifact is internally consistent (its content_hash
actually matches its arrays, and a fresh subset build reproduces the matching
rows). Determinism is checked on a small prefix so the test stays fast; the
committed artifact was built over the full test split by
scripts/build_reference_snapshot.py."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mixed

from wafer_deploy.snapshot import (
    _content_hash, build_snapshot, load_snapshot,
)

SUBSET = 32


@needs_mixed
def test_snapshot_build_is_deterministic(predictor, mixed_data):
    _, _, test_idx = mixed_data
    ids = test_idx[:SUBSET]
    a = build_snapshot(predictor, ids, batch_size=16)
    b = build_snapshot(predictor, ids, batch_size=16)
    assert a.meta["content_hash"] == b.meta["content_hash"]
    np.testing.assert_array_equal(a.embeddings, b.embeddings)
    np.testing.assert_array_equal(a.probs, b.probs)
    np.testing.assert_array_equal(a.preds, b.preds)


def test_committed_snapshot_integrity(cfg):
    """The committed artifact's stored hash matches its own arrays."""
    path = cfg.reference_snapshot_path
    if not path.exists():
        pytest.skip("reference snapshot not built yet")
    snap = load_snapshot(path)
    recomputed = _content_hash(
        snap.embeddings.astype(np.float16), snap.probs, snap.preds,
        snap.y_true, snap.map_ids)
    assert recomputed == snap.meta["content_hash"]
    assert snap.n == snap.meta["n"]
    assert snap.embeddings.shape[1] == snap.meta["embedding_dim"]
    # calibrated model → tiny reference ECE (sanity, not a tuned target).
    assert snap.meta["summary"]["reference_ece_mean"] < 0.05


@needs_mixed
def test_subset_build_reproduces_committed_rows(predictor, cfg, mixed_data):
    """A fresh build over the first SUBSET test indices equals the committed
    snapshot's first SUBSET rows — ties the committed artifact to a
    reproducible pipeline without re-running the full 7.6k-map build."""
    path = cfg.reference_snapshot_path
    if not path.exists():
        pytest.skip("reference snapshot not built yet")
    committed = load_snapshot(path)
    _, _, test_idx = mixed_data
    ids = test_idx[:SUBSET]
    assert list(committed.map_ids[:SUBSET]) == list(ids)

    fresh = build_snapshot(predictor, ids, batch_size=16)
    # Compare at the persisted float16 precision for embeddings.
    np.testing.assert_array_equal(fresh.embeddings.astype(np.float16),
                                  committed.embeddings[:SUBSET].astype(np.float16))
    np.testing.assert_allclose(fresh.probs, committed.probs[:SUBSET], atol=1e-6)
    np.testing.assert_array_equal(fresh.preds, committed.preds[:SUBSET])
