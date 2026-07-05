"""The correctness anchor: a served /predict must return the SAME multi-hot
(and matching calibrated probabilities) as wafer-mixed's own evaluate path —
collect_logits over a DataLoader → scale_probs → predict_multihot. If this
holds, the service is definitionally faithful to the model it wraps; every
later phase builds on that guarantee.

The two paths are genuinely independent: the service encodes one map at a time
through the HTTP layer, while the reference computes a batched forward pass with
wafer-mixed's DataLoader. Equality confirms neither the HTTP path nor the
one-at-a-time batching perturbs the result (BatchNorm runs on frozen running
stats in eval mode, so batch composition is irrelevant)."""
from __future__ import annotations

import numpy as np
from conftest import needs_mixed

from wafer_deploy.bridge import load_thresholds
from wafer_deploy.labels import LABELS

N_MAPS = 24  # a spread of single / mixed / normal maps from the test split


@needs_mixed
def test_predict_matches_wafer_mixed(predictor, mixed_data, client):
    maps, labels, test_idx = mixed_data
    # Deterministic spread across the split (not just the first N).
    sel = test_idx[np.linspace(0, len(test_idx) - 1, N_MAPS).astype(int)]

    # --- reference path: wafer-mixed's own batched evaluate pipeline ----------
    import torch
    from torch.utils.data import DataLoader
    wm = predictor.wm
    ds = wm.data.MixedWaferDataset(maps[sel], labels[sel],
                                   predictor.input_size, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    _, ref_logits = wm.evaluate.collect_logits(predictor.model, loader, "cpu",
                                               desc="parity", leave=False)
    T, tau = load_thresholds(predictor.cfg.thresholds_path)
    ref_probs = wm.calibrate.scale_probs(ref_logits, T)
    ref_pred = wm.metrics.predict_multihot(ref_probs, tau)

    # --- served path: one HTTP call per map -----------------------------------
    for row, idx in enumerate(sel):
        resp = client.post("/predict", json={"wafer_map": maps[idx].tolist()})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        served_pred = np.array([body["prediction"][n] for n in LABELS])
        served_prob = np.array([body["probabilities"][n] for n in LABELS])

        # Multi-hot must match exactly — the fab-facing decision.
        np.testing.assert_array_equal(
            served_pred, ref_pred[row],
            err_msg=f"multi-hot mismatch on test map {idx}")
        # Calibrated probabilities match to float tolerance.
        np.testing.assert_allclose(
            served_prob, ref_probs[row], atol=1e-5,
            err_msg=f"probability mismatch on test map {idx}")


@needs_mixed
def test_predicted_labels_field_agrees_with_multihot(client, mixed_data):
    maps, _, test_idx = mixed_data
    body = client.post("/predict",
                       json={"wafer_map": maps[int(test_idx[0])].tolist()}).json()
    active = {n for n in LABELS if body["prediction"][n] == 1}
    assert set(body["predicted_labels"]) == active
    assert body["is_normal"] == (len(active) == 0)
