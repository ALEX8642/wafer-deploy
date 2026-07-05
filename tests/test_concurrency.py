"""Concurrency regression: the predictor must be thread-safe under the uvicorn
threadpool. The penultimate-feature hook writes shared model state (`_captured`),
so before the inference lock two concurrent /predict calls raced on the buffer —
raising `embedding hook did not capture features` (a 500) or, worse, silently
returning one request's embedding for another's map (corrupting the drift
monitor). A concurrency-8 load test caught this in Phase 4.

This test pins the fix: predicting the same maps concurrently must return exactly
the single-threaded result — preds AND embeddings — with no failures."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from conftest import needs_mixed


@needs_mixed
def test_concurrent_predict_matches_serial(predictor, mixed_data):
    maps, _, test_idx = mixed_data
    # A spread of distinct maps so a swapped embedding between two in-flight
    # requests would differ from the serial reference.
    sel = test_idx[np.linspace(0, len(test_idx) - 1, 16).astype(int)]
    ref = {int(i): predictor.predict_one(maps[int(i)]) for i in sel}

    def one(idx: int):
        r = predictor.predict_one(maps[idx])
        return idx, r

    # 200 tasks over the 16 maps, 8 workers — enough overlap to trip the race.
    tasks = [int(sel[k % len(sel)]) for k in range(200)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, tasks))

    for idx, r in results:
        exp = ref[idx]
        np.testing.assert_array_equal(r.preds, exp.preds,
                                      err_msg=f"multi-hot differs under load, map {idx}")
        # The embedding must belong to *this* map — the silent-swap guard.
        np.testing.assert_array_equal(r.embeddings, exp.embeddings,
                                      err_msg=f"embedding swapped under load, map {idx}")
