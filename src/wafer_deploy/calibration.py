"""calibration.py — the delayed-label calibration-decay monitor (Phase 2).

Phase 1 leads with label-free signals (has the input moved, has the output
distribution moved). This module adds the signal you can only get *with* labels,
and in production labels arrive **late**: does the model's confidence still mean
what it meant? A calibrated model's 0.9 should be right ~90 % of the time; when
the input drifts, that contract erodes — sometimes *before* the hard decision
(and thus accuracy) visibly moves. This monitor tracks that erosion.

Two design choices carried over from Phase 1's ``drift.py`` on purpose:

    - **numpy-only** (``binary_ece`` reimplemented, not imported from wafer-mixed)
      so the monitor and its tests run off the committed snapshot with no
      checkpoint and no scipy — the online path stays in the lean serving image.
      ``binary_ece`` here is byte-for-byte wafer-mixed's binning (n_bins=15, same
      bin convention); ``test_calibration`` pins that equality when the sibling
      checkout is present, so the reference ECE in the snapshot and the windowed
      ECE here are the *same measurement*.
    - **the alarm threshold is calibrated from the reference itself** — the
      ``ece_quantile`` of the null distribution of *windowed* mean-ECE (random
      reference windows scored against their own labels). A 200-map window's ECE
      is noisier and positively biased vs the 7,603-map reference point, and rare
      labels (Near-full ~0.4 %) are noisier still; calibrating the threshold on
      like-sized windows folds all of that into an explicit false-alarm design
      quantile instead of a hand-picked constant.

Delayed labels: predictions are ``buffer_predictions``-ed as they are served;
ground truth arrives later via ``add_labels`` and is matched **FIFO** to the
oldest un-labelled predictions. A non-overlapping window is scored the moment its
last label lands — so a window is evaluated only once its (lagged) labels are in,
which is exactly the delayed-label regime. The await buffer is bounded by the
label lag (L windows × 8 floats/map — negligible); ``pending_labels`` exposes it.
"""
from __future__ import annotations

import collections
import dataclasses
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Calibration statistics (pure functions — the pieces the tests pin directly).
# --------------------------------------------------------------------------- #

def binary_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Binary Expected Calibration Error — identical binning to wafer-mixed.

    Bin by predicted probability; within each bin take |mean predicted prob −
    observed positive rate|, weighted by bin occupancy. Lower is better; a
    perfectly calibrated column is 0. Reimplemented here (not imported) so the
    monitor stays numpy-only and checkpoint-free; ``test_calibration`` pins it
    equal to ``wafer_mixed.calibrate.binary_ece`` when the checkout is present.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p > lo) & (p <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def per_label_ece(probs: np.ndarray, y_true: np.ndarray,
                  n_bins: int = 15) -> np.ndarray:
    """``binary_ece`` of each label column → shape (n_labels,)."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return np.array([binary_ece(probs[:, j], y_true[:, j], n_bins)
                     for j in range(probs.shape[1])])


def reliability_bins(p: np.ndarray, y: np.ndarray,
                     n_bins: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled reliability curve → per-bin (mean_conf, pos_rate, count).

    ``p`` and ``y`` are flattened (all labels pooled) — a single aggregate
    reliability read for the reference-vs-drifted diagram. Empty bins are NaN so
    a plotted line breaks there instead of inventing geometry across the
    saturated middle of a sigmoid output. Same bin convention as ``binary_ece``.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p > lo) & (p <= hi)
        c = int(mask.sum())
        cnt[b] = c
        if c:
            conf[b] = float(p[mask].mean())
            acc[b] = float(y[mask].mean())
    return conf, acc, cnt


# --------------------------------------------------------------------------- #
# Monitor result record.
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class CalibrationDriftResult:
    window_id: int
    ece_mean: float                 # mean per-label ECE this window
    ece_per_label: dict             # label → windowed ECE
    reference_ece_mean: float       # the frozen reference baseline
    ece_delta: float                # windowed − reference (the decay)
    threshold: float                # calibrated null quantile
    alarm: bool
    n: int


# --------------------------------------------------------------------------- #
# Monitor.
# --------------------------------------------------------------------------- #

class CalibrationMonitor:
    """Windowed ECE vs a frozen reference calibration, on delayed labels.

    Reference probabilities + labels (the snapshot) fix two things at
    construction: the reference per-label ECE (the baseline every window is
    compared against) and the calibrated alarm threshold (the ``ece_quantile``
    of windowed mean-ECE drawn from the reference itself). Serving buffers each
    prediction's probabilities; ``add_labels`` supplies delayed ground truth,
    FIFO-matched, and scores a non-overlapping window the moment its labels land.
    """

    def __init__(self, ref_probs: np.ndarray, ref_y_true: np.ndarray,
                 labels: Sequence[str], *, window_size: int = 200,
                 n_bins: int = 15, ece_quantile: float = 0.99,
                 calib_trials: int = 200, max_pending: int = 4000,
                 seed: int = 0) -> None:
        ref_probs = np.asarray(ref_probs, dtype=np.float64)
        ref_y_true = np.asarray(ref_y_true, dtype=np.float64)
        if ref_probs.ndim != 2 or ref_probs.shape != ref_y_true.shape:
            raise ValueError("ref_probs and ref_y_true must be matching 2-D (N, L)")
        if ref_probs.shape[1] != len(labels):
            raise ValueError("ref_probs column count must equal len(labels)")
        if max_pending < window_size:
            raise ValueError(
                f"max_pending {max_pending} < window_size {window_size}: a window "
                "could never accumulate its labels")
        self.labels = list(labels)
        self.window_size = int(window_size)
        self.n_bins = int(n_bins)
        self.ece_quantile = float(ece_quantile)
        self.max_pending = int(max_pending)

        self.reference_ece = per_label_ece(ref_probs, ref_y_true, self.n_bins)
        self.reference_ece_mean = float(self.reference_ece.mean())
        self.threshold = self._calibrate(ref_probs, ref_y_true, calib_trials, seed)

        # Delayed-label bookkeeping: served probs awaiting their labels (FIFO),
        # and the currently-forming scoring window of matched (prob, label) rows.
        self._await_probs: collections.deque = collections.deque()
        self._win_probs: list[np.ndarray] = []
        self._win_labels: list[np.ndarray] = []
        self._window_id = 0
        # Predictions evicted past max_pending; their labels are skipped on
        # arrival so FIFO alignment survives the bound. Cumulative, for /metrics.
        self._skip = 0            # labels still owed to already-dropped predictions
        self.dropped_total = 0

    def _calibrate(self, probs: np.ndarray, y_true: np.ndarray,
                   trials: int, seed: int) -> float:
        """Null windowed mean-ECE distribution → the ``ece_quantile`` alarm bar."""
        n = len(probs)
        if n < self.window_size:
            raise ValueError(
                f"reference has {n} rows < window_size {self.window_size}")
        rng = np.random.default_rng(seed)
        stats = [
            float(per_label_ece(probs[idx], y_true[idx], self.n_bins).mean())
            for idx in (rng.choice(n, size=self.window_size, replace=False)
                        for _ in range(trials))
        ]
        return float(np.quantile(stats, self.ece_quantile))

    def _score(self) -> CalibrationDriftResult:
        result = self.score_window(self._win_probs, self._win_labels,
                                   window_id=self._window_id)
        self._window_id += 1
        return result

    def score_window(self, probs: np.ndarray, y_true: np.ndarray,
                     window_id: int = -1) -> CalibrationDriftResult:
        """Score an explicit (probs, y_true) window against the reference.

        The delayed-label path (``buffer_predictions`` / ``add_labels``) is the
        online contract, where a window's labels dribble in FIFO. Offline — the
        Phase 3 scored sweep, where every window's labels are already in hand —
        this scores a fully-assembled window directly, using the *same* frozen
        reference ECE and calibrated threshold, so an offline ECE alarm and an
        online one are the identical measurement.
        """
        P = np.asarray(probs, dtype=np.float64)
        Y = np.asarray(y_true, dtype=np.float64)
        ece_pl = per_label_ece(P, Y, self.n_bins)
        ece_mean = float(ece_pl.mean())
        return CalibrationDriftResult(
            window_id=window_id,
            ece_mean=ece_mean,
            ece_per_label={self.labels[i]: float(ece_pl[i])
                           for i in range(len(self.labels))},
            reference_ece_mean=self.reference_ece_mean,
            ece_delta=ece_mean - self.reference_ece_mean,
            threshold=self.threshold,
            alarm=ece_mean > self.threshold,
            n=len(P))

    # ---- public API --------------------------------------------------------

    def buffer_predictions(self, probs: np.ndarray) -> int:
        """Record served probabilities in order, awaiting their delayed labels.

        Bounds the buffer at ``max_pending``: once full, the oldest awaiting
        prediction is evicted (its label is presumed lost to lag). Returns the
        number evicted this call so the caller can surface it.
        """
        dropped = 0
        for row in np.atleast_2d(np.asarray(probs, dtype=np.float64)):
            self._await_probs.append(row)
            if len(self._await_probs) > self.max_pending:
                self._await_probs.popleft()   # evict oldest, keep the cap
                self._skip += 1               # its label, on arrival, is skipped
                dropped += 1
        self.dropped_total += dropped
        return dropped

    def add_labels(self, labels: np.ndarray) -> list[CalibrationDriftResult]:
        """Supply delayed ground truth (FIFO) for the oldest served predictions.

        Each label row is matched to the next un-labelled buffered prediction; a
        window is scored (and drained) the moment ``window_size`` matched pairs
        accumulate. Labels owed to predictions already evicted by the retention
        cap are skipped (keeping FIFO alignment). Returns a result per window
        completed during the call.
        """
        results: list[CalibrationDriftResult] = []
        for y in np.atleast_2d(np.asarray(labels)):
            if self._skip > 0:           # this label's prediction was dropped
                self._skip -= 1
                continue
            if not self._await_probs:
                raise ValueError(
                    "received a label with no matching served prediction "
                    "(feedback arrived before /predict, or out of FIFO order)")
            self._win_probs.append(self._await_probs.popleft())
            self._win_labels.append(np.asarray(y, dtype=np.float64))
            if len(self._win_labels) == self.window_size:
                results.append(self._score())
                self._win_probs, self._win_labels = [], []
        return results

    @property
    def pending_labels(self) -> int:
        """Served predictions still awaiting a label (bounded by the lag)."""
        return len(self._await_probs)

    @property
    def pending_window(self) -> int:
        """Labelled rows accumulated toward the next scoring window."""
        return len(self._win_labels)

    @classmethod
    def from_snapshot(cls, snapshot, labels: Sequence[str], *,
                      window_size: int = 200, n_bins: int = 15,
                      ece_quantile: float = 0.99, calib_trials: int = 200,
                      max_pending: int = 4000, seed: int = 0) -> "CalibrationMonitor":
        """Build from a loaded ReferenceSnapshot (its probs + y_true)."""
        return cls(snapshot.probs, snapshot.y_true, labels,
                   window_size=window_size, n_bins=n_bins,
                   ece_quantile=ece_quantile, calib_trials=calib_trials,
                   max_pending=max_pending, seed=seed)
