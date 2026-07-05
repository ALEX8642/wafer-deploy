"""drift.py — unsupervised drift monitors (Phase 1: input + prediction-rate).

In production you rarely get labels immediately, so this repo *leads* with
signals that need no labels at all: has the input distribution moved, and has
the model's own output distribution moved? Both are computed against the frozen
reference snapshot (``snapshot.py``) — the known-good baseline from the
wafer-mixed test split.

Two monitors, both numpy-only (no scipy) so the online path stays in the lean
serving image, and both with **bounded state** (a fixed reference bank + a
rolling window capped at ``window_size``) so the sidecar is co-tenant-safe on
GB10:

    - CovariateDriftMonitor — embedding-space distance vs the reference bank.
      **MMD²** (RBF kernel) is the primary alarm; per-dimension **KS** is the
      interpretability read-out. The MMD² alarm threshold is *calibrated* from
      the reference itself (the null distribution of window-vs-bank MMD²), so
      the false-alarm rate is an explicit design quantile, not a guess.

    - PredictionDriftMonitor — the model's predicted label distribution vs the
      reference histogram via **PSI**, plus the windowed defect-rate. Cheap,
      always-available, and the usual first-alarm signal.

Monitors process **non-overlapping windows**: maps are buffered until a full
window of ``window_size`` accumulates, one result is emitted, and the buffer is
drained. ``update`` accepts a batch (or a single row) and returns a list of
results for every window that completed during the call (usually 0 or 1 on the
one-map-at-a-time serving path).
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Drift statistics (pure functions — the pieces the tests pin directly).
# --------------------------------------------------------------------------- #

def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape (len(a), len(b))."""
    a2 = np.einsum("ij,ij->i", a, a)[:, None]
    b2 = np.einsum("ij,ij->i", b, b)[None, :]
    d = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)  # numerical floor; tiny negatives → 0


def median_heuristic_gamma(x: np.ndarray, max_points: int = 512,
                           seed: int = 0) -> float:
    """RBF ``gamma`` from the median-of-squared-distances heuristic.

    ``gamma = 1 / median(||x_i - x_j||^2)`` — the standard bandwidth that keeps
    the kernel responsive at the reference's own scale. Subsamples to bound the
    O(n^2) distance matrix; falls back to 1.0 if the reference is degenerate.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) > max_points:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), size=max_points, replace=False)]
    d = _sq_dists(x, x)
    iu = np.triu_indices(len(x), k=1)
    med = float(np.median(d[iu])) if iu[0].size else 0.0
    return 1.0 / med if med > 0 else 1.0


def rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Unbiased squared MMD between samples ``x`` and ``y`` (RBF kernel).

    MMD²_u = mean_{i≠j} k(x_i,x_j) + mean_{i≠j} k(y_i,y_j) - 2 mean k(x_i,y_j),
    with the diagonal excluded from the within-sample terms (the unbiased form).
    ~0 when x and y are drawn from the same distribution; grows with separation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        raise ValueError("rbf_mmd2 needs at least 2 points in each sample")
    kxx = np.exp(-gamma * _sq_dists(x, x))
    kyy = np.exp(-gamma * _sq_dists(y, y))
    kxy = np.exp(-gamma * _sq_dists(x, y))
    # exclude diagonal (self-similarity = 1) from the within-sample means
    sum_xx = kxx.sum() - m
    sum_yy = kyy.sum() - n
    term_xx = sum_xx / (m * (m - 1))
    term_yy = sum_yy / (n * (n - 1))
    term_xy = kxy.mean()
    return float(term_xx + term_yy - 2.0 * term_xy)


def ks_per_dim(ref: np.ndarray, win: np.ndarray) -> tuple[float, float]:
    """Per-dimension two-sample KS statistic → (mean, max) across dimensions.

    KS is the max gap between the two empirical CDFs of one embedding
    coordinate; averaging over coordinates gives a broad "how far has the
    marginal geometry moved" read, while the max flags the single most-shifted
    coordinate. numpy-only (sorted-CDF via searchsorted), no scipy.
    """
    ref = np.asarray(ref, dtype=np.float64)
    win = np.asarray(win, dtype=np.float64)
    d = ref.shape[1]
    stats = np.empty(d, dtype=np.float64)
    nref, nwin = len(ref), len(win)
    for j in range(d):
        a = np.sort(ref[:, j])
        b = np.sort(win[:, j])
        grid = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, grid, side="right") / nref
        cdf_b = np.searchsorted(b, grid, side="right") / nwin
        stats[j] = np.max(np.abs(cdf_a - cdf_b))
    return float(stats.mean()), float(stats.max())


def psi(expected: Sequence[float], actual: Sequence[float],
        eps: float = 1e-6) -> float:
    """Population Stability Index between two histograms / distributions.

    PSI = Σ (a_i - e_i) · ln(a_i / e_i) over normalized bins. Inputs may be raw
    counts or proportions — each is normalized to sum 1 internally, then clipped
    at ``eps`` so an empty bin can't blow the log up. Rule-of-thumb reading:
    <0.1 stable, 0.1–0.25 moderate shift, >0.25 significant.
    """
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if e.shape != a.shape:
        raise ValueError(f"psi shape mismatch: {e.shape} vs {a.shape}")
    e_sum, a_sum = e.sum(), a.sum()
    if e_sum <= 0 or a_sum <= 0:
        raise ValueError("psi needs positive total mass in both histograms")
    e = np.clip(e / e_sum, eps, None)
    a = np.clip(a / a_sum, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


# --------------------------------------------------------------------------- #
# Monitor result records.
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class CovariateDriftResult:
    mmd2: float
    ks_mean: float
    ks_max: float
    threshold: float
    alarm: bool
    n: int


@dataclasses.dataclass
class PredictionDriftResult:
    psi: float
    defect_rate: float
    defect_rate_delta: float   # windowed − reference
    threshold: float
    alarm: bool
    per_label_rate: dict       # label → windowed fire-rate
    n: int


# --------------------------------------------------------------------------- #
# Monitors.
# --------------------------------------------------------------------------- #

class _WindowBuffer:
    """Bounded accumulator that yields non-overlapping windows of fixed size.

    State never exceeds one window (a full window is drained the moment it
    completes), which is what keeps the online monitor co-tenant-safe.
    """

    def __init__(self, window_size: int) -> None:
        self.window_size = int(window_size)
        self._rows: list[np.ndarray] = []

    def add(self, rows: np.ndarray) -> list[np.ndarray]:
        """Append rows; return a list of completed (window_size, D) windows."""
        for r in rows:
            self._rows.append(r)
        out = []
        while len(self._rows) >= self.window_size:
            out.append(np.asarray(self._rows[: self.window_size]))
            del self._rows[: self.window_size]
        return out

    @property
    def pending(self) -> int:
        return len(self._rows)


class CovariateDriftMonitor:
    """Embedding-space covariate drift vs a frozen reference bank.

    The reference embeddings are subsampled once to ``max_ref`` rows (the
    bounded comparison bank); the RBF bandwidth is fixed from that bank's median
    heuristic so MMD² values are comparable across windows. The alarm threshold
    is the ``mmd_quantile`` of the *null* MMD² distribution — reference windows
    scored against the bank — so a stationary input stream fires at an expected
    rate of ``1 − mmd_quantile`` by construction (the honest false-alarm knob).
    """

    def __init__(self, ref_embeddings: np.ndarray, *, window_size: int = 200,
                 max_ref: int = 1024, mmd_quantile: float = 0.99,
                 calib_trials: int = 200, gamma: Optional[float] = None,
                 seed: int = 0) -> None:
        ref = np.asarray(ref_embeddings, dtype=np.float64)
        if ref.ndim != 2:
            raise ValueError("ref_embeddings must be 2-D (N, D)")
        self.window_size = int(window_size)
        rng = np.random.default_rng(seed)

        # Bounded reference bank (fixed for the monitor's life).
        if len(ref) > max_ref:
            self.ref_bank = ref[rng.choice(len(ref), size=max_ref, replace=False)]
        else:
            self.ref_bank = ref
        self.embedding_dim = ref.shape[1]
        self.gamma = float(gamma) if gamma is not None else \
            median_heuristic_gamma(self.ref_bank, seed=seed)
        self.threshold = self._calibrate(ref, rng, mmd_quantile, calib_trials)
        self.mmd_quantile = float(mmd_quantile)
        self._buf = _WindowBuffer(window_size)

    def _calibrate(self, ref: np.ndarray, rng: np.random.Generator,
                   quantile: float, trials: int) -> float:
        """Null MMD² distribution: reference windows scored against the bank."""
        n = len(ref)
        if n < self.window_size:
            raise ValueError(
                f"reference has {n} rows < window_size {self.window_size}")
        stats = [
            rbf_mmd2(ref[rng.choice(n, size=self.window_size, replace=False)],
                     self.ref_bank, self.gamma)
            for _ in range(trials)
        ]
        return float(np.quantile(stats, quantile))

    def _score(self, window: np.ndarray) -> CovariateDriftResult:
        mmd2 = rbf_mmd2(window, self.ref_bank, self.gamma)
        ks_mean, ks_max = ks_per_dim(self.ref_bank, window)
        return CovariateDriftResult(
            mmd2=mmd2, ks_mean=ks_mean, ks_max=ks_max,
            threshold=self.threshold, alarm=mmd2 > self.threshold,
            n=len(window))

    def update(self, embeddings: np.ndarray) -> list[CovariateDriftResult]:
        """Feed one or more embedding rows; score each completed window."""
        rows = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        return [self._score(w) for w in self._buf.add(rows)]

    @property
    def pending(self) -> int:
        return self._buf.pending


class PredictionDriftMonitor:
    """Predicted-label distribution + defect-rate drift vs the reference.

    Multi-label maps don't form a single categorical histogram summing to one,
    so PSI here compares the **share of total positive predictions** carried by
    each label (the shape of *which* defects dominate). The windowed defect-rate
    (fraction of maps with ≥1 predicted defect) is tracked alongside as the
    simplest always-available prediction-drift read.
    """

    def __init__(self, ref_preds: np.ndarray, labels: Sequence[str], *,
                 window_size: int = 200, psi_threshold: float = 0.25) -> None:
        ref = np.asarray(ref_preds)
        if ref.ndim != 2 or ref.shape[1] != len(labels):
            raise ValueError("ref_preds must be (N, len(labels))")
        self.labels = list(labels)
        self.window_size = int(window_size)
        self.psi_threshold = float(psi_threshold)
        self.ref_histogram = ref.sum(axis=0).astype(np.float64)  # counts / label
        self.ref_defect_rate = float((ref.sum(axis=1) > 0).mean())
        self._buf = _WindowBuffer(window_size)

    def _score(self, window: np.ndarray) -> PredictionDriftResult:
        win_hist = window.sum(axis=0).astype(np.float64)
        psi_val = psi(self.ref_histogram, win_hist)
        defect_rate = float((window.sum(axis=1) > 0).mean())
        per_label = {self.labels[i]: float(window[:, i].mean())
                     for i in range(len(self.labels))}
        return PredictionDriftResult(
            psi=psi_val, defect_rate=defect_rate,
            defect_rate_delta=defect_rate - self.ref_defect_rate,
            threshold=self.psi_threshold, alarm=psi_val > self.psi_threshold,
            per_label_rate=per_label, n=len(window))

    def update(self, preds: np.ndarray) -> list[PredictionDriftResult]:
        """Feed one or more multi-hot prediction rows; score completed windows."""
        rows = np.atleast_2d(np.asarray(preds))
        return [self._score(w) for w in self._buf.add(rows)]

    @property
    def pending(self) -> int:
        return self._buf.pending


@dataclasses.dataclass
class DriftMonitors:
    """The pair of unsupervised monitors, fed together from the serving path."""
    covariate: CovariateDriftMonitor
    prediction: PredictionDriftMonitor

    @classmethod
    def from_snapshot(cls, snapshot, labels: Sequence[str], *,
                      window_size: int = 200, max_ref: int = 1024,
                      mmd_quantile: float = 0.99, calib_trials: int = 200,
                      psi_threshold: float = 0.25,
                      seed: int = 0) -> "DriftMonitors":
        """Build both monitors from a loaded ReferenceSnapshot."""
        return cls(
            covariate=CovariateDriftMonitor(
                snapshot.embeddings, window_size=window_size, max_ref=max_ref,
                mmd_quantile=mmd_quantile, calib_trials=calib_trials, seed=seed),
            prediction=PredictionDriftMonitor(
                snapshot.preds, labels, window_size=window_size,
                psi_threshold=psi_threshold),
        )

    def observe(self, embeddings: np.ndarray, preds: np.ndarray
                ) -> tuple[list[CovariateDriftResult], list[PredictionDriftResult]]:
        """Feed a batch of predictions to both monitors."""
        return self.covariate.update(embeddings), self.prediction.update(preds)
