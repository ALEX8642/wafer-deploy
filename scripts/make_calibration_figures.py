"""make_calibration_figures.py — the Phase 2 calibration figures for assets/.

Both figures are reproducible from the **committed reference snapshot alone** (no
checkpoint, no dataset) — the same self-contained guarantee the drift tests hold
to. The "drift" here is a controlled illustration: a monotone **confidence-erosion
warp** applied in probability space that pushes calibrated probabilities toward
0.5 (the model "hedges") with rising intensity. It isolates the *calibration*
channel — the fully coupled map-space shift that also moves embeddings and the
hard decisions is the Phase 3 job.

    python scripts/make_calibration_figures.py

    assets/reliability_reference_vs_drifted.png   reference vs an eroded window
    assets/calibration_ece_over_time.png          ECE per window as erosion ramps,
                                                   with the accuracy line flat —
                                                   "confidence decayed before the
                                                   decision did".

Because the warp is monotone and we hold the committed hard decisions (preds)
fixed, macro-F1 is constant by construction: the figure's point is precisely that
ECE (what the calibration monitor watches) moves while accuracy does not, so a
label-free accuracy check would stay silent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wafer_deploy.calibration import (  # noqa: E402
    CalibrationMonitor, per_label_ece, reliability_bins)
from wafer_deploy.config import DeployConfig  # noqa: E402
from wafer_deploy.labels import LABELS  # noqa: E402
from wafer_deploy.snapshot import load_snapshot  # noqa: E402

# House palette (shared with the sibling wafer-mixed figures).
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
C_REF, C_DRIFT, C_ALARM = "#2a78d6", "#eb6834", "#c0392b"


def erode(p: np.ndarray, gamma: float) -> np.ndarray:
    """Monotone confidence erosion toward 0.5. gamma=1 is identity; gamma<1
    flattens (the model becomes underconfident / hedges). Rankings are
    preserved, so the *decision* at any fixed threshold is essentially unchanged
    — only the meaning of the confidence degrades."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p ** gamma / (p ** gamma + (1 - p) ** gamma)


def macro_f1(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Label-averaged F1 (numpy-only) — the accuracy line held fixed here."""
    f1s = []
    for j in range(preds.shape[1]):
        tp = int((preds[:, j] & y_true[:, j]).sum())
        fp = int((preds[:, j] & (1 - y_true[:, j])).sum())
        fn = int(((1 - preds[:, j]) & y_true[:, j]).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 1.0)
    return float(np.mean(f1s))


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)


def _label_curve(p: np.ndarray, y: np.ndarray, n_bins: int,
                 min_count: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Per-label reliability curve with sparse bins broken (NaN), so the line
    doesn't invent geometry across the empty middle of a saturated label —
    the same convention as wafer-mixed's reliability grid."""
    conf, acc, cnt = reliability_bins(p, y, n_bins)
    keep = cnt >= min_count
    return np.where(keep, conf, np.nan), np.where(keep, acc, np.nan)


def reliability_figure(snap, gamma: float, n_bins: int, out: Path) -> None:
    """Per-label reliability grid: reference vs a confidence-eroded copy.

    Per-label (not pooled) because the monitor tracks per-label ECE and pooling
    labels of very different base rates distorts the aggregate curve — the
    reference is well-calibrated *within* each label (ECE≈0), which only shows
    on a per-label grid.
    """
    warped = erode(snap.probs, gamma)
    ref_pl = per_label_ece(snap.probs, snap.y_true, n_bins)
    drift_pl = per_label_ece(warped, snap.y_true, n_bins)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharex=True, sharey=True)
    for j, (ax, name) in enumerate(zip(axes.ravel(), LABELS)):
        _style(ax)
        ax.plot([0, 1], [0, 1], "--", lw=1, color=MUTED, zorder=1)
        for probs, color in ((snap.probs, C_REF), (warped, C_DRIFT)):
            cx, cy = _label_curve(probs[:, j], snap.y_true[:, j], n_bins)
            ax.plot(cx, cy, marker="o", ms=4, lw=1.8, color=color, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}\nECE {ref_pl[j]:.3f} → {drift_pl[j]:.3f}",
                     color=INK, fontsize=10)
    fig.text(0.5, 0.02, "mean predicted confidence", ha="center", color=INK)
    fig.text(0.02, 0.5, "observed positive rate", va="center",
             rotation="vertical", color=INK)
    handles = [plt.Line2D([], [], color=C_REF, marker="o", lw=1.8,
                          label=f"reference · mean ECE {ref_pl.mean():.4f}"),
               plt.Line2D([], [], color=C_DRIFT, marker="o", lw=1.8,
                          label=f"eroded (γ={gamma:g}) · mean ECE {drift_pl.mean():.4f}")]
    fig.suptitle("Reliability by label — reference vs confidence-eroded window",
                 color=INK, fontsize=13, y=0.99)
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.90))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  (reference mean ECE={ref_pl.mean():.4f}, "
          f"eroded mean ECE={drift_pl.mean():.4f})")


def ece_over_time_figure(snap, cfg, gamma_min: float, out: Path) -> None:
    """ECE per window as erosion ramps in, threshold + flat accuracy overlaid."""
    mon = CalibrationMonitor.from_snapshot(
        snap, LABELS, window_size=cfg.drift_window_size,
        n_bins=cfg.calibration_n_bins, ece_quantile=cfg.calibration_ece_quantile,
        calib_trials=cfg.calibration_calib_trials, seed=cfg.seed)
    w = cfg.drift_window_size
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(snap.n)
    n_win = snap.n // w
    half = n_win // 2  # first half null (γ=1), second half ramps to gamma_min

    eces, f1s, gammas, alarms = [], [], [], []
    for k in range(n_win):
        rows = perm[k * w:(k + 1) * w]
        gamma = 1.0 if k < half else \
            1.0 - (1.0 - gamma_min) * (k - half + 1) / (n_win - half)
        gammas.append(gamma)
        probs = erode(snap.probs[rows], gamma)
        y = snap.y_true[rows]
        eces.append(float(per_label_ece(probs, y, mon.n_bins).mean()))
        # Hard decisions held fixed (committed preds) → accuracy is constant.
        f1s.append(macro_f1(snap.preds[rows], y))
        alarms.append(eces[-1] > mon.threshold)

    x = np.arange(n_win)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    _style(ax)
    ax.axvspan(half - 0.5, n_win - 0.5, color=C_DRIFT, alpha=0.06, zorder=0)
    ax.axhline(mon.threshold, color=C_ALARM, lw=1.4, ls="--", zorder=2,
               label=f"alarm threshold ({mon.threshold:.4f})")
    ax.axhline(mon.reference_ece_mean, color=MUTED, lw=1.2, ls=":", zorder=2,
               label=f"reference ECE ({mon.reference_ece_mean:.4f})")
    ax.plot(x, eces, marker="o", ms=4, lw=2, color=C_REF, zorder=3,
            label="windowed ECE")
    fired = [i for i in x if alarms[i]]
    if fired:
        ax.scatter(fired, [eces[i] for i in fired], s=70, color=C_ALARM,
                   zorder=4, label="ECE alarm")
    ax.set_xlabel("window (time →)", color=INK)
    ax.set_ylabel("mean per-label ECE", color=INK)
    ax.set_title("Calibration decay over time — confidence erosion ramps in "
                 "at the midpoint", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(x, f1s, lw=1.6, color=INK, alpha=0.55, zorder=3)
    ax2.set_ylabel("macro-F1 (decisions held fixed)", color=INK)
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(MUTED)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    n_alarm = sum(alarms)
    print(f"wrote {out}  ({n_alarm}/{n_win} windows alarmed; "
          f"F1 flat at {np.mean(f1s):.4f} while ECE rose "
          f"{eces[0]:.4f} → {eces[-1]:.4f})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 calibration figures")
    ap.add_argument("--gamma", type=float, default=0.3,
                    help="final confidence-erosion strength (γ<1 flattens)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = DeployConfig.load()
    snap = load_snapshot(cfg.reference_snapshot_path)
    outdir = Path(args.outdir) if args.outdir else cfg.output_path.parent / "assets"
    outdir.mkdir(parents=True, exist_ok=True)

    reliability_figure(snap, args.gamma, cfg.calibration_n_bins,
                       outdir / "reliability_reference_vs_drifted.png")
    ece_over_time_figure(snap, cfg, args.gamma,
                         outdir / "calibration_ece_over_time.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
