"""make_shift_figures.py — the Phase 3 figures for assets/.

All three figures render from the committed ``experiments/shift_results.json``
alone — no checkpoint, no dataset, no re-inference — so a fresh clone reproduces
the paper figures exactly, the same self-contained guarantee the Phase 2 figures
hold to. The scored numbers were produced once by ``run_shift_experiments.py``
(which does need the checkpoint + data); this script only visualises them.

    python scripts/make_shift_figures.py

    assets/shift_detection_curve.png    per-corruption recall vs intensity, one
                                        line per monitor channel + the FA control
                                        at intensity 0.
    assets/shift_monitor_timeline.png   per headline shift, each monitor's signal
                                        vs its threshold over windows, with the
                                        shift onset and the retrain-trigger fire
                                        marked — the "which monitor fires first"
                                        story.
    assets/shift_alarm_table.png        the scored detection table (first channel,
                                        per-channel latency, trigger latency, FA).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wafer_deploy.config import DeployConfig  # noqa: E402

# House palette (shared with the sibling figures).
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
C_COV, C_PRED, C_CAL, C_TRIG = "#2a78d6", "#eb6834", "#159a7a", "#c0392b"
CH_COLOR = {"covariate": C_COV, "prediction": C_PRED, "calibration": C_CAL}


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)


# --------------------------------------------------------------------------- #
# (1) Detection curve — recall vs intensity, per corruption, per channel.
# --------------------------------------------------------------------------- #

def detection_curve(res: dict, out: Path) -> None:
    sweeps = res["sweeps"]
    intensities = res["meta"]["intensities"]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.2 * len(sweeps), 4.6),
                             sharey=True)
    if len(sweeps) == 1:
        axes = [axes]
    for ax, (cname, rows) in zip(axes, sweeps.items()):
        _style(ax)
        for ch in ("covariate", "prediction", "calibration"):
            recall = [r["summary"]["channel_alarm_rate_post"][ch] for r in rows]
            ax.plot(intensities, recall, marker="o", ms=5, lw=1.9,
                    color=CH_COLOR[ch], label=ch, zorder=3)
        # false-alarm control at intensity 0 (should sit near the floor)
        ax.axvline(0.0, color=MUTED, ls=":", lw=1)
        ax.set_title(cname, color=INK, fontsize=12)
        ax.set_xlabel("corruption intensity", color=INK)
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("post-onset alarm rate (recall)", color=INK)
    handles = [plt.Line2D([], [], color=CH_COLOR[c], marker="o", lw=1.9, label=c)
               for c in ("covariate", "prediction", "calibration")]
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Detection curve — monitor recall vs corruption intensity "
                 "(intensity 0 = false-alarm control)", color=INK, fontsize=13, y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# (2) Monitor-firing timeline per headline shift.
# --------------------------------------------------------------------------- #

def _headline_runs(res: dict) -> list[tuple[str, dict]]:
    order = [("noise_0.5", "synthetic noise (intensity 0.5)"),
             ("wm811k", "WM-811K cross-dataset"),
             ("campaign", "class-prior campaign (Edge-Ring)")]
    return [(title, res["headline"][key]) for key, title in order
            if res["headline"].get(key)]


def monitor_timeline(res: dict, out: Path) -> None:
    runs = _headline_runs(res)
    onset = res["meta"]["onset_window"]
    fig, axes = plt.subplots(len(runs), 1, figsize=(9.5, 3.0 * len(runs)),
                             sharex=False)
    if len(runs) == 1:
        axes = [axes]
    for ax, (title, run) in zip(axes, runs):
        _style(ax)
        recs = run["records"]
        x = [r["window"] for r in recs]
        # Each monitor's signal as a ratio to its own threshold → 1.0 == alarm bar.
        cov = [r["mmd2"] / r["mmd2_threshold"] if r["mmd2_threshold"] else 0 for r in recs]
        psi = [r["psi"] / r["psi_threshold"] if r["psi_threshold"] else 0 for r in recs]
        ece = [r["ece"] / r["ece_threshold"] if r["ece_threshold"] else 0 for r in recs]
        ax.axhline(1.0, color=MUTED, ls="--", lw=1, zorder=1)
        ax.axvspan(onset - 0.5, max(x) + 0.5, color=MUTED, alpha=0.07, zorder=0)
        ax.plot(x, cov, marker="o", ms=3, lw=1.6, color=C_COV, label="covariate MMD²", zorder=3)
        ax.plot(x, psi, marker="s", ms=3, lw=1.6, color=C_PRED, label="prediction PSI", zorder=3)
        ax.plot(x, ece, marker="^", ms=3, lw=1.6, color=C_CAL, label="calibration ECE", zorder=3)
        # retrain-trigger fire windows
        fires = [r["window"] for r in recs if r.get("trigger_just_fired")]
        for f in fires:
            ax.axvline(f, color=C_TRIG, lw=1.8, zorder=4)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_ylabel("signal ÷ threshold", color=INK, fontsize=9)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("window (time →)  ·  shaded = post-onset  ·  "
                        "red line = retrain trigger fires", color=INK)
    handles = [plt.Line2D([], [], color=C_COV, marker="o", lw=1.6, label="covariate MMD²"),
               plt.Line2D([], [], color=C_PRED, marker="s", lw=1.6, label="prediction PSI"),
               plt.Line2D([], [], color=C_CAL, marker="^", lw=1.6, label="calibration ECE"),
               plt.Line2D([], [], color=C_TRIG, lw=1.8, label="retrain trigger")]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Monitor firing timeline — unsupervised channels lead, "
                 "calibration confirms on delayed labels", color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# (3) Alarm table.
# --------------------------------------------------------------------------- #

def _fmt(v) -> str:
    return "—" if v is None else str(v)


def alarm_table(res: dict, out: Path) -> None:
    rows_data = table_rows(res)
    header = ["shift", "first channel", "cov lat", "pred lat", "cal lat",
              "trigger lat", "trigger FA@0"]
    cell = [[r[k] for k in ("shift", "first", "cov", "pred", "cal", "trig", "fa")]
            for r in rows_data]
    fig, ax = plt.subplots(figsize=(11.5, 0.5 * len(cell) + 1.4))
    ax.axis("off")
    col_w = [0.24, 0.15, 0.11, 0.11, 0.11, 0.13, 0.15]
    tbl = ax.table(cellText=cell, colLabels=header, loc="center", cellLoc="center",
                   colWidths=col_w)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.5)
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor(GRID)
        if r == 0:
            cellobj.set_facecolor("#f3f2ee")
            cellobj.set_text_props(color=INK, fontweight="bold")
        else:
            cellobj.set_text_props(color=INK)
    ax.set_title("Scored detection table — latency in windows after onset "
                 f"({res['meta']['window_size']} maps/window)",
                 color=INK, fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def table_rows(res: dict) -> list[dict]:
    """The alarm-table rows (also reused verbatim in docs/EXPERIMENTS.md)."""
    out = []
    for cname, rows in res["sweeps"].items():
        top = rows[-1]["summary"]                    # max-intensity row
        zero = rows[0]["summary"]                    # intensity-0 FA control
        out.append({
            "shift": f"{cname} @{rows[-1]['intensity']:.2f}",
            "first": _fmt(top["first_channel"]),
            "cov": _fmt(top["detection_latency"]["covariate"]),
            "pred": _fmt(top["detection_latency"]["prediction"]),
            "cal": _fmt(top["detection_latency"]["calibration"]),
            "trig": _fmt(top["trigger_latency"]),
            "fa": "yes" if zero["trigger_fired"] else "no",
        })
    for key, name in (("wm811k", "WM-811K cross-dataset"),
                      ("campaign", "class-prior campaign")):
        run = res["headline"].get(key)
        if not run:
            continue
        s = run["summary"]
        out.append({
            "shift": name, "first": _fmt(s["first_channel"]),
            "cov": _fmt(s["detection_latency"]["covariate"]),
            "pred": _fmt(s["detection_latency"]["prediction"]),
            "cal": _fmt(s["detection_latency"]["calibration"]),
            "trig": _fmt(s["trigger_latency"]), "fa": "n/a",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 shift figures")
    ap.add_argument("--results", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = DeployConfig.load()
    repo = Path(__file__).resolve().parents[1]
    results_path = Path(args.results) if args.results else repo / "experiments" / "shift_results.json"
    res = json.loads(results_path.read_text())
    outdir = Path(args.outdir) if args.outdir else repo / "assets"
    outdir.mkdir(parents=True, exist_ok=True)

    detection_curve(res, outdir / "shift_detection_curve.png")
    monitor_timeline(res, outdir / "shift_monitor_timeline.png")
    alarm_table(res, outdir / "shift_alarm_table.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
