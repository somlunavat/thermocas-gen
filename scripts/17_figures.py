#!/usr/bin/env python3
"""Generate the paper's figures from the committed result files.

Nothing here is transcribed. Bounds are read from the sweep tables written by
``15_ceiling_sweep.py`` and measured gains from the Gate 2 and test reports
written by ``11_train.py``, both of which are machine-generated with fixed
columns. Transcribing them into this file was the first version and it put the
external corpus's four-class bound on the primary corpus's panel, which is
exactly the class of error a figure script should not be able to make.

Error bars are one standard deviation of the shuffle spread, in the same
percentage units as the gain. The reports print the spread in absolute
perplexity and the gain as a percentage, so the conversion is gain / z, which
is exact: the report's z is the gain in perplexity units divided by that
spread.

Usage:
    python3 scripts/17_figures.py [--outdir paper/figures] [--format pdf]

The figures:

1. Two panels, corpus-wide bound against measured gain beside within-cluster
   bound against measured gain. The point of the pairing is that the left panel
   is a cloud and the right is close to a line, which is section 5.1b's claim.
2. The bound against threshold for three corpora, with the thermophilic
   residue mass underneath, because the uncapped corpus's peak is a feature of
   that row rather than of thermophilic composition.
3. Measured gain against threshold for the three trained configurations. This
   is the plateau result of sections 5.2b and 5.2c in one panel.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

# Where each configuration's bounds and reports live. Report directories are
# split because the primary corpus's eight thresholds were run in two batches.
SWEEPS = {
    "primary": RESULTS / "external_corpus_20261029" / "sweep_pretrain.txt",
    "external": (RESULTS / "external_corpus_20261029"
                 / "sweep_external_balanced_screened.txt"),
    "uncapped": RESULTS / "external_corpus_20261029" / "sweep_external.txt",
}
BATCH1 = RESULTS / "ceiling_sweep_20261027" / "runs"
BATCH2 = RESULTS / "ceiling_sweep_20261030" / "runs"

# run name -> (report directory, bound corpus, threshold)
RUNS = {
    "primary": {
        30: (BATCH1, "sweep_t30"), 35: (BATCH2, "sweep_t35"),
        45: (BATCH1, "sweep_t45"), 50: (BATCH2, "sweep_t50"),
        55: (BATCH1, "sweep_t55"), 65: (BATCH2, "sweep_t65"),
        70: (BATCH2, "sweep_t70"), 75: (BATCH1, "sweep_t75"),
    },
    "external": {
        30: (BATCH2, "ext_t30"), 40: (BATCH2, "ext_t40"),
        55: (BATCH2, "ext_t55"), 65: (BATCH2, "ext_t65"),
        75: (BATCH2, "ext_t75"),
    },
    "small": {
        30: (BATCH2, "small_t30"), 55: (BATCH2, "small_t55"),
        75: (BATCH2, "small_t75"),
    },
}
# The four-class granularity variant. Its bound is the primary corpus's
# four-class row, which is not comparable to a threshold's (section 6.2), so it
# appears once, marked, in the left panel only.
C4_RUN = (BATCH1, "sweep_c4")

# Spearman and exact permutation p for the primary corpus's eight thresholds,
# from results/ceiling_sweep_20261030/measured_n8.md. Printed on figure 1 so a
# reader is not asked to carry them from the text.
STATS = {"corpus": (0.381, 0.180), "within": (0.714, 0.029)}

# Colour-blind-safe, and separable in greyscale by marker.
INK = "#1a1a1a"
MUTED = "#8c8c8c"
COLOURS = {
    "primary": "#0f6ab4",
    "external": "#c1553b",
    "small": "#4a7c2f",
    "uncapped": "#8a6bbe",
}
MARKERS = {"primary": "o", "external": "s", "small": "^", "uncapped": "D"}
LABELS = {
    "primary": "primary corpus, progen2-base",
    "external": "second corpus, progen2-base",
    "small": "primary corpus, progen2-small",
    "uncapped": "external, uncapped",
}

_SWEEP_ROW = re.compile(
    r"^\s*(\d+)\s+([\d.]+)%\s+\+?([\d.]+)%\s+\+?([\d.]+)%")
_GRANULARITY_ROW = re.compile(
    r"^\s*(\d+)\s+[\d,\s]+\s+\+?([\d.]+)%\s+\+?([\d.]+)%")
_GAIN = re.compile(r"relative gain:\s*\+?(-?[\d.]+)%")
_Z = re.compile(r"z vs shuffle spread\s+([\d.]+)")


def read_sweep(path: Path) -> dict[int, tuple[float, float, float]]:
    """Threshold -> (corpus-wide bound, within-cluster bound, thermo mass %)."""
    out: dict[int, tuple[float, float, float]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("==="):
            if "threshold sweep" not in line:
                break  # the rebalanced and granularity tables follow
            continue
        m = _SWEEP_ROW.match(line)
        if m:
            edge, thermo, corpus, within = m.groups()
            out[int(edge)] = (float(corpus), float(within), float(thermo))
    if not out:
        raise SystemExit(f"no sweep rows parsed from {path}")
    return out


def read_granularity(path: Path, n_classes: int) -> tuple[float, float]:
    """(corpus-wide, within-cluster) bound for an n-class equal-mass binning."""
    block = path.read_text().split("=== granularity")[1]
    for line in block.splitlines():
        m = _GRANULARITY_ROW.match(line)
        if m and int(m.group(1)) == n_classes:
            return float(m.group(2)), float(m.group(3))
    raise SystemExit(f"no {n_classes}-class granularity row in {path}")


def read_run(directory: Path, name: str) -> dict[str, tuple[float, float]]:
    """Split -> (gain %, one sigma of the shuffle spread in the same units)."""
    out: dict[str, tuple[float, float]] = {}
    for split, suffix in (("val", "gate2"), ("test", "test")):
        path = directory / f"{name}_{suffix}.txt"
        text = path.read_text()
        gain = _GAIN.search(text)
        z = _Z.search(text)
        if not gain or not z:
            raise SystemExit(f"could not parse gain and z from {path}")
        g, zz = float(gain.group(1)), float(z.group(1))
        out[split] = (g, g / zz)
    return out


def load() -> dict:
    sweeps = {k: read_sweep(p) for k, p in SWEEPS.items()}
    gains = {
        cfg: {edge: read_run(d, name) for edge, (d, name) in runs.items()}
        for cfg, runs in RUNS.items()
    }
    # progen2-small trained on the primary corpus, so it shares its bounds.
    bounds = {"primary": sweeps["primary"], "external": sweeps["external"],
              "small": sweeps["primary"], "uncapped": sweeps["uncapped"]}
    c4_bound = read_granularity(SWEEPS["primary"], 4)
    return {
        "sweeps": sweeps,
        "bounds": bounds,
        "gains": gains,
        "c4": (c4_bound, read_run(*C4_RUN)),
    }


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def figure1(data: dict, outdir: Path, ext: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    panels = (
        ("corpus", 0, "corpus-wide: does not predict the ordering"),
        ("within", 1, "within-cluster: predicts the ordering"),
    )
    for ax, (key, col, title) in zip(axes, panels, strict=True):
        for cfg in ("primary", "external"):
            bounds, gains = data["bounds"][cfg], data["gains"][cfg]
            xs = [bounds[e][col] for e in sorted(gains)]
            ys = [gains[e]["val"][0] for e in sorted(gains)]
            errs = [gains[e]["val"][1] for e in sorted(gains)]
            ax.errorbar(xs, ys, yerr=errs, fmt=MARKERS[cfg], ms=4.4, lw=0,
                        elinewidth=0.7, capsize=1.6, color=COLOURS[cfg],
                        label=LABELS[cfg].replace(", progen2-base", ""),
                        zorder=3)
        # 35 C is the point that moves most between the panels, and the
        # mechanism in section 5.1b is about it.
        x35 = data["bounds"]["primary"][35][col]
        y35 = data["gains"]["primary"][35]["val"][0]
        ax.annotate("35 C", (x35, y35), textcoords="offset points",
                    xytext=(0, 11), fontsize=7, color=INK, ha="center",
                    va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
        if key == "corpus":
            (c4_corpus, _), c4_gain = data["c4"]
            ax.plot([c4_corpus], [c4_gain["val"][0]], "v", ms=5,
                    color=MUTED, zorder=3,
                    label="4 classes (not comparable; 6.2)")
        rho, p = STATS[key]
        ax.text(0.03, 0.96, f"$\\rho$ = {rho:+.3f}\np = {p:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                color=INK)
        ax.set_xlabel(f"{key.replace('corpus', 'corpus-wide')} bound (%)"
                      if key == "corpus" else "within-cluster bound (%)")
        ax.set_title(title, color=INK)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("measured conditioning gain, validation (%)")
    fig.text(0.5, -0.07,
             "$\\rho$ and p are the primary corpus's eight thresholds, exact "
             "permutation test over all 40,320 relabelings; the second "
             "corpus's five\nare overplotted for scale and scored separately "
             "in section 5.2b. Error bars: one standard deviation of the "
             "shuffle spread.",
             ha="center", fontsize=6.4, color=MUTED)
    path = outdir / f"figure1_bound_vs_measured.{ext}"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure2(data: dict, outdir: Path, ext: str) -> Path:
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(4.6, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.1},
    )
    names = {
        "primary": "primary (Gosha, capped + balanced)",
        "external": "external, design-matched",
        "uncapped": "external, uncapped",
    }
    for cfg, label in names.items():
        sweep = data["sweeps"][cfg]
        edges = sorted(sweep)
        bounds = [sweep[e][0] for e in edges]
        thermo = [sweep[e][2] for e in edges]
        ax.plot(edges, bounds, marker=MARKERS[cfg], ms=3.6, lw=1.1,
                color=COLOURS[cfg], label=label, zorder=3)
        peak = max(range(len(bounds)), key=lambda i: bounds[i])
        ax.plot([edges[peak]], [bounds[peak]], marker=MARKERS[cfg], ms=9,
                mfc="none", mec=COLOURS[cfg], mew=1.2, zorder=4)
        ax.annotate(f"{edges[peak]} C", (edges[peak], bounds[peak]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=7, color=COLOURS[cfg])
        bx.plot(edges, thermo, marker=MARKERS[cfg], ms=3.4, lw=1.1,
                color=COLOURS[cfg], zorder=3)
    ax.set_ylabel("corpus-wide bound (%)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)  # room for the peak labels
    ax.legend(loc="upper left")
    bx.set_ylabel("thermophilic\nresidue mass (%)")
    bx.set_xlabel("threshold, upper-exclusive (degrees C)")
    bx.set_xticks(sorted(data["sweeps"]["primary"]))
    # The uncapped corpus loses four fifths of its hot class between 35 and 40,
    # and that is where its peak sits: the cliff, not thermophilic composition.
    bx.annotate("37 C pile-up", (37.5, 29), xytext=(45, 55), fontsize=6.8,
                color=COLOURS["uncapped"],
                arrowprops=dict(arrowstyle="->", lw=0.6,
                                color=COLOURS["uncapped"]))
    path = outdir / f"figure2_bound_vs_threshold.{ext}"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure3(data: dict, outdir: Path, ext: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for cfg in ("primary", "external", "small"):
        gains = data["gains"][cfg]
        edges = sorted(gains)
        ys = [gains[e]["val"][0] for e in edges]
        errs = [gains[e]["val"][1] for e in edges]
        ax.errorbar(edges, ys, yerr=errs, marker=MARKERS[cfg], ms=4.0, lw=1.1,
                    elinewidth=0.7, capsize=1.6, color=COLOURS[cfg],
                    label=LABELS[cfg], zorder=3)
    ax.axvspan(45, 65, color=COLOURS["primary"], alpha=0.07, zorder=1)
    ax.set_xlabel("threshold, upper-exclusive (degrees C)")
    ax.set_ylabel("measured conditioning gain, validation (%)")
    ax.set_xticks([30, 35, 40, 45, 50, 55, 65, 70, 75])
    # Headroom for the legend, which otherwise sits on the second corpus's
    # line between 55 and 65 C.
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.60))
    ax.text(55, ax.get_ylim()[1] * 0.04, "plateau, primary corpus",
            ha="center", fontsize=6.8, color=MUTED)
    ax.legend(loc="upper right")
    path = outdir / f"figure3_measured_vs_threshold.{ext}"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(REPO_ROOT / "paper" / "figures"))
    ap.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load()
    _style()
    for fn in (figure1, figure2, figure3):
        print(f"wrote {fn(data, outdir, args.format).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
