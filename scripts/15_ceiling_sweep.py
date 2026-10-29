#!/usr/bin/env python3
"""Does the conditioning ceiling vary enough across corpora to be testable?

`13_conditioning_ceiling.py` computes one ceiling for one corpus. That gives a
number, not a claim: a single (predicted, measured) pair is consistent with
the ceiling predicting conditioning and equally consistent with coincidence.
The claim worth making -- that achievable conditioning gain is bounded by a
property of the corpus, computable before training -- needs the ceiling to
*move*, and measured gain to move with it.

This script asks the free half of that question: can the corpus be varied to
produce a spread of predicted ceilings wide enough for short training runs to
tell apart? If every variant lands between 0.15% and 0.19%, there is no curve
to fit and no experiment to buy. If they span an order of magnitude, there is.

Three dimensions, all of which change the label and none of which change the
sequences, so every variant trains on exactly the same residues:

**Threshold.** The meso/thermo edge is 45 C by convention. Moving it changes
both class balance and how much the label says. This is the main axis.

**Class balance.** Mutual information depends on the marginal, so a corpus
whose classes are balanced carries more information in its label than a
skewed one with the same per-class distributions. Reweighting the residue
counts gives the balanced-corpus ceiling without resampling anything.

**Granularity.** Equal-frequency bins at 2, 3 and 4 classes. More classes can
only raise the unigram ceiling, since a finer label is never less informative;
the question is by how much, and whether it is worth the data it costs per
class.

One pass over the FASTA accumulates residue counts per (cluster, OGT), and
every variant is then computed from those counts. The pass is the expensive
part -- 114M residues -- and doing it once is why a sweep costs about what a
single ceiling costs.

Usage:
    python scripts/15_ceiling_sweep.py
    python scripts/15_ceiling_sweep.py --config pretrain --out results/sweep.txt
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.fasta import read_fasta  # noqa: E402


def _ceiling_module():
    """Reuse 13's arithmetic rather than reimplementing it.

    A second implementation of mutual_information would be a second thing to
    keep correct, and the sweep's whole purpose is to be comparable with the
    single-corpus number 13 reports.
    """
    path = REPO_ROOT / "scripts" / "13_conditioning_ceiling.py"
    spec = importlib.util.spec_from_file_location("ceiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CEIL = _ceiling_module()
RESIDUES = CEIL.RESIDUES

# Residue counts keyed by (cluster_id, ogt_c) -> {residue: count}.
Counts = dict[tuple[str, float], dict[str, int]]


def collect(sequences: Path, rows: dict[str, dict[str, str]]) -> Counts:
    counts: Counts = {}
    kept = 0
    for record in read_fasta(sequences):
        row = rows.get(record.seq_id)
        if not row:
            continue
        try:
            ogt = float(row["ogt_c"])
        except (KeyError, TypeError, ValueError):
            continue
        cluster = row.get("cluster_id") or ""
        key = (cluster, ogt)
        bucket = counts.get(key)
        if bucket is None:
            bucket = collections.defaultdict(int)
            counts[key] = bucket
        for residue in record.sequence:
            if residue in RESIDUES:
                bucket[residue] += 1
        kept += 1
        if kept % 50000 == 0:
            print(f"  ...{kept:,} sequences", flush=True)
    return counts


def label_entropy(totals: dict[str, int]) -> float:
    """H(bin) in nats, from the residue mass in each class.

    Reported beside the ceiling because the obvious objection to the whole
    measure is that it only tracks class balance: mutual information is
    largest when the marginal is even, so a threshold sweep could be a
    roundabout way of finding the median OGT. ``I/H`` -- information per nat
    of label -- is what distinguishes the two. If the ceiling were only
    balance, I/H would be flat across thresholds and the ceiling would peak
    wherever H does.
    """
    total = sum(totals.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for mass in totals.values():
        if mass <= 0:
            continue
        p = mass / total
        entropy -= p * math.log(p)
    return entropy


def bin_of(ogt: float, edges: tuple[float, ...]) -> str:
    """Which class an OGT falls in. Edges are lower-exclusive upper bounds."""
    for index, edge in enumerate(edges):
        if ogt < edge:
            return f"b{index}"
    return f"b{len(edges)}"


def pool(
    counts: Counts, edges: tuple[float, ...], balance: bool = False
) -> dict[str, dict[str, int]]:
    """Corpus-wide residue counts per class."""
    by_bin: dict[str, dict[str, int]] = {}
    for (_, ogt), residues in counts.items():
        name = bin_of(ogt, edges)
        target = by_bin.setdefault(name, collections.defaultdict(int))
        for residue, n in residues.items():
            target[residue] += n
    if balance:
        by_bin = rebalance(by_bin)
    return by_bin


def rebalance(by_bin: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Reweight classes to equal total mass, preserving each one's shape.

    This is what resampling to a balanced corpus would do to the counts, done
    arithmetically. Mutual information is scale-free in the totals, so scaling
    every class to the same total is exactly the balanced-corpus calculation
    and needs no sequences moved. Counts become floats, which the entropy
    arithmetic handles since it only ever divides by the total.
    """
    totals = {name: sum(c.values()) for name, c in by_bin.items()}
    live = {n: t for n, t in totals.items() if t}
    if len(live) < 2:
        return by_bin
    target = min(live.values())
    out: dict[str, dict[str, int]] = {}
    for name, residues in by_bin.items():
        if not totals[name]:
            continue
        scale = target / totals[name]
        out[name] = {r: n * scale for r, n in residues.items()}
    return out


def within_cluster(
    counts: Counts, edges: tuple[float, ...], min_residues: int
) -> tuple[float, int, int]:
    """Weighted mean MI over clusters holding >=2 classes at >=min_residues."""
    per_cluster: dict[str, dict[str, dict[str, int]]] = {}
    for (cluster, ogt), residues in counts.items():
        if not cluster:
            continue
        name = bin_of(ogt, edges)
        by_bin = per_cluster.setdefault(cluster, {})
        target = by_bin.setdefault(name, collections.defaultdict(int))
        for residue, n in residues.items():
            target[residue] += n

    usable: list[tuple[float, int]] = []
    for by_bin in per_cluster.values():
        totals = {b: sum(c.values()) for b, c in by_bin.items()}
        present = [n for n in totals.values() if n >= min_residues]
        if len(present) < 2:
            continue
        info, residues_used = CEIL.mutual_information(by_bin)
        usable.append((info, residues_used))
    if not usable:
        return 0.0, 0, 0
    weight = sum(n for _, n in usable)
    return sum(i * n for i, n in usable) / weight, len(usable), weight


def quantile_edges(counts: Counts, n_bins: int) -> tuple[float, ...]:
    """Equal-frequency edges by residue mass, not by sequence count.

    The ceiling is a per-residue quantity, so the bins that make it comparable
    across granularities are the ones that split residues evenly. Splitting by
    sequence count would make long proteins quietly dominate a class.
    """
    mass: dict[float, int] = collections.defaultdict(int)
    for (_, ogt), residues in counts.items():
        mass[ogt] += sum(residues.values())
    ordered = sorted(mass.items())
    total = sum(mass.values())
    edges: list[float] = []
    seen = 0
    for index, (_, m) in enumerate(ordered):
        seen += m
        if len(edges) >= n_bins - 1:
            break
        if seen < total * (len(edges) + 1) / n_bins:
            continue
        if index + 1 >= len(ordered):
            break
        # The edge is the OGT that STARTS the next class, not the one that
        # finished this one. Bins are upper-exclusive (`ogt < edge`), so
        # appending the finishing value puts its whole mass in the class above
        # instead of below -- and when the first group alone exceeds the
        # target it yields an edge at the minimum OGT, which puts every
        # residue in one class and silently reports a ceiling for a corpus
        # with no contrast at all.
        candidate = ordered[index + 1][0]
        if not edges or candidate > edges[-1]:
            edges.append(candidate)
    return tuple(edges)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="pretrain")
    parser.add_argument("--min-cluster-residues", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    label_table = cfg.resolve_path("paths/label_table")
    sequences = cfg.resolve_path(
        "paths/screened_sequences"
        if "paths/screened_sequences" in cfg
        else "paths/labeled_fasta"
    )
    with label_table.open(newline="") as handle:
        rows = {row["seq_id"]: row for row in csv.DictReader(handle)}

    print(f"corpus:  {sequences.name}")
    print(f"labels:  {label_table.name} ({len(rows):,} rows)")
    print("\nsingle pass over the corpus (the expensive part)...", flush=True)
    counts = collect(sequences, rows)
    total_residues = sum(sum(c.values()) for c in counts.values())
    print(f"  {len(counts):,} (cluster, OGT) groups, "
          f"{total_residues:,} residues")

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit()
    emit("=== threshold sweep (2 classes), natural balance ===")
    emit(f"{'edge':>6} {'thermo%':>8} {'corpus':>9} {'within':>9} "
         f"{'H(bin)':>8} {'I/H':>9} {'clusters':>9} {'residues':>13}")
    thresholds = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0)
    for edge in thresholds:
        edges = (edge,)
        by_bin = pool(counts, edges)
        info, _ = CEIL.mutual_information(by_bin)
        totals = {b: sum(c.values()) for b, c in by_bin.items()}
        hot = totals.get("b1", 0) / max(sum(totals.values()), 1)
        wc_info, n_clusters, wc_residues = within_cluster(
            counts, edges, args.min_cluster_residues
        )
        entropy = label_entropy(totals)
        emit(f"{edge:>6.0f} {hot:>7.1%} "
             f"{CEIL.gain_from_information(info):>+8.3%} "
             f"{CEIL.gain_from_information(wc_info):>+8.3%} "
             f"{entropy:>8.4f} {info / entropy if entropy else 0.0:>9.5f} "
             f"{n_clusters:>9,} {wc_residues:>13,}")

    emit()
    emit("=== threshold sweep, classes rebalanced to equal mass ===")
    emit(f"{'edge':>6} {'corpus':>9} {'vs natural':>12}")
    for edge in thresholds:
        edges = (edge,)
        natural, _ = CEIL.mutual_information(pool(counts, edges))
        balanced, _ = CEIL.mutual_information(pool(counts, edges, balance=True))
        ratio = (
            CEIL.gain_from_information(balanced)
            / max(CEIL.gain_from_information(natural), 1e-12)
        )
        emit(f"{edge:>6.0f} {CEIL.gain_from_information(balanced):>+8.3%} "
             f"{ratio:>11.2f}x")

    emit()
    emit("=== granularity: equal-residue-mass bins ===")
    emit(f"{'classes':>8} {'edges (C)':>22} {'corpus':>9} {'within':>9} "
         f"{'clusters':>9}")
    for n_bins in (2, 3, 4, 5):
        edges = quantile_edges(counts, n_bins)
        if len(edges) != n_bins - 1:
            continue
        info, _ = CEIL.mutual_information(pool(counts, edges))
        wc_info, n_clusters, _ = within_cluster(
            counts, edges, args.min_cluster_residues
        )
        shown = ", ".join(f"{e:.0f}" for e in edges)
        emit(f"{n_bins:>8} {shown:>22} "
             f"{CEIL.gain_from_information(info):>+8.3%} "
             f"{CEIL.gain_from_information(wc_info):>+8.3%} {n_clusters:>9,}")

    # The spread is the whole point: it decides whether the GPU half of the
    # experiment can distinguish anything.
    corpus_gains = [
        CEIL.gain_from_information(CEIL.mutual_information(pool(counts, (e,)))[0])
        for e in thresholds
    ]
    low, high = min(corpus_gains), max(corpus_gains)
    emit()
    emit("=== is this testable? ===")
    emit(f"  corpus-wide ceiling ranges {low:+.3%} to {high:+.3%} "
         f"({high / max(low, 1e-12):.1f}x) across the threshold sweep")
    emit(f"  configured gate ceiling    {0.00172:+.3%} (45 C, within-cluster)")
    emit(f"  stage A measured gain      {0.00265:+.3%} (validation, 2026-10-26)")
    if high / max(low, 1e-12) < 2.0:
        emit("  SPREAD TOO NARROW: short runs could not separate these, so the "
             "predicted-vs-measured\n  experiment is not worth GPU time as "
             "designed. Widen the axes or drop the claim.")
    else:
        emit("  The spread is wide enough that short runs can separate the "
             "variants. What this\n  does NOT establish is that measured gain "
             "follows it -- that is the GPU half, and\n  it is the half that "
             "can falsify the claim.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
