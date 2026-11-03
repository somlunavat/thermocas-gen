#!/usr/bin/env python3
"""How much can ANY growth-temperature label be worth on this corpus?

`15_ceiling_sweep.py` answers that for a particular binning. This answers it
for the best binning there is, which is the number another group needs if they
want to read their own reported conditioning gain against a scale.

Mutual information is monotone non-decreasing under refinement of the label
partition, so I(residue; bin) rises with the number of classes and is largest
when the label is the recorded growth temperature itself, unbinned. That
quantity -- I(residue; OGT) at the database's native resolution -- is the
envelope over every possible binning of that database's labels, and no
threshold, no quantile scheme and no multi-class scheme can exceed it.

Monotonicity is also why the number needs a bias correction before it can be
quoted. A plug-in estimate of mutual information from counts is biased upward
by roughly (|R| - 1)(|B| - 1) / 2N nats, and |B| here is the number of
distinct temperatures rather than two, so the correction grows with exactly
the refinement that raises the estimate. The Miller-Madow correction is
reported beside every figure and subtracted in the `corrected` column. At the
corpus sizes here it is small -- which is itself worth establishing rather
than assuming.

What this does NOT do is convert anyone else's reported number into these
units. See `results/prime_denominator_20261102/findings.md`: the published
temperature-conditioned models report downstream correlations, not a
likelihood gain from a conditioning label, and nothing public bridges the two.

Usage:
    python scripts/18_label_envelope.py --config external
    python scripts/18_label_envelope.py --config pretrain --max-classes 24
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Both are reused rather than reimplemented: a second copy of this arithmetic
# would be a second thing to keep correct, and the point of the number is that
# it is comparable with what 13 and 15 already report.
CEIL = _module("ceiling", "13_conditioning_ceiling.py")
SWEEP = _module("sweep", "15_ceiling_sweep.py")


def miller_madow(by_bin: dict[str, dict[str, int]]) -> float:
    """Upward bias of the plug-in mutual information estimate, in nats.

    (|R| - 1)(|B| - 1) / 2N, counting only classes and residues that actually
    occur -- an alphabet letter absent from the corpus contributes no bias and
    including it would overstate the correction.
    """
    occupied_bins = sum(1 for counts in by_bin.values() if sum(counts.values()))
    residues = {r for counts in by_bin.values() for r, n in counts.items() if n}
    total = sum(sum(counts.values()) for counts in by_bin.values())
    if total == 0 or occupied_bins < 2 or len(residues) < 2:
        return 0.0
    return (len(residues) - 1) * (occupied_bins - 1) / (2 * total)


def residue_mass(counts) -> list[tuple[float, int]]:
    """(growth temperature, residues) ascending, one entry per recorded value."""
    mass: dict[float, int] = collections.defaultdict(int)
    for (_, ogt), residues in counts.items():
        mass[ogt] += sum(residues.values())
    return sorted(mass.items())


def balanced_edges(counts, n_bins: int) -> tuple[float, ...]:
    """Edges maximising the smallest class's share of residues.

    `15_ceiling_sweep.quantile_edges` aims for equal residue mass and cannot
    reach it here: a single recorded temperature can hold more mass than 1/k
    of the corpus, and an atom cannot be split. On the primary corpus 30 C is
    29.6% of all residues and 37 C is 18.5%, so from k=4 upward the quantile
    scheme degenerates -- at k=16 its smallest class is 0.1% of residues and
    313 sequences. A ceiling computed on that partition is real arithmetic on
    a binning no one could train on.

    This instead maximises min class mass over contiguous partitions, by the
    standard dynamic program: best[g][i] is the largest achievable minimum
    when the temperatures from i onward are cut into g classes. Contiguity is
    not a convenience -- a growth-temperature class that is not an interval is
    not a temperature class.

    Returns k-1 edges, upper-exclusive like every other binning here.
    """
    ordered = residue_mass(counts)
    masses = [m for _, m in ordered]
    n = len(masses)
    if n_bins < 2 or n_bins > n:
        return ()
    prefix = [0]
    for m in masses:
        prefix.append(prefix[-1] + m)

    # best[g] maps a start index to (achievable minimum, cut indices).
    best: list[dict[int, tuple[int, tuple[int, ...]]]] = [{} for _ in range(n_bins + 1)]
    for i in range(n):
        best[1][i] = (prefix[n] - prefix[i], ())
    for g in range(2, n_bins + 1):
        # A start index must leave at least g temperatures, one per class.
        for i in range(0, n - g + 1):
            chosen = (-1, ())
            for j in range(i + 1, n - g + 2):
                rest = best[g - 1].get(j)
                if rest is None:
                    continue
                value = min(prefix[j] - prefix[i], rest[0])
                if value > chosen[0]:
                    chosen = (value, (j,) + rest[1])
            best[g][i] = chosen
    _, cuts = best[n_bins][0]
    return tuple(ordered[j][0] for j in cuts)


def min_class_share(counts, edges: tuple[float, ...]) -> float:
    """Smallest class's share of residues under `edges`."""
    pooled = SWEEP.pool(counts, edges)
    sizes = [sum(c.values()) for c in pooled.values() if sum(c.values())]
    total = sum(sizes)
    return min(sizes) / total if total and sizes else 0.0


def by_native_ogt(counts) -> dict[str, dict[str, int]]:
    """Pool to one class per distinct recorded growth temperature."""
    out: dict[str, dict[str, int]] = {}
    for (_, ogt), residues in counts.items():
        bucket = out.setdefault(f"{ogt:g}", collections.defaultdict(int))
        for residue, n in residues.items():
            bucket[residue] += n
    return out


def report(by_bin: dict[str, dict[str, int]]) -> tuple[float, float, int, int]:
    """(raw gain %, bias-corrected gain %, classes, residues)."""
    info, residues = CEIL.mutual_information(by_bin)
    bias = miller_madow(by_bin)
    occupied = sum(1 for c in by_bin.values() if sum(c.values()))
    return (
        CEIL.gain_from_information(info),
        CEIL.gain_from_information(max(info - bias, 0.0)),
        occupied,
        residues,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="external")
    parser.add_argument(
        "--max-classes", type=int, default=16,
        help="largest equal-residue-mass binning to evaluate",
    )
    parser.add_argument(
        "--floor", type=float, default=0.05,
        help="smallest share of corpus residues a class may hold and still "
             "count as trainable; partitions below it are reported and marked",
    )
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
    print("\nsingle pass over the corpus...", flush=True)
    counts = SWEEP.collect(sequences, rows)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    native = by_native_ogt(counts)
    distinct = len(native)
    total = sum(sum(c.values()) for c in native.values())
    emit()
    emit(f"=== label envelope: {args.config} ===")
    emit(f"{distinct} distinct growth temperatures, {total:,} residues")
    emit()
    emit("Mutual information cannot fall when the label partition is refined,")
    emit("so the unbinned row is an upper bound on every binning of this")
    emit("corpus's labels -- threshold, quantile or otherwise.")
    emit()
    emit("The rows between are NOT nested: equal-residue-mass edges for k and")
    emit("k+1 classes are different partitions, not a refinement of one")
    emit("another, so this column can dip. A dip is a property of the")
    emit("quantile scheme, not a violation of monotonicity and not an error.")
    emit()
    emit(f"{'classes':>8} {'raw':>9} {'corrected':>10} {'bias':>9} "
         f"{'% of envelope':>14}")

    envelope_raw, envelope_corrected, _, _ = report(native)

    rows_out: list[tuple[str, float, float, float]] = []
    for k in range(2, args.max_classes + 1):
        edges = SWEEP.quantile_edges(counts, k)
        if len(edges) < k - 1:
            # The corpus ran out of distinct temperatures to split on; asking
            # for more classes than it can carry would silently report a
            # coarser binning under a finer label.
            emit(f"{k:>8}  (corpus cannot carry {k} equal-mass classes)")
            break
        raw, corrected, occupied, _ = report(SWEEP.pool(counts, edges))
        rows_out.append((str(k), raw, corrected, corrected / envelope_corrected))
        emit(f"{occupied:>8} {raw:>+8.3%} {corrected:>+9.3%} "
             f"{raw - corrected:>+8.3%} "
             f"{corrected / envelope_corrected:>13.1%}")

    emit(f"{distinct:>8} {envelope_raw:>+8.3%} {envelope_corrected:>+9.3%} "
         f"{envelope_raw - envelope_corrected:>+8.3%} {1.0:>13.1%}  "
         f"<- unbinned, the envelope")

    # The two-class row is what every paper in this area actually uses, so the
    # headline number is how much of the envelope a threshold throws away.
    if rows_out:
        two = rows_out[0]
        emit()
        emit(f"A two-class threshold reaches {two[3]:.1%} of the envelope.")
        emit(f"The conventional labelling therefore leaves a factor of "
             f"{envelope_corrected / two[2]:.1f} on the table -- not in "
             f"measured gain,")
        emit("which section 5.1 shows is a different and smaller quantity, "
             "but in")
        emit("what the corpus makes available to any label at all.")

    # The quantile table above answers "what does refinement make available".
    # It does not answer "what could anyone train on", because its own smallest
    # class collapses as k grows. This second table holds class mass up and
    # asks how much of the envelope survives that constraint.
    emit()
    emit(f"=== reachable under a {args.floor:.0%} minimum class mass ===")
    emit()
    emit("Edges here maximise the smallest class's share of residues rather")
    emit("than aiming for equal shares, which the corpus cannot deliver: its")
    emit("largest single recorded temperature alone holds more than 1/k of all")
    emit("residues for every k above 3.")
    emit()
    emit(f"{'classes':>8} {'min class':>10} {'corrected':>10} "
         f"{'% of envelope':>14}  edges")

    feasible: list[tuple[int, float, float]] = []
    for k in range(2, args.max_classes + 1):
        edges = balanced_edges(counts, k)
        if len(edges) < k - 1:
            emit(f"{k:>8}  (corpus has too few distinct temperatures)")
            break
        share = min_class_share(counts, edges)
        _, corrected, _, _ = report(SWEEP.pool(counts, edges))
        mark = "" if share >= args.floor else "  <- below floor"
        if share >= args.floor:
            feasible.append((k, corrected, share))
        emit(f"{k:>8} {share:>9.1%} {corrected:>+9.3%} "
             f"{corrected / envelope_corrected:>13.1%}  "
             f"{','.join(f'{e:g}' for e in edges)}{mark}")

    if feasible:
        best_k, best_gain, best_share = max(feasible, key=lambda row: row[1])
        largest_k = max(k for k, _, _ in feasible)
        emit()
        emit(f"Largest trainable granularity: {largest_k} classes.")
        emit(f"Best reachable bound: {best_gain:+.3%} at {best_k} classes "
             f"({best_gain / envelope_corrected:.1%} of the envelope,")
        emit(f"smallest class {best_share:.1%} of residues).")
        emit()
        emit("So the gap between a two-class threshold and the envelope is not")
        emit("all available to a label anyone could train on. The remainder")
        emit("sits in classes of a few hundred sequences, and a bound computed")
        emit("there describes a partition, not an experiment.")

    emit()
    emit("Bias is the Miller-Madow correction, (|R|-1)(|B|-1)/2N nats,")
    emit("converted to the same percentage units. It grows with the number of")
    emit("classes, which is why the envelope row needs it most.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
