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
