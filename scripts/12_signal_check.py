#!/usr/bin/env python3
"""Pre-flight for Gate 2: is the thermophilic signal actually in the corpus?

The stage A gate counts clusters and sequences. It says the corpus is large
enough; it does not say the corpus contains the signal the conditioning token
has to explain. This script checks that directly, before a GPU run.

Two measurements, because they answer different questions.

**Corpus-wide composition** compares all thermophilic sequences against all
mesophilic ones. A difference here is necessary but not sufficient: it is
confounded with phylogeny, since thermophiles are a different set of organisms
with different proteome-wide composition.

**Within-cluster paired shift** compares thermophilic and mesophilic members of
the *same* cluster, i.e. homologous proteins at different growth temperatures.
This controls for protein family, so a consistent shift here is evidence of
genuine thermal adaptation rather than a phylogenetic artifact. It is the
measurement that predicts whether Gate 2 can pass.

Reference signal (Zeldovich et al. 2007): thermophilic proteomes are enriched
in IVYWREL, enriched in charged residues that form salt bridges, and depleted
in the thermolabile polar residues.

Usage:
    python scripts/12_signal_check.py
    python scripts/12_signal_check.py --config pretrain
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.fasta import read_fasta  # noqa: E402

# Direction each group is expected to move in a thermophile: +1 up, -1 down.
GROUPS = {
    "IVYWREL": (set("IVYWREL"), +1),
    "charged": (set("DEKR"), +1),
    "polar": (set("STNQ"), -1),
}


def composition(sequence: str) -> dict[str, float]:
    length = len(sequence)
    if length == 0:
        return {name: 0.0 for name in GROUPS}
    return {
        name: sum(1 for c in sequence if c in residues) / length
        for name, (residues, _) in GROUPS.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="pretrain")
    parser.add_argument("--sequences", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument("--min-consistency", type=float, default=0.60,
                        help="fraction of clusters that must shift as expected")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    sequences = args.sequences or cfg.resolve_path("paths/sequences").with_name(
        cfg.resolve_path("paths/sequences").stem + "_screened.fasta"
    )
    label_table = args.label_table or cfg.resolve_path("paths/label_table")

    if not sequences.is_file():
        print(f"sequences not found: {sequences}", file=sys.stderr)
        return 1

    labels: dict[str, tuple[str, str]] = {}
    with label_table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["seq_id"]] = (row["temp_bin"], row.get("cluster_id", ""))

    totals: dict[str, list] = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
    per_cluster: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for record in read_fasta(sequences):
        meta = labels.get(record.seq_id)
        if meta is None:
            continue
        temp_bin, cluster_id = meta
        comp = composition(record.sequence)
        entry = totals[temp_bin]
        entry[0] += 1
        entry[1] += len(record)
        for name, value in comp.items():
            entry[2][name] += value * len(record)
        if cluster_id:
            per_cluster[cluster_id][temp_bin].append(comp)

    bins = sorted(totals)
    if len(bins) < 2:
        print(f"need two growth-temperature classes, found {bins}", file=sys.stderr)
        return 1

    print("=== corpus-wide composition (confounded with phylogeny) ===")
    header = f"{'bin':<8}{'seqs':>10}{'mean len':>10}" + "".join(f"{n:>10}" for n in GROUPS)
    print(header)
    fractions: dict[str, dict[str, float]] = {}
    for name in bins:
        count, residues, weighted = totals[name]
        fractions[name] = {g: weighted[g] / residues for g in GROUPS}
        row = f"{name:<8}{count:>10,}{residues / count:>10.1f}"
        row += "".join(f"{fractions[name][g]:>9.3%}" for g in GROUPS)
        print(row)

    cold, hot = bins[0], bins[-1]
    print(f"\ndelta ({hot} - {cold}):")
    for group, (_, direction) in GROUPS.items():
        delta = fractions[hot][group] - fractions[cold][group]
        arrow = "up" if direction > 0 else "down"
        agrees = (delta > 0) == (direction > 0)
        print(f"  {group:<9} {delta:+.3%}   expected {arrow:<5} {'OK' if agrees else 'WRONG DIRECTION'}")

    mixed = [c for c, d in per_cluster.items() if cold in d and hot in d]
    print(f"\n=== within-cluster paired shift (controls for protein family) ===")
    print(f"clusters: {len(per_cluster):,} total, {len(mixed):,} with both classes "
          f"({len(mixed) / max(len(per_cluster), 1):.1%})")

    if not mixed:
        print("\nno cluster contains both classes, so the within-family signal cannot be "
              "measured. The model could only learn a between-family association, which "
              "is unlikely to transfer to a single family such as Cas9.", file=sys.stderr)
        return 3

    failures = []
    print(f"\n{'group':<10}{'mean':>10}{'median':>10}{'as expected':>14}")
    for group, (_, direction) in GROUPS.items():
        deltas = [
            statistics.mean(c[group] for c in per_cluster[cluster][hot])
            - statistics.mean(c[group] for c in per_cluster[cluster][cold])
            for cluster in mixed
        ]
        consistent = sum(1 for d in deltas if (d > 0) == (direction > 0)) / len(deltas)
        print(f"{group:<10}{statistics.mean(deltas):>+10.3%}{statistics.median(deltas):>+10.3%}"
              f"{consistent:>13.1%}")
        if consistent < args.min_consistency:
            failures.append(f"{group} consistent in only {consistent:.1%} of clusters")

    print(
        "\n'as expected' is the fraction of families shifting in the direction\n"
        "thermophily predicts. 50% is chance; the signal is real above that."
    )

    if failures:
        print(f"\nWEAK SIGNAL: {'; '.join(failures)}", file=sys.stderr)
        print(
            "Gate 2 asks the model to lower perplexity using the growth-temperature\n"
            "token. If the corpus does not separate the classes within a family, there\n"
            "is little for the token to explain and Gate 2 is likely to fail.",
            file=sys.stderr,
        )
        return 3

    print("\nSIGNAL PRESENT: the conditioning token has something to explain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
