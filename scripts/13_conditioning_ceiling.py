#!/usr/bin/env python3
"""How much perplexity gain is a growth-temperature token worth at all?

Gate 2 requires validation perplexity with the correct growth-temperature
token to beat perplexity with a shuffled one by 2%. That threshold was set by
assertion. This script asks what the ceiling actually is, which is a property
of the corpus and needs no GPU.

The argument. A model that knows the bin scores each residue under
``p(residue | bin)``; one that does not scores it under the marginal
``p(residue)``. The best possible advantage is the mutual information between
residue identity and bin,

    I(residue; bin) = H(marginal) - sum_bin P(bin) H(p_bin)

and since perplexity is the exponential of cross-entropy, the relative gain a
perfect conditional model could show is

    1 - exp(-I)

which for small I is approximately I in nats. This is a *unigram* ceiling: a
real language model also uses context, which can only help it. But it
establishes the order of magnitude, and if the measured gain is close to it
then the model has learned essentially all the first-order signal available
and the threshold was never reachable.

Two versions, because they answer different questions:

**Corpus-wide** pools every sequence. It overstates what the token is worth to
a language model, because a thermophilic protein's own residues already
announce that it is thermophilic -- the model can infer the bin from the
prefix, making the token redundant rather than informative.

**Within-cluster** restricts to clusters holding both classes and computes the
information per cluster before averaging. This is the fairer comparison: it
asks what the token adds *given that the protein family is already known*,
which is the situation a language model is actually in a few dozen residues
into a sequence.

Usage:
    python scripts/13_conditioning_ceiling.py
    python scripts/13_conditioning_ceiling.py --config pretrain
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.fasta import read_fasta  # noqa: E402

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def entropy(counts: dict[str, int]) -> float:
    """Shannon entropy in nats of a residue count table."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    out = 0.0
    for n in counts.values():
        if n:
            p = n / total
            out -= p * math.log(p)
    return out


def mutual_information(by_bin: dict[str, dict[str, int]]) -> tuple[float, int]:
    """I(residue; bin) in nats, and the residue count it rests on."""
    totals = {bin_name: sum(c.values()) for bin_name, c in by_bin.items()}
    grand = sum(totals.values())
    if grand == 0 or len([b for b, n in totals.items() if n]) < 2:
        return 0.0, grand

    marginal: dict[str, int] = collections.defaultdict(int)
    for counts in by_bin.values():
        for residue, n in counts.items():
            marginal[residue] += n

    conditional = sum(
        (totals[bin_name] / grand) * entropy(counts)
        for bin_name, counts in by_bin.items()
        if totals[bin_name]
    )
    return entropy(marginal) - conditional, grand


def gain_from_information(nats: float) -> float:
    """The relative perplexity gain a perfect conditional model would show."""
    return 1.0 - math.exp(-nats)


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["seq_id"]: row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="pretrain")
    parser.add_argument(
        "--min-cluster-residues",
        type=int,
        default=2000,
        help="skip clusters with fewer residues per class; tiny clusters give "
             "mutual information estimates dominated by sampling noise, which "
             "is upward-biased",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    label_table = cfg.resolve_path("paths/label_table")
    if not label_table.exists():
        # The Cas9 corpus does not exist until the search and labelling
        # pipeline has been run, and a traceback here reads like a bug in the
        # arithmetic rather than a step that has not happened yet.
        print(
            f"no label table at {label_table}\n"
            f"The '{args.config}' corpus has not been built yet, so there is "
            "nothing to measure an\ninformation ceiling over. Build it first "
            "(02_search.py, then the labelling and\nscreening steps in "
            "README.md), or pass --config pretrain for the stage A corpus.",
            file=sys.stderr,
        )
        return 2
    rows = load_rows(label_table)
    sequences = cfg.resolve_path(
        "paths/screened_sequences"
        if "paths/screened_sequences" in cfg
        else "paths/labeled_fasta"
    )

    corpus: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    per_cluster: dict[str, dict[str, dict[str, int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(int))
    )

    kept = 0
    for record in read_fasta(sequences):
        row = rows.get(record.seq_id)
        if not row:
            continue
        bin_name = row.get("temp_bin") or ""
        cluster = row.get("cluster_id") or ""
        if not bin_name:
            continue
        kept += 1
        for residue in record.sequence:
            if residue in RESIDUES:
                corpus[bin_name][residue] += 1
                if cluster:
                    per_cluster[cluster][bin_name][residue] += 1

    print(f"sequences: {kept:,} from {sequences.name}")
    bins = ", ".join(
        f"{b}={sum(c.values()):,} residues" for b, c in sorted(corpus.items())
    )
    print(f"bins:      {bins}")

    info, residues = mutual_information(corpus)
    print("\n=== corpus-wide ceiling (overstates it; see the docstring) ===")
    print(f"  I(residue; bin)            {info:.6f} nats over {residues:,} residues")
    print(f"  best possible ppl gain     {gain_from_information(info):+.3%}")

    usable = []
    for by_bin in per_cluster.values():
        totals = {b: sum(c.values()) for b, c in by_bin.items()}
        if len(totals) < 2 or min(totals.values()) < args.min_cluster_residues:
            continue
        cluster_info, cluster_residues = mutual_information(by_bin)
        usable.append((cluster_info, cluster_residues))

    print("\n=== within-cluster ceiling (the token's value given the family) ===")
    if not usable:
        print(f"  no cluster has {args.min_cluster_residues:,}+ residues in both "
              "classes; nothing to estimate from")
        return 1

    weight = sum(n for _, n in usable)
    weighted = sum(i * n for i, n in usable) / weight
    print(f"  clusters used               {len(usable):,} "
          f"(both classes, >={args.min_cluster_residues:,} residues each)")
    print(f"  I(residue; bin | family)    {weighted:.6f} nats over {weight:,} residues")
    print(f"  best possible ppl gain      {gain_from_information(weighted):+.3%}")

    print("\n=== against the gate ===")
    train_cfg = Config.load("train")
    section = train_cfg.section("conditioning_gate")
    configured = (
        float(section["ceiling_gain"])
        if "ceiling_gain" in section and section["ceiling_gain"]
        else None
    )
    ceiling = gain_from_information(weighted)
    configured_corpus = (
        str(section["ceiling_corpus"])
        if "ceiling_corpus" in section and section["ceiling_corpus"]
        else None
    )
    print(f"  within-cluster ceiling      {ceiling:+.3%}")
    print(f"  put this in configs/train.yaml as conditioning_gate/ceiling_gain:"
          f"  {weighted:.5f}")
    print(f"  ...and conditioning_gate/ceiling_corpus: {args.config}")
    if configured is None:
        print("\n  no ceiling_gain is set, so Gate 2 cannot judge effect size.")
    elif configured_corpus != args.config:
        # Not a staleness problem: the configured number is about a different
        # corpus, so comparing the two values would be meaningless either way.
        print(f"\n  the configured ceiling belongs to the "
              f"'{configured_corpus}' corpus, not '{args.config}', so there "
              "is nothing\n  to compare. Gate 2 refuses a cross-corpus "
              "ceiling; see Gate2Params.check_corpus.")
    elif abs(configured - ceiling) / max(ceiling, 1e-12) > 0.10:
        print(f"\n  CONFIG IS STALE: conditioning_gate/ceiling_gain is "
              f"{configured:+.3%}, more than\n  10% away from the {ceiling:+.3%} "
              "this corpus gives. A ceiling measured on a\n  different corpus "
              "makes the effect-size criterion meaningless.")
    else:
        print(f"\n  configs/train.yaml agrees ({configured:+.3%}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
