#!/usr/bin/env python3
"""Stage A: alignment-level Cas9 screen, then cluster and split.

The annotation screen in 08_collect_broad.py catches anything labeled as
Cas9. This catches what annotation misses: unannotated or
differently-named homologs of the held-out Cas9 sequences.

This is the screen that protects the stage B leakage gate. That gate only
ever compares the Cas9 corpus against itself, so it cannot see a held-out
Cas9 that reached pre-training. If one did, the gate still passes and every
downstream number measures memorization. Thresholds here are deliberately
stricter than stage B's (30% identity over 50% coverage versus 45% over 80%):
a 400,000-sequence corpus hides a borderline homolog far more easily than a
1,000-sequence one, and over-excluding costs a few sequences while
under-excluding voids the experiment.

Usage:
    python scripts/09_screen_broad.py
    python scripts/09_screen_broad.py --skip-cluster
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config, GrowthTemperatureBinner  # noqa: E402
from thermocas.data.broad import Cas9Exclusion  # noqa: E402
from thermocas.data.cluster import (  # noqa: E402
    ClusterParams,
    SequenceMeta,
    SplitParams,
    assign_splits,
    mmseqs_cluster_commands,
    parse_cluster_tsv,
    verify_no_cluster_straddles_splits,
    write_split_files,
)
from thermocas.data.fasta import (  # noqa: E402
    FastaRecord,
    canonical_seq_id,
    read_fasta,
    write_fasta,
)
from thermocas.data.leakage import (  # noqa: E402
    LeakageParams,
    mmseqs_leakage_commands,
    parse_alignment_tsv,
)
from thermocas.data.search import require_tool  # noqa: E402


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {result.returncode}:\n{result.stderr.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="pretrain")
    parser.add_argument("--data-config", default="data")
    parser.add_argument("--sequences", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument("--heldout-cas9", type=Path, default=None,
                        help="FASTA of held-out Cas9 sequences to screen against")
    parser.add_argument("--out", type=Path, default=None, help="screened FASTA")
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    data_cfg = Config.load(args.data_config)
    binner = GrowthTemperatureBinner.from_config(data_cfg)
    exclusion = Cas9Exclusion.from_config(cfg)

    sequences = args.sequences or cfg.resolve_path("paths/sequences")
    label_table = args.label_table or cfg.resolve_path("paths/label_table")
    out_fasta = args.out or sequences.with_name(sequences.stem + "_screened.fasta")
    splits_dir = args.splits_dir or cfg.resolve_path("paths/splits_dir")

    if not sequences.is_file():
        print(f"stage A sequences not found: {sequences}", file=sys.stderr)
        print("run scripts/08_collect_broad.py first", file=sys.stderr)
        return 1

    # Canonicalize on the way in. MMseqs2 reports results under the bare
    # accession, so a corpus carrying full tr|ACC|NAME identifiers makes every
    # id comparison below silently fail to match -- which is how six Cas9
    # homologs survived an earlier run of this screen.
    records = [
        FastaRecord(canonical_seq_id(r.seq_id), r.description, r.sequence)
        for r in read_fasta(sequences)
    ]
    print(f"stage A corpus: {len(records):,} sequences from {sequences.name}")

    # Held-out Cas9 set. Prefer the real split; fall back to all seeds, since
    # screening against every seed is the conservative choice when the Cas9
    # corpus has not been built yet.
    heldout_path = args.heldout_cas9
    if heldout_path is None:
        cas9_splits = data_cfg.resolve_path("paths/splits_dir")
        test_ids_file = cas9_splits / "test.txt"
        cas9_fasta = data_cfg.resolve_path("paths/labeled_fasta")
        if test_ids_file.is_file() and cas9_fasta.is_file():
            ids = {ln.strip() for ln in test_ids_file.read_text().split() if ln.strip()}
            ids = {canonical_seq_id(i) for i in ids}
            heldout = [
                FastaRecord(canonical_seq_id(r.seq_id), r.description, r.sequence)
                for r in read_fasta(cas9_fasta)
                if canonical_seq_id(r.seq_id) in ids
            ]
            print(f"held-out Cas9: {len(heldout)} from the stage B test split")
        else:
            heldout = [
                FastaRecord(canonical_seq_id(r.seq_id), r.description, r.sequence)
                for r in read_fasta(data_cfg.resolve_path("paths/seeds"))
            ]
            print(f"held-out Cas9: {len(heldout)} reference seeds "
                  "(no Cas9 test split yet; screening against all seeds is the "
                  "conservative choice)")
    else:
        heldout = [
            FastaRecord(canonical_seq_id(r.seq_id), r.description, r.sequence)
            for r in read_fasta(heldout_path)
        ]
        print(f"held-out Cas9: {len(heldout)} from {heldout_path.name}")

    if not heldout:
        print("no held-out Cas9 sequences to screen against", file=sys.stderr)
        return 1

    print(f"screen: identity > {exclusion.max_identity_to_heldout:.0%} over "
          f">= {exclusion.alignment_coverage:.0%} coverage")
    stage_b = LeakageParams.from_config(data_cfg)
    print(f"  (stage B leakage gate is looser at {stage_b.max_identity_to_train:.0%} / "
          f"{stage_b.alignment_coverage:.0%}; this screen must be stricter)")

    # Align the stage A corpus against the held-out Cas9 sequences.
    require_tool("mmseqs")
    excluded: set[str] = set()
    worst: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="thermocas-screen-") as tmp:
        workdir = Path(tmp)
        query = workdir / "broad.fasta"
        target = workdir / "heldout.fasta"
        write_fasta(query, records)
        write_fasta(target, heldout)
        alignment = workdir / "aln.tsv"
        try:
            for command in mmseqs_leakage_commands(
                query, target, alignment, workdir, threads=args.threads
            ):
                run(command)
        except RuntimeError as exc:
            print(f"alignment failed: {exc}", file=sys.stderr)
            return 1

        if alignment.is_file():
            for pair in parse_alignment_tsv(alignment):
                worst[pair.query_id] = max(worst.get(pair.query_id, 0.0), pair.identity)
                if (
                    pair.coverage >= exclusion.alignment_coverage
                    and pair.identity > exclusion.max_identity_to_heldout
                ):
                    excluded.add(pair.query_id)

    print(f"\nalignment screen: {len(excluded)} sequence(s) excluded as Cas9 homologs")
    if excluded:
        ranked = sorted(excluded, key=lambda i: -worst.get(i, 0.0))[:10]
        for seq_id in ranked:
            print(f"  {seq_id}: identity {worst[seq_id]:.1%} to a held-out Cas9")

    kept = [r for r in records if r.seq_id not in excluded]

    # The screen reporting exclusions while writing every input sequence is
    # exactly the silent failure this guard exists for: it happened once,
    # because identifiers on the two sides of the comparison had different
    # formats, and nothing checked that the counts agreed.
    expected = len(records) - len(excluded)
    if len(kept) != expected:
        unmatched = sorted(excluded - {r.seq_id for r in records})
        print(
            f"\nID MISMATCH: {len(excluded)} sequence(s) were flagged for exclusion but "
            f"only {len(records) - len(kept)} could be removed from the corpus.\n"
            f"  {len(unmatched)} flagged id(s) do not appear in the corpus, e.g. "
            f"{', '.join(unmatched[:5])}\n"
            "  The alignment and the corpus are using different identifier formats, so "
            "Cas9 homologs would survive into pre-training. Refusing to write.",
            file=sys.stderr,
        )
        return 4

    write_fasta(out_fasta, kept)
    print(f"\nwrote {len(kept):,} screened sequences to {out_fasta} "
          f"({len(excluded)} excluded)")

    if exclusion.fail_on_violation and excluded and not kept:
        print("screening removed everything", file=sys.stderr)
        return 2

    # Update the label table to drop screened-out rows.
    labels: dict[str, dict[str, str]] = {}
    if label_table.is_file():
        with label_table.open(newline="") as handle:
            labels = {}
            for row in csv.DictReader(handle):
                row["seq_id"] = canonical_seq_id(row["seq_id"])
                labels[row["seq_id"]] = row
        for seq_id in excluded:
            labels.pop(seq_id, None)

    if args.skip_cluster:
        print("\n--skip-cluster: stopping before clustering")
        return 0

    # Cluster and split.
    cluster_params = ClusterParams.from_config(cfg)
    split_params = SplitParams.from_config(cfg)
    print(f"\nclustering at {cluster_params.min_seq_id:.0%} identity / "
          f"{cluster_params.coverage:.0%} coverage")

    splits_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="thermocas-bcluster-") as tmp:
        workdir = Path(tmp)
        prefix = workdir / "clusters"
        try:
            for command in mmseqs_cluster_commands(
                cluster_params, out_fasta, prefix, workdir, args.threads
            ):
                run(command)
        except RuntimeError as exc:
            print(f"clustering failed: {exc}", file=sys.stderr)
            return 1
        cluster_tsv = splits_dir / "clusters.tsv"
        cluster_tsv.write_text(prefix.with_suffix(".tsv").read_text())

    assignments_by_id = parse_cluster_tsv(cluster_tsv)
    n_clusters = len(set(assignments_by_id.values()))
    print(f"  {n_clusters:,} clusters from {len(assignments_by_id):,} sequences")

    # MMseqs2 keys its cluster table on the bare accession. If the corpus was
    # written with full sp|ACC|NAME identifiers, nothing would join and the
    # split would come out empty, so reconcile by canonical id rather than
    # failing on an id-format mismatch.
    matched = sum(1 for r in kept if r.seq_id in assignments_by_id)
    if matched == 0 and kept:
        canonical = {canonical_seq_id(k): v for k, v in assignments_by_id.items()}
        remapped = {
            r.seq_id: canonical[canonical_seq_id(r.seq_id)]
            for r in kept
            if canonical_seq_id(r.seq_id) in canonical
        }
        if remapped:
            print(f"  reconciled {len(remapped):,} identifiers by canonical accession")
            assignments_by_id = {**assignments_by_id, **remapped}

    metas = [
        SequenceMeta(
            seq_id=r.seq_id,
            cluster_id=assignments_by_id[r.seq_id],
            temp_bin=labels.get(r.seq_id, {}).get("temp_bin", "unlabeled"),
        )
        for r in kept
        if r.seq_id in assignments_by_id
    ]
    if not metas:
        print("nothing to split", file=sys.stderr)
        return 1

    assignments, report = assign_splits(metas, split_params)
    straddling = verify_no_cluster_straddles_splits(metas, assignments)
    if straddling:
        print(f"REFUSING TO WRITE: {len(straddling)} cluster(s) straddle splits", file=sys.stderr)
        return 3

    print(f"\nsplit assignment:\n{report.summary()}")
    write_split_files(splits_dir, assignments)
    print(f"wrote split files to {splits_dir}")

    if labels:
        for seq_id, split in assignments.items():
            if seq_id in labels:
                labels[seq_id]["split"] = split
                labels[seq_id]["cluster_id"] = assignments_by_id.get(seq_id, "")
        with label_table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(labels.values())).keys()))
            writer.writeheader()
            writer.writerows(labels.values())
        print(f"updated {label_table}")

    # Stage A gate.
    clusters_per_bin: dict[str, set[str]] = {}
    sequences_per_bin: dict[str, int] = {}
    for meta in metas:
        clusters_per_bin.setdefault(meta.temp_bin, set()).add(meta.cluster_id)
        sequences_per_bin[meta.temp_bin] = sequences_per_bin.get(meta.temp_bin, 0) + 1

    min_clusters = int(cfg["gate/min_clusters_per_bin"])
    min_sequences = int(cfg["gate/min_sequences_per_bin"])
    print(f"\nstage A gate (>= {min_clusters} clusters and >= {min_sequences:,} sequences per bin):")
    failures = []
    for name in binner.names:
        n_clu = len(clusters_per_bin.get(name, ()))
        n_seq = sequences_per_bin.get(name, 0)
        ok = n_clu >= min_clusters and n_seq >= min_sequences
        print(f"  {name:<8} {n_clu:>7,} clusters  {n_seq:>9,} sequences  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    if failures:
        print(
            f"\nstage A gate fails for {', '.join(failures)}. Raise "
            "species.max_per_bin or sampling.max_proteins_per_species and re-run.",
            file=sys.stderr,
        )
        return 3

    print("\nstage A gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
