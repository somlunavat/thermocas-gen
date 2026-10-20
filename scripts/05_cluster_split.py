#!/usr/bin/env python3
"""Stage 1e: cluster the corpus and split train/val/test by cluster.

Clusters with MMseqs2, then assigns whole clusters to splits so that a
sequence and its near-duplicate neighbour cannot land on opposite sides. The
held-out subfamily and the thermophilic positive controls are forced into
test, along with every other member of their clusters.

Refuses to write anything if a cluster would straddle two splits.

Usage:
    python scripts/05_cluster_split.py --sequences data/processed/cas9_labeled.fasta
    python scripts/05_cluster_split.py --sequences f.fasta --clusters existing.tsv
    python scripts/05_cluster_split.py --sequences f.fasta --dry-run
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

from thermocas.config import Config  # noqa: E402
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
from thermocas.data.fasta import read_fasta  # noqa: E402
from thermocas.data.search import require_tool, tool_version  # noqa: E402


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return {row["seq_id"]: row for row in csv.DictReader(handle)}


def run(command: list[str], dry_run: bool) -> None:
    print(f"  $ {' '.join(command)}", flush=True)
    if dry_run:
        return
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {result.returncode}:\n{result.stderr.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="data")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="label_table.csv from stage 1d")
    parser.add_argument("--clusters", type=Path, default=None, help="reuse an existing cluster TSV")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cluster_params = ClusterParams.from_config(cfg)
    split_params = SplitParams.from_config(cfg)
    labels_path = args.labels or cfg.resolve_path("paths/label_table")
    out_dir = args.out_dir or cfg.resolve_path("paths/splits_dir")

    if not args.sequences.is_file():
        print(f"sequences not found: {args.sequences}", file=sys.stderr)
        return 1

    records = list(read_fasta(args.sequences))
    labels = load_labels(labels_path)

    print(f"sequences: {len(records)} from {args.sequences}")
    print(f"labels:    {len(labels)} from {labels_path}" if labels else f"labels:    none at {labels_path}")
    print(f"cluster:   {cluster_params.min_seq_id:.0%} identity, {cluster_params.coverage:.0%} coverage")
    print(f"split:     {split_params.fractions}, seed {split_params.seed}")
    print(f"holdouts:  subfamily={split_params.holdout_subfamily}, "
          f"positive_controls={split_params.holdout_positive_controls} {split_params.positive_controls}")

    # Cluster
    if args.clusters is not None:
        cluster_tsv = args.clusters
        print(f"\nreusing clusters from {cluster_tsv}")
    else:
        require_tool("mmseqs")
        if not args.dry_run:
            print(f"\nmmseqs:    {tool_version('mmseqs')}")
        print("\ncommands:")
        with tempfile.TemporaryDirectory(prefix="thermocas-cluster-") as tmp:
            workdir = Path(tmp)
            prefix = workdir / "clusters"
            try:
                for command in mmseqs_cluster_commands(
                    cluster_params, args.sequences, prefix, workdir, args.threads
                ):
                    run(command, args.dry_run)
            except RuntimeError as exc:
                print(f"\nclustering failed: {exc}", file=sys.stderr)
                return 1
            if args.dry_run:
                print(f"\nwould write split files to {out_dir}")
                return 0
            cluster_tsv = out_dir / "clusters.tsv"
            out_dir.mkdir(parents=True, exist_ok=True)
            cluster_tsv.write_text(prefix.with_suffix(".tsv").read_text())
            print(f"\nwrote {cluster_tsv}")

    try:
        assignments_by_id = parse_cluster_tsv(cluster_tsv)
    except ValueError as exc:
        print(f"could not read clusters: {exc}", file=sys.stderr)
        return 1

    n_clusters = len(set(assignments_by_id.values()))
    print(f"clusters:  {n_clusters} from {len(assignments_by_id)} sequences")

    # Build splitter input
    control_names = {name.lower() for name in split_params.positive_controls}
    metas: list[SequenceMeta] = []
    unclustered = 0
    for record in records:
        cluster_id = assignments_by_id.get(record.seq_id)
        if cluster_id is None:
            unclustered += 1
            continue
        row = labels.get(record.seq_id, {})
        description = f"{record.seq_id} {record.description}".lower()
        subfamily = None
        for token in record.description.split("|"):
            token = token.strip()
            if token.startswith("type="):
                subfamily = token.split("=", 1)[1].strip()
        metas.append(
            SequenceMeta(
                seq_id=record.seq_id,
                cluster_id=cluster_id,
                temp_bin=row.get("temp_bin") or "unlabeled",
                subfamily=subfamily,
                is_positive_control=any(name in description for name in control_names),
            )
        )

    if unclustered:
        print(f"  WARNING {unclustered} sequence(s) absent from the cluster table; not split",
              file=sys.stderr)
    if not metas:
        print("no sequences could be split", file=sys.stderr)
        return 1

    assignments, report = assign_splits(metas, split_params)

    straddling = verify_no_cluster_straddles_splits(metas, assignments)
    if straddling:
        print(
            f"\nREFUSING TO WRITE: {len(straddling)} cluster(s) straddle splits: "
            f"{', '.join(straddling[:5])}\n"
            "A straddling cluster is leakage by construction.",
            file=sys.stderr,
        )
        return 3

    print(f"\nsplit assignment:\n{report.summary()}")

    written = write_split_files(out_dir, assignments)
    for name, path in written.items():
        print(f"wrote {path}")

    # Fold the assignments back into the label table so downstream stages
    # read one file rather than joining two.
    if labels:
        for seq_id, split in assignments.items():
            if seq_id in labels:
                labels[seq_id]["split"] = split
                labels[seq_id]["cluster_id"] = assignments_by_id.get(seq_id, "")
        with labels_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(labels.values())).keys()))
            writer.writeheader()
            writer.writerows(labels.values())
        print(f"updated {labels_path} with cluster_id and split")

    print("\nrun scripts/06_leakage_test.py before any training run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
