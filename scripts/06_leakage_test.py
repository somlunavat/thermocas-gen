#!/usr/bin/env python3
"""Stage 1f: the leakage gate. Must pass before any training run.

Verifies that no held-out sequence is a close homolog of a training sequence.
Two checks, cheapest first:

1. Split disjointness: no sequence id appears in two splits. An id in both
   train and test is leakage no identity threshold would catch, because the
   sequences are not similar, they are the same.
2. Alignment: searches val and test against train with MMseqs2 and flags any
   pair above the configured identity over the configured coverage.

Clustering at 45% identity should already make a violation impossible. This
script does not assume that. The clustering is the mechanism and this is the
verification, kept separate so a clustering mistake cannot disable its own
check.

Exits non-zero on any violation when leakage.fail_on_violation is set.

Usage:
    python scripts/06_leakage_test.py --sequences data/processed/cas9_labeled.fasta
    python scripts/06_leakage_test.py --sequences f.fasta --alignments existing.tsv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.fasta import read_fasta, write_fasta  # noqa: E402
from thermocas.data.leakage import (  # noqa: E402
    LeakageParams,
    check_leakage,
    check_split_disjointness,
    mmseqs_leakage_commands,
    parse_alignment_tsv,
)
from thermocas.data.search import require_tool  # noqa: E402


def read_split_ids(splits_dir: Path, name: str) -> set[str]:
    path = splits_dir / f"{name}.txt"
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} exited {result.returncode}:\n{result.stderr.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="data")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--alignments", type=Path, default=None, help="reuse an alignment TSV")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    params = LeakageParams.from_config(cfg)
    splits_dir = args.splits_dir or cfg.resolve_path("paths/splits_dir")

    if not args.sequences.is_file():
        print(f"sequences not found: {args.sequences}", file=sys.stderr)
        return 1

    splits = {name: read_split_ids(splits_dir, name) for name in ("train", "val", "test")}
    if not splits["train"]:
        print(
            f"no train split found in {splits_dir}\nrun scripts/05_cluster_split.py first",
            file=sys.stderr,
        )
        return 1

    print("leakage gate")
    print(f"  splits:    train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"  threshold: identity > {params.max_identity_to_train:.0%} "
          f"over >= {params.alignment_coverage:.0%} query coverage")

    # Check 1: disjointness.
    problems = check_split_disjointness({k: v for k, v in splits.items() if v})
    print("\ncheck 1: split disjointness")
    if problems:
        for problem in problems:
            print(f"  FAIL {problem}", file=sys.stderr)
        if params.fail_on_violation:
            return 3
    else:
        print("  PASS no sequence id appears in two splits")

    held_out = splits["val"] | splits["test"]
    if not held_out:
        print("\nno held-out sequences; nothing to align", file=sys.stderr)
        return 0

    # Check 2: alignment.
    print("\ncheck 2: alignment of held-out sequences against train")
    records = {r.seq_id: r for r in read_fasta(args.sequences)}
    missing = (splits["train"] | held_out) - set(records)
    if missing:
        print(
            f"  {len(missing)} split id(s) absent from {args.sequences.name}, "
            f"e.g. {', '.join(sorted(missing)[:5])}",
            file=sys.stderr,
        )
        return 1

    if args.alignments is not None:
        alignment_tsv = args.alignments
        print(f"  reusing {alignment_tsv}")
    else:
        require_tool("mmseqs")
        with tempfile.TemporaryDirectory(prefix="thermocas-leakage-") as tmp:
            workdir = Path(tmp)
            query_fasta = workdir / "heldout.fasta"
            target_fasta = workdir / "train.fasta"
            write_fasta(query_fasta, [records[i] for i in sorted(held_out)])
            write_fasta(target_fasta, [records[i] for i in sorted(splits["train"])])
            alignment_tsv = workdir / "aln.tsv"
            try:
                for command in mmseqs_leakage_commands(
                    query_fasta, target_fasta, alignment_tsv, workdir, threads=args.threads
                ):
                    run(command)
            except RuntimeError as exc:
                print(f"  alignment failed: {exc}", file=sys.stderr)
                return 1
            if not alignment_tsv.is_file():
                print("  mmseqs produced no alignment output", file=sys.stderr)
                return 1
            report = check_leakage(parse_alignment_tsv(alignment_tsv), params)
            _emit(report, params, args.report)
            return _verdict(report, params)

    report = check_leakage(parse_alignment_tsv(alignment_tsv), params)
    _emit(report, params, args.report)
    return _verdict(report, params)


def _emit(report, params, report_path: Path | None) -> None:
    for line in report.summary(params).splitlines():
        print(f"  {line}")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w") as handle:
            handle.write(report.summary(params) + "\n")
            handle.write("\nquery_id\ttarget_id\tidentity\tcoverage\tviolates\n")
            for pair in sorted(report.worst_by_query.values(), key=lambda p: -p.identity):
                handle.write(
                    f"{pair.query_id}\t{pair.target_id}\t{pair.identity:.4f}\t"
                    f"{pair.coverage:.4f}\t{int(pair.violates(params))}\n"
                )
        print(f"  wrote {report_path}")


def _verdict(report, params) -> int:
    if report.passed:
        print("\nLEAKAGE GATE PASSED")
        return 0
    print(
        f"\nLEAKAGE GATE FAILED: {len(report.leaked_queries)} held-out sequence(s) are close "
        f"homologs of training sequences.\n"
        "Re-cluster at a lower identity threshold (cluster.min_seq_id in configs/data.yaml) "
        "and re-split. Do not train on this split.",
        file=sys.stderr,
    )
    return 3 if params.fail_on_violation else 0


if __name__ == "__main__":
    raise SystemExit(main())
