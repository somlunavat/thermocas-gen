#!/usr/bin/env python3
"""Stage 1c: length filter and catalytic domain confirmation.

Takes the homolog sequences, applies the configured length and residue filter,
then confirms with HMMER that each one carries both nuclease domains (RuvC and
HNH). Writes the surviving sequences and a per-sequence domain report.

Every rejection is counted and reported with a reason. Nothing is dropped
silently.

Usage:
    python scripts/03_filter_domains.py --sequences data/processed/hits.fasta
    python scripts/03_filter_domains.py --sequences data/raw/seeds.fasta --report-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.domains import (  # noqa: E402
    DomainRequirements,
    HmmerError,
    call_domains,
    concatenate_profiles,
    parse_domtblout,
    run_hmmsearch,
)
from thermocas.data.fasta import read_fasta, write_fasta  # noqa: E402
from thermocas.data.sequence_filter import SequenceFilter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="data")
    parser.add_argument(
        "--sequences", type=Path, required=True, help="input FASTA of candidate homologs"
    )
    parser.add_argument("--out", type=Path, default=None, help="output FASTA of survivors")
    parser.add_argument("--report", type=Path, default=None, help="per-sequence domain report CSV")
    parser.add_argument("--workdir", type=Path, default=None, help="where to put HMMER artifacts")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument(
        "--report-only", action="store_true", help="report without writing a filtered FASTA"
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    requirements = DomainRequirements.from_config(cfg)
    seq_filter = SequenceFilter.from_config(cfg)

    if not args.sequences.is_file():
        print(f"input not found: {args.sequences}", file=sys.stderr)
        return 1

    out_path = args.out or cfg.resolve_path("paths/labeled_fasta").with_name("cas9_domain_filtered.fasta")
    report_path = args.report or out_path.with_name("domain_report.csv")
    workdir = args.workdir or out_path.parent / "hmmer"
    workdir.mkdir(parents=True, exist_ok=True)

    records = list(read_fasta(args.sequences))
    print(f"input:     {len(records)} sequences from {args.sequences}")
    print(f"length:    {seq_filter.min_length}-{seq_filter.max_length} aa")
    print(f"required:  {' + '.join(requirements.required)}")
    for name in requirements.required:
        print(f"  {name}: any of {', '.join(requirements.profiles[name])}")
    print(f"i-Evalue:  <= {requirements.max_evalue:g}")

    # Length and residue filter first: HMMER on a 100k-sequence corpus is the
    # expensive step, so there is no reason to search sequences that are
    # already excluded.
    length_passed, length_report = seq_filter.apply(records)
    print(f"\nlength and residue filter:\n{length_report.summary()}")

    if not length_passed:
        print("\nno sequences survived the length filter", file=sys.stderr)
        return 2

    staged_fasta = workdir / "length_passed.fasta"
    write_fasta(staged_fasta, length_passed)

    profiles_hmm = workdir / "profiles.hmm"
    profile_dir = cfg.resolve_path("domains/profile_dir")
    try:
        versions = concatenate_profiles(requirements.all_accessions(), profile_dir, profiles_hmm)
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print(f"\nprofiles:  {len(requirements.all_accessions())} from {profile_dir}")
    print(f"  versions (record in PROJECT_LOG.md): {', '.join(versions)}")

    domtbl = workdir / "domains.domtblout"
    try:
        command = run_hmmsearch(
            profiles_hmm, staged_fasta, domtbl, requirements.max_evalue, cpu=args.cpu
        )
    except HmmerError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print(f"  $ {' '.join(command)}")

    calls = call_domains(parse_domtblout(domtbl), requirements)

    kept = []
    missing_counts: Counter[str] = Counter()
    rows = []
    for record in length_passed:
        call = calls.get(record.seq_id)
        missing = requirements.required if call is None else call.missing(requirements)
        passed = not missing
        if passed:
            kept.append(record)
        else:
            missing_counts[",".join(missing)] += 1
        rows.append(
            {
                "seq_id": record.seq_id,
                "length": len(record),
                "passed": int(passed),
                "missing_domains": ";".join(missing),
                **{
                    f"{name}_evalue": (
                        f"{call.best_evalue[name]:g}" if call and name in call.best_evalue else ""
                    )
                    for name in requirements.required
                },
                **{
                    f"{name}_profile": (call.best_profile.get(name, "") if call else "")
                    for name in requirements.required
                },
                "informational": ";".join(
                    f"{k}={v:g}" for k, v in sorted((call.informational if call else {}).items())
                ),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\ndomain confirmation: {len(kept)}/{len(length_passed)} carry {' + '.join(requirements.required)}")
    for reason, count in missing_counts.most_common():
        print(f"  rejected {count:>7}  missing {reason}")
    print(f"\nwrote report {report_path}")

    if not args.report_only:
        write_fasta(out_path, kept)
        print(f"wrote {len(kept)} sequences to {out_path}")

    if not kept:
        print(
            "\nnothing passed domain confirmation; check that the Pfam profiles in "
            "configs/data.yaml match this protein family",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
