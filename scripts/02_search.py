#!/usr/bin/env python3
"""Stage 1b: homolog search of the Cas9 seeds against a protein database.

A blastp-style protein-vs-protein search (DIAMOND or MMseqs2), then the
configured e-value and coverage thresholds are re-applied to the resulting hit
table so the committed table is verifiably consistent with the config rather
than with whatever flags were typed by hand.

The reference database is a large download and is never fetched by this
script: pass an existing one with --database. On a 16 GB machine a UniRef50
DIAMOND index does not fit comfortably alongside the OS, so this stage is
expected to run on a cloud machine; --dry-run prints the exact commands to
take there.

Usage:
    python scripts/02_search.py --dry-run --database /path/uniref50.fasta
    python scripts/02_search.py --database /path/uniref50.fasta
    python scripts/02_search.py --database db.fasta --index-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.fasta import read_fasta  # noqa: E402
from thermocas.data.search import (  # noqa: E402
    HIT_COLUMNS,
    SearchParams,
    SearchToolError,
    diamond_blastp_command,
    diamond_makedb_command,
    filter_hits,
    mmseqs_search_commands,
    parse_hits,
    require_tool,
    tool_version,
    unique_subjects,
    write_hits,
)

TOOL_EXECUTABLE = {"diamond": "diamond", "mmseqs2": "mmseqs"}


def render(command: list[str]) -> str:
    return " ".join(command)


def run(command: list[str], dry_run: bool) -> None:
    print(f"  $ {render(command)}", flush=True)
    if dry_run:
        return
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SearchToolError(
            f"{command[0]} exited with status {result.returncode}; see its output above"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="data")
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="existing protein FASTA to search against; never downloaded here",
    )
    parser.add_argument("--index", type=Path, default=None, help="DIAMOND index prefix to build or reuse")
    parser.add_argument("--seeds", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--index-only", action="store_true", help="build the index and stop")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    params = SearchParams.from_config(cfg)
    seeds_path = args.seeds or cfg.resolve_path("paths/seeds")
    out_path = args.out or cfg.resolve_path("paths/hits")

    if not seeds_path.is_file():
        print(f"seeds not found: {seeds_path}\nrun scripts/01_collect_seeds.py first", file=sys.stderr)
        return 1
    if not args.dry_run and not args.database.exists():
        print(
            f"database not found: {args.database}\n"
            "This script never downloads a reference database. Fetch it on the machine "
            "that will run the search and record its version and date in PROJECT_LOG.md.",
            file=sys.stderr,
        )
        return 1

    seeds = list(read_fasta(seeds_path))
    executable = TOOL_EXECUTABLE[params.tool]

    print(f"tool:      {params.tool} ({executable})")
    print(f"database:  {args.database}")
    print(f"seeds:     {len(seeds)} from {seeds_path}")
    print(f"evalue:    <= {params.max_evalue:g}")
    print(f"coverage:  >= {params.min_query_coverage:.0%} of the query")
    print(f"threads:   {params.threads}")

    if not args.dry_run:
        require_tool(executable)
        version = tool_version(executable)
        print(f"version:   {version}")
        print(f"record in PROJECT_LOG.md: {executable} {version}, database {args.database.name}, {date.today().isoformat()}")

    print("\ncommands:")
    try:
        if params.tool == "diamond":
            index = args.index or args.database.with_suffix(".dmnd")
            if index.exists() and not args.dry_run:
                print(f"  reusing existing index {index}")
            else:
                run(diamond_makedb_command(args.database, index.with_suffix(""), params.threads), args.dry_run)
            if args.index_only:
                return 0
            run(diamond_blastp_command(params, index.with_suffix(""), seeds_path, out_path), args.dry_run)
        else:
            if args.index_only:
                print("  --index-only has no separate step for mmseqs2", file=sys.stderr)
            with tempfile.TemporaryDirectory(prefix="thermocas-mmseqs-") as tmp:
                for command in mmseqs_search_commands(
                    params, seeds_path, args.database, out_path, Path(tmp)
                ):
                    run(command, args.dry_run)
    except SearchToolError as exc:
        print(f"\nsearch failed: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"\nwould write filtered hits to {out_path}")
        return 0

    if not out_path.is_file():
        print(f"\n{params.tool} produced no output at {out_path}", file=sys.stderr)
        return 1

    raw_path = out_path.with_suffix(out_path.suffix + ".raw")
    out_path.replace(raw_path)
    hits, counts = filter_hits(parse_hits(raw_path, params.tool), params)
    write_hits(out_path, hits, params)

    print(f"\nparsed {counts['total']} hits from {raw_path.name}")
    print(f"  failed e-value:  {counts['failed_evalue']}")
    print(f"  failed coverage: {counts['failed_coverage']}")
    print(f"  kept:            {counts['kept']}")
    print(f"  unique subjects: {len(unique_subjects(hits))}")
    print(f"\nwrote {out_path}")

    if not hits:
        print(
            "\nno hits survived filtering; loosen search thresholds in the config "
            "or check that the database is a protein FASTA",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
