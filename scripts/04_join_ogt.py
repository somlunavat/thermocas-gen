#!/usr/bin/env python3
"""Stage 1d: join sequences to growth-temperature labels, and run Gate 1.

Builds ``label_table.csv`` by looking up each sequence's source organism in the
Gosha OGT database, binning the temperature, and recording whether independent
culture collections agree on that bin.

Gate 1 is the brief's data audit: if any growth-temperature bin holds fewer
sequences than ``growth_temperature.min_per_bin``, the script reports and exits
non-zero rather than letting training proceed on a corpus that cannot support
the claim.

Usage:
    python scripts/04_join_ogt.py --fetch                    # download Gosha
    python scripts/04_join_ogt.py --sequences data/raw/seeds.fasta
    python scripts/04_join_ogt.py --sequences f.fasta --strict-crosscheck
"""

from __future__ import annotations

import argparse
import csv
import gzip
import ssl
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config, GrowthTemperatureBinner  # noqa: E402
from thermocas.data.fasta import read_fasta  # noqa: E402
from thermocas.data.ogt import (  # noqa: E402
    JoinReport,
    OgtTable,
    gate1_failures,
    normalize_organism,
    parse_organism_from_header,
)

# figshare DOI 10.6084/m9.figshare.23257940, CC BY 4.0.
GOSHA_FILES = {
    "all_species_temperatures.csv": "https://ndownloader.figshare.com/files/40987052",
    "archaea_ncbi_temperatures.csv": "https://ndownloader.figshare.com/files/40987055",
    "bacteria_ncbi_temperatures.csv": "https://ndownloader.figshare.com/files/40987058",
    "eukaryotes_ncbi_temperatures.csv": "https://ndownloader.figshare.com/files/40987061",
}
USER_AGENT = "thermocas-gen/0.1 (OGT join; https://github.com/somlunavat/thermocas-gen)"

LABEL_COLUMNS = (
    "seq_id",
    "organism",
    "organism_key",
    "ogt_c",
    "temp_bin",
    "ogt_verified",
    "n_ogt_sources",
    "ogt_source_spread_c",
    "ogt_bin_agreement",
    "length",
    "cluster_id",
    "split",
)


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_gosha(destination_dir: Path, only_all_species: bool = True) -> list[Path]:
    """Download the Gosha CSVs (about 6 MB in total)."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    wanted = (
        ["all_species_temperatures.csv"] if only_all_species else list(GOSHA_FILES)
    )
    written: list[Path] = []
    for name in wanted:
        url = GOSHA_FILES[name]
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180, context=ssl_context()) as response:
                payload = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"could not download {name}: {exc}") from exc
        if payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        path = destination_dir / name
        path.write_bytes(payload)
        written.append(path)
        print(f"  {name}: {len(payload) / 1e6:.2f} MB")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="data")
    parser.add_argument("--sequences", type=Path, default=None, help="FASTA to label")
    parser.add_argument("--ogt-csv", type=Path, default=None, help="Gosha all-species CSV")
    parser.add_argument("--out", type=Path, default=None, help="label table CSV")
    parser.add_argument("--fetch", action="store_true", help="download the Gosha CSVs first")
    parser.add_argument(
        "--strict-crosscheck",
        action="store_true",
        help="run Gate 1 against cross-checked labels only (the honest reading)",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    binner = GrowthTemperatureBinner.from_config(cfg)
    ogt_csv = args.ogt_csv or (REPO_ROOT / "data/raw/gosha/all_species_temperatures.csv")

    if args.fetch:
        print(f"downloading Gosha to {ogt_csv.parent}")
        print("  source: figshare DOI 10.6084/m9.figshare.23257940 (CC BY 4.0)")
        try:
            fetch_gosha(ogt_csv.parent)
        except RuntimeError as exc:
            print(f"  {exc}", file=sys.stderr)
            return 1
        print(f"  record the download date ({date.today().isoformat()}) in PROJECT_LOG.md")

    if not ogt_csv.is_file():
        print(
            f"OGT database not found: {ogt_csv}\nrun with --fetch to download it",
            file=sys.stderr,
        )
        return 1
    if args.sequences is None:
        print("--sequences is required (or pass --fetch alone to download)", file=sys.stderr)
        return 1 if not args.fetch else 0
    if not args.sequences.is_file():
        print(f"sequences not found: {args.sequences}", file=sys.stderr)
        return 1

    try:
        table = OgtTable.from_gosha_csv(
            ogt_csv,
            min_plausible_c=float(cfg["ogt/min_plausible_c"]),
            max_plausible_c=float(cfg["ogt/max_plausible_c"]),
        )
    except ValueError as exc:
        print(f"could not load {ogt_csv}: {exc}", file=sys.stderr)
        return 1

    min_sources = int(cfg["ogt/crosscheck/min_sources"])
    min_agreement = float(cfg["ogt/crosscheck/min_bin_agreement"])

    print(f"OGT database: {ogt_csv}")
    print(table.report.summary())
    print(f"\ncross-check: >={min_sources} collections agreeing on the bin at >={min_agreement:.0%}")

    records = list(read_fasta(args.sequences))
    report = JoinReport(total=len(records))
    rows = []
    unlabeled_reasons: Counter[str] = Counter()

    for record in records:
        organism = parse_organism_from_header(record.description)
        if organism is None:
            report.unlabeled += 1
            unlabeled_reasons["no organism in header"] += 1
            if len(report.unlabeled_examples) < 5:
                report.unlabeled_examples.append(record.seq_id)
            continue

        found = table.lookup(organism)
        if found is None:
            report.unlabeled += 1
            unlabeled_reasons["organism not in OGT database"] += 1
            if len(report.unlabeled_examples) < 5:
                report.unlabeled_examples.append(f"{record.seq_id} ({organism})")
            continue

        temp_bin = binner.bin_for(found.ogt_c)
        verified = found.is_verified(
            binner.bin_for, min_sources=min_sources, min_bin_agreement=min_agreement
        )

        report.labeled += 1
        report.bin_counts[temp_bin] = report.bin_counts.get(temp_bin, 0) + 1
        if verified:
            report.verified += 1
            report.verified_bin_counts[temp_bin] = report.verified_bin_counts.get(temp_bin, 0) + 1

        rows.append(
            {
                "seq_id": record.seq_id,
                "organism": found.organism,
                "organism_key": normalize_organism(organism),
                "ogt_c": f"{found.ogt_c:g}",
                "temp_bin": temp_bin,
                "ogt_verified": int(verified),
                "n_ogt_sources": found.n_sources,
                "ogt_source_spread_c": f"{found.source_spread_c:g}",
                "ogt_bin_agreement": f"{found.bin_agreement(binner.bin_for):.3f}",
                "length": len(record),
                # Filled in by 05_cluster_split.
                "cluster_id": "",
                "split": "",
            }
        )

    # Keep every configured bin present, including empty ones, so Gate 1 sees
    # a missing class as zero rather than as absent.
    for name in binner.names:
        report.bin_counts.setdefault(name, 0)
        report.verified_bin_counts.setdefault(name, 0)

    print(f"\njoin against {args.sequences.name}:")
    print(report.summary())
    for reason, count in unlabeled_reasons.most_common():
        print(f"  unlabeled: {count} ({reason})")

    out_path = args.out or cfg.resolve_path("paths/label_table")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LABEL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} labeled rows to {out_path}")

    min_per_bin = int(cfg["growth_temperature/min_per_bin"])
    use_verified = args.strict_crosscheck or bool(cfg["growth_temperature/require_crosscheck"])
    failures = gate1_failures(report, min_per_bin, use_verified=use_verified)

    label = "cross-checked" if use_verified else "all labeled"
    print(f"\nGate 1 ({label} counts, minimum {min_per_bin} per bin):")
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        print(
            "\nGate 1 does not pass. Options, in the brief's order of preference:\n"
            "  - widen the homolog search (lower identity, larger database)\n"
            "  - merge the thermophilic bins, or treat OGT as a continuous target\n"
            "  - add a broader stage of temperature-labeled bacterial proteins\n"
            "  - reframe as an evaluation-benchmark paper\n"
            "Record the decision in PROJECT_LOG.md before changing the config.",
            file=sys.stderr,
        )
        return 3

    print("  PASS every bin meets the minimum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
