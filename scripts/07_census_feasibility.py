#!/usr/bin/env python3
"""Gate 1 pre-flight: estimate the thermophilic Cas9 corpus before searching.

The full pipeline needs a multi-gigabyte reference database and a cloud
machine. This script answers the question that decides whether that is worth
doing, using only small UniProt queries: how many *independent* thermophilic
Cas9 clusters exist at all?

Sequence counts are misleading here. Near-identical strain variants collapse
under clustering, and it is the cluster count that bounds both what the model
can learn and what cluster-level splitting can hold out.

Usage:
    python scripts/07_census_feasibility.py --fetch
    python scripts/07_census_feasibility.py --census data/raw/census/cas9_uniprot.tsv
"""

from __future__ import annotations

import argparse
import collections
import csv
import ssl
import sys
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config, GrowthTemperatureBinner  # noqa: E402
from thermocas.data.ogt import OgtTable  # noqa: E402

STREAM = "https://rest.uniprot.org/uniprotkb/stream"
CENSUS_QUERY = "(protein_name:cas9) AND length:[950 TO 1700]"
USER_AGENT = "thermocas-gen/0.1 (feasibility census)"


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_census(destination: Path) -> int:
    params = {
        "query": CENSUS_QUERY,
        "format": "tsv",
        "fields": "accession,organism_name,length,reviewed,protein_name",
    }
    url = f"{STREAM}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=300, context=ssl_context()) as response:
        destination.write_bytes(response.read())
    return sum(1 for _ in destination.open()) - 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data")
    parser.add_argument(
        "--census", type=Path, default=REPO_ROOT / "data/raw/census/cas9_uniprot.tsv"
    )
    parser.add_argument("--ogt-csv", type=Path, default=REPO_ROOT / "data/raw/gosha/all_species_temperatures.csv")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    if args.fetch:
        print(f"querying UniProt: {CENSUS_QUERY}")
        print(f"  wrote {fetch_census(args.census)} rows to {args.census}")

    if not args.census.is_file():
        print(f"census not found: {args.census}\nrun with --fetch", file=sys.stderr)
        return 1
    if not args.ogt_csv.is_file():
        print(f"OGT database not found: {args.ogt_csv}", file=sys.stderr)
        return 1

    cfg = Config.load(args.config)
    binner = GrowthTemperatureBinner.from_config(cfg)
    table = OgtTable.from_gosha_csv(
        args.ogt_csv,
        min_plausible_c=float(cfg["ogt/min_plausible_c"]),
        max_plausible_c=float(cfg["ogt/max_plausible_c"]),
    )
    min_sources = int(cfg["ogt/crosscheck/min_sources"])
    min_agreement = float(cfg["ogt/crosscheck/min_bin_agreement"])

    rows = list(csv.DictReader(args.census.open(), delimiter="\t"))
    bins: collections.Counter[str] = collections.Counter()
    verified: collections.Counter[str] = collections.Counter()
    organisms: dict[str, set[str]] = collections.defaultdict(set)
    thermo_accessions: list[tuple[str, float, str]] = []
    labeled = 0

    for row in rows:
        found = table.lookup(row["Organism"])
        if found is None:
            continue
        labeled += 1
        name = binner.bin_for(found.ogt_c)
        bins[name] += 1
        organisms[name].add(found.organism)
        if found.is_verified(binner.bin_for, min_sources, min_agreement):
            verified[name] += 1
        if name != binner.names[0]:
            thermo_accessions.append((row["Entry"], found.ogt_c, found.organism))

    print(f"\nCas9 proteins in UniProtKB (950-1700 aa): {len(rows)}")
    print(f"OGT coverage: {labeled}/{len(rows)} = {labeled / len(rows):.1%}")
    print(f"\n{'bin':<10}{'sequences':>11}{'cross-checked':>15}{'organisms':>12}")
    for name in binner.names:
        print(f"{name:<10}{bins[name]:>11}{verified[name]:>15}{len(organisms[name]):>12}")

    min_per_bin = int(cfg["growth_temperature/min_per_bin"])
    print(f"\nGate 1 threshold is {min_per_bin} per bin:")
    for name in binner.names:
        verdict = "PASS" if bins[name] >= min_per_bin else "FAIL"
        print(f"  {name}: {bins[name]} {verdict}")

    if thermo_accessions:
        hottest = max(ogt for _, ogt, _ in thermo_accessions)
        print(f"\nhottest thermophilic Cas9 host: {hottest:g} C")
        print("to count independent clusters, cluster these accessions at the configured identity:")
        print("  " + ",".join(a for a, _, _ in thermo_accessions[:40]))

    print(
        "\nNOTE: sequence counts overstate what is usable. Cluster the thermophilic\n"
        "set before trusting any of the numbers above; near-identical strain\n"
        "variants collapse, and the cluster count is what bounds learning and\n"
        "cluster-level splitting."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
