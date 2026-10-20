#!/usr/bin/env python3
"""Stage A: collect broad temperature-labeled bacterial proteins.

Selects a balanced set of mesophilic and thermophilic species from Gosha, then
downloads a capped sample of each species' proteins from UniProt.

Cas9-family proteins are screened out by annotation here; the alignment-level
screen against held-out Cas9 sequences runs in 09_screen_broad.py. Both are
needed: if a held-out Cas9 test cluster reaches this corpus, the stage B
leakage gate is measuring nothing, because the model will already have seen
the test sequence one stage earlier.

Downloads are sharded per species and resumable: re-running skips species
already fetched, so an interrupted run continues rather than starting over.

Usage:
    python scripts/08_collect_broad.py --dry-run
    python scripts/08_collect_broad.py --max-species 20      # pilot
    python scripts/08_collect_broad.py                       # full run
"""

from __future__ import annotations

import argparse
import csv
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config, GrowthTemperatureBinner  # noqa: E402
from thermocas.data.broad import (  # noqa: E402
    Cas9Exclusion,
    ExclusionReport,
    SpeciesSelection,
    select_species,
)
from thermocas.data.fasta import FastaRecord, read_fasta, write_fasta  # noqa: E402
from thermocas.data.ogt import OgtTable, parse_organism_from_header  # noqa: E402
from thermocas.data.sequence_filter import SequenceFilter  # noqa: E402

SEARCH = "https://rest.uniprot.org/uniprotkb/search"
USER_AGENT = "thermocas-gen/0.1 (stage A collection; https://github.com/somlunavat/thermocas-gen)"
UNIPROT_PAGE_MAX = 500


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


@dataclass
class SpeciesPlan:
    organism: str
    organism_key: str
    ogt_c: float
    temp_bin: str
    n_sources: int


def fetch_species_fasta(
    organism: str, limit: int, timeout: int = 120, attempts: int = 4
) -> str:
    """Download up to ``limit`` proteins for one organism, with retries.

    Retries matter here rather than being defensive boilerplate: a full run
    makes 800 requests, so a transient timeout or 5xx is close to certain.
    An early version caught only HTTPError and URLError and died 74 species in
    on a bare TimeoutError, which urlopen raises directly and which is not a
    URLError subclass. Catching OSError covers timeouts, resets and DNS
    failures alike.

    A 4xx other than 429 is not retried: it means the query is wrong, and
    repeating it would just be slower.
    """
    params = {
        "query": f'organism_name:"{organism}"',
        "format": "fasta",
        "size": str(min(limit, UNIPROT_PAGE_MAX)),
    }
    url = f"{SEARCH}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_error = "unknown"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=ssl_context()
            ) as response:
                return response.read().decode()
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code == 429 or exc.code >= 500:
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * attempt
            else:
                raise RuntimeError(last_error) from exc
        except urllib.error.URLError as exc:
            last_error = f"unreachable ({exc.reason})"
            delay = 2.0 * attempt
        except OSError as exc:
            # TimeoutError, ConnectionReset and friends arrive here.
            last_error = f"{type(exc).__name__}: {exc}"
            delay = 2.0 * attempt

        if attempt < attempts:
            time.sleep(delay)

    raise RuntimeError(f"{last_error} after {attempts} attempts")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="pretrain")
    parser.add_argument("--data-config", default="data")
    parser.add_argument("--ogt-csv", type=Path, default=REPO_ROOT / "data/raw/gosha/all_species_temperatures.csv")
    parser.add_argument("--shard-dir", type=Path, default=REPO_ROOT / "data/raw/broad_shards")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument("--species-table", type=Path, default=None)
    parser.add_argument("--max-species", type=int, default=None, help="per bin; for a pilot run")
    parser.add_argument("--sleep", type=float, default=0.34, help="seconds between requests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    data_cfg = Config.load(args.data_config)
    binner = GrowthTemperatureBinner.from_config(data_cfg)
    selection = SpeciesSelection.from_config(cfg)
    exclusion = Cas9Exclusion.from_config(cfg)

    out_fasta = args.out or cfg.resolve_path("paths/sequences")
    label_table = args.label_table or cfg.resolve_path("paths/label_table")
    species_table = args.species_table or cfg.resolve_path("paths/species_table")

    if not args.ogt_csv.is_file():
        print(f"OGT database not found: {args.ogt_csv}", file=sys.stderr)
        print("run: python scripts/04_join_ogt.py --fetch", file=sys.stderr)
        return 1

    table = OgtTable.from_gosha_csv(
        args.ogt_csv,
        min_plausible_c=float(data_cfg["ogt/min_plausible_c"]),
        max_plausible_c=float(data_cfg["ogt/max_plausible_c"]),
    )

    controls = [str(c) for c in data_cfg["split/positive_controls"]]
    control_hosts: list[str] = []
    if bool(cfg["species/exclude_positive_control_hosts"]):
        # Resolve control names to their host organisms via the seed FASTA.
        seeds_path = data_cfg.resolve_path("paths/seeds")
        if seeds_path.is_file():
            for record in read_fasta(seeds_path):
                if any(c.lower() in record.description.lower() for c in controls):
                    host = parse_organism_from_header(record.description)
                    if host:
                        control_hosts.append(host)

    chosen, species_report = select_species(
        table.records.values(),
        binner.bin_for,
        selection,
        min_sources=int(data_cfg["ogt/crosscheck/min_sources"]),
        min_bin_agreement=float(data_cfg["ogt/crosscheck/min_bin_agreement"]),
        excluded_hosts=control_hosts,
        seed=int(cfg["seed"]),
    )

    print("species selection")
    print(species_report.summary())
    if control_hosts:
        print(f"  excluded control hosts: {', '.join(control_hosts)}")

    plans: list[SpeciesPlan] = []
    for name in sorted(chosen):
        members = chosen[name]
        if args.max_species is not None:
            members = members[: args.max_species]
        for record in members:
            plans.append(
                SpeciesPlan(record.organism, record.organism_key, record.ogt_c, name, record.n_sources)
            )

    cap = int(cfg["sampling/max_proteins_per_species"])
    print(f"\nplan: {len(plans)} species x up to {cap} proteins "
          f"= at most {len(plans) * cap:,} sequences (~{len(plans) * cap * 350 / 1e6:.0f} MB)")

    species_table.parent.mkdir(parents=True, exist_ok=True)
    with species_table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["organism", "organism_key", "ogt_c", "temp_bin", "n_ogt_sources"])
        for plan in plans:
            writer.writerow([plan.organism, plan.organism_key, f"{plan.ogt_c:g}", plan.temp_bin, plan.n_sources])
    print(f"wrote {species_table}")

    if args.dry_run:
        print("\ndry run: nothing downloaded")
        return 0

    # Download, sharded per species so the run is resumable.
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    fetched = skipped = failed = 0
    failures: list[str] = []
    for index, plan in enumerate(plans, start=1):
        shard = args.shard_dir / f"{plan.organism_key.replace(' ', '_').replace('/', '_')}.fasta"
        if shard.is_file() and shard.stat().st_size > 0:
            skipped += 1
            continue
        try:
            text = fetch_species_fasta(plan.organism, cap)
        except Exception as exc:  # noqa: BLE001
            # One unreachable species must not end a run of 800. Failures are
            # counted and named; the run is resumable, so a later pass picks
            # up whatever is still missing.
            print(f"  [{index}/{len(plans)}] {plan.organism}: FAILED {exc}", file=sys.stderr)
            failed += 1
            failures.append(plan.organism)
            continue
        shard.write_text(text)
        fetched += 1
        if index % 25 == 0 or index == len(plans):
            print(f"  [{index}/{len(plans)}] fetched={fetched} cached={skipped} failed={failed}", flush=True)
        time.sleep(args.sleep)

    print(f"\ndownloaded {fetched} species ({skipped} already cached, {failed} failed)")
    if failures:
        print(f"  failed: {', '.join(failures[:10])}" + (" ..." if len(failures) > 10 else ""))
        print("  re-run to retry only the missing species (shards are cached)")

    # Assemble, filter and screen.
    seq_filter = SequenceFilter(
        min_length=int(cfg["sampling/min_length"]),
        max_length=int(cfg["sampling/max_length"]),
        allowed_residues=frozenset(str(cfg["sampling/allowed_residues"]).upper()),
        drop_exact_duplicates=True,
    )
    exclusion_report = ExclusionReport()
    kept: list[FastaRecord] = []
    rows: list[dict[str, object]] = []
    by_key = {plan.organism_key: plan for plan in plans}
    seen_sequences: set[str] = set()

    for shard in sorted(args.shard_dir.glob("*.fasta")):
        key = shard.stem.replace("_", " ")
        plan = by_key.get(key)
        if plan is None:
            continue
        try:
            records = list(read_fasta(shard))
        except ValueError as exc:
            print(f"  skipping malformed shard {shard.name}: {exc}", file=sys.stderr)
            continue
        for record in records:
            exclusion_report.examined += 1

            pattern = exclusion.matches_annotation(record.description)
            if pattern is not None:
                exclusion_report.excluded_by_name += 1
                exclusion_report.name_hits[pattern] = exclusion_report.name_hits.get(pattern, 0) + 1
                continue

            if seq_filter.check(record.sequence) is not None:
                continue
            sequence = record.sequence.strip().upper()
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)

            kept.append(FastaRecord(record.seq_id, record.description, sequence))
            rows.append(
                {
                    "seq_id": record.seq_id,
                    "organism": plan.organism,
                    "organism_key": plan.organism_key,
                    "ogt_c": f"{plan.ogt_c:g}",
                    "temp_bin": plan.temp_bin,
                    "n_ogt_sources": plan.n_sources,
                    "length": len(sequence),
                    "cluster_id": "",
                    "split": "",
                }
            )

    print(f"\n{exclusion_report.summary()}")

    if not kept:
        print("no sequences survived collection", file=sys.stderr)
        return 2

    write_fasta(out_fasta, kept)
    with label_table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bins: dict[str, int] = {}
    for row in rows:
        bins[row["temp_bin"]] = bins.get(row["temp_bin"], 0) + 1

    print(f"\nwrote {len(kept):,} sequences to {out_fasta}")
    print(f"wrote {label_table}")
    for name in binner.names:
        print(f"  {name:<8} {bins.get(name, 0):>9,} sequences")
    print(f"\nrecord the download date ({date.today().isoformat()}) in PROJECT_LOG.md")
    print("next: python scripts/09_screen_broad.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
