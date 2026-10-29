#!/usr/bin/env python3
"""Build the external corpus the conditioning ceiling is validated against.

The ceiling sweep showed measured conditioning gain tracking a corpus-only
ceiling across threshold choices. It showed that on one corpus: ours, built by
our own species balancing and per-species capping. Two explanations survive
that result. Either the ceiling is a property of corpora, or the curve is a
property of how stage A was assembled. They are distinguished by measuring a
corpus nobody here designed.

This script builds that corpus by joining UniProtKB reviewed bacteria and
archaea to Engqvist's growth temperatures, and writes it in the same schema as
the stage A label table so 15_ceiling_sweep.py runs on it unchanged:

    python scripts/16_external_corpus.py --download           # ~357k sequences
    python scripts/16_external_corpus.py --dry-run
    python scripts/16_external_corpus.py                      # uncapped
    python scripts/16_external_corpus.py --config external_balanced
    python scripts/16_external_corpus.py --cluster            # fill cluster_id
    python scripts/15_ceiling_sweep.py --config external

NEVER TRAIN ON WHAT THIS WRITES. The corpus has had no Cas9 annotation screen
and no alignment screen against the held-out Cas9 sequences. It is a
measurement corpus; configs/pretrain.yaml is the one training reads. The
sequences never enter a model here -- only their residue counts, per label
class, do.

Two passes over the FASTA rather than one. The first reads headers and lengths
and decides which ids survive filtering, capping and balancing; the second
writes the sequences for those ids. Balancing cannot be decided until every
species has been seen, and holding 120M residues in memory to avoid a second
40-second pass over a gzip stream is the wrong trade.
"""

from __future__ import annotations

import argparse
import collections
import csv
import functools
import random
import re
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config, GrowthTemperatureBinner  # noqa: E402
from thermocas.data.cluster import (  # noqa: E402
    ClusterParams,
    mmseqs_cluster_commands,
    parse_cluster_tsv,
)
from thermocas.data.engqvist import EngqvistTable  # noqa: E402
from thermocas.data.fasta import (  # noqa: E402
    FastaRecord,
    canonical_seq_id,
    read_fasta,
    write_fasta,
)
from thermocas.data.ogt import parse_organism_from_header  # noqa: E402
from thermocas.data.sequence_filter import SequenceFilter  # noqa: E402

# UniProt FASTA headers carry the NCBI taxon as `OX=1314`.
_TAXID = re.compile(r"\bOX=(\d+)\b")

COLUMNS = (
    "seq_id",
    "organism",
    "organism_key",
    "ogt_c",
    "temp_bin",
    "n_ogt_sources",
    "length",
    "cluster_id",
    "split",
)


USER_AGENT = (
    "thermocas-gen/0.1 (external ceiling-validation corpus; "
    "https://github.com/somlunavat/thermocas-gen)"
)


@functools.lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    """Same fallback as 08_collect_broad.py: this Python build ships without a
    usable root store, so without certifi every UniProt request fails
    verification."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch(url: str, destination: Path, timeout: float, retries: int = 5) -> int:
    """Download to a temporary name and rename, so a partial file is never
    mistaken for a complete one on the next run."""
    staging = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=ssl_context()
            ) as response, staging.open("wb") as out:
                copied = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    copied += len(chunk)
                    if copied % (32 << 20) < (1 << 20):
                        print(f"  {copied / 1e6:,.0f} MB", flush=True)
            staging.rename(destination)
            return 0
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as exc:
            staging.unlink(missing_ok=True)
            if attempt == retries - 1:
                print(f"giving up on {url}: {exc}", file=sys.stderr)
                return 3
            wait = 2 ** attempt
            print(f"  retry in {wait}s after {type(exc).__name__}", flush=True)
            time.sleep(wait)
    return 3


def download(cfg: Config, timeout: float) -> int:
    """Fetch the UniProt release file this corpus is built from.

    The whole of Swiss-Prot in one versioned file, rather than a REST query
    for the prokaryotic subset. Two reasons, in order of how much time they
    cost to learn: the bulk `stream` endpoint dies partway through a
    357k-entry response with "HTTP/2 stream 1 was not closed cleanly:
    INTERNAL_ERROR", and again over HTTP/1.1, leaving a truncated gzip that
    only fails when something reads the end of it; and cursor pagination,
    which does work, needed ~715 requests at ~27 s each. The release file is
    94 MB over one connection and carries a release number, which is better
    provenance than a query run on an unrecorded date.

    Nothing is lost by taking all taxa: the Engqvist table is loaded
    prokaryotes-only, so every eukaryotic entry drops out of the join as
    unlabeled.
    """
    sequences = cfg.resolve_path("paths/sequences")
    url = str(cfg["download/url"])
    sequences.parent.mkdir(parents=True, exist_ok=True)

    if sequences.is_file():
        print(f"{sequences} exists; checking it reads cleanly")
    else:
        print(f"fetching {url}")
        code = fetch(url, sequences, timeout)
        if code:
            return code

    release = cfg["download/release_file"]
    release_path = sequences.parent / "reldate.txt"
    if not release_path.is_file():
        fetch(str(release), release_path, timeout)
    if release_path.is_file():
        print("\n" + release_path.read_text().strip()[:200])

    # Read it back rather than trusting the write: a truncated gzip member is
    # the failure mode that cost two downloads, and it is silent until
    # something reaches the end of the file.
    seen = sum(1 for _ in read_fasta(sequences))
    print(f"\n{sequences}: {seen:,} sequences readable")
    return 0 if seen else 4


@dataclass
class Candidate:
    seq_id: str
    organism: str
    organism_key: str
    ogt_c: float
    temp_bin: str
    length: int


@dataclass
class ScanReport:
    """Every sequence in the download ends in exactly one of these."""

    seen: int = 0
    no_organism: int = 0
    no_ogt: int = 0
    filtered: collections.Counter = None  # type: ignore[assignment]
    duplicate_sequence: int = 0
    over_species_cap: int = 0
    kept: int = 0
    matched_by_taxid: int = 0
    matched_by_name: int = 0

    def __post_init__(self) -> None:
        if self.filtered is None:
            self.filtered = collections.Counter()

    def summary(self) -> str:
        lines = [
            f"scanned {self.seen:,} sequences",
            f"  no parseable organism      {self.no_organism:,}",
            f"  organism not in Engqvist   {self.no_ogt:,}",
        ]
        for reason, count in sorted(self.filtered.items()):
            lines.append(f"  rejected {reason:<17} {count:,}")
        if self.duplicate_sequence:
            lines.append(f"  exact duplicate sequence   {self.duplicate_sequence:,}")
        if self.over_species_cap:
            lines.append(f"  over the per-species cap   {self.over_species_cap:,}")
        lines.append(f"  labeled and kept           {self.kept:,}")
        # Counted at lookup, so these total the sequences that found an
        # organism -- before length, alphabet and duplicate filtering -- not
        # the ones kept.
        lines.append(
            f"  of those labeled: {self.matched_by_taxid:,} matched by taxid, "
            f"{self.matched_by_name:,} by organism name"
        )
        return "\n".join(lines)


def scan(
    sequences: Path,
    table: EngqvistTable,
    binner: GrowthTemperatureBinner,
    filt: SequenceFilter,
    max_per_species: int,
) -> tuple[list[Candidate], ScanReport]:
    """Pass one: decide which sequences are labeled, in range and under cap."""
    report = ScanReport()
    kept: list[Candidate] = []
    per_species: collections.Counter = collections.Counter()
    seen_sequences: set[int] = set()

    for record in read_fasta(sequences):
        report.seen += 1
        if report.seen % 100000 == 0:
            print(f"  ...{report.seen:,} sequences", flush=True)

        organism = parse_organism_from_header(record.description)
        taxid_match = _TAXID.search(record.description)
        taxid = taxid_match.group(1) if taxid_match else None
        if not organism and not taxid:
            report.no_organism += 1
            continue

        found = table.lookup(taxid, organism)
        if found is None:
            report.no_ogt += 1
            continue
        if taxid and table.by_taxid.get(taxid) is found:
            report.matched_by_taxid += 1
        else:
            report.matched_by_name += 1

        rejection = filt.check(record.sequence)
        if rejection is not None:
            report.filtered[rejection.value] += 1
            continue

        if filt.drop_exact_duplicates:
            # Hashing rather than storing the sequences: this pass is about
            # ids, and 350k full sequences in a set is the memory the
            # two-pass design exists to avoid. A hash collision would drop
            # one sequence out of a corpus of 350k, which is not a risk worth
            # 120 MB.
            digest = hash(record.sequence)
            if digest in seen_sequences:
                report.duplicate_sequence += 1
                continue
            seen_sequences.add(digest)

        if max_per_species and per_species[found.key] >= max_per_species:
            report.over_species_cap += 1
            continue
        per_species[found.key] += 1

        kept.append(
            Candidate(
                seq_id=canonical_seq_id(record.seq_id),
                organism=found.organism,
                organism_key=found.key,
                ogt_c=found.ogt_c,
                temp_bin=binner.bin_for(found.ogt_c),
                length=len(record.sequence),
            )
        )
        report.kept += 1

    return kept, report


def balance_species(
    candidates: list[Candidate], seed: int
) -> tuple[list[Candidate], dict[str, int]]:
    """Cut the larger class's *species* list down to the smaller class's size.

    Species rather than sequences, matching how stage A balances: dropping
    sequences at random from an over-represented class would leave its species
    in the corpus with thinner coverage, which changes per-class residue
    composition as well as the marginal. Dropping whole species keeps each
    surviving species intact.
    """
    by_bin: dict[str, set[str]] = collections.defaultdict(set)
    for candidate in candidates:
        by_bin[candidate.temp_bin].add(candidate.organism_key)
    if len(by_bin) < 2:
        return candidates, {b: len(s) for b, s in by_bin.items()}

    target = min(len(species) for species in by_bin.values())
    rng = random.Random(seed)
    chosen: set[str] = set()
    counts: dict[str, int] = {}
    for temp_bin, species in sorted(by_bin.items()):
        ordered = sorted(species)
        if len(ordered) > target:
            ordered = rng.sample(ordered, target)
        counts[temp_bin] = len(ordered)
        chosen.update(ordered)
    return [c for c in candidates if c.organism_key in chosen], counts


def write_corpus(
    sequences: Path, out_fasta: Path, out_table: Path, candidates: list[Candidate]
) -> int:
    """Pass two: write the kept sequences, then the label table beside them."""
    wanted = {c.seq_id for c in candidates}

    def selected() -> Iterator[FastaRecord]:
        for record in read_fasta(sequences):
            seq_id = canonical_seq_id(record.seq_id)
            if seq_id in wanted:
                yield FastaRecord(seq_id, record.description, record.sequence)

    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    written = write_fasta(out_fasta, selected())

    with out_table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "seq_id": candidate.seq_id,
                    "organism": candidate.organism,
                    "organism_key": candidate.organism_key,
                    "ogt_c": f"{candidate.ogt_c:g}",
                    "temp_bin": candidate.temp_bin,
                    # Engqvist publishes one consolidated temperature per
                    # organism with no per-collection breakdown, so the
                    # cross-check column stage A fills has no counterpart.
                    # Left empty rather than written as 1, which would claim
                    # a single agreeing source where the truth is unknown.
                    "n_ogt_sources": "",
                    "length": candidate.length,
                    "cluster_id": "",
                    "split": "",
                }
            )
    return written


def cluster(cfg: Config, threads: int, dry_run: bool) -> int:
    """Fill cluster_id in the label table from an MMseqs2 clustering.

    The sweep's within-cluster column compares sequences that share a cluster,
    which is the comparison that holds family identity fixed. Without this the
    external sweep reports the corpus-wide column only.
    """
    fasta = cfg.resolve_path("paths/labeled_fasta")
    table_path = cfg.resolve_path("paths/label_table")
    clusters = cfg.resolve_path("paths/clusters")
    if not fasta.is_file() or not table_path.is_file():
        print(
            f"build the corpus first: {fasta} or {table_path} is missing",
            file=sys.stderr,
        )
        return 2

    params = ClusterParams.from_config(cfg)
    with tempfile.TemporaryDirectory(prefix="external-cluster-") as workdir:
        commands = mmseqs_cluster_commands(
            params, fasta, clusters.with_suffix(""), Path(workdir), threads=threads
        )
        for command in commands:
            print("  " + " ".join(command), flush=True)
            if dry_run:
                continue
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(result.stdout[-2000:], file=sys.stderr)
                print(result.stderr[-2000:], file=sys.stderr)
                return result.returncode
    if dry_run:
        return 0

    assignments = parse_cluster_tsv(clusters)
    with table_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = 0
    for row in rows:
        representative = assignments.get(row["seq_id"])
        if representative is None:
            missing += 1
            continue
        row["cluster_id"] = representative
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    sizes = collections.Counter(assignments.values())
    print(f"\nclusters:  {len(sizes):,} over {len(assignments):,} sequences")
    print(f"singletons {sum(1 for n in sizes.values() if n == 1):,}")
    if missing:
        # MMseqs2 keys on the middle field of a pipe-separated header, which
        # is why the corpus is written with canonical ids. If this is ever
        # non-zero the two sides are disagreeing about identifiers again.
        print(f"WARNING {missing:,} table rows absent from the cluster table")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="external")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch the UniProt sequences for this config and exit; resumable",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="fill cluster_id in an already-built label table and exit",
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.download:
        return download(cfg, args.timeout)
    if args.cluster:
        return cluster(cfg, args.threads, args.dry_run)

    ogt_source = cfg.resolve_path("paths/ogt_source")
    sequences = cfg.resolve_path("paths/sequences")
    out_fasta = cfg.resolve_path("paths/labeled_fasta")
    out_table = cfg.resolve_path("paths/label_table")
    for required in (ogt_source, sequences):
        if not required.is_file():
            print(f"missing input: {required}", file=sys.stderr)
            return 2

    bounds = cfg.section("ogt")
    table = EngqvistTable.load(
        ogt_source,
        min_plausible_c=float(bounds["min_plausible_c"]),
        max_plausible_c=float(bounds["max_plausible_c"]),
    )
    binner = GrowthTemperatureBinner.from_config(cfg)
    filt = SequenceFilter.from_config(cfg, "filter")
    sampling = cfg.section("sampling")
    max_per_species = int(sampling["max_proteins_per_species"])
    balance = str(sampling["balance"])

    print(f"config:    {args.config}")
    print(f"ogt:       {ogt_source.name}")
    print(table.report.summary())
    print(f"sequences: {sequences.name}")
    print(f"bins:      {binner.names}")
    print(f"cap:       {max_per_species or 'none'} per species, balance {balance}")
    if args.dry_run:
        print("\n--dry-run: stopping before the corpus pass")
        return 0

    print("\npass one, over the download...", flush=True)
    candidates, report = scan(sequences, table, binner, filt, max_per_species)
    print(report.summary())
    if not candidates:
        print("\nnothing labeled; the organism join found no overlap", file=sys.stderr)
        return 4

    if balance == "match_smallest":
        before = len(candidates)
        candidates, species_counts = balance_species(candidates, int(cfg["seed"]))
        print(
            f"\nbalanced to {species_counts} species per class: "
            f"{before:,} -> {len(candidates):,} sequences"
        )
    elif balance != "none":
        print(f"\nunknown sampling/balance: {balance!r}", file=sys.stderr)
        return 2

    bins = collections.Counter(c.temp_bin for c in candidates)
    species = len({c.organism_key for c in candidates})
    residues = sum(c.length for c in candidates)
    distinct_ogt = len({c.ogt_c for c in candidates})
    print(f"\ncorpus:    {len(candidates):,} sequences, {species:,} species")
    print(f"residues:  {residues:,}")
    print(f"bins:      {dict(bins)}")
    print(f"distinct OGT values: {distinct_ogt}")

    print("\npass two, writing the corpus...", flush=True)
    written = write_corpus(sequences, out_fasta, out_table, candidates)
    print(f"wrote {written:,} sequences to {out_fasta}")
    print(f"wrote {len(candidates):,} rows to {out_table}")
    if written != len(candidates):
        print(
            f"\n{len(candidates) - written:,} kept ids were not found on the "
            f"second pass; the two passes disagree about identifiers",
            file=sys.stderr,
        )
        return 4
    print("\nNOT training data: no Cas9 screen has been applied to this corpus.")
    print("next: python scripts/16_external_corpus.py --config "
          f"{args.config} --cluster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
