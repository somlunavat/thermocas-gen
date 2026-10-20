#!/usr/bin/env python3
"""Stage 1a: collect reference Cas9 seed sequences from UniProt.

Reads ``data/raw/seed_definitions.tsv``, resolves each entry against the
UniProt REST API, and writes ``data/raw/seeds.fasta`` plus a provenance table
recording exactly which accession each seed resolved to and when.

Seed definitions carry an accession only where it has been verified. Entries
without one are resolved by query, and the accession that comes back is
written to the provenance table rather than being assumed. Nothing here
fabricates an identifier.

Usage:
    python scripts/01_collect_seeds.py                   # fetch and write
    python scripts/01_collect_seeds.py --dry-run         # show what it would do
    python scripts/01_collect_seeds.py --offline FILE    # reuse a prior download
"""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.fasta import FastaRecord, read_fasta, write_fasta  # noqa: E402
from thermocas.data.sequence_filter import SequenceFilter  # noqa: E402

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
USER_AGENT = "thermocas-gen/0.1 (seed collection; https://github.com/somlunavat/thermocas-gen)"

SOURCES = ("uniprot", "ncbi")


@dataclass
class SeedDefinition:
    name: str
    source: str
    accession: str
    organism: str
    cas9_type: str
    role: str
    query: str

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(
                f"seed {self.name!r}: unknown source {self.source!r}; expected one of {SOURCES}"
            )


@dataclass
class ResolvedSeed:
    name: str
    source: str
    requested_accession: str
    resolved_accession: str
    organism_requested: str
    organism_resolved: str
    cas9_type: str
    role: str
    length: int
    protein_name: str
    retrieved_on: str


def load_definitions(path: Path) -> list[SeedDefinition]:
    definitions: list[SeedDefinition] = []
    with path.open() as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 7:
                raise ValueError(
                    f"{path}:{lineno}: expected 7 tab-separated fields, got {len(fields)}"
                )
            try:
                definitions.append(SeedDefinition(*(f.strip() for f in fields)))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not definitions:
        raise ValueError(f"{path}: no seed definitions found")
    return definitions


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    """TLS context backed by certifi when the system store is unusable.

    The python.org macOS installers ship without wiring OpenSSL to the system
    keychain (that is what Install Certificates.command does), so a stock
    interpreter raises CERTIFICATE_VERIFY_FAILED against UniProt even though
    the network is fine. Preferring certifi's bundle makes the script work on
    a fresh machine without disabling verification.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def uniprot_search(query: str, timeout: int = 60) -> list[dict]:
    """Query UniProtKB, preferring reviewed entries.

    Returns the parsed ``results`` list. Reviewed (Swiss-Prot) entries are
    requested first because an unreviewed hit is a weaker reference sequence.
    """
    params = {
        "query": query,
        "format": "json",
        "fields": "accession,id,protein_name,organism_name,sequence,reviewed,length",
        "size": "25",
    }
    url = f"{UNIPROT_SEARCH}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"UniProt returned HTTP {exc.code} for query {query!r}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach UniProt ({exc.reason}). Re-run with --offline to "
            "use a previously downloaded FASTA."
        ) from exc
    return payload.get("results", [])


def pick_best(results: list[dict], seed_filter: SequenceFilter) -> dict | None:
    """Choose the most suitable hit: reviewed first, then plausible length.

    Length plausibility uses the same filter the pipeline applies to homologs,
    so a seed cannot be something the pipeline would itself reject.
    """
    scored = []
    for entry in results:
        sequence = entry.get("sequence", {}).get("value", "")
        if not sequence:
            continue
        reviewed = entry.get("entryType", "").startswith("UniProtKB reviewed")
        passes = seed_filter.check(sequence) is None
        scored.append((reviewed, passes, len(sequence), entry))

    if not scored:
        return None
    # Reviewed and length-plausible first; break ties on longer sequence, which
    # favors a full-length ortholog over a fragment.
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return scored[0][3]


def _get(url: str, params: dict[str, str], timeout: int = 60) -> str:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            return response.read().decode()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach {url} ({exc.reason})") from exc


def resolve_from_ncbi(
    definition: SeedDefinition, today: str
) -> tuple[FastaRecord, ResolvedSeed]:
    """Fetch one protein from NCBI by accession.

    Used for seeds absent from UniProtKB. ThermoCas9 is the motivating case:
    the Geobacillus thermodenitrificans T12 strain is GenBank/RefSeq-only, so
    the positive control cannot be obtained from UniProt at all.
    """
    accession = definition.accession or definition.query
    if not accession:
        raise RuntimeError(
            f"seed {definition.name!r} uses source=ncbi but has no accession; "
            "NCBI resolution requires an explicit accession"
        )

    fasta_text = _get(
        NCBI_EFETCH,
        {"db": "protein", "id": accession, "rettype": "fasta", "retmode": "text"},
    )
    lines = [ln.strip() for ln in fasta_text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith(">"):
        raise RuntimeError(f"NCBI returned no FASTA for {accession!r}")

    header = lines[0][1:]
    sequence = "".join(lines[1:]).upper()
    if not sequence:
        raise RuntimeError(f"NCBI returned an empty sequence for {accession!r}")

    # The organism is bracketed at the end of an NCBI protein defline.
    organism = ""
    if "[" in header and header.rstrip().endswith("]"):
        organism = header[header.rindex("[") + 1 : -1].strip()
    protein_name = header.split(" ", 1)[1] if " " in header else header
    if organism:
        protein_name = protein_name.replace(f"[{organism}]", "").strip()

    description = (
        f"{definition.name} | {organism or definition.organism} "
        f"| type={definition.cas9_type} | role={definition.role} | len={len(sequence)}"
    )
    record = FastaRecord(seq_id=accession, description=description, sequence=sequence)
    resolved = ResolvedSeed(
        name=definition.name,
        source="ncbi",
        requested_accession=definition.accession or "",
        resolved_accession=accession,
        organism_requested=definition.organism,
        organism_resolved=organism,
        cas9_type=definition.cas9_type,
        role=definition.role,
        length=len(sequence),
        protein_name=protein_name,
        retrieved_on=today,
    )
    return record, resolved


def resolve_seed(
    definition: SeedDefinition, seed_filter: SequenceFilter, today: str
) -> tuple[FastaRecord, ResolvedSeed]:
    if definition.source == "ncbi":
        record, resolved = resolve_from_ncbi(definition, today)
        rejection = seed_filter.check(record.sequence)
        if rejection is not None:
            print(
                f"  WARNING {definition.name}: resolved sequence would be rejected "
                f"as {rejection.value} (length {len(record.sequence)}). Review before use.",
                file=sys.stderr,
            )
        return record, resolved

    results = uniprot_search(definition.query)
    best = pick_best(results, seed_filter)
    if best is None:
        raise RuntimeError(
            f"no usable UniProt hit for seed {definition.name!r} "
            f"(query: {definition.query!r}). Fill in a verified accession in "
            "seed_definitions.tsv."
        )

    accession = best["primaryAccession"]
    sequence = best["sequence"]["value"].upper()
    organism = best.get("organism", {}).get("scientificName", "")
    protein_name = (
        best.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "")
    )

    if definition.accession and accession != definition.accession:
        raise RuntimeError(
            f"seed {definition.name!r} pins accession {definition.accession} but "
            f"the query resolved to {accession}; reconcile before continuing"
        )

    rejection = seed_filter.check(sequence)
    if rejection is not None:
        # A seed the pipeline's own filter would reject is a real problem: it
        # means either the wrong entry resolved or the filter is miscalibrated.
        print(
            f"  WARNING {definition.name}: resolved sequence would be rejected "
            f"as {rejection.value} (length {len(sequence)}). Review before use.",
            file=sys.stderr,
        )

    description = (
        f"{definition.name} | {organism} | type={definition.cas9_type} "
        f"| role={definition.role} | len={len(sequence)}"
    )
    record = FastaRecord(seq_id=accession, description=description, sequence=sequence)
    resolved = ResolvedSeed(
        name=definition.name,
        source="uniprot",
        requested_accession=definition.accession or "",
        resolved_accession=accession,
        organism_requested=definition.organism,
        organism_resolved=organism,
        cas9_type=definition.cas9_type,
        role=definition.role,
        length=len(sequence),
        protein_name=protein_name,
        retrieved_on=today,
    )
    return record, resolved


def write_provenance(path: Path, rows: list[ResolvedSeed]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="data", help="config name under configs/")
    parser.add_argument(
        "--definitions",
        type=Path,
        default=REPO_ROOT / "data/raw/seed_definitions.tsv",
    )
    parser.add_argument("--out", type=Path, default=None, help="override output FASTA path")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=REPO_ROOT / "data/raw/seed_provenance.csv",
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve nothing; list the plan")
    parser.add_argument(
        "--offline",
        type=Path,
        default=None,
        help="validate and re-filter an already downloaded FASTA instead of fetching",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    seed_filter = SequenceFilter.from_config(cfg)
    out_path = args.out or cfg.resolve_path("paths/seeds")
    definitions = load_definitions(args.definitions)

    print(f"{len(definitions)} seed definitions from {args.definitions}")
    print(f"length filter: {seed_filter.min_length}-{seed_filter.max_length} aa")

    if args.dry_run:
        for definition in definitions:
            pinned = definition.accession or "(resolve by query)"
            print(f"  {definition.name:<12} {definition.source:<8} {pinned:<20} {definition.query}")
        print(f"\nwould write {out_path}")
        return 0

    if args.offline:
        records = list(read_fasta(args.offline))
        kept, report = seed_filter.apply(records)
        print(report.summary())
        written = write_fasta(out_path, kept)
        print(f"wrote {written} seeds to {out_path}")
        return 0 if written else 1

    today = date.today().isoformat()
    records: list[FastaRecord] = []
    resolved_rows: list[ResolvedSeed] = []
    failures: list[str] = []

    for definition in definitions:
        print(f"  resolving {definition.name} ...", flush=True)
        try:
            record, resolved = resolve_seed(definition, seed_filter, today)
        except RuntimeError as exc:
            print(f"  FAILED {definition.name}: {exc}", file=sys.stderr)
            failures.append(definition.name)
            continue
        records.append(record)
        resolved_rows.append(resolved)
        print(
            f"    -> {resolved.resolved_accession} "
            f"({resolved.organism_resolved}, {resolved.length} aa)"
        )

    if not records:
        print("no seeds resolved; nothing written", file=sys.stderr)
        return 1

    written = write_fasta(out_path, records)
    write_provenance(args.provenance, resolved_rows)
    print(f"\nwrote {written} seeds to {out_path}")
    print(f"wrote provenance for {len(resolved_rows)} seeds to {args.provenance}")
    print(f"record the retrieval date ({today}) in PROJECT_LOG.md")

    if failures:
        print(f"\n{len(failures)} seed(s) unresolved: {', '.join(failures)}", file=sys.stderr)
        print("fill in verified accessions in seed_definitions.tsv", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
