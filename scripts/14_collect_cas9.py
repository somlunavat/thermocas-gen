#!/usr/bin/env python3
"""Stage 1a-alt: collect Cas9-family sequences by domain annotation.

This replaces `02_search.py` as the way the stage B corpus is built, and the
reason is the census. `07_census_feasibility.py` established that the
thermophilic Cas9 shortage is biological rather than a database-coverage
problem: the hottest host of any known Cas9 is 60 C, the classic thermophile
genera have no Cas9 at all, and nr holds ~33x more Cas9 proteins than
UniProtKB without being able to manufacture diversity that does not exist. A
multi-gigabyte UniRef50 download and a sensitive DIAMOND search buys remote
homologs that annotation misses; it does not buy thermophiles. So the cheap
path is run first and measured, and the expensive one is only justified by
what this leaves missing.

**Why Pfam cross-reference and not a name match.** The census queried
`protein_name:cas9`, which is right for counting but wrong for collecting: an
ortholog annotated only as "type II CRISPR RNA-guided endonuclease" is a Cas9
and is invisible to that query. Querying UniProt for the Cas9-*specific* Pfam
families instead is the same criterion `configs/data.yaml` already uses to
confirm a domain, applied one step earlier.

Only the Cas9-specific families are used. `configs/data.yaml` also lists
PF13395, PF01844 (HNH) and PF18541 (RuvC_III) as alternatives for domain
confirmation, and those are generic nuclease models that hit many thousands of
unrelated proteins -- useful for confirming a candidate, useless for finding
one. The union here is deliberately wide and the filter is `03_filter_domains`,
which still requires both nuclease domains by HMMER.

Nothing here is a substitute for that filter: this script collects candidates
and records where they came from. It does not decide what counts as a Cas9.

Usage:
    python scripts/14_collect_cas9.py --dry-run
    python scripts/14_collect_cas9.py
    python scripts/14_collect_cas9.py --out data/processed/cas9_candidates.fasta
"""

from __future__ import annotations

import argparse
import collections
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.fasta import (  # noqa: E402
    FastaRecord,
    canonical_seq_id,
    read_fasta,
    write_fasta,
)

STREAM = "https://rest.uniprot.org/uniprotkb/stream"
USER_AGENT = "thermocas-gen/0.1 (cas9 collection)"

# The Cas9-specific Pfam families from configs/data.yaml. Kept as a literal
# list rather than read from the config because the config's groupings mix
# these with generic nuclease models (see the module docstring), and silently
# including PF01844 here would return a large fraction of all HNH nucleases.
CAS9_FAMILIES = (
    "PF22702",  # Cas9_RuvC, the Cas9-specific RuvC domain
    "PF16592",  # Cas9_REC, REC lobe
    "PF16593",  # Cas9_BH, bridge helix
    "PF16595",  # Cas9_PI, PAM-interacting domain
    "PF21069",  # Csx12, type II-B alpha-helical lobe
    "PF22385",  # Cas9_C_2, C-terminal domain
)


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def build_query(min_length: int, max_length: int) -> str:
    """The UniProt query: any Cas9-specific family, or a Cas9 name, in range.

    The length bound is the same one `configs/data.yaml` uses, applied at the
    server so the download is not dominated by fragments. `03_filter_domains`
    re-applies it, which is the check that the two agree.
    """
    families = " OR ".join(f"xref:pfam-{acc}" for acc in CAS9_FAMILIES)
    return (
        f"(({families}) OR protein_name:cas9) "
        f"AND length:[{min_length} TO {max_length}]"
    )


def fetch(query: str, destination: Path, fmt: str = "fasta") -> int:
    params = {"query": query, "format": fmt}
    if fmt == "fasta":
        params["includeIsoform"] = "false"
    url = f"{STREAM}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(
            request, timeout=900, context=ssl_context()
        ) as response:
            destination.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        # The stream endpoint reports a malformed query as 400 with a body
        # that names the offending token, which is the only useful thing here.
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(f"UniProt returned HTTP {exc.code}\n{detail}", file=sys.stderr)
        return -1
    return destination.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="data")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cas9_candidates.fasta",
        help="candidate sequences; feed this to 03_filter_domains.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the query and the URL without downloading",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    min_length = int(cfg["filter/min_length"])
    max_length = int(cfg["filter/max_length"])
    query = build_query(min_length, max_length)

    print(f"query:  {query}")
    print(f"length: {min_length}-{max_length} aa (filter/min_length, max_length)")
    print(f"out:    {args.out}")
    if args.dry_run:
        params = urllib.parse.urlencode({"query": query, "format": "fasta"})
        print(f"\nurl:    {STREAM}?{params}")
        print("\ndry run: nothing downloaded")
        return 0

    print("\ndownloading from UniProtKB...", flush=True)
    size = fetch(query, args.out)
    if size < 0:
        return 1
    print(f"wrote {size:,} bytes")

    # Canonicalize identifiers before anything else reads the file. UniProt
    # serves `tr|A0A069QG82|A0A069QG82_HOYLO`; MMseqs2 parses that header and
    # keys its cluster table on the middle field alone, so a corpus written
    # with the three-part id produces a cluster table that joins to nothing,
    # 05_cluster_split reports every sequence as absent, and the split is
    # empty. That is not hypothetical: it is what this pipeline did on first
    # run. The stage A collection (08_collect_broad.py) already canonicalizes
    # for the same reason, and the Cas9 path did not because it had never been
    # run end to end.
    records = [
        FastaRecord(canonical_seq_id(r.seq_id), r.description, r.sequence)
        for r in read_fasta(args.out)
    ]
    collisions = len(records) - len({r.seq_id for r in records})
    if collisions:
        print(
            f"{collisions} identifiers collided after canonicalization; "
            "writing them would drop sequences from a join keyed on id",
            file=sys.stderr,
        )
        return 1
    write_fasta(args.out, records)
    print(f"canonicalized {len(records):,} identifiers (MMseqs2 keys on these)")

    # Report what arrived, by organism and by length, because the only numbers
    # that matter next are how many distinct organisms are present (the OGT
    # join is per-organism) and whether the length bound did anything.
    lengths: list[int] = []
    organisms: collections.Counter[str] = collections.Counter()
    for record in read_fasta(args.out):
        lengths.append(len(record.sequence))
        marker = "OS="
        text = record.description
        if marker in text:
            tail = text.split(marker, 1)[1]
            # OS= runs until the next two-letter KEY= token.
            for token in (" OX=", " GN=", " PE=", " SV="):
                if token in tail:
                    tail = tail.split(token, 1)[0]
            organisms[tail.strip()] += 1
        else:
            organisms["(no organism in header)"] += 1

    if not lengths:
        print("no sequences returned; nothing to filter", file=sys.stderr)
        return 1

    lengths.sort()
    print(f"\nsequences:        {len(lengths):,}")
    print(f"distinct organisms: {len(organisms):,}")
    print(f"length:           {lengths[0]}-{lengths[-1]} aa, "
          f"median {lengths[len(lengths) // 2]}")

    provenance = args.out.with_suffix(".provenance.txt")
    provenance.write_text(
        f"source:    UniProtKB REST stream, {STREAM}\n"
        f"query:     {query}\n"
        f"families:  {', '.join(CAS9_FAMILIES)}\n"
        f"retrieved: {date.today().isoformat()}\n"
        f"sequences: {len(lengths):,}\n"
        f"organisms: {len(organisms):,}\n"
        "note:      candidates only. Both nuclease domains are confirmed by\n"
        "           03_filter_domains.py, not by this query.\n"
    )
    print(f"provenance written to {provenance}")
    print(f"\nnext: python scripts/03_filter_domains.py --sequences {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
