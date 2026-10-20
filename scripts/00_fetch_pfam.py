#!/usr/bin/env python3
"""Fetch the Pfam HMM profiles named in configs/data.yaml.

Profiles are reference data: re-downloaded rather than committed. This script
records the exact profile versions it fetched so they can be pasted into
PROJECT_LOG.md, since a Pfam release bump changes which sequences match.

Usage:
    python scripts/00_fetch_pfam.py
    python scripts/00_fetch_pfam.py --force
"""

from __future__ import annotations

import argparse
import gzip
import ssl
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thermocas.config import Config  # noqa: E402
from thermocas.data.domains import DomainRequirements  # noqa: E402

INTERPRO_HMM = "https://www.ebi.ac.uk/interpro/api/entry/pfam/{accession}/?annotation=hmm"
USER_AGENT = "thermocas-gen/0.1 (profile fetch; https://github.com/somlunavat/thermocas-gen)"


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    """Prefer certifi's bundle; the python.org macOS build has no keychain wiring."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch(accession: str, destination: Path, timeout: int = 60) -> str:
    """Download one profile, returning its declared NAME and ACC."""
    request = urllib.request.Request(
        INTERPRO_HMM.format(accession=accession), headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"InterPro returned HTTP {exc.code} for {accession}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach InterPro ({exc.reason})") from exc

    # The annotation endpoint serves gzip; tolerate plain text in case it changes.
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    text = payload.decode()
    if not text.lstrip().startswith("HMMER3"):
        raise RuntimeError(f"{accession}: response is not a HMMER3 profile")

    destination.write_text(text)
    name = version = ""
    for line in text.splitlines():
        if line.startswith("NAME"):
            name = line.split(maxsplit=1)[1].strip()
        elif line.startswith("ACC"):
            version = line.split(maxsplit=1)[1].strip()
            break
    return f"{name} ({version})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data")
    parser.add_argument("--force", action="store_true", help="re-download profiles already present")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    requirements = DomainRequirements.from_config(cfg)
    profile_dir = cfg.resolve_path("domains/profile_dir")
    profile_dir.mkdir(parents=True, exist_ok=True)

    accessions = requirements.all_accessions()
    print(f"{len(accessions)} profiles -> {profile_dir}")

    versions: list[str] = []
    failures: list[str] = []
    for accession in accessions:
        destination = profile_dir / f"{accession}.hmm"
        if destination.is_file() and not args.force:
            print(f"  {accession}: already present (--force to refresh)")
            header = [ln for ln in destination.read_text().splitlines()[:5] if ln.startswith("ACC")]
            versions.append(f"{accession} ({header[0].split()[-1] if header else '?'})")
            continue
        try:
            recorded = fetch(accession, destination)
        except RuntimeError as exc:
            print(f"  {accession}: FAILED {exc}", file=sys.stderr)
            failures.append(accession)
            continue
        print(f"  {accession}: {recorded}")
        versions.append(f"{accession} {recorded}")

    print("\nrecord these versions in PROJECT_LOG.md:")
    print("  " + "; ".join(versions))

    if failures:
        print(f"\n{len(failures)} profile(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
