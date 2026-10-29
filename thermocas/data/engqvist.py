"""Optimal growth temperature labels from the Engqvist database.

Engqvist (BMC Microbiology 18:177, 2018; data at 10.5281/zenodo.1175609,
CC BY 4.0) assigns a growth temperature to 21,498 microorganisms by mining
culture-collection catalogues. The distributed ``temperature_data.tsv`` is one
row per organism with a single ``temperature`` column, an NCBI ``taxid`` and
the full lineage.

This exists alongside :mod:`thermocas.data.ogt` (Gosha) because the
conditioning-ceiling claim is a claim about corpora in general, and a measure
validated on one corpus is a measure validated on one corpus. The external
corpus built from this table is a *measurement* corpus: it is never trained on
and never screened for Cas9, so nothing here may be fed to stage A or B. See
``scripts/16_external_corpus.py``.

Two differences from Gosha matter when the two are compared, and both run in
the direction of making Engqvist labels noisier rather than cleaner:

**There is no per-collection breakdown, so there is no cross-check.** Gosha
ships one column per source collection, which is what
``ogt.crosscheck.min_sources`` requires agreement across. Engqvist ships a
single consolidated value, so ``require_crosscheck`` has no counterpart here
and the external corpus cannot be filtered on label agreement.

**The two are not independent at the source.** Both scrape overlapping
culture-collection catalogues (DSMZ, ATCC, JCM), so agreement between them is
evidence that the ceiling is not an artifact of one *corpus*, not evidence
that it survives a genuinely independent *measurement* of growth temperature.

What is genuinely different is resolution: Engqvist carries 95 distinct
temperatures over bacteria and archaea where the Gosha-labeled stage A corpus
carries 47, and its values are not organism-rounded in the same places. That
is the property the threshold sweep is limited by, so it is the property worth
having a second opinion on.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from thermocas.data.ogt import (
    DEFAULT_MAX_PLAUSIBLE_C,
    DEFAULT_MIN_PLAUSIBLE_C,
    normalize_organism,
)

ORGANISM_COLUMN = "organism"
TEMPERATURE_COLUMN = "temperature"
TAXID_COLUMN = "taxid"
DOMAIN_COLUMN = "domain"

PROKARYOTE_DOMAINS = frozenset({"Bacteria", "Archaea"})


def engqvist_organism_key(name: str) -> str:
    """Reduce an Engqvist organism name to the same key Gosha names reduce to.

    Engqvist writes ``abiotrophia_adiacens`` where UniProt writes
    ``Abiotrophia adiacens``, so the underscores have to go before
    :func:`normalize_organism` sees the name -- otherwise the whole binomial
    is one token and every join misses.
    """
    return normalize_organism(name.replace("_", " "))


@dataclass
class EngqvistRecord:
    organism: str
    key: str
    ogt_c: float
    taxid: str
    domain: str


@dataclass
class EngqvistReport:
    """What the loader accepted, rejected and why."""

    rows: int = 0
    loaded: int = 0
    rejected_domain: int = 0
    rejected_nonnumeric: int = 0
    rejected_implausible: int = 0
    rejected_no_name: int = 0
    duplicate_keys: int = 0
    implausible_examples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"loaded {self.loaded}/{self.rows} Engqvist OGT records"]
        if self.rejected_domain:
            lines.append(f"  skipped {self.rejected_domain} rows outside the requested domains")
        if self.rejected_nonnumeric:
            lines.append(f"  rejected {self.rejected_nonnumeric} non-numeric temperatures")
        if self.rejected_implausible:
            shown = ", ".join(self.implausible_examples[:5])
            lines.append(
                f"  rejected {self.rejected_implausible} physically implausible values: {shown}"
            )
        if self.rejected_no_name:
            lines.append(f"  rejected {self.rejected_no_name} rows with an unusable organism name")
        if self.duplicate_keys:
            lines.append(
                f"  {self.duplicate_keys} organism keys seen more than once "
                f"(kept the first; Engqvist has one row per organism, so a "
                f"collision means two strains reduced to one binomial)"
            )
        return "\n".join(lines)


class EngqvistTable:
    """Organism-to-OGT lookup over Engqvist's ``temperature_data.tsv``."""

    def __init__(
        self,
        by_key: dict[str, EngqvistRecord],
        by_taxid: dict[str, EngqvistRecord],
        report: EngqvistReport,
    ) -> None:
        self.by_key = by_key
        self.by_taxid = by_taxid
        self.report = report

    def __len__(self) -> int:
        return len(self.by_key)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        domains: frozenset[str] | None = PROKARYOTE_DOMAINS,
        min_plausible_c: float = DEFAULT_MIN_PLAUSIBLE_C,
        max_plausible_c: float = DEFAULT_MAX_PLAUSIBLE_C,
    ) -> EngqvistTable:
        report = EngqvistReport()
        by_key: dict[str, EngqvistRecord] = {}
        by_taxid: dict[str, EngqvistRecord] = {}

        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                report.rows += 1
                domain = (row.get(DOMAIN_COLUMN) or "").strip()
                if domains is not None and domain not in domains:
                    report.rejected_domain += 1
                    continue
                name = (row.get(ORGANISM_COLUMN) or "").strip()
                key = engqvist_organism_key(name)
                if not key:
                    report.rejected_no_name += 1
                    continue
                raw = (row.get(TEMPERATURE_COLUMN) or "").strip()
                try:
                    ogt = float(raw)
                except (TypeError, ValueError):
                    report.rejected_nonnumeric += 1
                    continue
                if not min_plausible_c <= ogt <= max_plausible_c:
                    report.rejected_implausible += 1
                    if len(report.implausible_examples) < 5:
                        report.implausible_examples.append(f"{name} at {ogt} C")
                    continue

                record = EngqvistRecord(
                    organism=name.replace("_", " "),
                    key=key,
                    ogt_c=ogt,
                    taxid=(row.get(TAXID_COLUMN) or "").strip(),
                    domain=domain,
                )
                if key in by_key:
                    report.duplicate_keys += 1
                else:
                    by_key[key] = record
                    report.loaded += 1
                if record.taxid:
                    by_taxid.setdefault(record.taxid, record)

        return cls(by_key, by_taxid, report)

    def lookup(self, taxid: str | None, organism: str | None) -> EngqvistRecord | None:
        """Find a record by NCBI taxid first, then by normalized binomial.

        Taxid is tried first because it is exact, but it is the weaker key in
        practice: UniProt carries strain-level taxids (``OX=1314`` for one
        *Streptococcus pyogenes* strain among many) while Engqvist is indexed
        at species level, so most hits come from the name. Trying taxid first
        costs nothing and is right when the strain happens to be the species
        type strain.
        """
        if taxid:
            found = self.by_taxid.get(str(taxid).strip())
            if found:
                return found
        if organism:
            return self.by_key.get(normalize_organism(organism))
        return None
