"""Optimal growth temperature labels from the Gosha database.

Gosha (Helena-Bueno, Brown & Melnikov, bioRxiv 2021.12.21.473645) assigns an
OGT to 25,324 species by scraping culture-collection catalogues. The
distributed ``all_species_temperatures.csv`` carries a consensus column plus
one column per source collection (ATCC, DSMZ, JCM, ...), and the consensus is
the median of whichever collections reported a value.

Two properties of the real data shape this module:

**It contains physically impossible values.** *Bacillus haloalkaliphilus* is
listed at 230 °C and *Halomonas aerodenitrificans* at 187 °C, each from a
single collection. No organism grows above roughly 122 °C. Values are
therefore bounded, and anything outside the plausible range is rejected and
counted rather than binned.

**Scrape errors inflate temperature, which contaminates the thermophilic
classes specifically.** *Streptococcus massilliensis* is listed at 96 °C from
one collection, which would label a mesophile as a hyperthermophile. The
genuine hyperthermophiles (Pyrococcus, Pyrolobus, Methanopyrus) all carry
several agreeing collections. Agreement across independent collections is
therefore computed per species and can be required for a label to be used.

That makes the per-collection columns the cross-check the project brief wanted
a second database for, without leaving Gosha.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

ORGANISM_COLUMN = "Organism"
CONSENSUS_COLUMN = "Consensus optimal temperature"

# No cultured organism is known to grow above ~122 C (Methanopyrus kandleri
# strain 116). The default ceiling leaves headroom above that; the floor
# admits psychrophiles well below zero growth optima.
DEFAULT_MIN_PLAUSIBLE_C = -20.0
DEFAULT_MAX_PLAUSIBLE_C = 130.0

_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
# Strain designations, culture-collection codes and subspecies qualifiers that
# appear after the binomial in NCBI-style organism names.
_STRAIN_NOISE = re.compile(
    r"\s+(?:subsp\.|sp\.|str\.|strain|var\.|serovar|serotype|biovar|pv\.)\s+.*$",
    re.IGNORECASE,
)


def normalize_organism(name: str) -> str:
    """Reduce an organism name to a joinable key.

    Gosha's species file uses bare binomials (``Abditibacterium utsteinense``)
    while NCBI-derived sources carry strain suffixes (``Polaribacter irgensii
    23-P``). Reducing both to the first two tokens, lowercased, is what lets
    them join. This is a heuristic: it cannot distinguish strains with
    genuinely different growth optima, and that limitation is reported by the
    join rather than hidden.
    """
    cleaned = name.strip().strip('"').replace("#", "")
    cleaned = _STRAIN_NOISE.sub("", cleaned)
    cleaned = re.sub(r"[\[\]()]", " ", cleaned)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    # Candidatus is a status prefix, not a genus.
    if tokens[0].lower() == "candidatus" and len(tokens) > 1:
        tokens = tokens[1:]
    return " ".join(tokens[:2]).lower()


def parse_organism_from_header(description: str) -> str | None:
    """Pull the source organism out of a FASTA description.

    Three header conventions appear in this pipeline's inputs:

    * UniRef:  ``... n=1 Tax=Streptococcus pyogenes TaxID=1314 RepID=...``
    * UniProt: ``... OS=Streptococcus pyogenes OX=1314 GN=cas9 ...``
    * NCBI:    ``type II CRISPR ... endonuclease Cas9 [Geobacillus ...]``

    Returns None when no organism can be found, which the caller counts as
    unlabeled rather than guessing.
    """
    text = description.strip()
    if not text:
        return None

    # UniRef Tax= runs until the next KEY= token.
    match = re.search(r"\bTax=(.+?)(?=\s+\w+=|$)", text)
    if match:
        return match.group(1).strip() or None

    # UniProt OS= likewise.
    match = re.search(r"\bOS=(.+?)(?=\s+\w\w=|$)", text)
    if match:
        return match.group(1).strip() or None

    # NCBI defline: the trailing bracketed group.
    if text.endswith("]") and "[" in text:
        candidate = text[text.rindex("[") + 1 : -1].strip()
        return candidate or None

    # This project's own seed headers: "Name | Organism | type=... | ...".
    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
        if len(parts) >= 2 and parts[1] and "=" not in parts[1]:
            return parts[1]

    return None


@dataclass(frozen=True)
class OgtRecord:
    """One species' OGT, with the evidence behind it."""

    organism: str
    organism_key: str
    ogt_c: float
    source_values: dict[str, float] = field(default_factory=dict)

    @property
    def n_sources(self) -> int:
        return len(self.source_values)

    @property
    def source_spread_c(self) -> float:
        """Max minus min across reporting collections; 0.0 for a single source."""
        if self.n_sources < 2:
            return 0.0
        values = list(self.source_values.values())
        return max(values) - min(values)

    @property
    def source_median_c(self) -> float | None:
        if not self.source_values:
            return None
        return statistics.median(self.source_values.values())

    def bin_agreement(self, bin_for) -> float:
        """Fraction of reporting collections that agree on the consensus's bin.

        Agreement is measured on the bin rather than on absolute degrees
        because spread scales with temperature. *Pyrococcus furiosus* has a
        20 °C spread across five collections and is beyond doubt a
        hyperthermophile; a 5 °C absolute tolerance would reject it along with
        most other genuine thermophiles, while catching nothing that the
        source count does not already catch.

        A low value is meaningful: *Thermus thermophilus* sits at 67 °C with
        sources straddling the 70 °C boundary, so its class genuinely is
        ambiguous and it should not be used as a confident label.
        """
        if not self.source_values:
            return 0.0
        consensus_bin = bin_for(self.ogt_c)
        agreeing = sum(1 for v in self.source_values.values() if bin_for(v) == consensus_bin)
        return agreeing / len(self.source_values)

    def is_verified(
        self, bin_for, min_sources: int = 2, min_bin_agreement: float = 0.67
    ) -> bool:
        """Whether independent collections support the binned label.

        This is the within-Gosha cross-check, and it rests on two separate
        signals:

        * ``min_sources`` catches single-collection scrape errors, which are
          the actual error mode. *Streptococcus massilliensis* is listed at
          96 °C by one collection alone, which would label a mesophile as a
          hyperthermophile.
        * ``min_bin_agreement`` catches genuine class ambiguity near a bin
          boundary.
        """
        return (
            self.n_sources >= min_sources
            and self.bin_agreement(bin_for) >= min_bin_agreement
        )


@dataclass
class OgtTableReport:
    """What the loader accepted, rejected and why."""

    rows: int = 0
    loaded: int = 0
    rejected_nonnumeric: int = 0
    rejected_implausible: int = 0
    rejected_no_name: int = 0
    duplicate_keys: int = 0
    implausible_examples: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"loaded {self.loaded}/{self.rows} OGT records"]
        if self.rejected_nonnumeric:
            lines.append(f"  rejected {self.rejected_nonnumeric} non-numeric consensus values")
        if self.rejected_implausible:
            shown = ", ".join(self.implausible_examples[:5])
            lines.append(
                f"  rejected {self.rejected_implausible} physically implausible values: {shown}"
            )
        if self.rejected_no_name:
            lines.append(f"  rejected {self.rejected_no_name} rows with an unusable organism name")
        if self.duplicate_keys:
            lines.append(
                f"  {self.duplicate_keys} duplicate organism keys (kept the better-supported row)"
            )
        return "\n".join(lines)


class OgtTable:
    """Species-to-OGT lookup built from Gosha's all-species CSV."""

    def __init__(self, records: dict[str, OgtRecord], report: OgtTableReport) -> None:
        self.records = records
        self.report = report

    def __len__(self) -> int:
        return len(self.records)

    def lookup(self, organism: str) -> OgtRecord | None:
        return self.records.get(normalize_organism(organism))

    @classmethod
    def from_gosha_csv(
        cls,
        path: str | Path,
        min_plausible_c: float = DEFAULT_MIN_PLAUSIBLE_C,
        max_plausible_c: float = DEFAULT_MAX_PLAUSIBLE_C,
    ) -> OgtTable:
        """Load Gosha's ``all_species_temperatures.csv``.

        Every column other than the organism and consensus columns is treated
        as an independent culture collection, so new collections in a later
        release are picked up without a code change.
        """
        path = Path(path)
        report = OgtTableReport()
        records: dict[str, OgtRecord] = {}

        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: file is empty")
            missing = {ORGANISM_COLUMN, CONSENSUS_COLUMN} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"{path}: missing expected column(s) {sorted(missing)}. "
                    f"Found: {reader.fieldnames}. Gosha's schema may have changed; "
                    "check the download recorded in PROJECT_LOG.md."
                )
            collection_columns = [
                name
                for name in reader.fieldnames
                if name not in {ORGANISM_COLUMN, CONSENSUS_COLUMN} and name
            ]

            for row in reader:
                report.rows += 1
                organism = (row.get(ORGANISM_COLUMN) or "").strip()
                key = normalize_organism(organism)
                if not key:
                    report.rejected_no_name += 1
                    continue

                raw = (row.get(CONSENSUS_COLUMN) or "").strip()
                if not _NUMERIC.match(raw):
                    report.rejected_nonnumeric += 1
                    continue
                ogt = float(raw)

                if not min_plausible_c <= ogt <= max_plausible_c:
                    report.rejected_implausible += 1
                    if len(report.implausible_examples) < 5:
                        report.implausible_examples.append(f"{organism} ({ogt:g} C)")
                    continue

                sources = {
                    column: float(value.strip())
                    for column in collection_columns
                    if _NUMERIC.match((value := row.get(column) or "").strip())
                }
                record = OgtRecord(
                    organism=organism, organism_key=key, ogt_c=ogt, source_values=sources
                )

                existing = records.get(key)
                if existing is not None:
                    report.duplicate_keys += 1
                    # Prefer the row backed by more collections; a bare
                    # binomial and a strain row can normalize to one key.
                    if record.n_sources <= existing.n_sources:
                        continue
                else:
                    report.loaded += 1
                records[key] = record

        return cls(records, report)


@dataclass
class JoinReport:
    """Coverage of an OGT join, which Gate 1 reads."""

    total: int = 0
    labeled: int = 0
    unlabeled: int = 0
    verified: int = 0
    bin_counts: dict[str, int] = field(default_factory=dict)
    verified_bin_counts: dict[str, int] = field(default_factory=dict)
    unlabeled_examples: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.labeled / self.total if self.total else 0.0

    @property
    def verified_fraction(self) -> float:
        return self.verified / self.labeled if self.labeled else 0.0

    def summary(self) -> str:
        lines = [
            f"labeled {self.labeled}/{self.total} sequences ({self.coverage:.1%})",
            f"  cross-checked by >=2 agreeing collections: {self.verified} "
            f"({self.verified_fraction:.1%} of labeled)",
        ]
        for name, count in self.bin_counts.items():
            verified = self.verified_bin_counts.get(name, 0)
            lines.append(f"  {name:<8} {count:>7}  (verified {verified})")
        if self.unlabeled_examples:
            lines.append(f"  unlabeled e.g. {', '.join(self.unlabeled_examples[:5])}")
        return "\n".join(lines)


def gate1_failures(report: JoinReport, min_per_bin: int, use_verified: bool = False) -> list[str]:
    """Bins that fall short of the Gate 1 minimum.

    An empty list means Gate 1 passes. Checking against the verified counts is
    the stricter reading, and the honest one if the labels are to be trusted at
    the thermophilic end.
    """
    counts = report.verified_bin_counts if use_verified else report.bin_counts
    return [
        f"{name}: {counts.get(name, 0)} < {min_per_bin}"
        for name in report.bin_counts
        if counts.get(name, 0) < min_per_bin
    ]


def iter_unique(values: Iterator[str]) -> Iterator[str]:
    """Yield values once, in first-seen order."""
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            yield value
