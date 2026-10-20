"""Stage A: species selection and Cas9 exclusion for broad pre-training.

Temperature conditioning cannot be learned from Cas9 alone — the Gate 1
pre-flight found roughly ten independent thermophilic Cas9 clusters, and the
scarcity is biological rather than a database artifact. This stage builds a
corpus of general temperature-labeled bacterial proteins where the signal does
exist, so the growth-temperature token means something before stage B ever
sees a Cas9.

Two properties of this corpus are load-bearing:

**It must be balanced.** Gosha holds 13,898 cross-checked mesophilic species
against 672 thermophilic ones. Sampling proportionally would let the model
minimize its loss by ignoring the conditioning token entirely and always
predicting mesophilic features, which is precisely the failure Gate 2 exists
to catch. The scarce class is taken in full and the abundant one matched to
it.

**It must contain no Cas9.** If a held-out Cas9 test cluster appears in
pre-training, the stage B leakage gate is measuring nothing: the model has
already seen the test sequence, just in an earlier stage. Screening is done by
annotation and by alignment, and the alignment threshold is stricter than the
stage B one because a large corpus makes a borderline homolog both likelier
and harder to notice.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from thermocas.data.ogt import OgtRecord


@dataclass(frozen=True)
class SpeciesSelection:
    """Settings for choosing which species contribute proteins."""

    require_crosscheck: bool
    max_per_bin: int
    balance: str
    exclude_positive_control_hosts: bool

    VALID_BALANCE = ("match_smallest", "max_per_bin", "none")

    def __post_init__(self) -> None:
        if self.balance not in self.VALID_BALANCE:
            raise ValueError(
                f"unknown balance mode {self.balance!r}; expected one of {self.VALID_BALANCE}"
            )
        if self.max_per_bin < 1:
            raise ValueError(f"max_per_bin must be at least 1, got {self.max_per_bin}")

    @classmethod
    def from_config(cls, cfg, path: str = "species") -> SpeciesSelection:
        section = cfg.section(path)
        return cls(
            require_crosscheck=bool(section["require_crosscheck"]),
            max_per_bin=int(section["max_per_bin"]),
            balance=str(section["balance"]),
            exclude_positive_control_hosts=bool(section["exclude_positive_control_hosts"]),
        )


@dataclass
class SpeciesSelectionReport:
    considered: int = 0
    per_bin_available: dict[str, int] = field(default_factory=dict)
    per_bin_selected: dict[str, int] = field(default_factory=dict)
    dropped_uncrosschecked: int = 0
    dropped_excluded_host: int = 0

    def summary(self) -> str:
        lines = [f"considered {self.considered} species"]
        if self.dropped_uncrosschecked:
            lines.append(f"  dropped {self.dropped_uncrosschecked} without a cross-checked OGT")
        if self.dropped_excluded_host:
            lines.append(f"  dropped {self.dropped_excluded_host} positive-control host(s)")
        for name in sorted(self.per_bin_available):
            available = self.per_bin_available[name]
            selected = self.per_bin_selected.get(name, 0)
            lines.append(f"  {name:<8} {selected:>5} selected of {available} available")
        return "\n".join(lines)


def select_species(
    records: Iterable[OgtRecord],
    bin_for,
    selection: SpeciesSelection,
    min_sources: int = 2,
    min_bin_agreement: float = 0.67,
    excluded_hosts: Iterable[str] = (),
    seed: int = 0,
) -> tuple[dict[str, list[OgtRecord]], SpeciesSelectionReport]:
    """Choose species per growth-temperature bin.

    Returns ``bin name -> species records`` and a report. Selection is
    deterministic for a given seed: species are ordered by how well supported
    their OGT is, so that when the abundant class has to be cut down, the
    best-evidenced species survive rather than an arbitrary sample.
    """
    import random

    report = SpeciesSelectionReport()
    excluded = {host.strip().lower() for host in excluded_hosts if host.strip()}

    by_bin: dict[str, list[OgtRecord]] = {}
    for record in records:
        report.considered += 1

        if selection.require_crosscheck and not record.is_verified(
            bin_for, min_sources=min_sources, min_bin_agreement=min_bin_agreement
        ):
            report.dropped_uncrosschecked += 1
            continue

        if selection.exclude_positive_control_hosts and excluded:
            key = record.organism.strip().lower()
            if any(key.startswith(host) or host.startswith(key) for host in excluded):
                report.dropped_excluded_host += 1
                continue

        name = bin_for(record.ogt_c)
        by_bin.setdefault(name, []).append(record)

    for name, members in by_bin.items():
        report.per_bin_available[name] = len(members)

    if not by_bin:
        return {}, report

    if selection.balance == "match_smallest":
        quota = min(min(len(m) for m in by_bin.values()), selection.max_per_bin)
        quotas = {name: quota for name in by_bin}
    elif selection.balance == "max_per_bin":
        quotas = {name: min(len(by_bin[name]), selection.max_per_bin) for name in by_bin}
    else:
        quotas = {name: len(by_bin[name]) for name in by_bin}

    rng = random.Random(seed)
    selected: dict[str, list[OgtRecord]] = {}
    for name, members in by_bin.items():
        # Best-evidenced first: more reporting collections, then tighter
        # agreement. Ties shuffle deterministically so the cut is not
        # alphabetical by organism, which would bias toward certain genera.
        ordered = sorted(members, key=lambda r: r.organism_key)
        rng.shuffle(ordered)
        ordered.sort(key=lambda r: (-r.n_sources, r.source_spread_c))
        selected[name] = ordered[: quotas[name]]
        report.per_bin_selected[name] = len(selected[name])

    return selected, report


@dataclass(frozen=True)
class Cas9Exclusion:
    """How stage A screens Cas9-family proteins out of the pre-training set."""

    enabled: bool
    name_patterns: tuple[str, ...]
    pfam_profiles: tuple[str, ...]
    max_identity_to_heldout: float
    alignment_coverage: float
    fail_on_violation: bool

    def __post_init__(self) -> None:
        # Normalize here rather than in from_config so the invariants hold for
        # every construction path. PF22702.3 and PF22702 must behave alike, or
        # a Pfam release bump silently stops excluding a Cas9 model and lets
        # Cas9 into the pre-training corpus.
        object.__setattr__(
            self, "pfam_profiles", tuple(str(p).split(".")[0] for p in self.pfam_profiles)
        )
        object.__setattr__(
            self, "name_patterns", tuple(str(p).lower() for p in self.name_patterns)
        )

        if not 0.0 < self.max_identity_to_heldout <= 1.0:
            raise ValueError(
                f"max_identity_to_heldout must be a fraction in (0, 1], got "
                f"{self.max_identity_to_heldout}. Use 0.30 for 30%, not 30."
            )
        if not 0.0 <= self.alignment_coverage <= 1.0:
            raise ValueError(
                f"alignment_coverage must be a fraction in [0, 1], got {self.alignment_coverage}"
            )

    @classmethod
    def from_config(cls, cfg, path: str = "exclude_cas9") -> Cas9Exclusion:
        section = cfg.section(path)
        return cls(
            enabled=bool(section["enabled"]),
            name_patterns=tuple(str(p).lower() for p in section["name_patterns"]),
            pfam_profiles=tuple(str(p).split(".")[0] for p in section["pfam_profiles"]),
            max_identity_to_heldout=float(section["max_identity_to_heldout"]),
            alignment_coverage=float(section["alignment_coverage"]),
            fail_on_violation=bool(section["fail_on_violation"]),
        )

    def matches_annotation(self, description: str) -> str | None:
        """Return the pattern that marks this as Cas9-family, or None.

        Matching is on word-ish boundaries so that an unrelated protein whose
        description merely contains the letters "cas9" inside a longer token is
        not excluded, while "Cas9", "cas9-like" and "CRISPR-associated
        endonuclease Cas9" all are.
        """
        if not self.enabled:
            return None
        text = description.lower()
        for pattern in self.name_patterns:
            if re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", text):
                return pattern
        return None


@dataclass
class ExclusionReport:
    """What the Cas9 screen removed, and why."""

    examined: int = 0
    excluded_by_name: int = 0
    excluded_by_domain: int = 0
    excluded_by_alignment: int = 0
    name_hits: dict[str, int] = field(default_factory=dict)
    alignment_examples: list[str] = field(default_factory=list)

    @property
    def total_excluded(self) -> int:
        return self.excluded_by_name + self.excluded_by_domain + self.excluded_by_alignment

    def summary(self) -> str:
        lines = [
            f"screened {self.examined} sequences, excluded {self.total_excluded} Cas9-family",
            f"  by annotation: {self.excluded_by_name}",
            f"  by Pfam domain: {self.excluded_by_domain}",
            f"  by alignment to held-out Cas9: {self.excluded_by_alignment}",
        ]
        for pattern, count in sorted(self.name_hits.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {pattern!r}: {count}")
        if self.alignment_examples:
            lines.append(f"  alignment hits e.g. {', '.join(self.alignment_examples[:5])}")
        return "\n".join(lines)
