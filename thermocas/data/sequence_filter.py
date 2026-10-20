"""Sequence-level filtering for the data stage.

Every rejection is recorded with a reason and counted. Nothing is dropped
silently. The prior codebase discarded over-length sequences inside a boolean
predicate, which is how every full-length Cas9 left its training set without
anyone noticing; this module exists so that cannot recur.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from thermocas.data.fasta import FastaRecord


class Rejection(str, Enum):
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    NONSTANDARD_RESIDUE = "nonstandard_residue"
    DUPLICATE_SEQUENCE = "duplicate_sequence"
    EMPTY = "empty"


@dataclass
class FilterReport:
    """Counts and examples for every rejection reason."""

    kept: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)
    length_min: int | None = None
    length_max: int | None = None

    @property
    def total(self) -> int:
        return self.kept + sum(self.rejected.values())

    def record_rejection(self, reason: Rejection, seq_id: str, max_examples: int = 5) -> None:
        self.rejected[reason.value] += 1
        bucket = self.examples.setdefault(reason.value, [])
        if len(bucket) < max_examples:
            bucket.append(seq_id)

    def summary(self) -> str:
        lines = [f"kept {self.kept}/{self.total} sequences"]
        if self.kept:
            lines.append(f"  length range: {self.length_min}-{self.length_max} aa")
        for reason, count in self.rejected.most_common():
            shown = ", ".join(self.examples.get(reason, []))
            lines.append(f"  rejected {count:>7}  {reason}  e.g. {shown}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SequenceFilter:
    """Length, residue-alphabet and duplicate filtering.

    Built from ``configs/data.yaml``'s ``filter`` section.
    """

    min_length: int
    max_length: int
    allowed_residues: frozenset[str]
    drop_exact_duplicates: bool = True

    @classmethod
    def from_config(cls, cfg, path: str = "filter") -> SequenceFilter:
        section = cfg.section(path)
        min_length = int(section["min_length"])
        max_length = int(section["max_length"])
        if min_length > max_length:
            raise ValueError(f"min_length {min_length} exceeds max_length {max_length}")
        return cls(
            min_length=min_length,
            max_length=max_length,
            allowed_residues=frozenset(str(section["allowed_residues"]).upper()),
            drop_exact_duplicates=bool(section["drop_exact_duplicates"]),
        )

    def check(self, sequence: str) -> Rejection | None:
        """Return the reason this sequence fails, or None if it passes."""
        seq = sequence.strip().upper()
        if not seq:
            return Rejection.EMPTY
        if len(seq) < self.min_length:
            return Rejection.TOO_SHORT
        if len(seq) > self.max_length:
            return Rejection.TOO_LONG
        if not set(seq) <= self.allowed_residues:
            return Rejection.NONSTANDARD_RESIDUE
        return None

    def apply(
        self, records: Iterable[FastaRecord]
    ) -> tuple[list[FastaRecord], FilterReport]:
        """Filter records, returning survivors and a full accounting."""
        report = FilterReport()
        kept: list[FastaRecord] = []
        seen_sequences: dict[str, str] = {}

        for record in records:
            reason = self.check(record.sequence)
            if reason is not None:
                report.record_rejection(reason, record.seq_id)
                continue

            seq = record.sequence.strip().upper()
            if self.drop_exact_duplicates and seq in seen_sequences:
                # Keep the first id so the choice is deterministic in file order.
                report.record_rejection(
                    Rejection.DUPLICATE_SEQUENCE,
                    f"{record.seq_id}(=={seen_sequences[seq]})",
                )
                continue
            seen_sequences[seq] = record.seq_id

            kept.append(FastaRecord(record.seq_id, record.description, seq))
            report.kept += 1
            length = len(seq)
            report.length_min = length if report.length_min is None else min(report.length_min, length)
            report.length_max = length if report.length_max is None else max(report.length_max, length)

        return kept, report
