"""Minimal FASTA reading and writing.

Deliberately dependency-free so the data stage runs before Biopython is
installed, and so there is no ambiguity about how records are parsed.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


def canonical_seq_id(raw_id: str) -> str:
    """Reduce a FASTA identifier to the bare accession.

    UniProt serves headers as ``sp|P33247|SQHC_ALIAD`` or ``tr|A0A123|...``.
    MMseqs2 parses those and uses only the middle field as its identifier,
    so a corpus written with the full three-part id produces cluster tables
    keyed on something else, every downstream join silently finds nothing,
    and a split ends up empty. Canonicalizing identifiers on the way in means
    every tool in the pipeline agrees on what a sequence is called.

    Identifiers without pipes, such as ``Q99ZW2``, ``UniRef50_X`` or
    ``WP_099233044.1``, are returned unchanged; MMseqs2 keeps those as-is.
    """
    text = raw_id.strip()
    if "|" not in text:
        return text
    parts = [p for p in text.split("|") if p]
    if not parts:
        return text
    # sp|ACC|NAME and tr|ACC|NAME both carry the accession in the middle.
    if len(parts) >= 3 and parts[0] in {"sp", "tr", "gb", "emb", "dbj", "ref", "pir", "prf"}:
        return parts[1]
    # Two-field forms such as ACC|NAME keep the first field.
    return parts[0]


@dataclass(frozen=True)
class FastaRecord:
    seq_id: str
    description: str
    sequence: str

    @property
    def header(self) -> str:
        return f"{self.seq_id} {self.description}".strip()

    def __len__(self) -> int:
        return len(self.sequence)


def read_fasta(path: str | Path) -> Iterator[FastaRecord]:
    """Stream records from a FASTA file, gzipped or not.

    Sequence lines are concatenated with whitespace stripped and uppercased.
    A duplicate id raises: downstream joins are keyed by id, and a silent
    overwrite would drop a sequence.

    A ``.gz`` suffix is read through gzip rather than decompressed to disk;
    UniProt serves bulk downloads compressed and the external corpus is 180 MB
    expanded, which is not worth writing out twice.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else None
    seen: set[str] = set()
    seq_id: str | None = None
    description = ""
    chunks: list[str] = []

    def finish() -> FastaRecord | None:
        if seq_id is None:
            return None
        return FastaRecord(seq_id, description, "".join(chunks))

    with (opener(path, "rt") if opener else path.open()) as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                record = finish()
                if record is not None:
                    yield record
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{path}:{lineno}: empty FASTA header")
                seq_id, _, description = header.partition(" ")
                if seq_id in seen:
                    raise ValueError(f"{path}:{lineno}: duplicate sequence id {seq_id!r}")
                seen.add(seq_id)
                description = description.strip()
                chunks = []
            else:
                if seq_id is None:
                    raise ValueError(f"{path}:{lineno}: sequence data before any header")
                chunks.append(line.upper().replace(" ", ""))

    record = finish()
    if record is not None:
        yield record


def write_fasta(path: str | Path, records: Iterable[FastaRecord], line_width: int = 60) -> int:
    """Write records with wrapped sequence lines. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w") as handle:
        for record in records:
            handle.write(f">{record.header}\n")
            seq = record.sequence
            if line_width > 0:
                for start in range(0, len(seq), line_width):
                    handle.write(seq[start : start + line_width] + "\n")
            else:
                handle.write(seq + "\n")
            written += 1
    return written
