"""Homolog search command construction and hit-table parsing.

The search itself is a protein query against a protein database (blastp-style).
Not blastx: both the seeds and the database are protein, and blastx would
translate a nucleotide query that does not exist here.

Command construction lives in Python rather than in shell so that the unit
conversions are covered by tests. The one that matters: coverage is a fraction
in ``configs/data.yaml`` (0.70) but DIAMOND's ``--query-cover`` and MMseqs2's
``-c`` take different units, and passing 0.70 where 70 is expected would
quietly accept almost every fragment in the database.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# DIAMOND tabular fields, in the order they are emitted. qcovhsp is required
# because coverage filtering is checked again when the table is parsed.
DIAMOND_FIELDS = (
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
    "qcovhsp",
    "scovhsp",
    "slen",
)

MMSEQS_FIELDS = (
    "query",
    "target",
    "pident",
    "alnlen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
    "qcov",
    "tcov",
    "tlen",
)

# Normalized column names used downstream, independent of which tool ran.
HIT_COLUMNS = (
    "query_id",
    "subject_id",
    "percent_identity",
    "alignment_length",
    "evalue",
    "bitscore",
    "query_coverage",
    "subject_coverage",
    "subject_length",
)


class SearchToolError(RuntimeError):
    """A search tool is missing, or returned a non-zero exit status."""


@dataclass(frozen=True)
class SearchParams:
    """Homolog search settings, read from ``configs/data.yaml``."""

    tool: str
    database: str
    max_evalue: float
    min_query_coverage: float  # fraction in [0, 1]
    max_target_seqs: int
    sensitivity: str
    threads: int

    SUPPORTED_TOOLS = ("diamond", "mmseqs2")

    def __post_init__(self) -> None:
        # Validated here rather than in from_config so the invariants hold for
        # every construction path, including direct use in tests and notebooks.
        if self.tool not in self.SUPPORTED_TOOLS:
            raise ValueError(
                f"unsupported search tool {self.tool!r}; expected one of "
                f"{self.SUPPORTED_TOOLS}. This is a protein-vs-protein search, "
                "so blastx-style tools do not apply."
            )
        if not 0.0 <= self.min_query_coverage <= 1.0:
            raise ValueError(
                f"min_query_coverage must be a fraction in [0, 1], got "
                f"{self.min_query_coverage}. Use 0.70 for 70%, not 70."
            )
        if self.threads < 1:
            raise ValueError(f"threads must be at least 1, got {self.threads}")
        if self.max_evalue <= 0:
            raise ValueError(f"max_evalue must be positive, got {self.max_evalue}")
        if self.max_target_seqs < 1:
            raise ValueError(f"max_target_seqs must be at least 1, got {self.max_target_seqs}")

    @classmethod
    def from_config(cls, cfg, path: str = "search") -> SearchParams:
        section = cfg.section(path)
        return cls(
            tool=str(section["tool"]).lower(),
            database=str(section["database"]),
            max_evalue=float(section["max_evalue"]),
            min_query_coverage=float(section["min_query_coverage"]),
            max_target_seqs=int(section["max_target_seqs"]),
            sensitivity=str(section["sensitivity"]),
            threads=int(section["threads"]),
        )

    @property
    def query_coverage_percent(self) -> float:
        """DIAMOND's --query-cover is a percentage, not a fraction."""
        return self.min_query_coverage * 100.0


def require_tool(name: str) -> str:
    """Resolve an executable, with an actionable message when it is absent."""
    path = shutil.which(name)
    if path is None:
        raise SearchToolError(
            f"{name!r} is not on PATH. Install it into the conda env "
            f"(environment-local.yml / environment-gpu.yml). On Apple Silicon some "
            f"bioconda packages have no osx-arm64 build, in which case run this stage "
            f"on a cloud machine and note it in PROJECT_LOG.md."
        )
    return path


def tool_version(name: str) -> str:
    """Best-effort version string, for the record in PROJECT_LOG.md."""
    executable = shutil.which(name)
    if executable is None:
        return "not installed"
    for flag in ("--version", "version", "-h"):
        try:
            result = subprocess.run(
                [executable, flag], capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (result.stdout or result.stderr).strip().splitlines()
        if output:
            return output[0].strip()
    return "unknown"


def diamond_makedb_command(database_fasta: Path, index_prefix: Path, threads: int) -> list[str]:
    return [
        "diamond",
        "makedb",
        "--in",
        str(database_fasta),
        "-d",
        str(index_prefix),
        "--threads",
        str(threads),
    ]


def diamond_blastp_command(
    params: SearchParams, index_prefix: Path, query_fasta: Path, out_tsv: Path
) -> list[str]:
    """blastp-style protein-vs-protein search."""
    sensitivity_flag = f"--{params.sensitivity}" if params.sensitivity else None
    command = [
        "diamond",
        "blastp",
        "-d",
        str(index_prefix),
        "-q",
        str(query_fasta),
        "-o",
        str(out_tsv),
        "--outfmt",
        "6",
        *DIAMOND_FIELDS,
        "--evalue",
        repr(params.max_evalue),
        # Percentage, not a fraction. See SearchParams.query_coverage_percent.
        "--query-cover",
        f"{params.query_coverage_percent:g}",
        "--max-target-seqs",
        str(params.max_target_seqs),
        "--threads",
        str(params.threads),
    ]
    if sensitivity_flag:
        command.append(sensitivity_flag)
    return command


def mmseqs_search_commands(
    params: SearchParams,
    query_fasta: Path,
    database_fasta: Path,
    out_tsv: Path,
    workdir: Path,
) -> list[list[str]]:
    """MMseqs2 equivalent: createdb, createdb, search, convertalis.

    MMseqs2's ``-c`` takes a fraction, unlike DIAMOND's percentage, and
    ``--cov-mode 1`` makes it query coverage specifically.
    """
    query_db = workdir / "queryDB"
    target_db = workdir / "targetDB"
    result_db = workdir / "resultDB"
    tmp = workdir / "tmp"
    return [
        ["mmseqs", "createdb", str(query_fasta), str(query_db)],
        ["mmseqs", "createdb", str(database_fasta), str(target_db)],
        [
            "mmseqs",
            "search",
            str(query_db),
            str(target_db),
            str(result_db),
            str(tmp),
            "-e",
            repr(params.max_evalue),
            "-c",
            f"{params.min_query_coverage:g}",
            "--cov-mode",
            "1",
            "--max-seqs",
            str(params.max_target_seqs),
            "--threads",
            str(params.threads),
        ],
        [
            "mmseqs",
            "convertalis",
            str(query_db),
            str(target_db),
            str(result_db),
            str(out_tsv),
            "--format-output",
            ",".join(MMSEQS_FIELDS),
            "--threads",
            str(params.threads),
        ],
    ]


@dataclass(frozen=True)
class Hit:
    query_id: str
    subject_id: str
    percent_identity: float
    alignment_length: int
    evalue: float
    bitscore: float
    query_coverage: float  # fraction in [0, 1]
    subject_coverage: float
    subject_length: int


def _coverage_to_fraction(value: float) -> float:
    """Both tools can report coverage as a percentage; normalize to a fraction.

    DIAMOND's qcovhsp is 0-100. MMseqs2's qcov is 0-1. Normalizing on parse
    means the downstream threshold comparison has one unit, whichever tool ran.
    """
    return value / 100.0 if value > 1.0 else value


def parse_hits(path: str | Path, tool: str) -> Iterator[Hit]:
    """Parse a tabular hit file from either tool into normalized Hit rows."""
    tool = tool.lower()
    if tool == "diamond":
        n_fields = len(DIAMOND_FIELDS)
        indices = {name: i for i, name in enumerate(DIAMOND_FIELDS)}
        keys = ("qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qcovhsp", "scovhsp", "slen")
    elif tool == "mmseqs2":
        n_fields = len(MMSEQS_FIELDS)
        indices = {name: i for i, name in enumerate(MMSEQS_FIELDS)}
        keys = ("query", "target", "pident", "alnlen", "evalue", "bits", "qcov", "tcov", "tlen")
    else:
        raise ValueError(f"unsupported search tool {tool!r}")

    with Path(path).open() as handle:
        reader = csv.reader(handle, delimiter="\t")
        for lineno, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != n_fields:
                raise ValueError(
                    f"{path}:{lineno}: expected {n_fields} columns for {tool}, got {len(row)}. "
                    "The hit table was probably produced with a different --outfmt."
                )
            get = lambda name: row[indices[name]]  # noqa: E731
            try:
                yield Hit(
                    query_id=get(keys[0]),
                    subject_id=get(keys[1]),
                    percent_identity=float(get(keys[2])),
                    alignment_length=int(get(keys[3])),
                    evalue=float(get(keys[4])),
                    bitscore=float(get(keys[5])),
                    query_coverage=_coverage_to_fraction(float(get(keys[6]))),
                    subject_coverage=_coverage_to_fraction(float(get(keys[7]))),
                    subject_length=int(get(keys[8])),
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: malformed hit row: {exc}") from exc


def filter_hits(hits: Iterator[Hit], params: SearchParams) -> tuple[list[Hit], dict[str, int]]:
    """Re-apply the configured thresholds to parsed hits.

    The search tool already filtered, but re-checking here means the committed
    hit table is verifiably consistent with the config that is supposed to have
    produced it, rather than with whatever flags were typed by hand.
    """
    kept: list[Hit] = []
    counts = {"total": 0, "failed_evalue": 0, "failed_coverage": 0, "kept": 0}
    for hit in hits:
        counts["total"] += 1
        if hit.evalue > params.max_evalue:
            counts["failed_evalue"] += 1
            continue
        if hit.query_coverage < params.min_query_coverage:
            counts["failed_coverage"] += 1
            continue
        kept.append(hit)
        counts["kept"] += 1
    return kept, counts


def write_hits(path: str | Path, hits: list[Hit], params: SearchParams) -> int:
    """Write the normalized hit table that stage 1c reads.

    The header comments record the tool, database and thresholds that produced
    the table, so a committed table can be checked against the config that
    claims to have generated it. Coverage columns are always fractions here,
    whichever tool ran.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"# tool={params.tool} database={params.database}\n")
        handle.write(
            f"# max_evalue={params.max_evalue:g} "
            f"min_query_coverage={params.min_query_coverage:g}\n"
        )
        handle.write("# coverage columns are fractions in [0, 1], normalized on parse\n")
        handle.write("\t".join(HIT_COLUMNS) + "\n")
        for hit in hits:
            handle.write(
                "\t".join(
                    (
                        hit.query_id,
                        hit.subject_id,
                        f"{hit.percent_identity:g}",
                        str(hit.alignment_length),
                        f"{hit.evalue:g}",
                        f"{hit.bitscore:g}",
                        f"{hit.query_coverage:.4f}",
                        f"{hit.subject_coverage:.4f}",
                        str(hit.subject_length),
                    )
                )
                + "\n"
            )
    return len(hits)


def read_hit_table(path: str | Path) -> Iterator[Hit]:
    """Read back a table written by :func:`write_hits`."""
    with Path(path).open() as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            row = line.split("\t")
            if row[0] == HIT_COLUMNS[0]:
                continue
            if len(row) != len(HIT_COLUMNS):
                raise ValueError(
                    f"{path}:{lineno}: expected {len(HIT_COLUMNS)} columns, got {len(row)}"
                )
            try:
                yield Hit(
                    query_id=row[0],
                    subject_id=row[1],
                    percent_identity=float(row[2]),
                    alignment_length=int(row[3]),
                    evalue=float(row[4]),
                    bitscore=float(row[5]),
                    query_coverage=float(row[6]),
                    subject_coverage=float(row[7]),
                    subject_length=int(row[8]),
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: malformed hit row: {exc}") from exc


def unique_subjects(hits: list[Hit]) -> list[str]:
    """Subject ids in first-seen order, keeping the best hit's ordering."""
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(hit.subject_id, None)
    return list(seen)
