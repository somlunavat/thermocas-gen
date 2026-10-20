"""Train/test leakage detection.

This is the gate that protects every downstream number. The corpus is homologs
of a handful of seeds, so without cluster-level splitting a test sequence and
its 95%-identical neighbour end up on opposite sides of the split and every
reported metric measures memorization.

Clustering at 45% identity should already make a cross-split pair above that
threshold impossible. This module does not assume that: it searches the test
set against the training set with MMseqs2 and reports any pair that exceeds
the configured identity over the configured coverage. The clustering is the
mechanism; this is the verification, and the two are kept separate on purpose
so that a clustering mistake cannot also disable its own check.

Identity is compared over an alignment covering at least
``alignment_coverage`` of the sequence. A 99%-identical 40-residue stretch
shared between two 1,300-residue Cas9s is a conserved motif, not leakage, and
filtering on identity alone would flag the entire corpus.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LeakageParams:
    """Leakage thresholds from ``configs/data.yaml``."""

    max_identity_to_train: float
    alignment_coverage: float
    fail_on_violation: bool

    def __post_init__(self) -> None:
        if not 0.0 < self.max_identity_to_train <= 1.0:
            raise ValueError(
                f"max_identity_to_train must be a fraction in (0, 1], got "
                f"{self.max_identity_to_train}. Use 0.45 for 45%, not 45."
            )
        if not 0.0 <= self.alignment_coverage <= 1.0:
            raise ValueError(
                f"alignment_coverage must be a fraction in [0, 1], got {self.alignment_coverage}"
            )

    @classmethod
    def from_config(cls, cfg, path: str = "leakage") -> LeakageParams:
        section = cfg.section(path)
        return cls(
            max_identity_to_train=float(section["max_identity_to_train"]),
            alignment_coverage=float(section["alignment_coverage"]),
            fail_on_violation=bool(section["fail_on_violation"]),
        )


@dataclass(frozen=True)
class SequencePair:
    """One test-versus-train alignment."""

    query_id: str  # held-out sequence
    target_id: str  # training sequence
    identity: float  # fraction in [0, 1]
    coverage: float  # fraction of the query aligned

    def violates(self, params: LeakageParams) -> bool:
        return (
            self.coverage >= params.alignment_coverage
            and self.identity > params.max_identity_to_train
        )


def mmseqs_leakage_commands(
    query_fasta: Path,
    target_fasta: Path,
    out_tsv: Path,
    workdir: Path,
    sensitivity: float = 7.5,
    threads: int = 4,
) -> list[list[str]]:
    """Search held-out sequences against training sequences.

    Sensitivity is set high deliberately. A missed alignment here reads as
    "no leakage", which is the dangerous direction to be wrong in, so the
    search errs toward finding too much and letting the thresholds decide.
    """
    query_db = workdir / "queryDB"
    target_db = workdir / "targetDB"
    result_db = workdir / "resultDB"
    tmp = workdir / "tmp"
    return [
        ["mmseqs", "createdb", str(query_fasta), str(query_db)],
        ["mmseqs", "createdb", str(target_fasta), str(target_db)],
        [
            "mmseqs",
            "search",
            str(query_db),
            str(target_db),
            str(result_db),
            str(tmp),
            "-s",
            f"{sensitivity:g}",
            # No e-value or coverage prefilter: thresholds are applied after
            # parsing so the report can show near-misses too.
            "-e",
            "10000",
            "--max-seqs",
            "1000",
            "--threads",
            str(threads),
        ],
        [
            "mmseqs",
            "convertalis",
            str(query_db),
            str(target_db),
            str(result_db),
            str(out_tsv),
            "--format-output",
            "query,target,fident,qcov,tcov,alnlen,evalue,bits",
            "--threads",
            str(threads),
        ],
    ]


def parse_alignment_tsv(path: str | Path) -> Iterator[SequencePair]:
    """Parse the ``convertalis`` output produced by the command above.

    MMseqs2's ``fident`` and ``qcov`` are fractions, unlike DIAMOND's
    percentages. Values above 1 are treated as percentages and normalized, so
    a format change cannot silently turn a 90% identity into 0.9% and hide
    leakage.
    """
    with Path(path).open() as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                raise ValueError(
                    f"{path}:{lineno}: expected at least 4 columns "
                    f"(query, target, fident, qcov), got {len(fields)}"
                )
            try:
                identity = float(fields[2])
                coverage = float(fields[3])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: malformed alignment row: {exc}") from exc
            yield SequencePair(
                query_id=fields[0],
                target_id=fields[1],
                identity=identity / 100.0 if identity > 1.0 else identity,
                coverage=coverage / 100.0 if coverage > 1.0 else coverage,
            )


@dataclass
class LeakageReport:
    """What the leakage check found."""

    pairs_examined: int = 0
    violations: list[SequencePair] = field(default_factory=list)
    worst_by_query: dict[str, SequencePair] = field(default_factory=dict)
    self_matches_ignored: int = 0

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def leaked_queries(self) -> list[str]:
        return sorted({pair.query_id for pair in self.violations})

    def summary(self, params: LeakageParams) -> str:
        lines = [
            f"examined {self.pairs_examined} test-vs-train alignments",
            f"threshold: identity > {params.max_identity_to_train:.0%} "
            f"over >= {params.alignment_coverage:.0%} query coverage",
        ]
        if self.self_matches_ignored:
            lines.append(f"ignored {self.self_matches_ignored} self-matches")
        if self.violations:
            lines.append(f"VIOLATIONS: {len(self.violations)} pairs, {len(self.leaked_queries)} held-out sequences")
            for pair in sorted(self.violations, key=lambda p: -p.identity)[:10]:
                lines.append(
                    f"  {pair.query_id} ~ {pair.target_id}: "
                    f"identity {pair.identity:.1%}, coverage {pair.coverage:.1%}"
                )
        else:
            lines.append("no violations")
            worst = max(
                self.worst_by_query.values(), key=lambda p: p.identity, default=None
            )
            if worst is not None:
                lines.append(
                    f"closest pair: {worst.query_id} ~ {worst.target_id} "
                    f"at identity {worst.identity:.1%}, coverage {worst.coverage:.1%}"
                )
        return "\n".join(lines)


def check_leakage(
    pairs: Iterable[SequencePair], params: LeakageParams
) -> LeakageReport:
    """Evaluate alignments against the leakage thresholds."""
    report = LeakageReport()
    for pair in pairs:
        if pair.query_id == pair.target_id:
            # The same accession on both sides is not leakage between splits;
            # it means the same sequence was handed to the check twice.
            report.self_matches_ignored += 1
            continue
        report.pairs_examined += 1

        existing = report.worst_by_query.get(pair.query_id)
        if existing is None or pair.identity > existing.identity:
            report.worst_by_query[pair.query_id] = pair

        if pair.violates(params):
            report.violations.append(pair)
    return report


def check_split_disjointness(splits: dict[str, set[str]]) -> list[str]:
    """Return problems if any sequence id appears in more than one split.

    A cheap check that runs before any alignment: an id in both train and test
    is leakage that no identity threshold would catch, because the sequences
    are not similar, they are the same.
    """
    problems: list[str] = []
    names = sorted(splits)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = splits[first] & splits[second]
            if shared:
                examples = ", ".join(sorted(shared)[:5])
                problems.append(
                    f"{len(shared)} id(s) in both {first} and {second}: {examples}"
                )
    return problems
