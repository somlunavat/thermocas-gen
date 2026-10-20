"""Sequence clustering and cluster-level train/val/test splitting.

The corpus is homologs of a handful of seeds, so near-duplicate sequences are
everywhere. Splitting by sequence would place a sequence in train and its
95%-identical neighbour in test, and every reported metric would be measuring
memorization. Splitting by cluster is what makes the test set mean anything,
and it is the single most important structural decision in the data stage.

Three things are held out deliberately:

* a whole subfamily, so the test set measures generalization to an unseen Cas9
  type rather than interpolation within known ones;
* the thermophilic positive controls, because a natural thermophilic Cas9 in
  the training set would make "shifted toward thermophilic" partly recall;
* everything in a cluster together, never split across sets.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ClusterParams:
    """MMseqs2 clustering settings from ``configs/data.yaml``."""

    tool: str
    min_seq_id: float
    coverage: float
    cov_mode: int
    cluster_mode: int

    def __post_init__(self) -> None:
        if not 0.0 < self.min_seq_id <= 1.0:
            raise ValueError(
                f"min_seq_id must be a fraction in (0, 1], got {self.min_seq_id}. "
                "Use 0.45 for 45%, not 45."
            )
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError(f"coverage must be a fraction in [0, 1], got {self.coverage}")

    @classmethod
    def from_config(cls, cfg, path: str = "cluster") -> ClusterParams:
        section = cfg.section(path)
        return cls(
            tool=str(section["tool"]).lower(),
            min_seq_id=float(section["min_seq_id"]),
            coverage=float(section["coverage"]),
            cov_mode=int(section["cov_mode"]),
            cluster_mode=int(section["cluster_mode"]),
        )


def mmseqs_cluster_commands(
    params: ClusterParams, sequences_fasta: Path, out_prefix: Path, workdir: Path, threads: int = 4
) -> list[list[str]]:
    """createdb, cluster, createtsv.

    MMseqs2's ``--min-seq-id`` takes a fraction, so 0.45 means 45%. Passing 45
    would ask for 4500% identity and silently produce singleton clusters,
    which would then split like sequences and reintroduce the leakage this
    module exists to prevent.
    """
    db = workdir / "seqDB"
    clu = workdir / "cluDB"
    tmp = workdir / "tmp"
    return [
        ["mmseqs", "createdb", str(sequences_fasta), str(db)],
        [
            "mmseqs",
            "cluster",
            str(db),
            str(clu),
            str(tmp),
            "--min-seq-id",
            f"{params.min_seq_id:g}",
            "-c",
            f"{params.coverage:g}",
            "--cov-mode",
            str(params.cov_mode),
            "--cluster-mode",
            str(params.cluster_mode),
            "--threads",
            str(threads),
        ],
        ["mmseqs", "createtsv", str(db), str(db), str(clu), str(out_prefix.with_suffix(".tsv"))],
    ]


def parse_cluster_tsv(path: str | Path) -> dict[str, str]:
    """Parse MMseqs2 ``createtsv`` output into member -> representative.

    The file is two columns, representative then member, with the
    representative also listed as a member of its own cluster.
    """
    assignments: dict[str, str] = {}
    with Path(path).open() as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"{path}:{lineno}: expected 2 tab-separated columns in a cluster TSV, "
                    f"got {len(parts)}"
                )
            representative, member = parts[0].strip(), parts[1].strip()
            if member in assignments and assignments[member] != representative:
                raise ValueError(
                    f"{path}:{lineno}: {member!r} assigned to two clusters "
                    f"({assignments[member]!r} and {representative!r})"
                )
            assignments[member] = representative
    if not assignments:
        raise ValueError(f"{path}: no cluster assignments found")
    return assignments


@dataclass(frozen=True)
class SplitParams:
    """Split fractions and holdout rules."""

    fractions: dict[str, float]
    holdout_subfamily: str | None
    holdout_positive_controls: bool
    positive_controls: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        missing = set(SPLIT_NAMES) - set(self.fractions)
        if missing:
            raise ValueError(f"split fractions missing {sorted(missing)}")
        total = sum(self.fractions[name] for name in SPLIT_NAMES)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        for name in SPLIT_NAMES:
            if self.fractions[name] < 0:
                raise ValueError(f"split fraction for {name} is negative")

    @classmethod
    def from_config(cls, cfg, path: str = "split") -> SplitParams:
        section = cfg.section(path)
        subfamily = section["holdout_subfamily"] if "holdout_subfamily" in section else None
        return cls(
            fractions={k: float(v) for k, v in section["fractions"].items()},
            holdout_subfamily=str(subfamily) if subfamily else None,
            holdout_positive_controls=bool(section["holdout_positive_controls"]),
            positive_controls=tuple(section["positive_controls"]),
            seed=int(cfg["seed"]),
        )


@dataclass
class SequenceMeta:
    """What the splitter needs to know about one sequence."""

    seq_id: str
    cluster_id: str
    temp_bin: str
    subfamily: str | None = None
    is_positive_control: bool = False


@dataclass
class SplitReport:
    counts: dict[str, int] = field(default_factory=dict)
    clusters: dict[str, int] = field(default_factory=dict)
    bin_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    forced_to_test: int = 0
    forced_reasons: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        for name in SPLIT_NAMES:
            bins = self.bin_counts.get(name, {})
            detail = ", ".join(f"{k}={v}" for k, v in sorted(bins.items())) or "empty"
            lines.append(
                f"  {name:<6} {self.counts.get(name, 0):>7} sequences  "
                f"{self.clusters.get(name, 0):>6} clusters  [{detail}]"
            )
        if self.forced_to_test:
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.forced_reasons.items()))
            lines.append(f"  forced into test: {self.forced_to_test} ({reasons})")
        return "\n".join(lines)


def assign_splits(
    sequences: Iterable[SequenceMeta], params: SplitParams
) -> tuple[dict[str, str], SplitReport]:
    """Assign every sequence to train, val or test at cluster granularity.

    Returns ``seq_id -> split`` and a report. Guarantees, each of which has a
    test:

    * a cluster never straddles two splits;
    * held-out subfamilies and positive controls land in test, along with
      every other member of their clusters;
    * assignment is deterministic given the configured seed;
    * clusters are allocated in a stratified order so a rare bin is not
      swallowed entirely by train.
    """
    members = list(sequences)
    if not members:
        raise ValueError("no sequences to split")

    by_cluster: dict[str, list[SequenceMeta]] = defaultdict(list)
    for item in members:
        by_cluster[item.cluster_id].append(item)

    forced: set[str] = set()
    report = SplitReport()

    for cluster_id, cluster_members in by_cluster.items():
        reasons: list[str] = []
        if params.holdout_subfamily and any(
            m.subfamily == params.holdout_subfamily for m in cluster_members
        ):
            reasons.append("subfamily")
        if params.holdout_positive_controls and any(
            m.is_positive_control for m in cluster_members
        ):
            reasons.append("positive_control")
        if reasons:
            forced.add(cluster_id)
            for reason in reasons:
                report.forced_reasons[reason] = report.forced_reasons.get(reason, 0) + 1

    # Allocate the remaining clusters. Clusters are grouped by their dominant
    # growth-temperature bin and dealt out per bin, so the thermophilic side --
    # which is scarce -- is represented in val and test rather than landing
    # wholly in train by chance.
    assignments: dict[str, str] = {}
    rng = random.Random(params.seed)

    free_by_bin: dict[str, list[str]] = defaultdict(list)
    for cluster_id, cluster_members in by_cluster.items():
        if cluster_id in forced:
            continue
        dominant = _dominant_bin(cluster_members)
        free_by_bin[dominant].append(cluster_id)

    cluster_split: dict[str, str] = {cluster_id: "test" for cluster_id in forced}

    for bin_name in sorted(free_by_bin):
        cluster_ids = sorted(free_by_bin[bin_name])
        rng.shuffle(cluster_ids)
        n = len(cluster_ids)
        n_val = int(round(n * params.fractions["val"]))
        n_test = int(round(n * params.fractions["test"]))
        # With very few clusters, rounding can claim more than exist; train
        # gives way first so val and test are never empty by rounding alone.
        if n_val + n_test > n:
            n_test = min(n_test, n)
            n_val = max(0, n - n_test)
        for index, cluster_id in enumerate(cluster_ids):
            if index < n_test:
                cluster_split[cluster_id] = "test"
            elif index < n_test + n_val:
                cluster_split[cluster_id] = "val"
            else:
                cluster_split[cluster_id] = "train"

    for cluster_id, cluster_members in by_cluster.items():
        split = cluster_split[cluster_id]
        report.clusters[split] = report.clusters.get(split, 0) + 1
        for item in cluster_members:
            assignments[item.seq_id] = split
            report.counts[split] = report.counts.get(split, 0) + 1
            bins = report.bin_counts.setdefault(split, {})
            bins[item.temp_bin] = bins.get(item.temp_bin, 0) + 1
            if cluster_id in forced:
                report.forced_to_test += 1

    return assignments, report


def _dominant_bin(members: list[SequenceMeta]) -> str:
    counts: dict[str, int] = {}
    for item in members:
        counts[item.temp_bin] = counts.get(item.temp_bin, 0) + 1
    # Ties break on the bin name so the result is deterministic.
    return max(sorted(counts), key=lambda name: (counts[name], name))


def verify_no_cluster_straddles_splits(
    sequences: Iterable[SequenceMeta], assignments: dict[str, str]
) -> list[str]:
    """Return clusters appearing in more than one split; empty means clean.

    Called by the split script before it writes anything. A straddling
    cluster is leakage by construction, so this is checked rather than
    assumed.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for item in sequences:
        split = assignments.get(item.seq_id)
        if split is not None:
            seen[item.cluster_id].add(split)
    return sorted(cluster for cluster, splits in seen.items() if len(splits) > 1)


def write_split_files(out_dir: Path, assignments: dict[str, str]) -> dict[str, Path]:
    """Write one id-per-line file per split, plus a combined CSV.

    These files are committed: they are what makes a run reproducible, and
    re-deriving them later from a re-clustered corpus would not give the same
    split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        ids = sorted(seq_id for seq_id, split in assignments.items() if split == name)
        path = out_dir / f"{name}.txt"
        path.write_text("".join(f"{seq_id}\n" for seq_id in ids))
        written[name] = path

    combined = out_dir / "splits.csv"
    with combined.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq_id", "split"])
        for seq_id in sorted(assignments):
            writer.writerow([seq_id, assignments[seq_id]])
    written["combined"] = combined
    return written
