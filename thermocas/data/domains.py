"""Catalytic domain confirmation with HMMER.

A Cas9 must carry both nuclease domains, RuvC and HNH. Each requirement lists
alternative Pfam profiles and is satisfied when any of them hits, because no
single Pfam model covers the family: the Cas9-specific PF22702 matches St1Cas9
at 1.2e-66 while the generic PF18541 misses it entirely.

Parsing targets HMMER 3's ``--domtblout`` table. The value compared against the
threshold is the per-domain i-Evalue (column 13), not the full-sequence E-value
(column 7). The distinction matters: a long multi-domain protein can show a
strong full-sequence E-value from one good domain while a second, weak domain
hit rides along, and filtering on the sequence-level value would accept it.
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Column indices in HMMER 3 --domtblout, zero-based. The file is
# whitespace-aligned with a free-text description at the end, so rows are
# split with maxsplit and the description is not parsed.
COL_TARGET_NAME = 0
COL_TARGET_LENGTH = 2
COL_QUERY_NAME = 3
COL_QUERY_ACCESSION = 4
COL_DOMAIN_IEVALUE = 12
COL_DOMAIN_SCORE = 13
COL_ALI_FROM = 17
COL_ALI_TO = 18
N_FIXED_COLUMNS = 22


class HmmerError(RuntimeError):
    """HMMER is missing or returned a non-zero exit status."""


@dataclass(frozen=True)
class DomainHit:
    """One domain alignment from a --domtblout row.

    ``sequence_id`` is the target and ``profile_name``/``profile_accession``
    the query, which is the orientation hmmsearch produces. hmmscan swaps
    them; this module assumes hmmsearch.
    """

    sequence_id: str
    sequence_length: int
    profile_name: str
    profile_accession: str
    i_evalue: float
    score: float
    ali_from: int
    ali_to: int

    @property
    def profile_base_accession(self) -> str:
        """``PF22702`` from ``PF22702.3``, so configs need not pin versions."""
        return self.profile_accession.split(".")[0]

    @property
    def ali_length(self) -> int:
        return self.ali_to - self.ali_from + 1


@dataclass(frozen=True)
class DomainRequirements:
    """Which domains are required, and which profiles satisfy each."""

    required: tuple[str, ...]
    profiles: dict[str, tuple[str, ...]]
    max_evalue: float
    informational: dict[str, tuple[str, ...]] = field(default_factory=dict)
    profile_dir: Path | None = None

    def __post_init__(self) -> None:
        # Strip profile versions here, not in from_config, so the invariant
        # holds for every construction path. PF22702.3 and PF22702 must behave
        # identically; a Pfam release bump should not silently stop matching.
        for attribute in ("profiles", "informational"):
            mapping = getattr(self, attribute)
            normalized = {
                name: tuple(str(a).split(".")[0] for a in accessions)
                for name, accessions in (mapping or {}).items()
            }
            object.__setattr__(self, attribute, normalized)

        if not self.required:
            raise ValueError("no required domains configured")
        missing = [name for name in self.required if not self.profiles.get(name)]
        if missing:
            raise ValueError(
                f"required domains {missing} have no profiles configured; "
                "a requirement with no profile can never be satisfied"
            )
        if self.max_evalue <= 0:
            raise ValueError(f"hmm_evalue must be positive, got {self.max_evalue}")

    @classmethod
    def from_config(cls, cfg, path: str = "domains") -> DomainRequirements:
        section = cfg.section(path)

        def normalize(raw: dict) -> dict[str, tuple[str, ...]]:
            out: dict[str, tuple[str, ...]] = {}
            for name, value in (raw or {}).items():
                # Tolerate a bare string so a single-profile domain can be
                # written without a list, but store it uniformly.
                accessions = [value] if isinstance(value, str) else list(value)
                out[name] = tuple(str(a).split(".")[0] for a in accessions)
            return out

        profile_dir = None
        if "profile_dir" in section:
            profile_dir = Path(section["profile_dir"])

        return cls(
            required=tuple(section["require"]),
            profiles=normalize(section["hmm_profiles"]),
            max_evalue=float(section["hmm_evalue"]),
            informational=normalize(
                section["informational_profiles"] if "informational_profiles" in section else {}
            ),
            profile_dir=profile_dir,
        )

    def all_accessions(self) -> list[str]:
        """Every accession to search with, required and informational."""
        seen: dict[str, None] = {}
        for group in (self.profiles, self.informational):
            for accessions in group.values():
                for accession in accessions:
                    seen.setdefault(accession, None)
        return list(seen)

    def domain_for_accession(self, accession: str) -> str | None:
        """Which required domain an accession satisfies, if any."""
        base = accession.split(".")[0]
        for name, accessions in self.profiles.items():
            if base in accessions:
                return name
        return None


def parse_domtblout(path: str | Path) -> Iterator[DomainHit]:
    """Parse a HMMER 3 --domtblout file."""
    with Path(path).open() as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split(maxsplit=N_FIXED_COLUMNS)
            if len(fields) < N_FIXED_COLUMNS:
                raise ValueError(
                    f"{path}:{lineno}: expected at least {N_FIXED_COLUMNS} columns in a "
                    f"--domtblout row, got {len(fields)}. Was this produced with "
                    "--tblout instead of --domtblout?"
                )
            try:
                yield DomainHit(
                    sequence_id=fields[COL_TARGET_NAME],
                    sequence_length=int(fields[COL_TARGET_LENGTH]),
                    profile_name=fields[COL_QUERY_NAME],
                    profile_accession=fields[COL_QUERY_ACCESSION],
                    i_evalue=float(fields[COL_DOMAIN_IEVALUE]),
                    score=float(fields[COL_DOMAIN_SCORE]),
                    ali_from=int(fields[COL_ALI_FROM]),
                    ali_to=int(fields[COL_ALI_TO]),
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: malformed domtblout row: {exc}") from exc


@dataclass
class DomainCall:
    """The domain verdict for one sequence."""

    sequence_id: str
    best_evalue: dict[str, float]  # required domain -> best i-Evalue seen
    best_profile: dict[str, str]  # required domain -> the profile that hit
    informational: dict[str, float] = field(default_factory=dict)

    @property
    def satisfied(self) -> set[str]:
        return set(self.best_evalue)

    def passes(self, requirements: DomainRequirements) -> bool:
        return all(name in self.best_evalue for name in requirements.required)

    def missing(self, requirements: DomainRequirements) -> list[str]:
        return [name for name in requirements.required if name not in self.best_evalue]


def call_domains(
    hits: Iterable[DomainHit], requirements: DomainRequirements
) -> dict[str, DomainCall]:
    """Reduce domain hits to one verdict per sequence.

    Only hits at or below the configured i-Evalue count. For each required
    domain the best-scoring satisfying profile is kept, so the report says
    which model actually matched.
    """
    calls: dict[str, DomainCall] = {}
    by_sequence: dict[str, list[DomainHit]] = defaultdict(list)
    for hit in hits:
        by_sequence[hit.sequence_id].append(hit)

    info_lookup: dict[str, str] = {}
    for group, accessions in requirements.informational.items():
        for accession in accessions:
            info_lookup[accession] = group

    for sequence_id, sequence_hits in by_sequence.items():
        call = DomainCall(sequence_id=sequence_id, best_evalue={}, best_profile={})
        for hit in sequence_hits:
            if hit.i_evalue > requirements.max_evalue:
                continue
            domain = requirements.domain_for_accession(hit.profile_base_accession)
            if domain is not None:
                if hit.i_evalue < call.best_evalue.get(domain, float("inf")):
                    call.best_evalue[domain] = hit.i_evalue
                    call.best_profile[domain] = hit.profile_name
                continue
            group = info_lookup.get(hit.profile_base_accession)
            if group is not None and hit.i_evalue < call.informational.get(group, float("inf")):
                call.informational[group] = hit.i_evalue
        calls[sequence_id] = call
    return calls


def concatenate_profiles(
    accessions: Iterable[str], profile_dir: Path, out_path: Path
) -> list[str]:
    """Build one searchable HMM file from individual Pfam profiles.

    Returns the ``NAME``/``ACC`` lines found, so the exact profile versions
    used can be recorded in PROJECT_LOG.md.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recorded: list[str] = []
    missing: list[str] = []
    with out_path.open("w") as out:
        for accession in accessions:
            source = profile_dir / f"{accession}.hmm"
            if not source.is_file():
                missing.append(str(source))
                continue
            text = source.read_text()
            for line in text.splitlines():
                if line.startswith(("NAME", "ACC")):
                    recorded.append(line.split(maxsplit=1)[-1].strip())
                if line.startswith("ACC"):
                    break
            out.write(text if text.endswith("\n") else text + "\n")
    if missing:
        raise FileNotFoundError(
            "missing Pfam profiles: "
            + ", ".join(missing)
            + ". Fetch each with: curl -sSL "
            "'https://www.ebi.ac.uk/interpro/api/entry/pfam/<ACC>/?annotation=hmm' "
            "| gunzip > <ACC>.hmm"
        )
    return recorded


def run_hmmsearch(
    profiles_hmm: Path,
    sequences_fasta: Path,
    domtblout: Path,
    max_evalue: float,
    cpu: int = 4,
) -> list[str]:
    """Run hmmsearch, returning the command for the log.

    ``-E`` prefilters on the full-sequence E-value; the authoritative
    per-domain filtering happens in :func:`call_domains`, which compares the
    i-Evalue.
    """
    domtblout.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "hmmsearch",
        "--domtblout",
        str(domtblout),
        "-E",
        f"{max_evalue:g}",
        "--cpu",
        str(cpu),
        str(profiles_hmm),
        str(sequences_fasta),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise HmmerError(
            f"hmmsearch exited with status {result.returncode}:\n{result.stderr.strip()}"
        )
    return command
