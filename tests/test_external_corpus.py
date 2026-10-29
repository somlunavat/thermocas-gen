"""Tests for the external corpus builder.

A wrong corpus here does not crash anything. It produces a smaller or
differently composed set of residues, the sweep over it prints a perfectly
plausible curve, and the conclusion drawn -- that the ceiling is or is not a
property of corpora in general -- is drawn from the wrong corpus. The
properties that make the comparison with stage A valid are therefore asserted
directly: that the join uses both keys, that filtering matches stage A's
bounds, that the cap and the balancing drop whole species rather than
reshaping classes, and that the FASTA and the label table agree.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from thermocas.config import GrowthTemperatureBin, GrowthTemperatureBinner
from thermocas.data.engqvist import EngqvistTable
from thermocas.data.sequence_filter import SequenceFilter

REPO_ROOT = Path(__file__).resolve().parent.parent


def load():
    path = REPO_ROOT / "scripts" / "16_external_corpus.py"
    spec = importlib.util.spec_from_file_location("external_corpus", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because @dataclass resolves annotations
    # through sys.modules[cls.__module__]; without this the module's own
    # dataclasses fail to build at import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOD = load()

BINNER = GrowthTemperatureBinner(
    [
        GrowthTemperatureBin(name="meso", min_c=float("-inf"), max_c=45.0),
        GrowthTemperatureBin(name="thermo", min_c=45.0, max_c=float("inf")),
    ]
)
FILTER = SequenceFilter(
    min_length=60,
    max_length=1024,
    allowed_residues=frozenset("ACDEFGHIKLMNPQRSTVWY"),
    drop_exact_duplicates=True,
)


def ogt_table(tmp_path: Path, *organisms: tuple[str, str, str]) -> EngqvistTable:
    header = "organism\tdomain\ttemperature\ttaxid\n"
    rows = "".join(
        f"{name}\tBacteria\t{temp}\t{taxid}\n" for name, temp, taxid in organisms
    )
    path = tmp_path / "temperature_data.tsv"
    path.write_text(header + rows)
    return EngqvistTable.load(path)


def fasta(tmp_path: Path, *records: tuple[str, str, str], name: str = "in.fasta") -> Path:
    """Records are (accession, organism-and-taxid description, sequence)."""
    path = tmp_path / name
    path.write_text("".join(f">{acc} {desc}\n{seq}\n" for acc, desc, seq in records))
    return path


def protein(letter: str = "A", length: int = 100) -> str:
    return letter * length


class TestScan:
    def test_it_labels_a_sequence_from_its_organism(self, tmp_path):
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        path = fasta(
            tmp_path,
            ("sp|P1|X_THEAQ", "Protein OS=Thermus aquaticus OX=271", protein()),
        )
        kept, report = MOD.scan(path, table, BINNER, FILTER, 0)
        assert [c.seq_id for c in kept] == ["P1"]
        assert kept[0].ogt_c == 70.0
        assert kept[0].temp_bin == "thermo"
        assert report.kept == 1

    def test_the_id_is_canonicalized(self, tmp_path):
        """MMseqs2 keys on the middle field of a piped header, so the corpus
        has to be written with bare accessions or the cluster join finds
        nothing -- the bug that produced an empty Cas9 split."""
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        path = fasta(
            tmp_path, ("sp|P1|X_THEAQ", "OS=Thermus aquaticus OX=271", protein())
        )
        kept, _ = MOD.scan(path, table, BINNER, FILTER, 0)
        assert kept[0].seq_id == "P1"

    def test_an_unlabeled_organism_is_dropped_and_counted(self, tmp_path):
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        path = fasta(tmp_path, ("sp|P1|X", "OS=Homo sapiens OX=9606", protein()))
        kept, report = MOD.scan(path, table, BINNER, FILTER, 0)
        assert kept == []
        assert report.no_ogt == 1

    def test_a_strain_level_taxid_still_joins_on_the_name(self, tmp_path):
        """Most real hits take this path: UniProt carries strain taxids,
        Engqvist is indexed at species level."""
        table = ogt_table(tmp_path, ("thermus_thermophilus", "68", "274"))
        path = fasta(
            tmp_path,
            ("sp|P1|X", "OS=Thermus thermophilus HB8 OX=300852", protein()),
        )
        kept, report = MOD.scan(path, table, BINNER, FILTER, 0)
        assert len(kept) == 1
        assert report.matched_by_name == 1
        assert report.matched_by_taxid == 0

    def test_too_short_is_rejected_with_a_reason(self, tmp_path):
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        path = fasta(
            tmp_path, ("sp|P1|X", "OS=Thermus aquaticus OX=271", protein(length=10))
        )
        kept, report = MOD.scan(path, table, BINNER, FILTER, 0)
        assert kept == []
        assert sum(report.filtered.values()) == 1

    def test_a_nonstandard_residue_is_rejected(self, tmp_path):
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        path = fasta(
            tmp_path, ("sp|P1|X", "OS=Thermus aquaticus OX=271", "X" * 100)
        )
        kept, _ = MOD.scan(path, table, BINNER, FILTER, 0)
        assert kept == []

    def test_an_exact_duplicate_sequence_is_dropped_once(self, tmp_path):
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        path = fasta(
            tmp_path,
            ("sp|P1|X", "OS=Thermus aquaticus OX=271", protein()),
            ("sp|P2|Y", "OS=Thermus aquaticus OX=271", protein()),
        )
        kept, report = MOD.scan(path, table, BINNER, FILTER, 0)
        assert len(kept) == 1
        assert report.duplicate_sequence == 1

    def test_the_per_species_cap_is_applied(self, tmp_path):
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        path = fasta(
            tmp_path,
            *[
                (f"sp|P{i}|X", "OS=Thermus aquaticus OX=271", protein("ACDEFG"[i % 6]))
                for i in range(4)
            ],
        )
        kept, report = MOD.scan(path, table, BINNER, FILTER, 2)
        assert len(kept) == 2
        assert report.over_species_cap == 2

    def test_the_cap_is_per_species_not_global(self, tmp_path):
        table = ogt_table(
            tmp_path, ("thermus_aquaticus", "70", "271"), ("escherichia_coli", "37", "562")
        )
        path = fasta(
            tmp_path,
            ("sp|P1|A", "OS=Thermus aquaticus OX=271", protein("A")),
            ("sp|P2|B", "OS=Thermus aquaticus OX=271", protein("C")),
            ("sp|P3|C", "OS=Escherichia coli OX=562", protein("D")),
        )
        kept, _ = MOD.scan(path, table, BINNER, FILTER, 1)
        assert sorted(c.organism_key for c in kept) == [
            "escherichia coli",
            "thermus aquaticus",
        ]

    def test_zero_means_no_cap(self, tmp_path):
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        path = fasta(
            tmp_path,
            *[
                (f"sp|P{i}|X", "OS=Thermus aquaticus OX=271", protein("ACDEFG"[i]))
                for i in range(6)
            ],
        )
        kept, _ = MOD.scan(path, table, BINNER, FILTER, 0)
        assert len(kept) == 6


def candidate(key: str, temp_bin: str, seq_id: str) -> object:
    return MOD.Candidate(
        seq_id=seq_id,
        organism=key.title(),
        organism_key=key,
        ogt_c=70.0 if temp_bin == "thermo" else 30.0,
        temp_bin=temp_bin,
        length=100,
    )


class TestBalanceSpecies:
    def test_it_equalizes_the_species_count_per_class(self, tmp_path):
        candidates = [
            candidate(f"hot{i}", "thermo", f"h{i}") for i in range(2)
        ] + [candidate(f"cold{i}", "meso", f"c{i}") for i in range(10)]
        out, counts = MOD.balance_species(candidates, seed=1)
        assert counts == {"meso": 2, "thermo": 2}
        assert len({c.organism_key for c in out}) == 4

    def test_it_keeps_surviving_species_whole(self, tmp_path):
        """Dropping sequences rather than species would thin a class's
        coverage and change its residue composition, which is the thing the
        ceiling measures."""
        candidates = [
            candidate("hot1", "thermo", "h1"),
            candidate("cold1", "meso", "c1"),
            candidate("cold1", "meso", "c2"),
            candidate("cold2", "meso", "c3"),
        ]
        out, _ = MOD.balance_species(candidates, seed=1)
        surviving = {c.organism_key for c in out}
        assert len(surviving) == 2
        for key in surviving:
            before = [c for c in candidates if c.organism_key == key]
            after = [c for c in out if c.organism_key == key]
            assert len(before) == len(after)

    def test_it_is_deterministic_for_a_seed(self, tmp_path):
        candidates = [candidate("hot1", "thermo", "h1")] + [
            candidate(f"cold{i}", "meso", f"c{i}") for i in range(20)
        ]
        first, _ = MOD.balance_species(candidates, seed=7)
        second, _ = MOD.balance_species(candidates, seed=7)
        assert [c.seq_id for c in first] == [c.seq_id for c in second]

    def test_a_single_class_is_returned_unchanged(self, tmp_path):
        candidates = [candidate(f"cold{i}", "meso", f"c{i}") for i in range(3)]
        out, counts = MOD.balance_species(candidates, seed=1)
        assert out == candidates
        assert counts == {"meso": 3}

    def test_it_preserves_input_order(self, tmp_path):
        """The second pass streams the FASTA and matches on ids, so order is
        not load-bearing -- but a reordering here would make the written table
        disagree with any earlier printed summary for no reason."""
        candidates = [
            candidate("hot1", "thermo", "h1"),
            candidate("cold1", "meso", "c1"),
            candidate("cold2", "meso", "c2"),
        ]
        out, _ = MOD.balance_species(candidates, seed=1)
        assert [c.seq_id for c in out] == [
            c.seq_id for c in candidates if c in out
        ]


class TestWriteCorpus:
    def test_the_table_and_the_fasta_hold_the_same_ids(self, tmp_path):
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        source = fasta(
            tmp_path,
            ("sp|P1|X", "OS=Thermus aquaticus OX=271", protein("A")),
            ("sp|P2|Y", "OS=Thermus aquaticus OX=271", protein("C")),
        )
        kept, _ = MOD.scan(source, table, BINNER, FILTER, 0)
        out_fasta = tmp_path / "out.fasta"
        out_table = tmp_path / "out.csv"
        written = MOD.write_corpus(source, out_fasta, out_table, kept)
        assert written == 2
        with out_table.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [r["seq_id"] for r in rows] == ["P1", "P2"]
        assert out_fasta.read_text().count(">") == 2

    def test_the_cross_check_column_is_left_empty(self, tmp_path):
        """Engqvist publishes one consolidated temperature per organism with
        no per-collection breakdown. Writing 1 would claim a single agreeing
        source where the number of sources is simply unknown."""
        table = ogt_table(tmp_path, ("thermus_aquaticus", "70", "271"))
        source = fasta(
            tmp_path, ("sp|P1|X", "OS=Thermus aquaticus OX=271", protein())
        )
        kept, _ = MOD.scan(source, table, BINNER, FILTER, 0)
        out_table = tmp_path / "out.csv"
        MOD.write_corpus(source, tmp_path / "out.fasta", out_table, kept)
        with out_table.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        assert row["n_ogt_sources"] == ""
        assert row["cluster_id"] == ""
        assert row["split"] == ""

    def test_it_writes_the_schema_the_sweep_reads(self, tmp_path):
        """15_ceiling_sweep.py joins on seq_id and reads ogt_c and cluster_id.
        Matching stage A's column set is what lets it run unchanged."""
        broad = REPO_ROOT / "data" / "processed" / "broad_label_table.csv"
        if not broad.exists():
            pytest.skip("stage A corpus not present")
        with broad.open(newline="") as handle:
            assert next(csv.reader(handle)) == list(MOD.COLUMNS)
