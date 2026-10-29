"""Tests for the Engqvist OGT loader.

The external corpus exists to tell "the ceiling is a property of corpora"
apart from "the ceiling is a property of how we built stage A". That only
works if the join is right: an organism-name mismatch does not fail loudly,
it quietly produces a smaller and differently composed corpus, and the sweep
over it still prints a plausible curve. So the join key and the rejection
paths are asserted directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thermocas.data.engqvist import (
    EngqvistTable,
    engqvist_organism_key,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER = "organism\tdomain\ttemperature\ttaxid\tlineage_text\n"


def table(tmp_path: Path, *rows: str, **kwargs) -> EngqvistTable:
    path = tmp_path / "temperature_data.tsv"
    path.write_text(HEADER + "".join(rows))
    return EngqvistTable.load(path, **kwargs)


def row(organism: str, domain: str, temperature: str, taxid: str = "1") -> str:
    return f"{organism}\t{domain}\t{temperature}\t{taxid}\tcellular organisms\n"


class TestOrganismKey:
    def test_it_replaces_the_underscore_engqvist_uses(self):
        """Engqvist writes `abiotrophia_adiacens`; without the substitution the
        whole binomial is one token and every join against UniProt misses."""
        assert engqvist_organism_key("abiotrophia_adiacens") == "abiotrophia adiacens"

    def test_it_agrees_with_the_gosha_key_for_the_same_species(self):
        from thermocas.data.ogt import normalize_organism

        assert engqvist_organism_key("thermus_thermophilus") == normalize_organism(
            "Thermus thermophilus HB8"
        )

    def test_it_drops_strain_suffixes(self):
        assert engqvist_organism_key("escherichia_coli_k12") == "escherichia coli"

    def test_it_lowercases(self):
        assert engqvist_organism_key("Thermus_aquaticus") == "thermus aquaticus"


class TestLoad:
    def test_it_loads_a_prokaryote(self, tmp_path):
        loaded = table(tmp_path, row("thermus_aquaticus", "Bacteria", "70"))
        assert len(loaded) == 1
        assert loaded.by_key["thermus aquaticus"].ogt_c == 70.0

    def test_eukaryotes_are_skipped_by_default(self, tmp_path):
        """7,682 of the 21,498 rows are eukaryotic. A corpus of reviewed
        bacteria and archaea has nothing to join them to, and counting them as
        unmatched would misreport the join rate."""
        loaded = table(tmp_path, row("saccharomyces_cerevisiae", "Eukaryota", "30"))
        assert len(loaded) == 0
        assert loaded.report.rejected_domain == 1

    def test_archaea_are_kept(self, tmp_path):
        loaded = table(tmp_path, row("pyrococcus_furiosus", "Archaea", "100"))
        assert len(loaded) == 1

    def test_a_nonnumeric_temperature_is_rejected(self, tmp_path):
        loaded = table(tmp_path, row("x_y", "Bacteria", "warm"))
        assert len(loaded) == 0
        assert loaded.report.rejected_nonnumeric == 1

    def test_an_implausible_temperature_is_rejected(self, tmp_path):
        """The same bound the Gosha loader applies, for the same reason:
        scraped catalogues carry values nothing can grow at, and they land in
        the thermophilic class specifically."""
        loaded = table(tmp_path, row("b_haloalkaliphilus", "Bacteria", "230"))
        assert len(loaded) == 0
        assert loaded.report.rejected_implausible == 1
        assert "230" in loaded.report.implausible_examples[0]

    def test_the_bounds_are_configurable(self, tmp_path):
        loaded = table(
            tmp_path, row("x_y", "Bacteria", "90"), max_plausible_c=80.0
        )
        assert loaded.report.rejected_implausible == 1

    def test_a_blank_name_is_rejected(self, tmp_path):
        loaded = table(tmp_path, row("", "Bacteria", "37"))
        assert loaded.report.rejected_no_name == 1

    def test_duplicate_keys_keep_the_first_and_are_counted(self, tmp_path):
        loaded = table(
            tmp_path,
            row("escherichia_coli", "Bacteria", "37", "562"),
            row("escherichia_coli_k12", "Bacteria", "30", "83333"),
        )
        assert len(loaded) == 1
        assert loaded.by_key["escherichia coli"].ogt_c == 37.0
        assert loaded.report.duplicate_keys == 1


class TestLookup:
    def test_taxid_matches_exactly(self, tmp_path):
        loaded = table(tmp_path, row("thermus_aquaticus", "Bacteria", "70", "271"))
        assert loaded.lookup("271", None).ogt_c == 70.0

    def test_the_name_is_the_fallback(self, tmp_path):
        """UniProt carries strain-level taxids while Engqvist is indexed at
        species level, so most real hits come through the name."""
        loaded = table(tmp_path, row("thermus_aquaticus", "Bacteria", "70", "271"))
        assert loaded.lookup("999999", "Thermus aquaticus YT-1").ogt_c == 70.0

    def test_an_unknown_organism_returns_none(self, tmp_path):
        loaded = table(tmp_path, row("thermus_aquaticus", "Bacteria", "70"))
        assert loaded.lookup(None, "Homo sapiens") is None

    def test_nothing_to_look_up_returns_none(self, tmp_path):
        loaded = table(tmp_path, row("thermus_aquaticus", "Bacteria", "70"))
        assert loaded.lookup(None, None) is None


class TestAgainstTheRealTable:
    """The downloaded table, when it is present. These are the numbers the
    external corpus's provenance claims rest on."""

    @pytest.fixture
    def real(self):
        path = REPO_ROOT / "data" / "raw" / "engqvist" / "temperature_data.tsv"
        if not path.exists():
            pytest.skip("Engqvist table not downloaded")
        return EngqvistTable.load(path)

    def test_it_holds_the_published_row_count(self, real):
        assert real.report.rows == 21498

    def test_prokaryotes_are_the_majority_of_it(self, real):
        assert real.report.rejected_domain == 7682

    def test_it_resolves_finer_than_the_stage_a_corpus(self, real):
        """The reason this corpus is worth building: the stage A label table
        carries 47 distinct OGT values, which is what limits the threshold
        sweep to ~9 effective thresholds."""
        assert len({r.ogt_c for r in real.by_key.values()}) > 47

    def test_a_known_hyperthermophile_is_labeled_hot(self, real):
        found = real.lookup(None, "Pyrococcus furiosus")
        assert found is not None and found.ogt_c >= 90
