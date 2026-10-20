"""OGT loading, organism-name normalization and the within-Gosha cross-check.

The class at the bottom runs against the real downloaded Gosha CSV when it is
present, which is what surfaced the implausible values and the fact that an
absolute spread tolerance rejects genuine thermophiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thermocas.config import Config, GrowthTemperatureBinner
from thermocas.data.ogt import (
    CONSENSUS_COLUMN,
    OgtRecord,
    OgtTable,
    JoinReport,
    gate1_failures,
    normalize_organism,
)

GOSHA_CSV = Path("data/raw/gosha/all_species_temperatures.csv")


def bin_for(value: float) -> str:
    return "meso" if value < 45 else ("thermo" if value < 70 else "hyper")


def record(ogt_c: float, sources: dict[str, float] | None = None) -> OgtRecord:
    return OgtRecord(
        organism="Test organism",
        organism_key="test organism",
        ogt_c=ogt_c,
        source_values=sources or {},
    )


class TestNormalizeOrganism:
    def test_reduces_to_genus_and_species(self):
        assert normalize_organism("Streptococcus pyogenes") == "streptococcus pyogenes"

    def test_strips_strain_suffixes(self):
        """Gosha uses bare binomials; NCBI sources carry strain designations."""
        assert normalize_organism("Polaribacter irgensii 23-P") == "polaribacter irgensii"
        assert (
            normalize_organism("Streptococcus pyogenes serotype M1") == "streptococcus pyogenes"
        )
        assert (
            normalize_organism("Francisella tularensis subsp. novicida U112")
            == "francisella tularensis"
        )
        assert normalize_organism("Geobacillus sp. LC300") == "geobacillus"

    def test_strips_the_ncbi_hash_prefix(self):
        """The NCBI-derived files head their name column '#Organism Name'."""
        assert normalize_organism("#Polaribacter irgensii") == "polaribacter irgensii"

    def test_candidatus_is_a_status_not_a_genus(self):
        assert normalize_organism("Candidatus Pelagibacter ubique") == "pelagibacter ubique"

    def test_case_and_whitespace_insensitive(self):
        assert normalize_organism("  STREPTOCOCCUS   PYOGENES  ") == "streptococcus pyogenes"

    def test_brackets_removed(self):
        assert normalize_organism("[Clostridium] difficile") == "clostridium difficile"

    def test_empty_input_yields_empty_key(self):
        assert normalize_organism("   ") == ""
        assert normalize_organism("") == ""


class TestCrossCheck:
    def test_single_source_is_never_verified(self):
        """The actual Gosha error mode: one collection reporting a wild value.

        Streptococcus massilliensis is listed at 96 C by one collection, which
        would label a mesophile as a hyperthermophile.
        """
        assert not record(96.0, {"NCTC": 96.0}).is_verified(bin_for)

    def test_agreeing_sources_verify(self):
        assert record(37.0, {"ATCC": 37.0, "DSMZ": 37.0, "JCM": 37.0}).is_verified(bin_for)

    def test_wide_spread_within_one_bin_still_verifies(self):
        """Spread grows with temperature; the bin is what gets trained on.

        Pyrococcus furiosus spans 20 C across five collections and is
        unambiguously a hyperthermophile. An absolute 5 C tolerance would
        reject it and most other genuine thermophiles.
        """
        pyrococcus = record(93.0, {"ATCC": 100.0, "DSMZ": 95.0, "JCM": 90.0, "CABRI": 85.0, "VKM": 93.0})
        assert pyrococcus.source_spread_c == 15.0
        assert pyrococcus.bin_agreement(bin_for) == 1.0
        assert pyrococcus.is_verified(bin_for)

    def test_sources_straddling_a_bin_boundary_do_not_verify(self):
        """Thermus thermophilus sits at 67 C with sources either side of 70."""
        straddling = record(67.0, {"ATCC": 85.0, "DSMZ": 65.0, "JCM": 60.0, "CABRI": 75.0})
        assert straddling.bin_agreement(bin_for) == 0.5
        assert not straddling.is_verified(bin_for, min_bin_agreement=0.67)

    def test_agreement_threshold_is_respected(self):
        partial = record(50.0, {"a": 50.0, "b": 50.0, "c": 30.0})
        assert partial.bin_agreement(bin_for) == pytest.approx(2 / 3)
        assert partial.is_verified(bin_for, min_bin_agreement=0.6)
        assert not partial.is_verified(bin_for, min_bin_agreement=0.9)

    def test_no_sources_means_no_agreement(self):
        assert record(37.0).bin_agreement(bin_for) == 0.0
        assert not record(37.0).is_verified(bin_for)

    def test_spread_of_a_single_source_is_zero(self):
        assert record(37.0, {"ATCC": 37.0}).source_spread_c == 0.0


class TestLoading:
    def write_csv(self, tmp_path, rows, columns=("Organism", CONSENSUS_COLUMN, "ATCC", "DSMZ")):
        path = tmp_path / "gosha.csv"
        lines = [",".join(columns)]
        lines += [",".join(str(c) for c in row) for row in rows]
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_loads_and_looks_up(self, tmp_path):
        path = self.write_csv(tmp_path, [("Streptococcus pyogenes", 37, 37, 37)])
        table = OgtTable.from_gosha_csv(path)
        found = table.lookup("Streptococcus pyogenes serotype M1")
        assert found is not None
        assert found.ogt_c == 37.0
        assert found.source_values == {"ATCC": 37.0, "DSMZ": 37.0}

    def test_implausible_values_rejected_and_counted(self, tmp_path):
        """Gosha really does contain 230 C and 187 C entries."""
        path = self.write_csv(
            tmp_path,
            [
                ("Bacillus haloalkaliphilus", 230, "", ""),
                ("Halomonas aerodenitrificans", 187, "", ""),
                ("Streptococcus pyogenes", 37, 37, ""),
            ],
        )
        table = OgtTable.from_gosha_csv(path, max_plausible_c=130.0)
        assert len(table) == 1
        assert table.report.rejected_implausible == 2
        assert table.lookup("Bacillus haloalkaliphilus") is None
        assert "230" in table.report.summary()

    def test_nonnumeric_consensus_rejected(self, tmp_path):
        path = self.write_csv(tmp_path, [("A b", "Unknown", "", ""), ("C d", 30, "", "")])
        table = OgtTable.from_gosha_csv(path)
        assert len(table) == 1
        assert table.report.rejected_nonnumeric == 1

    def test_negative_temperatures_allowed_for_psychrophiles(self, tmp_path):
        path = self.write_csv(tmp_path, [("Psychro bacter", -5, -5, "")])
        table = OgtTable.from_gosha_csv(path)
        assert table.lookup("Psychro bacter").ogt_c == -5.0

    def test_schema_change_is_diagnosed(self, tmp_path):
        path = tmp_path / "gosha.csv"
        path.write_text("species,temp\nA b,37\n")
        with pytest.raises(ValueError, match="missing expected column"):
            OgtTable.from_gosha_csv(path)

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "gosha.csv"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            OgtTable.from_gosha_csv(path)

    def test_unknown_collections_are_picked_up_automatically(self, tmp_path):
        """A later Gosha release adding a collection must not need a code change."""
        path = self.write_csv(
            tmp_path,
            [("A b", 37, 37, 37, 38)],
            columns=("Organism", CONSENSUS_COLUMN, "ATCC", "DSMZ", "NEWCOLLECTION"),
        )
        table = OgtTable.from_gosha_csv(path)
        assert "NEWCOLLECTION" in table.lookup("A b").source_values

    def test_duplicate_keys_keep_the_better_supported_row(self, tmp_path):
        path = self.write_csv(
            tmp_path,
            [("Geobacillus stearothermophilus", 55, 55, ""), ("Geobacillus stearothermophilus str. X", 60, 60, 61)],
        )
        table = OgtTable.from_gosha_csv(path)
        assert len(table) == 1
        assert table.report.duplicate_keys == 1
        assert table.lookup("Geobacillus stearothermophilus").n_sources == 2


class TestGate1:
    def test_passes_when_every_bin_meets_the_minimum(self):
        report = JoinReport(total=100, labeled=100, bin_counts={"meso": 500, "thermo": 300, "hyper": 250})
        assert gate1_failures(report, min_per_bin=200) == []

    def test_names_the_short_bins(self):
        report = JoinReport(bin_counts={"meso": 5000, "thermo": 120, "hyper": 30})
        failures = gate1_failures(report, min_per_bin=200)
        assert len(failures) == 2
        assert any("thermo" in f for f in failures)
        assert any("hyper: 30 < 200" in f for f in failures)

    def test_verified_counts_are_the_stricter_reading(self):
        report = JoinReport(
            bin_counts={"meso": 900, "thermo": 400},
            verified_bin_counts={"meso": 800, "thermo": 150},
        )
        assert gate1_failures(report, 200, use_verified=False) == []
        assert len(gate1_failures(report, 200, use_verified=True)) == 1


@pytest.mark.skipif(
    not GOSHA_CSV.is_file(), reason="Gosha CSV not downloaded (scripts/04_join_ogt.py --fetch)"
)
class TestAgainstRealGosha:
    """Runs against the actual downloaded database."""

    @pytest.fixture(scope="class")
    def table(self):
        cfg = Config.load("data")
        return OgtTable.from_gosha_csv(
            GOSHA_CSV,
            min_plausible_c=cfg["ogt/min_plausible_c"],
            max_plausible_c=cfg["ogt/max_plausible_c"],
        )

    def test_loads_the_expected_number_of_species(self, table):
        # The preprint reports 25,324 species; two are rejected as implausible.
        assert len(table) == 25322
        assert table.report.rejected_implausible == 2

    def test_the_known_bad_rows_are_the_ones_rejected(self, table):
        assert table.lookup("Bacillus haloalkaliphilus") is None
        assert table.lookup("Halomonas aerodenitrificans") is None

    def test_every_seed_organism_resolves(self, table):
        """If a seed's host cannot be labeled, neither can its homologs."""
        for organism in (
            "Streptococcus pyogenes",
            "Staphylococcus aureus",
            "Geobacillus stearothermophilus",
            "Geobacillus thermodenitrificans",
            "Neisseria meningitidis",
            "Campylobacter jejuni",
            "Streptococcus thermophilus",
            "Francisella tularensis subsp. novicida",
        ):
            assert table.lookup(organism) is not None, organism

    def test_thermophilic_hosts_land_in_thermophilic_bins(self, table):
        binner = GrowthTemperatureBinner.from_config(Config.load("data"))
        assert binner.bin_for(table.lookup("Geobacillus stearothermophilus").ogt_c) == "thermo"
        assert binner.bin_for(table.lookup("Geobacillus thermodenitrificans").ogt_c) == "thermo"
        assert binner.bin_for(table.lookup("Pyrococcus furiosus").ogt_c) == "hyper"
        assert binner.bin_for(table.lookup("Streptococcus pyogenes").ogt_c) == "meso"

    def test_genuine_hyperthermophiles_survive_the_crosscheck(self, table):
        """The regression that an absolute spread tolerance would have caused."""
        binner = GrowthTemperatureBinner.from_config(Config.load("data"))
        for organism in ("Pyrococcus furiosus", "Pyrolobus fumarii", "Methanopyrus kandleri"):
            assert table.lookup(organism).is_verified(binner.bin_for), organism

    def test_single_source_thermophile_is_rejected(self, table):
        binner = GrowthTemperatureBinner.from_config(Config.load("data"))
        massilliensis = table.lookup("Streptococcus massilliensis")
        assert massilliensis.ogt_c == 96.0  # one collection, implausible for the genus
        assert not massilliensis.is_verified(binner.bin_for)

    def test_thermophilic_classes_are_scarce_at_the_species_level(self, table):
        """Gate 1 context: thermophiles are ~4% of species before any Cas9 filter.

        Recorded as a test so a Gosha release that changes the picture is
        noticed, since the whole project depends on these counts surviving
        intersection with the Cas9 homolog set.
        """
        binner = GrowthTemperatureBinner.from_config(Config.load("data"))
        counts: dict[str, int] = {}
        for rec in table.records.values():
            name = binner.bin_for(rec.ogt_c)
            counts[name] = counts.get(name, 0) + 1
        assert counts["meso"] > 20000
        assert counts["thermo"] < 1500
        assert counts["hyper"] < 500


class TestParseOrganismFromHeader:
    """Each input source in this pipeline labels organisms differently."""

    def test_uniref_tax_field(self):
        from thermocas.data.ogt import parse_organism_from_header

        header = (
            "CRISPR-associated endonuclease Cas9 n=1 "
            "Tax=Streptococcus pyogenes TaxID=1314 RepID=CAS9_STRP1"
        )
        assert parse_organism_from_header(header) == "Streptococcus pyogenes"

    def test_uniref_tax_with_multiword_organism(self):
        from thermocas.data.ogt import parse_organism_from_header

        header = "Cas9 n=3 Tax=Francisella tularensis subsp. novicida TaxID=401 RepID=X"
        assert parse_organism_from_header(header) == "Francisella tularensis subsp. novicida"

    def test_uniprot_os_field(self):
        from thermocas.data.ogt import parse_organism_from_header

        header = "CRISPR-associated endonuclease Cas9 OS=Streptococcus pyogenes OX=1314 GN=cas9 PE=1 SV=1"
        assert parse_organism_from_header(header) == "Streptococcus pyogenes"

    def test_ncbi_bracketed_organism(self):
        from thermocas.data.ogt import parse_organism_from_header

        header = "type II CRISPR RNA-guided endonuclease Cas9 [Geobacillus thermodenitrificans]"
        assert parse_organism_from_header(header) == "Geobacillus thermodenitrificans"

    def test_this_projects_seed_headers(self):
        from thermocas.data.ogt import parse_organism_from_header

        header = "SpCas9 | Streptococcus pyogenes serotype M1 | type=II-A | role=meso_control | len=1368"
        assert parse_organism_from_header(header) == "Streptococcus pyogenes serotype M1"

    def test_returns_none_when_absent(self):
        from thermocas.data.ogt import parse_organism_from_header

        assert parse_organism_from_header("hypothetical protein") is None
        assert parse_organism_from_header("") is None
        assert parse_organism_from_header("   ") is None

    def test_committed_seeds_all_yield_an_organism(self):
        """Every seed must be labelable, or its homologs cannot be either."""
        from thermocas.data.fasta import read_fasta
        from thermocas.data.ogt import parse_organism_from_header

        seeds = list(read_fasta(Config.load("data").resolve_path("paths/seeds")))
        for seed in seeds:
            assert parse_organism_from_header(seed.description), seed.seq_id
