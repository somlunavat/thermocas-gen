"""Stage A species selection and Cas9 exclusion.

The two things tested hardest are the two that would quietly ruin the
experiment: an unbalanced corpus that lets the model ignore the conditioning
token, and Cas9 leaking into pre-training, which would make the stage B
leakage gate meaningless.
"""

from __future__ import annotations

import pytest

from thermocas.config import Config, GrowthTemperatureBinner
from thermocas.data.broad import (
    Cas9Exclusion,
    SpeciesSelection,
    select_species,
)
from thermocas.data.ogt import OgtRecord


def bin_for(value: float) -> str:
    return "meso" if value < 45 else "thermo"


def rec(name: str, ogt: float, n_sources: int = 3, spread: float = 0.0) -> OgtRecord:
    # Build source values that agree on the bin and give the requested spread.
    values = {f"C{i}": ogt for i in range(n_sources)}
    if n_sources >= 2 and spread:
        values["C1"] = ogt + spread
    return OgtRecord(organism=name, organism_key=name.lower(), ogt_c=ogt, source_values=values)


def selection(**overrides) -> SpeciesSelection:
    base = dict(
        require_crosscheck=True,
        max_per_bin=400,
        balance="match_smallest",
        exclude_positive_control_hosts=True,
    )
    base.update(overrides)
    return SpeciesSelection(**base)


class TestSelectionConfig:
    def test_loads_from_shipped_config(self):
        loaded = SpeciesSelection.from_config(Config.load("pretrain"))
        assert loaded.require_crosscheck is True
        assert loaded.balance == "match_smallest"

    def test_unknown_balance_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown balance"):
            selection(balance="proportional")

    def test_zero_quota_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            selection(max_per_bin=0)


class TestSelectSpecies:
    def test_balances_the_scarce_class_against_the_abundant_one(self):
        """Gosha has ~20x more mesophilic species than thermophilic.

        Sampling proportionally would let the model minimize loss by ignoring
        the conditioning token and always predicting mesophilic features,
        which is exactly what Gate 2 is designed to catch.
        """
        records = [rec(f"meso{i}", 30.0) for i in range(500)]
        records += [rec(f"thermo{i}", 60.0) for i in range(40)]
        selected, report = select_species(records, bin_for, selection())
        assert len(selected["meso"]) == len(selected["thermo"]) == 40
        assert report.per_bin_available["meso"] == 500

    def test_max_per_bin_caps_even_the_scarce_class(self):
        records = [rec(f"m{i}", 30.0) for i in range(100)]
        records += [rec(f"t{i}", 60.0) for i in range(100)]
        selected, _ = select_species(records, bin_for, selection(max_per_bin=25))
        assert len(selected["meso"]) == len(selected["thermo"]) == 25

    def test_max_per_bin_mode_does_not_balance(self):
        records = [rec(f"m{i}", 30.0) for i in range(100)]
        records += [rec(f"t{i}", 60.0) for i in range(10)]
        selected, _ = select_species(
            records, bin_for, selection(balance="max_per_bin", max_per_bin=50)
        )
        assert len(selected["meso"]) == 50
        assert len(selected["thermo"]) == 10

    def test_uncrosschecked_species_dropped_when_required(self):
        records = [
            rec("good", 60.0, n_sources=3),
            OgtRecord("lonely", "lonely", 96.0, {"NCTC": 96.0}),  # single source
        ]
        selected, report = select_species(records, bin_for, selection(require_crosscheck=True))
        names = [r.organism for members in selected.values() for r in members]
        assert names == ["good"]
        assert report.dropped_uncrosschecked == 1

    def test_uncrosschecked_kept_when_not_required(self):
        records = [OgtRecord("lonely", "lonely", 96.0, {"NCTC": 96.0})]
        selected, report = select_species(records, bin_for, selection(require_crosscheck=False))
        assert report.dropped_uncrosschecked == 0
        assert len(selected["thermo"]) == 1

    def test_positive_control_hosts_are_excluded(self):
        """Stage A must not teach the model GeoCas9's own genomic context."""
        records = [
            rec("Geobacillus stearothermophilus", 55.0),
            rec("Geobacillus thermodenitrificans", 55.0),
            rec("Other thermophile", 60.0),
        ]
        selected, report = select_species(
            records,
            bin_for,
            selection(),
            excluded_hosts=["Geobacillus stearothermophilus", "Geobacillus thermodenitrificans"],
        )
        assert [r.organism for r in selected["thermo"]] == ["Other thermophile"]
        assert report.dropped_excluded_host == 2

    def test_exclusion_can_be_disabled(self):
        records = [rec("Geobacillus stearothermophilus", 55.0), rec("Other", 60.0)]
        selected, report = select_species(
            records,
            bin_for,
            selection(exclude_positive_control_hosts=False),
            excluded_hosts=["Geobacillus stearothermophilus"],
        )
        assert report.dropped_excluded_host == 0
        assert len(selected["thermo"]) == 2

    def test_best_evidenced_species_survive_the_cut(self):
        """When the abundant class is trimmed, keep the best-supported OGTs."""
        records = [rec(f"weak{i}", 30.0, n_sources=2) for i in range(10)]
        records += [rec(f"strong{i}", 30.0, n_sources=18) for i in range(3)]
        records += [rec(f"t{i}", 60.0, n_sources=5) for i in range(3)]
        selected, _ = select_species(records, bin_for, selection())
        kept = {r.organism for r in selected["meso"]}
        assert kept == {"strong0", "strong1", "strong2"}

    def test_deterministic_for_a_fixed_seed(self):
        records = [rec(f"m{i}", 30.0, n_sources=3) for i in range(50)]
        records += [rec(f"t{i}", 60.0, n_sources=3) for i in range(5)]
        first, _ = select_species(records, bin_for, selection(), seed=11)
        second, _ = select_species(records, bin_for, selection(), seed=11)
        assert [r.organism for r in first["meso"]] == [r.organism for r in second["meso"]]

    def test_empty_input_returns_nothing_without_crashing(self):
        selected, report = select_species([], bin_for, selection())
        assert selected == {}
        assert report.considered == 0


class TestCas9Exclusion:
    def exclusion(self, **overrides) -> Cas9Exclusion:
        base = dict(
            enabled=True,
            name_patterns=("cas9", "csn1", "csx12", "crispr-associated endonuclease"),
            pfam_profiles=("PF22702", "PF13395"),
            max_identity_to_heldout=0.30,
            alignment_coverage=0.50,
            fail_on_violation=True,
        )
        base.update(overrides)
        return Cas9Exclusion(**base)

    def test_loads_from_shipped_config(self):
        loaded = Cas9Exclusion.from_config(Config.load("pretrain"))
        assert loaded.enabled is True
        assert "PF22702" in loaded.pfam_profiles
        # Stricter than the stage B leakage threshold of 0.45, because a large
        # pre-training corpus hides a borderline homolog more easily.
        assert loaded.max_identity_to_heldout < 0.45

    def test_identity_as_percent_rejected(self):
        with pytest.raises(ValueError, match="fraction"):
            self.exclusion(max_identity_to_heldout=30)

    def test_pfam_versions_stripped(self):
        assert self.exclusion(pfam_profiles=("PF22702.3",)).pfam_profiles == ("PF22702",)

    @pytest.mark.parametrize(
        "description",
        [
            "CRISPR-associated endonuclease Cas9",
            "Cas9",
            "type II CRISPR RNA-guided endonuclease Cas9",
            "cas9-like protein",
            "Csn1 family protein",
            "Cas9/Csx12 endonuclease",
        ],
    )
    def test_cas9_annotations_are_caught(self, description):
        assert self.exclusion().matches_annotation(description) is not None

    @pytest.mark.parametrize(
        "description",
        [
            "DNA polymerase III subunit alpha",
            "50S ribosomal protein L2",
            "hypothetical protein",
            "CRISPR-associated protein Cas1",
            "Cas12a endonuclease",
            "cas9000 hypothetical transporter",
        ],
    )
    def test_unrelated_proteins_are_not_caught(self, description):
        """Cas1 and Cas12a are CRISPR proteins but not Cas9.

        Over-excluding would thin the pre-training corpus for no benefit; the
        last case checks that 'cas9' inside a longer token does not match.
        """
        assert self.exclusion().matches_annotation(description) is None

    def test_disabled_exclusion_matches_nothing(self):
        assert self.exclusion(enabled=False).matches_annotation("Cas9") is None

    def test_matching_is_case_insensitive(self):
        assert self.exclusion().matches_annotation("CAS9 ENDONUCLEASE") is not None


class TestAgainstShippedConfigs:
    def test_stage_a_splits_have_no_test_set(self):
        """Stage A is not evaluated on its own; all held-out evaluation
        happens on the Cas9 corpus where the controls live."""
        cfg = Config.load("pretrain")
        assert cfg["split/fractions/test"] == 0.0
        assert cfg["split/fractions/train"] + cfg["split/fractions/val"] == 1.0

    def test_stage_a_does_not_hold_out_positive_controls(self):
        cfg = Config.load("pretrain")
        assert cfg["split/holdout_positive_controls"] is False

    def test_stage_a_context_is_shorter_than_cas9(self):
        """Stage A proteins need not be Cas9-length, which keeps the corpus
        affordable; stage B supplies the long-context training."""
        assert Config.load("pretrain")["sampling/max_length"] <= 1024

    def test_bins_match_the_cas9_stage(self):
        """A conditioning token learned in stage A is reused in stage B, so
        the bin definitions must be identical or the token changes meaning."""
        data_bins = GrowthTemperatureBinner.from_config(Config.load("data")).names
        assert data_bins == ["meso", "thermo"]
