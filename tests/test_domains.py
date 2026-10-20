"""Domain confirmation: config handling, domtblout parsing, and verdicts.

The integration test at the bottom runs real HMMER against the committed
reference seeds, so the Pfam profile choice is checked against actual Cas9
sequences rather than against a mock.
"""

from __future__ import annotations

import shutil

import pytest

from thermocas.config import Config
from thermocas.data.domains import (
    DomainHit,
    DomainRequirements,
    call_domains,
    concatenate_profiles,
    parse_domtblout,
    run_hmmsearch,
)

REPO_PROFILE_DIR = "data/raw/pfam"


def requirements(**overrides) -> DomainRequirements:
    base = dict(
        required=("RuvC", "HNH"),
        profiles={"RuvC": ("PF22702", "PF18541"), "HNH": ("PF13395", "PF01844")},
        max_evalue=1e-3,
        informational={"Cas9_PI": ("PF16595",)},
    )
    base.update(overrides)
    return DomainRequirements(**base)


def hit(**overrides) -> DomainHit:
    base = dict(
        sequence_id="seq1",
        sequence_length=1368,
        profile_name="Cas9_RuvC",
        profile_accession="PF22702.3",
        i_evalue=1e-50,
        score=200.0,
        ali_from=760,
        ali_to=980,
    )
    base.update(overrides)
    return DomainHit(**base)


class TestRequirements:
    def test_loads_from_shipped_config(self):
        req = DomainRequirements.from_config(Config.load("data"))
        assert req.required == ("RuvC", "HNH")
        # PF22702 is the Cas9-specific RuvC model and must be present: the
        # generic PF18541 alone misses St1Cas9 entirely.
        assert "PF22702" in req.profiles["RuvC"]
        assert "PF13395" in req.profiles["HNH"]

    def test_versions_are_stripped_so_configs_need_not_pin_them(self):
        req = DomainRequirements(
            required=("RuvC",), profiles={"RuvC": ("PF22702.3",)}, max_evalue=1e-3
        )
        assert req.profiles["RuvC"] == ("PF22702",)

    def test_bare_string_profile_is_accepted(self):
        cfg = Config(
            {"domains": {"require": ["RuvC"], "hmm_profiles": {"RuvC": "PF22702"}, "hmm_evalue": 1e-3}}
        )
        assert DomainRequirements.from_config(cfg).profiles["RuvC"] == ("PF22702",)

    def test_requirement_without_profiles_is_rejected(self):
        """A requirement with no profile can never be satisfied, so every
        sequence would fail with no indication of why."""
        with pytest.raises(ValueError, match="never be satisfied"):
            DomainRequirements(required=("RuvC", "HNH"), profiles={"RuvC": ("PF22702",)}, max_evalue=1e-3)

    def test_empty_requirements_rejected(self):
        with pytest.raises(ValueError, match="no required domains"):
            DomainRequirements(required=(), profiles={}, max_evalue=1e-3)

    def test_nonpositive_evalue_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            requirements(max_evalue=0)

    def test_accession_maps_back_to_its_domain(self):
        req = requirements()
        assert req.domain_for_accession("PF22702") == "RuvC"
        assert req.domain_for_accession("PF18541.8") == "RuvC"
        assert req.domain_for_accession("PF13395") == "HNH"
        assert req.domain_for_accession("PF16595") is None  # informational only

    def test_all_accessions_includes_informational_without_duplicates(self):
        accessions = requirements().all_accessions()
        assert set(accessions) == {"PF22702", "PF18541", "PF13395", "PF01844", "PF16595"}
        assert len(accessions) == len(set(accessions))


class TestParsing:
    def row(self, **kw) -> str:
        fields = [
            kw.get("target", "Q99ZW2"), "-", str(kw.get("tlen", 1368)),
            kw.get("query", "Cas9_RuvC"), kw.get("acc", "PF22702.3"), "220",
            kw.get("seq_evalue", "1e-60"), "200.0", "0.1",
            "1", "1",
            "1e-55", kw.get("i_evalue", "9.9e-70"), "199.0", "0.1",
            "1", "220",
            str(kw.get("ali_from", 760)), str(kw.get("ali_to", 980)),
            "758", "982", "0.95",
            "Cas9 RuvC domain",
        ]
        return " ".join(fields)

    def test_parses_a_domtblout_row(self, tmp_path):
        path = tmp_path / "d.tbl"
        path.write_text("# comment\n" + self.row() + "\n")
        (parsed,) = list(parse_domtblout(path))
        assert parsed.sequence_id == "Q99ZW2"
        assert parsed.profile_name == "Cas9_RuvC"
        assert parsed.profile_accession == "PF22702.3"
        assert parsed.profile_base_accession == "PF22702"
        assert parsed.i_evalue == 9.9e-70
        assert parsed.ali_length == 221

    def test_uses_domain_ievalue_not_sequence_evalue(self, tmp_path):
        """Column 13 (i-Evalue), not column 7.

        A long multi-domain protein can carry a strong sequence-level E-value
        from one good domain while a weak second hit rides along; filtering on
        the sequence value would accept that weak hit as real.
        """
        path = tmp_path / "d.tbl"
        path.write_text(self.row(seq_evalue="1e-99", i_evalue="0.5") + "\n")
        (parsed,) = list(parse_domtblout(path))
        assert parsed.i_evalue == 0.5

    def test_description_with_spaces_does_not_shift_columns(self, tmp_path):
        path = tmp_path / "d.tbl"
        path.write_text(self.row() + "\n")
        (parsed,) = list(parse_domtblout(path))
        assert parsed.ali_from == 760 and parsed.ali_to == 980

    def test_tblout_instead_of_domtblout_is_diagnosed(self, tmp_path):
        path = tmp_path / "d.tbl"
        path.write_text("Q99ZW2 - Cas9_RuvC PF22702.3 1e-60 200.0 0.1 1e-55 199 0.1 1 0 0 0 0 0 0 desc\n")
        with pytest.raises(ValueError, match="--domtblout"):
            list(parse_domtblout(path))

    def test_blank_and_comment_lines_skipped(self, tmp_path):
        path = tmp_path / "d.tbl"
        path.write_text(f"#\n\n{self.row()}\n#\n")
        assert len(list(parse_domtblout(path))) == 1


class TestCalls:
    def test_both_domains_present_passes(self):
        req = requirements()
        calls = call_domains(
            [hit(), hit(profile_name="HNH_4", profile_accession="PF13395.13", i_evalue=2.4e-19)], req
        )
        call = calls["seq1"]
        assert call.passes(req)
        assert call.missing(req) == []
        assert call.best_profile == {"RuvC": "Cas9_RuvC", "HNH": "HNH_4"}

    def test_missing_domain_fails_and_is_named(self):
        """FnCas9's real failure mode: HNH present, no RuvC model matches."""
        req = requirements()
        calls = call_domains(
            [hit(profile_name="HNH", profile_accession="PF01844.30", i_evalue=1.2e-05)], req
        )
        call = calls["seq1"]
        assert not call.passes(req)
        assert call.missing(req) == ["RuvC"]

    def test_hits_above_threshold_do_not_satisfy_a_requirement(self):
        req = requirements(max_evalue=1e-3)
        calls = call_domains([hit(i_evalue=2.0)], req)
        assert calls["seq1"].missing(req) == ["RuvC", "HNH"]

    def test_threshold_is_inclusive(self):
        req = requirements(max_evalue=1e-3)
        calls = call_domains([hit(i_evalue=1e-3)], req)
        assert "RuvC" in calls["seq1"].satisfied

    def test_best_profile_wins_among_alternatives(self):
        """Both RuvC models may hit; the report must name the stronger one."""
        req = requirements()
        calls = call_domains(
            [
                hit(profile_name="RuvC_III", profile_accession="PF18541.8", i_evalue=9.3e-05),
                hit(profile_name="Cas9_RuvC", profile_accession="PF22702.3", i_evalue=9.9e-70),
            ],
            req,
        )
        assert calls["seq1"].best_profile["RuvC"] == "Cas9_RuvC"
        assert calls["seq1"].best_evalue["RuvC"] == 9.9e-70

    def test_informational_profiles_recorded_but_not_required(self):
        req = requirements()
        calls = call_domains(
            [
                hit(),
                hit(profile_name="HNH_4", profile_accession="PF13395.13", i_evalue=1e-19),
                hit(profile_name="Cas9_PI", profile_accession="PF16595.11", i_evalue=2.8e-63),
            ],
            req,
        )
        call = calls["seq1"]
        assert call.passes(req)
        assert call.informational["Cas9_PI"] == 2.8e-63
        # An informational profile must never stand in for a required domain.
        assert "Cas9_PI" not in call.satisfied

    def test_multiple_sequences_kept_separate(self):
        req = requirements()
        calls = call_domains(
            [
                hit(sequence_id="a"),
                hit(sequence_id="a", profile_name="HNH_4", profile_accession="PF13395.13", i_evalue=1e-20),
                hit(sequence_id="b"),
            ],
            req,
        )
        assert calls["a"].passes(req)
        assert not calls["b"].passes(req)


@pytest.mark.skipif(shutil.which("hmmsearch") is None, reason="HMMER not installed")
class TestAgainstRealSeeds:
    """Runs HMMER on the committed reference seeds.

    This is what caught the original profile error: the config pinned the
    generic PF18541 for RuvC, which misses St1Cas9 completely.
    """

    @pytest.fixture
    def calls(self, tmp_path):
        cfg = Config.load("data")
        req = DomainRequirements.from_config(cfg)
        profiles = tmp_path / "profiles.hmm"
        concatenate_profiles(req.all_accessions(), cfg.resolve_path("domains/profile_dir"), profiles)
        domtbl = tmp_path / "dom.tbl"
        run_hmmsearch(profiles, cfg.resolve_path("paths/seeds"), domtbl, req.max_evalue, cpu=2)
        return call_domains(parse_domtblout(domtbl), req), req

    def test_canonical_cas9s_pass_both_domains(self, calls):
        results, req = calls
        for accession, name in {
            "Q99ZW2": "SpCas9",
            "J7RUA5": "SaCas9",
            "A0ABF7PG96": "GeoCas9",
            "WP_099233044.1": "ThermoCas9",
            "C9X1G5": "NmeCas9",
            "Q0P897": "CjCas9",
            "G3ECR1": "St1Cas9",
        }.items():
            call = results.get(accession)
            assert call is not None, f"{name} produced no domain hits at all"
            assert call.passes(req), f"{name} missing {call.missing(req)}"

    def test_st1cas9_requires_the_cas9_specific_ruvc_model(self, calls):
        """St1Cas9 is matched by PF22702 and not by PF18541.

        This is the regression that motivated the profile change: with
        PF18541 alone, a canonical type II-A Cas9 is rejected.
        """
        results, _ = calls
        assert results["G3ECR1"].best_profile["RuvC"] == "Cas9_RuvC"

    def test_thermophilic_positive_controls_pass(self, calls):
        """GeoCas9 and ThermoCas9 carry the evaluation stage's claim."""
        results, req = calls
        for accession in ("A0ABF7PG96", "WP_099233044.1"):
            assert results[accession].passes(req)

    def test_type_iib_ruvc_gap_is_still_present(self, calls):
        """FnCas9 matches no Pfam RuvC model; documented limitation.

        If a future Pfam release closes this gap the assertion fails, which is
        the signal to update the limitation note in PROJECT_LOG.md rather than
        leave a stale claim in the paper.
        """
        results, req = calls
        call = results["A0Q5Y3"]
        assert "HNH" in call.satisfied
        assert call.missing(req) == ["RuvC"]
