"""Tests for the Cas9 candidate collection query.

The query decides what the stage B corpus can contain, and a wrong one fails
quietly: a name-only query silently omits orthologs annotated as "type II
CRISPR RNA-guided endonuclease", and a query including the generic nuclease
families returns a large fraction of all HNH proteins. Neither mistake
announces itself downstream -- both just change the corpus size.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load():
    path = REPO_ROOT / "scripts" / "14_collect_cas9.py"
    spec = importlib.util.spec_from_file_location("collect_cas9", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load()

# The generic nuclease models configs/data.yaml allows as alternatives when
# *confirming* a domain. Using them to *find* candidates would return many
# thousands of unrelated nucleases.
GENERIC = ("PF13395", "PF01844", "PF18541")


class TestTheFamilyList:
    def test_it_uses_the_cas9_specific_families(self):
        assert "PF22702" in MOD.CAS9_FAMILIES  # Cas9_RuvC

    def test_it_excludes_the_generic_nuclease_models(self):
        for accession in GENERIC:
            assert accession not in MOD.CAS9_FAMILIES

    def test_the_families_are_distinct(self):
        assert len(set(MOD.CAS9_FAMILIES)) == len(MOD.CAS9_FAMILIES)


class TestTheQuery:
    def test_every_family_appears_as_a_pfam_cross_reference(self):
        query = MOD.build_query(950, 1700)
        for accession in MOD.CAS9_FAMILIES:
            assert f"xref:pfam-{accession}" in query

    def test_no_generic_family_appears(self):
        query = MOD.build_query(950, 1700)
        for accession in GENERIC:
            assert accession not in query

    def test_the_families_are_an_or(self):
        """A conjunction would require every Cas9 domain to be annotated on
        every entry, which UniProt's Pfam coverage does not support -- only
        601 of 1,490 entries carry any Cas9-specific xref at all."""
        query = MOD.build_query(950, 1700)
        assert " AND xref:pfam-" not in query
        assert " OR xref:pfam-" in query

    def test_the_name_match_is_kept_as_an_alternative(self):
        """Measured: the name query alone returns 1,479 entries against the
        families' 601, so dropping it would shrink the corpus by more than
        half."""
        assert "protein_name:cas9" in MOD.build_query(950, 1700)

    def test_the_length_bound_is_applied_server_side(self):
        assert "length:[950 TO 1700]" in MOD.build_query(950, 1700)

    def test_the_length_bound_follows_its_arguments(self):
        """It comes from filter/min_length and max_length, not a literal."""
        assert "length:[984 TO 1629]" in MOD.build_query(984, 1629)

    def test_the_length_bound_is_conjunctive(self):
        """ORing it would return every Cas9-family protein at any length,
        including fragments the filter is there to exclude."""
        query = MOD.build_query(950, 1700)
        assert "AND length:" in query

    def test_the_length_bound_matches_the_config(self):
        from thermocas.config import Config

        cfg = Config.load("data")
        query = MOD.build_query(
            int(cfg["filter/min_length"]), int(cfg["filter/max_length"])
        )
        assert f"length:[{cfg['filter/min_length']} TO {cfg['filter/max_length']}]" in query
