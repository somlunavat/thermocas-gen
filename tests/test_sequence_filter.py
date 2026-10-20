"""Sequence filtering, and the accounting that keeps rejections visible."""

from __future__ import annotations

import pytest

from thermocas.config import Config
from thermocas.data.fasta import FastaRecord, read_fasta
from thermocas.data.sequence_filter import Rejection, SequenceFilter


@pytest.fixture
def filt() -> SequenceFilter:
    return SequenceFilter(
        min_length=10, max_length=20, allowed_residues=frozenset("ACDEFGHIKLMNPQRSTVWY")
    )


def test_length_bounds_are_inclusive(filt):
    assert filt.check("A" * 10) is None
    assert filt.check("A" * 20) is None
    assert filt.check("A" * 9) is Rejection.TOO_SHORT
    assert filt.check("A" * 21) is Rejection.TOO_LONG


def test_nonstandard_residues_rejected(filt):
    for bad in ("X", "B", "Z", "U", "O", "-", "*"):
        assert filt.check("A" * 15 + bad) is Rejection.NONSTANDARD_RESIDUE


def test_empty_rejected(filt):
    assert filt.check("   ") is Rejection.EMPTY


def test_min_greater_than_max_rejected_at_construction():
    cfg = Config(
        {
            "filter": {
                "min_length": 100,
                "max_length": 50,
                "allowed_residues": "ACDEFGHIKLMNPQRSTVWY",
                "drop_exact_duplicates": True,
            }
        }
    )
    with pytest.raises(ValueError, match="exceeds"):
        SequenceFilter.from_config(cfg)


class TestAccounting:
    """Every rejection must be counted. Silent drops are the bug being guarded."""

    def test_over_length_sequences_are_counted_not_silently_dropped(self, filt):
        """Regression against the prior codebase's central failure.

        There, over-length sequences vanished inside a boolean predicate, which
        is how every full-length Cas9 left the training set unnoticed.
        """
        records = [
            FastaRecord("keep", "", "A" * 15),
            FastaRecord("long1", "", "A" * 500),
            FastaRecord("long2", "", "A" * 900),
        ]
        kept, report = filt.apply(records)
        assert [r.seq_id for r in kept] == ["keep"]
        assert report.rejected[Rejection.TOO_LONG.value] == 2
        assert report.total == 3
        assert "long1" in report.examples[Rejection.TOO_LONG.value]
        assert "too_long" in report.summary()

    def test_counts_sum_to_total(self, filt):
        records = [
            FastaRecord("ok", "", "A" * 15),
            FastaRecord("short", "", "A"),
            FastaRecord("long", "", "A" * 99),
            FastaRecord("weird", "", "X" * 15),
        ]
        kept, report = filt.apply(records)
        assert report.kept == len(kept) == 1
        assert report.total == 4
        assert sum(report.rejected.values()) == 3

    def test_exact_duplicates_dropped_keeping_first(self, filt):
        records = [
            FastaRecord("first", "", "ACDEFGHIKL"),
            FastaRecord("second", "", "ACDEFGHIKL"),
            FastaRecord("other", "", "MNPQRSTVWY"),
        ]
        kept, report = filt.apply(records)
        assert [r.seq_id for r in kept] == ["first", "other"]
        assert report.rejected[Rejection.DUPLICATE_SEQUENCE.value] == 1
        # The example names both ids so the discarded duplicate is traceable.
        assert "second" in report.examples[Rejection.DUPLICATE_SEQUENCE.value][0]
        assert "first" in report.examples[Rejection.DUPLICATE_SEQUENCE.value][0]

    def test_duplicates_kept_when_disabled(self):
        filt = SequenceFilter(
            min_length=1,
            max_length=99,
            allowed_residues=frozenset("ACDEFGHIKLMNPQRSTVWY"),
            drop_exact_duplicates=False,
        )
        kept, report = filt.apply(
            [FastaRecord("a", "", "ACD"), FastaRecord("b", "", "ACD")]
        )
        assert len(kept) == 2
        assert not report.rejected

    def test_length_range_recorded(self, filt):
        kept, report = filt.apply(
            [FastaRecord("a", "", "A" * 12), FastaRecord("b", "", "A" * 18)]
        )
        assert (report.length_min, report.length_max) == (12, 18)

    def test_examples_are_capped(self, filt):
        records = [FastaRecord(f"s{i}", "", "A") for i in range(50)]
        _, report = filt.apply(records)
        assert report.rejected[Rejection.TOO_SHORT.value] == 50
        assert len(report.examples[Rejection.TOO_SHORT.value]) == 5

    def test_sequences_are_normalized_on_the_way_out(self, filt):
        kept, _ = filt.apply([FastaRecord("a", "", "  acdefghikl  ")])
        assert kept[0].sequence == "ACDEFGHIKL"


class TestShippedSeeds:
    """The committed reference seeds must survive the pipeline's own filter."""

    def test_every_seed_passes_the_configured_filter(self):
        cfg = Config.load("data")
        seeds = list(read_fasta(cfg.resolve_path("paths/seeds")))
        assert len(seeds) == 8, "expected 8 reference seeds"
        kept, report = SequenceFilter.from_config(cfg).apply(seeds)
        assert report.kept == len(seeds), report.summary()

    def test_seed_lengths_match_published_values(self):
        """Guards against a resolver silently picking a fragment or a paralog."""
        expected = {
            "Q99ZW2": 1368,  # SpCas9
            "J7RUA5": 1053,  # SaCas9
            "WP_099233044.1": 1082,  # ThermoCas9
            "C9X1G5": 1082,  # NmeCas9
            "Q0P897": 984,  # CjCas9
            "G3ECR1": 1409,  # St1Cas9
            "A0Q5Y3": 1629,  # FnCas9, type II-B and the longest seed
        }
        seeds = {r.seq_id: len(r) for r in read_fasta(Config.load("data").resolve_path("paths/seeds"))}
        for accession, length in expected.items():
            assert seeds[accession] == length, f"{accession} length changed"

    def test_filter_bounds_admit_the_longest_and_shortest_seed(self):
        """FnCas9 at 1,629 aa and CjCas9 at 984 aa set the real bounds.

        The brief estimated 1,000-1,400 aa; the verified seeds are wider than
        that at both ends, so the config must not be tightened back.
        """
        cfg = Config.load("data")
        assert cfg["filter/min_length"] <= 984
        assert cfg["filter/max_length"] >= 1629
