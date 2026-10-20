"""The leakage gate.

A leak invalidates every downstream result silently, so these tests focus on
the ways a check like this can appear to pass while being wrong: wrong units,
short high-identity motifs mistaken for duplicates, and ids shared outright
between splits.
"""

from __future__ import annotations

import pytest

from thermocas.config import Config
from thermocas.data.leakage import (
    LeakageParams,
    SequencePair,
    check_leakage,
    check_split_disjointness,
    mmseqs_leakage_commands,
    parse_alignment_tsv,
)


def params(**overrides) -> LeakageParams:
    base = dict(max_identity_to_train=0.45, alignment_coverage=0.80, fail_on_violation=True)
    base.update(overrides)
    return LeakageParams(**base)


class TestParams:
    def test_loads_from_shipped_config(self):
        loaded = LeakageParams.from_config(Config.load("data"))
        assert loaded.max_identity_to_train == 0.45
        assert loaded.fail_on_violation is True

    def test_identity_as_percent_is_rejected(self):
        """45 instead of 0.45 would make the gate unreachable.

        Identity is never above 1.0 as a fraction, so a threshold of 45 would
        pass everything and report no leakage however bad the split was.
        """
        with pytest.raises(ValueError, match="fraction"):
            params(max_identity_to_train=45)

    def test_coverage_as_percent_is_rejected(self):
        with pytest.raises(ValueError, match="fraction"):
            params(alignment_coverage=80)


class TestViolation:
    def test_high_identity_over_high_coverage_violates(self):
        pair = SequencePair("test1", "train1", identity=0.95, coverage=0.98)
        assert pair.violates(params())

    def test_short_conserved_motif_does_not_violate(self):
        """A 99%-identical 40-residue stretch between two 1,300 aa Cas9s is a
        conserved catalytic motif. Filtering on identity alone would flag the
        whole corpus as leaked."""
        pair = SequencePair("test1", "train1", identity=0.99, coverage=0.03)
        assert not pair.violates(params())

    def test_low_identity_over_full_coverage_does_not_violate(self):
        """Distant homologs aligning end to end are the corpus, not a leak."""
        pair = SequencePair("test1", "train1", identity=0.32, coverage=0.95)
        assert not pair.violates(params())

    def test_threshold_is_exclusive_on_identity(self):
        at_threshold = SequencePair("t", "r", identity=0.45, coverage=0.9)
        just_over = SequencePair("t", "r", identity=0.4501, coverage=0.9)
        assert not at_threshold.violates(params())
        assert just_over.violates(params())

    def test_coverage_threshold_is_inclusive(self):
        pair = SequencePair("t", "r", identity=0.9, coverage=0.80)
        assert pair.violates(params())


class TestCheckLeakage:
    def test_clean_split_passes(self):
        pairs = [
            SequencePair("test1", "train1", 0.31, 0.92),
            SequencePair("test2", "train2", 0.28, 0.88),
        ]
        report = check_leakage(pairs, params())
        assert report.passed
        assert report.violations == []
        assert "no violations" in report.summary(params())

    def test_leak_is_detected_and_reported(self):
        pairs = [
            SequencePair("test1", "train1", 0.31, 0.92),
            SequencePair("test2", "train9", 0.97, 0.99),
        ]
        report = check_leakage(pairs, params())
        assert not report.passed
        assert report.leaked_queries == ["test2"]
        assert "VIOLATIONS" in report.summary(params())
        assert "train9" in report.summary(params())

    def test_closest_pair_is_reported_even_when_clean(self):
        """Useful signal: a clean run at 44% identity is near the threshold."""
        pairs = [
            SequencePair("test1", "train1", 0.20, 0.9),
            SequencePair("test1", "train2", 0.44, 0.9),
        ]
        report = check_leakage(pairs, params())
        assert report.passed
        assert report.worst_by_query["test1"].identity == 0.44
        assert "44.0%" in report.summary(params())

    def test_self_matches_are_ignored_not_counted_as_leakage(self):
        pairs = [SequencePair("same", "same", 1.0, 1.0)]
        report = check_leakage(pairs, params())
        assert report.passed
        assert report.self_matches_ignored == 1
        assert report.pairs_examined == 0

    def test_multiple_leaks_for_one_query_all_recorded(self):
        pairs = [
            SequencePair("test1", "train1", 0.90, 0.95),
            SequencePair("test1", "train2", 0.85, 0.95),
        ]
        report = check_leakage(pairs, params())
        assert len(report.violations) == 2
        assert report.leaked_queries == ["test1"]

    def test_no_pairs_passes_vacuously(self):
        report = check_leakage([], params())
        assert report.passed
        assert report.pairs_examined == 0


class TestParseAlignmentTsv:
    def test_parses_fractional_values(self, tmp_path):
        path = tmp_path / "aln.tsv"
        path.write_text("q1\tt1\t0.412\t0.88\t0.91\t1200\t1e-120\t800\n")
        (pair,) = list(parse_alignment_tsv(path))
        assert pair.query_id == "q1"
        assert pair.identity == pytest.approx(0.412)
        assert pair.coverage == pytest.approx(0.88)

    def test_percentages_are_normalized(self, tmp_path):
        """Guards the dangerous direction: a 90% identity read as 0.9% would
        report no leakage on a badly leaked split."""
        path = tmp_path / "aln.tsv"
        path.write_text("q1\tt1\t90.0\t95.0\t95.0\t1200\t1e-120\t800\n")
        (pair,) = list(parse_alignment_tsv(path))
        assert pair.identity == pytest.approx(0.90)
        assert pair.coverage == pytest.approx(0.95)
        assert pair.violates(params())

    def test_identity_of_exactly_one_stays_full(self, tmp_path):
        path = tmp_path / "aln.tsv"
        path.write_text("q1\tt1\t1.0\t1.0\t1.0\t1200\t0\t900\n")
        (pair,) = list(parse_alignment_tsv(path))
        assert pair.identity == 1.0

    def test_truncated_row_raises(self, tmp_path):
        path = tmp_path / "aln.tsv"
        path.write_text("q1\tt1\n")
        with pytest.raises(ValueError, match="at least 4 columns"):
            list(parse_alignment_tsv(path))

    def test_malformed_number_reports_the_line(self, tmp_path):
        path = tmp_path / "aln.tsv"
        path.write_text("q1\tt1\tNA\t0.9\n")
        with pytest.raises(ValueError, match="aln.tsv:1"):
            list(parse_alignment_tsv(path))


class TestSplitDisjointness:
    def test_disjoint_splits_report_nothing(self):
        splits = {"train": {"a", "b"}, "val": {"c"}, "test": {"d"}}
        assert check_split_disjointness(splits) == []

    def test_shared_id_is_reported(self):
        """An id in two splits is leakage no identity threshold would catch:
        the sequences are not similar, they are identical."""
        splits = {"train": {"a", "b"}, "val": {"c"}, "test": {"b", "d"}}
        problems = check_split_disjointness(splits)
        assert len(problems) == 1
        assert "train" in problems[0] and "test" in problems[0]
        assert "b" in problems[0]

    def test_empty_splits_are_fine(self):
        assert check_split_disjointness({"train": set(), "test": set()}) == []


class TestMmseqsCommands:
    def test_pipeline_order(self, tmp_path):
        commands = mmseqs_leakage_commands(
            tmp_path / "q.fasta", tmp_path / "t.fasta", tmp_path / "o.tsv", tmp_path
        )
        assert [c[1] for c in commands] == ["createdb", "createdb", "search", "convertalis"]

    def test_search_is_sensitive_by_default(self, tmp_path):
        """A missed alignment reads as 'no leakage', the dangerous direction."""
        commands = mmseqs_leakage_commands(
            tmp_path / "q", tmp_path / "t", tmp_path / "o", tmp_path
        )
        search = next(c for c in commands if c[1] == "search")
        assert float(search[search.index("-s") + 1]) >= 7.0

    def test_output_includes_identity_and_coverage(self, tmp_path):
        commands = mmseqs_leakage_commands(
            tmp_path / "q", tmp_path / "t", tmp_path / "o", tmp_path
        )
        convert = next(c for c in commands if c[1] == "convertalis")
        fields = convert[convert.index("--format-output") + 1]
        assert "fident" in fields
        assert "qcov" in fields
