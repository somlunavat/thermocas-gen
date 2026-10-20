"""Clustering parameters and cluster-level splitting.

The guarantees tested here are what make the test set interpretable. If a
cluster can straddle train and test, every downstream number is measuring
memorization instead of generalization.
"""

from __future__ import annotations

import pytest

from thermocas.config import Config
from thermocas.data.cluster import (
    ClusterParams,
    SequenceMeta,
    SplitParams,
    assign_splits,
    mmseqs_cluster_commands,
    parse_cluster_tsv,
    verify_no_cluster_straddles_splits,
    write_split_files,
)


def split_params(**overrides) -> SplitParams:
    base = dict(
        fractions={"train": 0.8, "val": 0.1, "test": 0.1},
        holdout_subfamily=None,
        holdout_positive_controls=False,
        positive_controls=(),
        seed=20261019,
    )
    base.update(overrides)
    return SplitParams(**base)


class TestClusterParams:
    def test_loads_from_shipped_config(self):
        params = ClusterParams.from_config(Config.load("data"))
        assert params.tool == "mmseqs2"
        # The brief specifies 40-50% identity for cluster-level splitting.
        assert 0.40 <= params.min_seq_id <= 0.50

    def test_identity_given_as_percent_is_rejected(self):
        """45 instead of 0.45 would ask for 4500% identity.

        MMseqs2 would not error; it would return singleton clusters, which
        then split like individual sequences and reintroduce exactly the
        leakage clustering exists to prevent.
        """
        with pytest.raises(ValueError, match="fraction"):
            ClusterParams(tool="mmseqs2", min_seq_id=45, coverage=0.8, cov_mode=0, cluster_mode=0)

    def test_zero_identity_rejected(self):
        with pytest.raises(ValueError, match="fraction"):
            ClusterParams(tool="mmseqs2", min_seq_id=0.0, coverage=0.8, cov_mode=0, cluster_mode=0)

    def test_coverage_bounds(self):
        with pytest.raises(ValueError, match="coverage"):
            ClusterParams(tool="mmseqs2", min_seq_id=0.45, coverage=80, cov_mode=0, cluster_mode=0)

    def test_min_seq_id_passed_as_a_fraction(self, tmp_path):
        params = ClusterParams(tool="mmseqs2", min_seq_id=0.45, coverage=0.8, cov_mode=0, cluster_mode=0)
        commands = mmseqs_cluster_commands(params, tmp_path / "s.fasta", tmp_path / "out", tmp_path)
        cluster = next(c for c in commands if c[1] == "cluster")
        assert cluster[cluster.index("--min-seq-id") + 1] == "0.45"
        assert cluster[cluster.index("-c") + 1] == "0.8"

    def test_command_order(self, tmp_path):
        params = ClusterParams(tool="mmseqs2", min_seq_id=0.45, coverage=0.8, cov_mode=0, cluster_mode=0)
        commands = mmseqs_cluster_commands(params, tmp_path / "s", tmp_path / "o", tmp_path)
        assert [c[1] for c in commands] == ["createdb", "cluster", "createtsv"]


class TestParseClusterTsv:
    def test_parses_representative_and_members(self, tmp_path):
        path = tmp_path / "clu.tsv"
        path.write_text("repA\trepA\nrepA\tmem1\nrepA\tmem2\nrepB\trepB\n")
        assignments = parse_cluster_tsv(path)
        assert assignments == {"repA": "repA", "mem1": "repA", "mem2": "repA", "repB": "repB"}

    def test_conflicting_assignment_raises(self, tmp_path):
        path = tmp_path / "clu.tsv"
        path.write_text("repA\tmem1\nrepB\tmem1\n")
        with pytest.raises(ValueError, match="two clusters"):
            parse_cluster_tsv(path)

    def test_truncated_row_raises(self, tmp_path):
        path = tmp_path / "clu.tsv"
        path.write_text("repA\n")
        with pytest.raises(ValueError, match="2 tab-separated"):
            parse_cluster_tsv(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "clu.tsv"
        path.write_text("")
        with pytest.raises(ValueError, match="no cluster assignments"):
            parse_cluster_tsv(path)


class TestSplitParams:
    def test_loads_from_shipped_config(self):
        params = SplitParams.from_config(Config.load("data"))
        assert params.holdout_positive_controls is True
        assert "GeoCas9" in params.positive_controls
        assert "ThermoCas9" in params.positive_controls

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_params(fractions={"train": 0.8, "val": 0.1, "test": 0.2})

    def test_missing_fraction_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            split_params(fractions={"train": 0.9, "val": 0.1})


def make_corpus(n_clusters: int = 30, per_cluster: int = 4, thermo_every: int = 5):
    sequences = []
    for c in range(n_clusters):
        temp_bin = "thermo" if c % thermo_every == 0 else "meso"
        for m in range(per_cluster):
            sequences.append(
                SequenceMeta(seq_id=f"c{c}_m{m}", cluster_id=f"clu{c}", temp_bin=temp_bin)
            )
    return sequences


class TestAssignSplits:
    def test_no_cluster_straddles_two_splits(self):
        """The central guarantee. A straddling cluster is leakage by
        construction, so it is verified rather than assumed."""
        sequences = make_corpus()
        assignments, _ = assign_splits(sequences, split_params())
        assert verify_no_cluster_straddles_splits(sequences, assignments) == []

    def test_every_sequence_is_assigned_exactly_once(self):
        sequences = make_corpus()
        assignments, report = assign_splits(sequences, split_params())
        assert len(assignments) == len(sequences)
        assert sum(report.counts.values()) == len(sequences)

    def test_deterministic_for_a_fixed_seed(self):
        sequences = make_corpus()
        first, _ = assign_splits(sequences, split_params(seed=7))
        second, _ = assign_splits(sequences, split_params(seed=7))
        assert first == second

    def test_different_seed_gives_a_different_split(self):
        sequences = make_corpus()
        first, _ = assign_splits(sequences, split_params(seed=1))
        second, _ = assign_splits(sequences, split_params(seed=2))
        assert first != second

    def test_held_out_subfamily_lands_in_test(self):
        sequences = make_corpus()
        sequences.append(
            SequenceMeta("iic_1", "clu_iic", "thermo", subfamily="II-C")
        )
        sequences.append(SequenceMeta("iic_2", "clu_iic", "thermo", subfamily=None))
        assignments, report = assign_splits(
            sequences, split_params(holdout_subfamily="II-C")
        )
        assert assignments["iic_1"] == "test"
        # The whole cluster goes with it, not just the tagged member.
        assert assignments["iic_2"] == "test"
        assert report.forced_reasons["subfamily"] == 1

    def test_positive_controls_land_in_test_with_their_clusters(self):
        """A natural thermophilic Cas9 in training would make the central
        claim partly recall rather than generation."""
        sequences = make_corpus()
        sequences.append(SequenceMeta("geo", "clu_geo", "thermo", is_positive_control=True))
        sequences.append(SequenceMeta("geo_neighbour", "clu_geo", "thermo"))
        assignments, report = assign_splits(
            sequences,
            split_params(holdout_positive_controls=True, positive_controls=("GeoCas9",)),
        )
        assert assignments["geo"] == "test"
        assert assignments["geo_neighbour"] == "test"
        assert report.forced_reasons["positive_control"] == 1

    def test_positive_controls_not_held_out_when_disabled(self):
        sequences = [
            SequenceMeta(f"s{i}", f"clu{i}", "meso") for i in range(20)
        ]
        sequences.append(SequenceMeta("geo", "clu_geo", "thermo", is_positive_control=True))
        assignments, report = assign_splits(
            sequences, split_params(holdout_positive_controls=False)
        )
        assert report.forced_to_test == 0
        assert assignments["geo"] in {"train", "val", "test"}

    def test_rare_bin_is_represented_outside_train(self):
        """Thermophilic clusters are scarce; allocating them without
        stratifying would let train swallow them all by chance."""
        sequences = make_corpus(n_clusters=40, thermo_every=4)
        _, report = assign_splits(sequences, split_params())
        for name in ("val", "test"):
            assert report.bin_counts[name].get("thermo", 0) > 0, f"{name} has no thermophiles"

    def test_fractions_are_approximately_honoured(self):
        sequences = make_corpus(n_clusters=100, per_cluster=1)
        _, report = assign_splits(sequences, split_params())
        assert 70 <= report.counts["train"] <= 90
        assert 5 <= report.counts["val"] <= 15
        assert 5 <= report.counts["test"] <= 15

    def test_tiny_corpus_does_not_crash_or_empty_test(self):
        sequences = [SequenceMeta(f"s{i}", f"clu{i}", "meso") for i in range(3)]
        assignments, report = assign_splits(sequences, split_params())
        assert len(assignments) == 3
        assert sum(report.counts.values()) == 3

    def test_single_cluster_corpus_is_assigned_coherently(self):
        sequences = [SequenceMeta(f"s{i}", "only", "meso") for i in range(5)]
        assignments, _ = assign_splits(sequences, split_params())
        assert len(set(assignments.values())) == 1

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="no sequences"):
            assign_splits([], split_params())


class TestStraddleDetection:
    def test_detects_a_straddling_cluster(self):
        sequences = [
            SequenceMeta("a", "clu1", "meso"),
            SequenceMeta("b", "clu1", "meso"),
        ]
        bad = {"a": "train", "b": "test"}
        assert verify_no_cluster_straddles_splits(sequences, bad) == ["clu1"]

    def test_clean_assignment_reports_nothing(self):
        sequences = [SequenceMeta("a", "clu1", "meso"), SequenceMeta("b", "clu1", "meso")]
        assert verify_no_cluster_straddles_splits(sequences, {"a": "train", "b": "train"}) == []


class TestWriteSplitFiles:
    def test_writes_one_file_per_split_plus_a_combined_csv(self, tmp_path):
        assignments = {"a": "train", "b": "val", "c": "test", "d": "train"}
        written = write_split_files(tmp_path, assignments)
        assert (tmp_path / "train.txt").read_text().split() == ["a", "d"]
        assert (tmp_path / "val.txt").read_text().split() == ["b"]
        assert (tmp_path / "test.txt").read_text().split() == ["c"]
        assert "seq_id,split" in written["combined"].read_text()

    def test_ids_are_sorted_so_the_files_are_stable(self, tmp_path):
        write_split_files(tmp_path, {"z": "train", "a": "train"})
        assert (tmp_path / "train.txt").read_text().split() == ["a", "z"]
