"""Tests for the ceiling sweep.

The sweep's output is the evidence for whether the predicted-vs-measured
experiment is worth GPU time, so a bug here spends money or declines to. Two
of its pieces are easy to get subtly wrong and impossible to notice from the
output: rebalancing, which must change the marginal without changing any
class's shape, and quantile binning, which must split residues rather than
sequences.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def load():
    path = REPO_ROOT / "scripts" / "15_ceiling_sweep.py"
    spec = importlib.util.spec_from_file_location("sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load()


def counts(*groups):
    """Build the (cluster, ogt) -> residue-count mapping the sweep consumes."""
    return {(cluster, ogt): dict(residues) for cluster, ogt, residues in groups}


class TestBinning:
    def test_below_the_edge_is_the_low_class(self):
        assert MOD.bin_of(30.0, (45.0,)) == "b0"

    def test_on_the_edge_is_the_high_class(self):
        """configs/data.yaml documents bins as upper-exclusive, so an OGT
        exactly at the edge belongs above it. 45 C is a real and common value
        in the corpus, so this boundary moves thousands of sequences."""
        assert MOD.bin_of(45.0, (45.0,)) == "b1"

    def test_multiple_edges_give_ordered_classes(self):
        edges = (30.0, 50.0)
        assert MOD.bin_of(25.0, edges) == "b0"
        assert MOD.bin_of(40.0, edges) == "b1"
        assert MOD.bin_of(60.0, edges) == "b2"


class TestRebalance:
    def test_it_equalizes_the_class_totals(self):
        out = MOD.rebalance({"a": {"A": 900, "C": 100}, "b": {"A": 50, "C": 50}})
        assert sum(out["a"].values()) == pytest.approx(sum(out["b"].values()))

    def test_it_preserves_each_class_shape(self):
        """Rebalancing may change how much mass a class has and must not
        change what that class looks like -- otherwise it is not measuring the
        same corpus with a different prior."""
        out = MOD.rebalance({"a": {"A": 900, "C": 100}, "b": {"A": 50, "C": 50}})
        assert out["a"]["A"] / sum(out["a"].values()) == pytest.approx(0.9)

    def test_balancing_identical_classes_still_carries_no_information(self):
        shape = {"A": 60, "C": 40}
        out = MOD.rebalance({"a": dict(shape), "b": {k: v * 9 for k, v in shape.items()}})
        info, _ = MOD.CEIL.mutual_information(out)
        assert info == pytest.approx(0.0, abs=1e-12)

    def test_balancing_raises_information_for_a_skewed_corpus(self):
        """The reason the balance axis is in the sweep at all: same per-class
        distributions, different prior, more information in the label."""
        skewed = {"a": {"A": 9000, "C": 1000}, "b": {"A": 100, "C": 900}}
        before, _ = MOD.CEIL.mutual_information(skewed)
        after, _ = MOD.CEIL.mutual_information(MOD.rebalance(skewed))
        assert after > before

    def test_a_single_class_is_returned_unchanged(self):
        only = {"a": {"A": 10}}
        assert MOD.rebalance(only) == only


class TestPool:
    def test_it_merges_clusters_into_classes(self):
        data = counts(
            ("c1", 30.0, {"A": 10}),
            ("c2", 30.0, {"A": 5}),
            ("c3", 60.0, {"C": 7}),
        )
        out = MOD.pool(data, (45.0,))
        assert out["b0"]["A"] == 15
        assert out["b1"]["C"] == 7


class TestQuantileEdges:
    def test_it_splits_by_residue_mass_not_sequence_count(self):
        """One long low-temperature group against many short high-temperature
        ones. Splitting by group count would put the edge in the wrong place;
        splitting by residues puts it between them."""
        data = counts(
            ("c1", 20.0, {"A": 1000}),
            ("c2", 80.0, {"A": 500}),
            ("c3", 81.0, {"A": 500}),
        )
        edges = MOD.quantile_edges(data, 2)
        assert edges == (80.0,)

    def test_two_bins_give_one_edge(self):
        data = counts(("c1", 20.0, {"A": 100}), ("c2", 80.0, {"A": 100}))
        assert len(MOD.quantile_edges(data, 2)) == 1

    def test_edges_are_ascending(self):
        data = counts(
            *[(f"c{i}", float(10 * i), {"A": 100}) for i in range(1, 11)]
        )
        edges = MOD.quantile_edges(data, 4)
        assert list(edges) == sorted(edges)


class TestWithinCluster:
    def test_a_cluster_with_one_class_is_skipped(self):
        data = counts(("c1", 30.0, {"A": 5000}), ("c1", 31.0, {"A": 5000}))
        info, n_clusters, residues = MOD.within_cluster(data, (45.0,), 2000)
        assert (n_clusters, residues) == (0, 0)
        assert info == 0.0

    def test_a_cluster_below_the_residue_floor_is_skipped(self):
        data = counts(("c1", 30.0, {"A": 100}), ("c1", 60.0, {"C": 100}))
        _, n_clusters, _ = MOD.within_cluster(data, (45.0,), 2000)
        assert n_clusters == 0

    def test_a_qualifying_cluster_is_counted(self):
        data = counts(("c1", 30.0, {"A": 3000}), ("c1", 60.0, {"C": 3000}))
        info, n_clusters, residues = MOD.within_cluster(data, (45.0,), 2000)
        assert n_clusters == 1
        assert residues == 6000
        # Disjoint residues, equal mass: the label is fully determined.
        assert info == pytest.approx(math.log(2))

    def test_unclustered_groups_are_ignored(self):
        """An empty cluster_id is not a cluster, and treating it as one would
        pool unrelated families into a single enormous fake cluster."""
        data = counts(("", 30.0, {"A": 5000}), ("", 60.0, {"C": 5000}))
        _, n_clusters, _ = MOD.within_cluster(data, (45.0,), 2000)
        assert n_clusters == 0


class TestAgainstTheMeasuredSweep:
    """The 45 C within-cluster ceiling is computed independently by
    13_conditioning_ceiling.py, which reported 0.001720 nats / +0.172%. The
    sweep reimplements the per-cluster aggregation, so the two must agree or
    one of them is wrong."""

    def test_the_sweep_reproduces_the_configured_ceiling(self):
        from thermocas.config import Config

        gate = Config.load("train").section("conditioning_gate")
        assert float(gate["ceiling_gain"]) == pytest.approx(0.00172, abs=5e-6)

    def test_the_measured_gain_exceeded_that_ceiling(self):
        """Recorded so a future change that silently lowers the ceiling below
        the measured gain does not pass unnoticed."""
        assert 0.00265 > MOD.CEIL.gain_from_information(0.00172)
