"""Tests for the ceiling sweep's relabeling path.

Every sweep variant is a relabeling: same sequences, same clusters, same
splits, different meaning for the control token. If the relabeling is wrong
the five runs still complete and still report gains, and the comparison
between them is silently meaningless -- there is no symptom to notice. So the
properties that make the comparison valid are asserted directly.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def load():
    path = REPO_ROOT / "scripts" / "11_train.py"
    spec = importlib.util.spec_from_file_location("train_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load()


class TestBinnerFromEdges:
    def test_one_edge_gives_the_configured_names(self):
        """Two classes keep meso/thermo so a threshold variant is directly
        comparable with every number already in PROJECT_LOG.md."""
        assert MOD.binner_from_edges((45.0,)).names == ["meso", "thermo"]

    def test_more_edges_give_positional_names(self):
        assert MOD.binner_from_edges((32.0, 45.0, 56.0)).names == [
            "t0", "t1", "t2", "t3",
        ]

    def test_the_bins_tile_the_real_line(self):
        """GrowthTemperatureBinner rejects gaps at construction, so reaching
        this means the ends are infinite and nothing can fall outside."""
        binner = MOD.binner_from_edges((45.0,))
        assert binner.bins[0].min_c == -math.inf
        assert binner.bins[-1].max_c == math.inf

    def test_the_edge_is_upper_exclusive(self):
        """An OGT exactly at the edge is thermophilic, matching
        configs/data.yaml. 45 C is a common value in the corpus -- 14,054
        rows -- so this boundary is not academic."""
        binner = MOD.binner_from_edges((45.0,))
        assert binner.bin_for(44.9) == "meso"
        assert binner.bin_for(45.0) == "thermo"

    def test_it_reproduces_the_configured_binner(self):
        from thermocas.config import Config, GrowthTemperatureBinner

        configured = GrowthTemperatureBinner.from_config(Config.load("data"))
        derived = MOD.binner_from_edges((45.0,))
        assert derived.names == configured.names
        for ogt in (15.0, 37.0, 44.0, 45.0, 60.0, 103.0):
            assert derived.bin_for(ogt) == configured.bin_for(ogt)

    def test_four_classes_split_where_asked(self):
        binner = MOD.binner_from_edges((32.0, 45.0, 56.0))
        assert binner.bin_for(30.0) == "t0"
        assert binner.bin_for(37.0) == "t1"
        assert binner.bin_for(50.0) == "t2"
        assert binner.bin_for(70.0) == "t3"


def rows(*ogts):
    return [{"seq_id": f"s{i}", "ogt_c": str(v)} for i, v in enumerate(ogts)]


class TestRebinLabels:
    def test_it_rewrites_the_bin_from_the_temperature(self):
        labels = rows(30.0, 60.0)
        MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        assert [r["temp_bin"] for r in labels] == ["meso", "thermo"]

    def test_moving_the_edge_moves_the_label(self):
        """The whole experiment: identical rows, different threshold,
        different label."""
        labels = rows(50.0)
        MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        assert labels[0]["temp_bin"] == "thermo"
        MOD.rebin_labels(labels, MOD.binner_from_edges((55.0,)))
        assert labels[0]["temp_bin"] == "meso"

    def test_it_overwrites_an_existing_label(self):
        """The corpus ships with temp_bin already filled in at 45 C, so a
        variant that only filled blanks would silently train on the
        configured bins while reporting someone else's ceiling."""
        labels = [{"seq_id": "s0", "ogt_c": "50.0", "temp_bin": "meso"}]
        MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        assert labels[0]["temp_bin"] == "thermo"

    def test_it_returns_the_counts(self):
        labels = rows(30.0, 31.0, 60.0)
        counts = MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        assert counts == {"meso": 2, "thermo": 1}

    def test_an_unreadable_temperature_is_emptied_not_guessed(self):
        """The dataset counts an empty temp_bin as dropped_no_label. Guessing
        a bin here would put a sequence in a class on no evidence."""
        labels = [{"seq_id": "s0", "ogt_c": ""}, {"seq_id": "s1", "ogt_c": "oops"}]
        counts = MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        assert [r["temp_bin"] for r in labels] == ["", ""]
        assert counts == {"(unlabeled)": 2}

    def test_a_missing_column_is_emptied_not_guessed(self):
        labels = [{"seq_id": "s0"}]
        MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        assert labels[0]["temp_bin"] == ""

    def test_it_touches_nothing_but_the_label(self):
        """What makes the variants comparable: the split and the cluster are
        what hold the corpus fixed across runs."""
        labels = [
            {"seq_id": "s0", "ogt_c": "60.0", "split": "train", "cluster_id": "c1"}
        ]
        MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        assert labels[0]["split"] == "train"
        assert labels[0]["cluster_id"] == "c1"
        assert labels[0]["seq_id"] == "s0"

    def test_every_row_gets_a_decision(self):
        labels = rows(*[float(t) for t in range(15, 104)])
        counts = MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        assert sum(counts.values()) == len(labels)
        assert all(r["temp_bin"] for r in labels)

    def test_the_counts_match_the_corpus_at_the_configured_edge(self):
        """Against the measured corpus: 41.4% of rows are at or above 45 C."""
        import csv

        table = REPO_ROOT / "data" / "processed" / "broad_label_table.csv"
        if not table.exists():
            pytest.skip("stage A corpus not present")
        with table.open(newline="") as handle:
            labels = list(csv.DictReader(handle))
        counts = MOD.rebin_labels(labels, MOD.binner_from_edges((45.0,)))
        hot = counts["thermo"] / sum(counts.values())
        assert hot == pytest.approx(0.414, abs=0.005)
