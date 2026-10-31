"""Tests for training stage A on a corpus other than the configured one.

``11_train.py --corpus external_balanced`` exists so the measured-gain
experiment can be repeated on a corpus built from a different
growth-temperature database. Two things can go wrong silently and both cost a
GPU run to discover:

A corpus config can be missing a key stage A needs. ``paths/screened_sequences``
is the dangerous one -- without it training reads the unscreened corpus, and
the difference between those two files is exactly the Cas9 contamination that
would void the stage B leakage gate while leaving it passing.

A split can be degenerate in a held-out set. The Cas9 corpus failed this way:
its test split held zero thermophilic sequences, so shuffling the
growth-temperature token across it is a no-op and the measured gain is ~0 by
construction, whatever the model learned. That is not a number that can be
reported, and it is invisible until Gate 2 prints it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from thermocas.config import Config
from thermocas.data.cluster import ClusterParams, SplitParams

REPO_ROOT = Path(__file__).resolve().parent.parent

# Trained on by the cross-corpus experiment. configs/external.yaml is the
# uncapped counterpart and is measurement-only, so it is held to the config
# contract but not to the trainability one.
TRAINABLE = ("pretrain", "external_balanced")
MEASUREMENT_ONLY = ("external",)


class TestConfigContract:
    """Every key stage A reads, checked without a GPU."""

    @pytest.mark.parametrize("name", TRAINABLE)
    def test_it_names_a_screened_corpus(self, name):
        """Stage A must read the screened file. A config without this key
        falls back to the raw corpus, which is the one failure here that
        passes every downstream gate while invalidating the experiment."""
        cfg = Config.load(name)
        assert "paths/screened_sequences" in cfg

    @pytest.mark.parametrize("name", TRAINABLE)
    def test_the_cas9_screen_is_enabled(self, name):
        cfg = Config.load(name)
        assert cfg["exclude_cas9/enabled"] is True

    @pytest.mark.parametrize("name", TRAINABLE)
    def test_the_screen_is_stricter_than_the_stage_b_leakage_gate(self, name):
        """A pre-training corpus is large, so a borderline homolog is both
        likelier to be present and harder to notice than in the 1,019-sequence
        Cas9 corpus. 09_screen_broad.py documents this; here it is asserted."""
        cfg = Config.load(name)
        stage_b = Config.load("data")
        assert float(cfg["exclude_cas9/max_identity_to_heldout"]) < float(
            stage_b["leakage/max_identity_to_train"]
        )

    @pytest.mark.parametrize("name", TRAINABLE)
    def test_it_has_a_stage_a_gate(self, name):
        cfg = Config.load(name)
        assert int(cfg["gate/min_clusters_per_bin"]) > 0
        assert int(cfg["gate/min_sequences_per_bin"]) > 0

    @pytest.mark.parametrize("name", TRAINABLE + MEASUREMENT_ONLY)
    def test_split_params_parse(self, name):
        SplitParams.from_config(Config.load(name))

    @pytest.mark.parametrize("name", TRAINABLE + MEASUREMENT_ONLY)
    def test_it_has_a_splits_dir(self, name):
        assert "paths/splits_dir" in Config.load(name)

    def test_the_external_corpus_clusters_like_stage_a(self):
        """The within-cluster ceiling column is only comparable across the two
        corpora if they were clustered at the same thresholds."""
        a = ClusterParams.from_config(Config.load("pretrain"))
        b = ClusterParams.from_config(Config.load("external_balanced"))
        assert (a.min_seq_id, a.coverage) == (b.min_seq_id, b.coverage)


def split_bins(name: str) -> dict[str, dict[str, int]]:
    """Per-split bin counts, read from the committed label table."""
    table = Config.load(name).resolve_path("paths/label_table")
    counts: dict[str, dict[str, int]] = {}
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if not split:
                continue
            counts.setdefault(split, {})
            bin_name = row["temp_bin"]
            counts[split][bin_name] = counts[split].get(bin_name, 0) + 1
    return counts


class TestSplitsAreMeasurable:
    """Skipped where the label table is absent: it is gitignored data, so a
    fresh clone has the configs and the split files but not this."""

    @pytest.mark.parametrize("name", TRAINABLE + MEASUREMENT_ONLY)
    def test_every_split_carries_both_bins(self, name):
        cfg = Config.load(name)
        if not cfg.resolve_path("paths/label_table").is_file():
            pytest.skip(f"{name} label table not built locally")
        counts = split_bins(name)
        assert set(counts) == {"train", "val", "test"}
        for split, bins in sorted(counts.items()):
            missing = {"meso", "thermo"} - set(bins)
            assert not missing, (
                f"{name} {split} split has no {sorted(missing)} sequence. "
                "Gate 2 shuffles the growth-temperature token across the "
                "held-out set, so a split with one bin reports ~0 gain by "
                "construction -- this is how the Cas9 corpus failed."
            )

    @pytest.mark.parametrize("name", TRAINABLE)
    def test_the_held_out_bins_are_not_token_sized(self, name):
        """Both bins present is not enough; a handful of sequences gives a
        gain estimate dominated by noise. 200 is the Gate 1 floor."""
        cfg = Config.load(name)
        if not cfg.resolve_path("paths/label_table").is_file():
            pytest.skip(f"{name} label table not built locally")
        counts = split_bins(name)
        for split in ("val", "test"):
            for bin_name, n in sorted(counts[split].items()):
                assert n >= 200, f"{name} {split} has only {n} {bin_name}"


class TestSecondBackbone:
    """conditioned_lm_small is the cross-backbone check. It has a 1,024-token
    context, which is the configuration that shipped the prior repository's
    central bug, so the guards that keep it to stage A are asserted here."""

    def test_it_is_a_different_checkpoint(self):
        model = Config.load("model")
        assert (
            model["conditioned_lm_small/checkpoint"]
            != model["conditioned_lm/checkpoint"]
        )

    def test_it_conditions_the_same_way_as_the_primary_model(self):
        """A backbone comparison has to vary the backbone and nothing else.
        If the conditioning differed, the comparison would measure that."""
        model = Config.load("model")
        for key in ("use_growth_temp_token", "growth_temp_token_dropout",
                    "lora/r", "lora/alpha", "lora/dropout", "ogt_head/enabled"):
            assert (
                model[f"conditioned_lm_small/{key}"]
                == model[f"conditioned_lm/{key}"]
            ), key

    def test_it_cannot_hold_a_full_length_cas9(self):
        """Not a defect -- it is why this backbone is stage A only. Recorded
        so that nobody promotes it to stage B on the grounds that it trained
        fine on stage A, where sequences cap at 1,024 residues."""
        model = Config.load("model")
        data = Config.load("data")
        longest_cas9 = int(data["filter/max_length"]) + 3
        assert int(model["conditioned_lm_small/required_context"]) < longest_cas9

    def test_the_token_budget_cap_binds_for_it_and_not_for_the_primary(self):
        """The dataset budget is min(configured, this backbone's context).
        If the cap did not bind, over-long sequences would reach a model that
        cannot index them; if it bound for the primary model, every run so far
        would have been silently truncated."""
        model = Config.load("model")
        configured = int(Config.load("train")["data/max_length"])
        assert int(model["conditioned_lm_small/required_context"]) < configured
        assert int(model["conditioned_lm/required_context"]) >= configured
