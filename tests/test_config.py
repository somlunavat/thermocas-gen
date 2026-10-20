"""Config loading and growth-temperature binning."""

from __future__ import annotations

import math

import pytest
import yaml

from thermocas.config import (
    Config,
    ConfigError,
    GrowthTemperatureBin,
    GrowthTemperatureBinner,
)


def test_shipped_configs_parse():
    for name in ("data", "model", "train", "generate", "eval"):
        cfg = Config.load(name)
        assert isinstance(cfg.as_dict(), dict)


def test_missing_key_raises_rather_than_defaulting():
    cfg = Config({"a": {"b": 1}})
    assert cfg["a/b"] == 1
    with pytest.raises(ConfigError):
        cfg["a/missing"]
    with pytest.raises(ConfigError):
        cfg["nope/at/all"]


def test_contains_does_not_raise():
    cfg = Config({"a": {"b": 1}})
    assert "a/b" in cfg
    assert "a/c" not in cfg


def test_section_rejects_non_mapping():
    cfg = Config({"a": [1, 2]})
    with pytest.raises(ConfigError):
        cfg.section("a")


def test_as_dict_is_a_copy():
    cfg = Config({"a": {"b": 1}})
    snapshot = cfg.as_dict()
    snapshot["a"]["b"] = 99
    assert cfg["a/b"] == 1


class TestGrowthTemperatureBinning:
    """The shipped bins must tile the real line with no gaps."""

    def test_shipped_bins_cover_every_plausible_ogt(self):
        """Two bins: the hyper class was merged into thermo.

        Gosha holds only 171 cross-checked hyperthermophilic species, so a
        separate hyper class could never have reached min_per_bin and Gate 1
        would have failed on the threshold rather than on the science.
        """
        binner = GrowthTemperatureBinner.from_config(Config.load("data"))
        assert binner.names == ["meso", "thermo"]
        # Boundaries are half-open upward, so 45 is thermo and not meso.
        assert binner.bin_for(4) == "meso"
        assert binner.bin_for(37) == "meso"
        assert binner.bin_for(44.9) == "meso"
        assert binner.bin_for(45) == "thermo"
        assert binner.bin_for(69.9) == "thermo"
        # Former hyper range now falls in thermo.
        assert binner.bin_for(70) == "thermo"
        assert binner.bin_for(103) == "thermo"

    def test_unbounded_edges_parsed_as_floats_not_strings(self):
        """Regression: PyYAML parses `Infinity` as a str, `.inf` as a float.

        A string edge makes every comparison raise or silently mis-bin, so the
        shipped config must use `.inf`.
        """
        raw = yaml.safe_load(open("configs/data.yaml"))
        edges = raw["growth_temperature"]["bins"]
        assert math.isinf(float(edges[0]["min"]))
        assert math.isinf(float(edges[-1]["max"]))
        for edge in edges:
            assert isinstance(edge["min"], (int, float))
            assert isinstance(edge["max"], (int, float))

    def test_string_edges_are_coerced_not_compared_as_strings(self):
        """`from_config` coerces every edge through float() before comparing.

        Without the coercion a YAML-quoted edge stays a str, and `str <= int`
        raises TypeError deep inside binning rather than at load time.
        float("Infinity") happens to parse, so coercion makes this case work
        instead of exploding.
        """
        cfg = Config(
            {
                "growth_temperature": {
                    "bins": [
                        {"name": "low", "min": "-Infinity", "max": "45"},
                        {"name": "high", "min": "45", "max": "Infinity"},
                    ]
                }
            }
        )
        binner = GrowthTemperatureBinner.from_config(cfg)
        assert all(isinstance(b.min_c, float) for b in binner.bins)
        assert binner.bin_for(20) == "low"
        assert binner.bin_for(80) == "high"

    def test_non_numeric_edge_is_rejected_loudly(self):
        cfg = Config({"growth_temperature": {"bins": [{"name": "x", "min": "hot", "max": 1}]}})
        with pytest.raises(ConfigError, match="malformed"):
            GrowthTemperatureBinner.from_config(cfg)

    def test_bin_missing_a_name_is_rejected(self):
        cfg = Config({"growth_temperature": {"bins": [{"min": 0, "max": 1}]}})
        with pytest.raises(ConfigError, match="malformed"):
            GrowthTemperatureBinner.from_config(cfg)

    def test_gap_between_bins_is_rejected(self):
        with pytest.raises(ConfigError, match="gaps"):
            GrowthTemperatureBinner(
                [
                    GrowthTemperatureBin("low", -math.inf, 45),
                    GrowthTemperatureBin("high", 50, math.inf),  # 45-50 unassigned
                ]
            )

    def test_overlapping_bins_are_rejected(self):
        with pytest.raises(ConfigError):
            GrowthTemperatureBinner(
                [
                    GrowthTemperatureBin("low", -math.inf, 50),
                    GrowthTemperatureBin("high", 45, math.inf),
                ]
            )

    def test_nan_ogt_is_rejected(self):
        binner = GrowthTemperatureBinner.from_config(Config.load("data"))
        with pytest.raises(ValueError, match="NaN"):
            binner.bin_for(float("nan"))

    def test_empty_bin_list_is_rejected(self):
        with pytest.raises(ConfigError):
            GrowthTemperatureBinner([])


class TestContextLengthCoversCas9:
    """The prior repository's central bug was a context too short for Cas9.

    It capped max_length at 512 and silently dropped every full-length Cas9.
    These tests make the same class of mistake impossible to reintroduce,
    including by swapping in a pretrained checkpoint with a shorter context.
    """

    def required_tokens(self) -> int:
        # Residues plus BOS, the growth-temperature control token, and EOS.
        return Config.load("data")["filter/max_length"] + 3

    def test_pretrained_context_covers_the_longest_admissible_sequence(self):
        model = Config.load("model")
        for name in ("conditioned_lm", "unconditioned_lm"):
            assert model[f"{name}/required_context"] >= self.required_tokens(), name

    def test_scratch_baseline_context_covers_it_too(self):
        model = Config.load("model")
        assert model["scratch_transformer/max_length"] >= self.required_tokens()

    def test_training_max_length_covers_it(self):
        assert Config.load("train")["data/max_length"] >= self.required_tokens()

    def test_checkpoint_is_one_with_a_long_enough_context(self):
        """progen2-small and progen2-large both declare n_positions=1024,
        which cannot hold SpCas9 at 1,368 residues. Only progen2-base, at
        2048, can. The size ordering is counter-intuitive, so the choice is
        pinned here rather than left to whoever edits the config next.
        """
        model = Config.load("model")
        for name in ("conditioned_lm", "unconditioned_lm"):
            checkpoint = model[f"{name}/checkpoint"]
            assert not checkpoint.endswith(("progen2-small", "progen2-large")), (
                f"{name} uses {checkpoint}, whose context is 1024 tokens and "
                "cannot fit a full-length Cas9"
            )

    def test_ablation_uses_the_same_checkpoint_as_the_main_model(self):
        """Otherwise the ablation measures the checkpoint, not the token."""
        model = Config.load("model")
        assert model["conditioned_lm/checkpoint"] == model["unconditioned_lm/checkpoint"]

    def test_sequences_that_must_fit_actually_fit(self):
        """Checked against the real committed seeds, not against a constant."""
        from thermocas.data.fasta import read_fasta

        data = Config.load("data")
        budget = Config.load("model")["conditioned_lm/required_context"]
        for record in read_fasta(data.resolve_path("paths/seeds")):
            if len(record) <= data["filter/max_length"]:
                assert len(record) + 3 <= budget, f"{record.seq_id} ({len(record)} aa) does not fit"
