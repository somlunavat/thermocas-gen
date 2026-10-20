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
        binner = GrowthTemperatureBinner.from_config(Config.load("data"))
        assert binner.names == ["meso", "thermo", "hyper"]
        # Boundaries are half-open upward, so 45 is thermo and not meso.
        assert binner.bin_for(4) == "meso"
        assert binner.bin_for(37) == "meso"
        assert binner.bin_for(44.9) == "meso"
        assert binner.bin_for(45) == "thermo"
        assert binner.bin_for(69.9) == "thermo"
        assert binner.bin_for(70) == "hyper"
        assert binner.bin_for(103) == "hyper"

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
