"""Tests for the conditioning ceiling calculation.

This number decides how a failed Gate 2 is read -- a measured gain near the
ceiling means the conditioning worked and the threshold was wrong, which is
the opposite reading from "the model ignored its token". So the arithmetic is
checked against cases whose answers are known independently rather than
trusted because it looked plausible.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def load():
    path = REPO_ROOT / "scripts" / "13_conditioning_ceiling.py"
    spec = importlib.util.spec_from_file_location("ceiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load()


class TestEntropy:
    def test_a_certain_outcome_has_no_entropy(self):
        assert MOD.entropy({"A": 10}) == 0.0

    def test_a_fair_coin_is_one_bit(self):
        assert MOD.entropy({"A": 5, "B": 5}) == pytest.approx(math.log(2))

    def test_uniform_over_twenty_is_log_twenty(self):
        assert MOD.entropy(dict.fromkeys(MOD.RESIDUES, 1)) == pytest.approx(
            math.log(20)
        )

    def test_empty_is_zero_rather_than_an_error(self):
        assert MOD.entropy({}) == 0.0

    def test_zero_counts_do_not_contribute(self):
        assert MOD.entropy({"A": 5, "B": 5, "C": 0}) == pytest.approx(math.log(2))


class TestMutualInformation:
    def test_identical_distributions_carry_no_information(self):
        """The case that matters most: if the bins look the same, the token is
        worth nothing, and a nonzero answer here would inflate the ceiling."""
        counts = {"A": 50, "B": 30, "C": 20}
        info, _ = MOD.mutual_information({"meso": dict(counts), "thermo": dict(counts)})
        assert info == pytest.approx(0.0, abs=1e-12)

    def test_disjoint_distributions_carry_the_bin_entropy(self):
        """Perfectly separated residues: knowing the residue determines the
        bin, so the information equals the entropy of the bin prior."""
        info, _ = MOD.mutual_information({"meso": {"A": 50}, "thermo": {"B": 50}})
        assert info == pytest.approx(math.log(2))

    def test_unbalanced_disjoint_matches_the_prior_entropy(self):
        info, _ = MOD.mutual_information({"meso": {"A": 75}, "thermo": {"B": 25}})
        expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        assert info == pytest.approx(expected)

    def test_a_single_bin_carries_no_information(self):
        info, _ = MOD.mutual_information({"meso": {"A": 5, "B": 5}})
        assert info == 0.0

    def test_information_is_never_negative(self):
        info, _ = MOD.mutual_information(
            {"meso": {"A": 52, "B": 48}, "thermo": {"A": 48, "B": 52}}
        )
        assert info >= 0.0

    def test_a_small_shift_gives_the_quadratic_approximation(self):
        """For a small perturbation d, I ~= d^2/(2 p (1-p)) nats. This is the
        regime the corpus is in, so it is the regime worth pinning down."""
        info, _ = MOD.mutual_information(
            {"meso": {"A": 5100, "B": 4900}, "thermo": {"A": 4900, "B": 5100}}
        )
        d = 0.01
        assert info == pytest.approx(d * d / (2 * 0.5 * 0.5), rel=0.05)

    def test_it_reports_the_residue_count_it_used(self):
        _, n = MOD.mutual_information({"meso": {"A": 10}, "thermo": {"B": 7}})
        assert n == 17


class TestGainFromInformation:
    def test_no_information_is_no_gain(self):
        assert MOD.gain_from_information(0.0) == 0.0

    def test_small_information_is_approximately_itself(self):
        """Why the ceiling can be quoted in percent directly."""
        assert MOD.gain_from_information(0.00172) == pytest.approx(0.00172, rel=1e-3)

    def test_the_gain_is_bounded_by_one(self):
        """1 - exp(-I) < 1 mathematically, but exp(-50) underflows the gap, so
        the representable assertion is <=."""
        assert MOD.gain_from_information(50.0) <= 1.0
        assert MOD.gain_from_information(5.0) < 1.0

    def test_it_matches_the_perplexity_ratio_definition(self):
        """gain = (ppl_shuffled - ppl_true) / ppl_shuffled, with
        ppl = exp(cross entropy) and the two entropies differing by I."""
        info = 0.05
        ppl_true, ppl_shuffled = math.exp(2.0), math.exp(2.0 + info)
        direct = (ppl_shuffled - ppl_true) / ppl_shuffled
        assert MOD.gain_from_information(info) == pytest.approx(direct)


class TestAgainstTheMeasuredRun:
    """The Stage A run of 2026-10-26 measured +0.264% on validation against a
    computed unigram ceiling of +0.172%. Exceeding the unigram ceiling is the
    expected behaviour of a language model, which also has context, and is the
    evidence that the token was being used rather than ignored."""

    def test_the_measured_validation_gain_is_what_the_perplexities_say(self):
        true, shuffled = 3.4373, 3.4464
        assert (shuffled - true) / shuffled == pytest.approx(0.00264, abs=5e-5)

    def test_the_measured_test_gain_is_what_the_perplexities_say(self):
        true, shuffled = 3.4071, 3.4164
        assert (shuffled - true) / shuffled == pytest.approx(0.00272, abs=5e-5)

    def test_the_measured_gain_exceeds_the_unigram_ceiling(self):
        unigram_ceiling = MOD.gain_from_information(0.001720)
        measured = (3.4464 - 3.4373) / 3.4464
        assert measured > unigram_ceiling
