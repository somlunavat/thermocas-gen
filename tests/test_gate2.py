"""Gate 2, the conditioning check.

Gate 2 decides whether anything downstream is worth generating, so the tests
focus on ways it could pass while being meaningless: averaging perplexity
wrongly, comparing against a vacuous shuffle, or a threshold expressed in the
wrong units.
"""

from __future__ import annotations

import math

import pytest

from thermocas.config import Config
from thermocas.model.gate2 import (
    Gate2Params,
    Gate2Result,
    evaluate_gate2,
    perplexity,
)


def params(**overrides) -> Gate2Params:
    base = dict(
        enabled=True,
        n_shuffles=5,
        fail_on_violation=True,
        min_z=3.0,
        ceiling_gain=0.00172,
        min_ceiling_fraction=0.5,
    )
    base.update(overrides)
    return Gate2Params(**base)


# Perplexities from the 20,000-step Stage A run of 2026-10-26, which the old
# gate failed and the corrected gate passes. Kept as constants because they
# are the regression this rewrite exists for.
MEASURED_VAL_TRUE = 3.4373
MEASURED_VAL_SHUFFLED = [3.4464, 3.4465, 3.4464, 3.4463, 3.4466]
MEASURED_TEST_TRUE = 3.4071
MEASURED_TEST_SHUFFLED = [3.4165, 3.4166, 3.4164, 3.4160, 3.4165]
# scripts/13_conditioning_ceiling.py on the stage A corpus.
STAGE_A_CEILING = 0.00172


def result(true, shuffled, **overrides) -> Gate2Result:
    kwargs = dict(
        min_z=3.0, ceiling_gain=STAGE_A_CEILING, min_ceiling_fraction=0.5
    )
    kwargs.update(overrides)
    return Gate2Result(
        true_perplexity=true, shuffled_perplexities=list(shuffled), **kwargs
    )


class TestParams:
    def test_loads_from_shipped_config(self):
        loaded = Gate2Params.from_config(Config.load("train"))
        assert loaded.enabled is True
        assert loaded.fail_on_violation is True
        assert loaded.min_z > 0

    def test_the_shipped_config_carries_a_measured_ceiling(self):
        """Without one, effect size is unjudged and the gate is half a gate."""
        loaded = Gate2Params.from_config(Config.load("train"))
        assert loaded.ceiling_gain is not None
        assert 0.0 < loaded.ceiling_gain < 0.01

    def test_a_ceiling_expressed_as_a_percent_is_rejected(self):
        """0.172 instead of 0.00172 would demand a 17% gain on a corpus whose
        ceiling is 0.172%, which is the original mistake in new units."""
        with pytest.raises(ValueError, match="relative perplexity gain"):
            params(ceiling_gain=0.172)

    def test_one_shuffle_is_rejected(self):
        """The gate measures a margin against the spread across shuffles, and
        one shuffle has no spread."""
        with pytest.raises(ValueError, match="at least 2"):
            params(n_shuffles=1)

    def test_a_negative_z_floor_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            params(min_z=-1.0)

    def test_a_ceiling_fraction_above_one_is_allowed(self):
        """A language model has context and can beat a unigram ceiling, so
        demanding more than 1.0x is a legitimate, if strict, choice."""
        assert params(min_ceiling_fraction=1.5).min_ceiling_fraction == 1.5

    def test_an_absurd_ceiling_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="units mistake"):
            params(min_ceiling_fraction=50.0)

    def test_no_ceiling_is_allowed(self):
        assert params(ceiling_gain=None).ceiling_gain is None


class TestSignificance:
    def test_the_measured_run_is_significant(self):
        r = result(MEASURED_VAL_TRUE, MEASURED_VAL_SHUFFLED)
        assert r.is_significant
        assert r.z > 50

    def test_no_effect_is_not_significant(self):
        """Shuffles and truth landing alike is the null this gate tests."""
        r = result(3.4464, MEASURED_VAL_SHUFFLED)
        assert not r.is_significant

    def test_beating_the_mean_while_losing_to_a_shuffle_is_not_significant(self):
        r = result(3.4464, [3.4470, 3.4480, 3.4460, 3.4475, 3.4478])
        assert not r.beats_every_shuffle
        assert not r.is_significant

    def test_identical_shuffles_with_a_real_margin_are_infinitely_significant(self):
        r = result(1.0, [2.0, 2.0, 2.0, 2.0, 2.0])
        assert r.z == math.inf
        assert r.is_significant

    def test_identical_shuffles_with_no_margin_are_not(self):
        """Zero spread must not read as infinite significance on a tie."""
        r = result(2.0, [2.0, 2.0, 2.0, 2.0, 2.0])
        assert r.z == 0.0
        assert not r.is_significant

    def test_the_wrong_direction_is_not_significant(self):
        r = result(3.4560, MEASURED_VAL_SHUFFLED)
        assert r.relative_gain < 0
        assert not r.is_significant


class TestEffectSize:
    def test_the_measured_run_exceeds_the_unigram_ceiling(self):
        """Expected of a model that also has context, not a red flag."""
        r = result(MEASURED_VAL_TRUE, MEASURED_VAL_SHUFFLED)
        assert r.ceiling_fraction > 1.0
        assert r.is_large_enough

    def test_a_real_but_tiny_effect_is_too_small(self):
        """The case the old gate could not express: unambiguously significant
        and still a fraction of the available signal."""
        r = result(3.4450, MEASURED_VAL_SHUFFLED)
        assert r.is_significant
        assert not r.is_large_enough
        assert not r.passed

    def test_without_a_ceiling_effect_size_is_unjudged_not_failed(self):
        r = result(3.4450, MEASURED_VAL_SHUFFLED, ceiling_gain=None)
        assert r.ceiling_fraction is None
        assert r.is_large_enough
        assert "UNJUDGED" in r.summary()

    def test_the_summary_names_the_script_that_computes_the_ceiling(self):
        r = result(3.4450, MEASURED_VAL_SHUFFLED, ceiling_gain=None)
        assert "13_conditioning_ceiling.py" in r.summary()


class TestAgainstTheMeasuredRun:
    """The rewrite's whole point: this run failed the old gate and passes the
    corrected one, on both splits, without the numbers changing."""

    def test_validation_passes(self):
        assert result(MEASURED_VAL_TRUE, MEASURED_VAL_SHUFFLED).passed

    def test_held_out_test_passes(self):
        assert result(MEASURED_TEST_TRUE, MEASURED_TEST_SHUFFLED).passed

    def test_both_splits_agree_on_effect_size(self):
        """Validation and test within 10% of each other is the evidence that
        the 0.26% is a property of the model, not of the split."""
        val = result(MEASURED_VAL_TRUE, MEASURED_VAL_SHUFFLED).relative_gain
        test = result(MEASURED_TEST_TRUE, MEASURED_TEST_SHUFFLED).relative_gain
        assert abs(val - test) / val < 0.10

    def test_the_old_two_percent_threshold_would_still_fail_it(self):
        """Documents why the threshold was the problem: the measured gain is
        an order of magnitude under the asserted 2%, and also 1.5x the most
        any model could get from residue composition on this corpus."""
        gain = result(MEASURED_VAL_TRUE, MEASURED_VAL_SHUFFLED).relative_gain
        assert gain < 0.02
        assert gain > STAGE_A_CEILING

    def test_a_failed_gate_says_which_criterion_failed(self):
        """'Real but small' and 'not using the token' call for opposite
        responses, so the diagnosis has to distinguish them."""
        small = result(3.4450, MEASURED_VAL_SHUFFLED).summary()
        absent = result(3.4464, MEASURED_VAL_SHUFFLED).summary()
        assert "real but small" in small
        assert "not using its growth-temperature token" in absent


class TestPerplexity:
    def test_matches_the_definition(self):
        assert perplexity(total_nll=0.0, total_tokens=10) == pytest.approx(1.0)
        assert perplexity(total_nll=10.0, total_tokens=10) == pytest.approx(math.e)

    def test_is_a_token_level_average_not_a_sequence_level_one(self):
        """Summing NLL and tokens before exponentiating weights by length.

        Averaging per-sequence perplexities instead would weight a 20-residue
        sequence the same as a 1,400-residue one, which can move the Gate 2
        comparison across its threshold on its own.
        """
        # One short confident sequence, one long uncertain one.
        short_nll, short_tokens = 2.0, 20
        long_nll, long_tokens = 2800.0, 1400

        token_level = perplexity(short_nll + long_nll, short_tokens + long_tokens)
        sequence_level = (
            perplexity(short_nll, short_tokens) + perplexity(long_nll, long_tokens)
        ) / 2

        assert token_level == pytest.approx(math.exp(2802.0 / 1420))
        # The long sequence dominates the token-level figure, as it should.
        assert token_level > sequence_level

    def test_zero_tokens_rejected(self):
        with pytest.raises(ValueError, match="zero tokens"):
            perplexity(1.0, 0)


class TestEvaluate:
    """evaluate_gate2 on synthetic NLLs, where the right answer is arithmetic.

    A 10% perplexity gain is used for the passing cases rather than the ~0.26%
    the real corpus allows: these tests check the plumbing, and the measured
    numbers are exercised separately in TestAgainstTheMeasuredRun.
    """

    def test_conditioning_that_helps_passes(self):
        result = evaluate_gate2(
            true_nll=100.0,
            true_tokens=100,
            shuffled_runs=[(110.0, 100), (110.5, 100), (109.5, 100),
                           (110.2, 100), (109.8, 100)],
            params=params(ceiling_gain=0.05),
        )
        assert result.passed
        assert "GATE 2 PASSED" in result.summary()

    def test_conditioning_that_does_nothing_fails(self):
        result = evaluate_gate2(
            true_nll=100.0,
            true_tokens=100,
            shuffled_runs=[(100.0, 100), (100.1, 100), (99.9, 100),
                           (100.05, 100), (99.95, 100)],
            params=params(),
        )
        assert not result.passed
        assert "GATE 2 FAILED" in result.summary()

    def test_failure_message_says_what_to_check(self):
        result = evaluate_gate2(
            100.0, 100,
            [(100.0, 100), (100.1, 100), (99.9, 100), (100.05, 100), (99.95, 100)],
            params(),
        )
        summary = result.summary()
        assert "not using its growth-temperature token" in summary
        assert "label dropout" in summary

    def test_conditioning_that_hurts_fails_with_a_negative_gain(self):
        result = evaluate_gate2(
            true_nll=120.0,
            true_tokens=100,
            shuffled_runs=[(100.0, 100), (100.5, 100), (99.5, 100),
                           (100.2, 100), (99.8, 100)],
            params=params(),
        )
        assert not result.passed
        assert result.relative_gain < 0

    def test_an_effect_under_its_share_of_the_ceiling_fails(self):
        """Replaces the old 'just below the threshold' test. The gain is real
        and significant; it is a fifth of what the corpus allows."""
        result = evaluate_gate2(
            true_nll=99.0,
            true_tokens=100,
            shuffled_runs=[(100.0, 100), (100.1, 100), (99.9, 100),
                           (100.05, 100), (99.95, 100)],
            params=params(ceiling_gain=0.05),
        )
        assert result.is_significant
        assert result.ceiling_fraction < 0.5
        assert not result.passed

    def test_the_ceiling_criterion_is_inclusive(self):
        """Exactly its share passes, so the boundary is not off by one."""
        result = Gate2Result(
            true_perplexity=0.99,
            shuffled_perplexities=[1.0, 1.0001, 0.9999, 1.00005, 0.99995],
            min_z=3.0,
            ceiling_gain=0.02,
            min_ceiling_fraction=0.5,
        )
        assert result.ceiling_fraction == pytest.approx(0.5, rel=1e-3)
        assert result.passed

    def test_compares_against_the_mean_of_shuffles(self):
        """One unlucky shuffle must not decide the gate."""
        result = evaluate_gate2(
            true_nll=100.0,
            true_tokens=100,
            shuffled_runs=[(118.0, 100), (120.0, 100), (115.0, 100),
                           (118.0, 100), (119.0, 100)],
            params=params(ceiling_gain=0.05),
        )
        assert len(result.shuffled_perplexities) == 5
        assert result.mean_shuffled_perplexity > result.true_perplexity
        assert result.passed

    def test_losing_to_a_single_shuffle_now_fails_the_gate(self):
        """Changed deliberately. This used to pass on the mean while losing to
        one shuffle, with 'beats every shuffle: False' reported alongside as a
        note. Beating every shuffle is part of the significance criterion now:
        a result inside the spread of which shuffle was drawn is not an
        effect, whatever the mean says."""
        losing = evaluate_gate2(
            true_nll=100.0,
            true_tokens=100,
            shuffled_runs=[(90.0, 100), (130.0, 100), (130.0, 100),
                           (130.0, 100), (130.0, 100)],
            params=params(ceiling_gain=0.05),
        )
        assert not losing.beats_every_shuffle
        assert not losing.passed
        assert "beats every shuffle   False" in losing.summary()

    def test_shuffle_count_must_match_the_config(self):
        """Silently running fewer shuffles than configured would weaken the gate."""
        with pytest.raises(ValueError, match="expected 5 shuffled runs"):
            evaluate_gate2(100.0, 100, [(110.0, 100)] * 2, params(n_shuffles=5))

    def test_no_shuffles_rejected(self):
        with pytest.raises(ValueError, match="at least one shuffled run"):
            evaluate_gate2(100.0, 100, [], params(n_shuffles=2))

    def test_differing_token_counts_are_handled(self):
        """Dropping over-length sequences can change token counts between runs."""
        result = evaluate_gate2(
            true_nll=200.0,
            true_tokens=200,
            shuffled_runs=[(230.0, 200), (231.0, 200), (229.0, 200),
                           (230.5, 200), (229.5, 200)],
            params=params(ceiling_gain=0.05),
        )
        assert result.passed


class TestConditioningProbe:
    """The probe that turns one end-of-run pass/fail into a curve.

    It lives in scripts/11_train.py rather than the package, so it is loaded
    the way the Modal launcher loads that script. Worth testing despite the
    awkward import: its failure mode is silence, which is the failure mode
    this project keeps finding.
    """

    @staticmethod
    def probe_class():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "11_train.py"
        spec = importlib.util.spec_from_file_location("thermocas_train_probe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ConditioningProbe

    class FakeResult:
        def __init__(self, gain):
            self.relative_gain = gain

    class FakeTrainer:
        def __init__(self, step):
            self.global_step = step

    class FakeModule:
        def __init__(self):
            self.training = True
            self.logged = {}
            self.train_called = 0

        def train(self):
            self.training = True
            self.train_called += 1

        def log(self, key, value):
            self.logged[key] = value

    def build(self, gains, every_n_steps=100):
        calls = []

        def check(dataset, label, limit, quiet=False):
            calls.append((label, limit, quiet))
            return TestConditioningProbe.FakeResult(gains[len(calls) - 1])

        probe = self.probe_class()(
            check, dataset=["a", "b"], limit=2, every_n_steps=every_n_steps
        )
        return probe, calls

    def fire(self, probe, module, step):
        probe.on_train_batch_end(self.FakeTrainer(step), module, None, None, 0)

    def test_it_does_not_run_between_intervals(self):
        probe, calls = self.build([0.01])
        module = self.FakeModule()
        for step in (1, 50, 99, 101):
            self.fire(probe, module, step)
        assert calls == []

    def test_it_runs_on_the_interval(self):
        probe, calls = self.build([0.01, 0.02])
        module = self.FakeModule()
        self.fire(probe, module, 100)
        self.fire(probe, module, 200)
        assert len(calls) == 2
        assert [step for step, _ in probe.history] == [100, 200]

    def test_it_never_probes_step_zero(self):
        """Lightning calls the hook with global_step 0 before any optimizer
        step, and 0 % anything is 0."""
        probe, calls = self.build([0.01])
        self.fire(probe, self.FakeModule(), 0)
        assert calls == []

    def test_it_does_not_probe_the_same_step_twice(self):
        """With gradient accumulation the hook fires once per batch while
        global_step stays put, so the interval alone is not enough."""
        probe, calls = self.build([0.01, 0.02])
        module = self.FakeModule()
        self.fire(probe, module, 100)
        self.fire(probe, module, 100)
        assert len(calls) == 1

    def test_it_puts_the_model_back_in_train_mode(self):
        """evaluate_perplexity calls module.eval() and Lightning does not
        restore it. Training the rest of the run in eval mode would disable
        dropout silently and be invisible in the loss curve."""
        probe, _ = self.build([0.01])
        module = self.FakeModule()
        self.fire(probe, module, 100)
        assert module.train_called == 1
        assert module.training is True

    def test_it_asks_for_a_quiet_check(self):
        """The probe runs eight times in a real run; the smoke-check warning
        and the timing line belong to the gate, not to every probe."""
        probe, calls = self.build([0.01])
        self.fire(probe, self.FakeModule(), 100)
        assert calls[0][2] is True

    def test_it_logs_the_gain_as_a_metric(self):
        probe, _ = self.build([0.0123])
        module = self.FakeModule()
        self.fire(probe, module, 100)
        assert module.logged["cond_gain"] == 0.0123

    def curve(self, gains):
        probe, _ = self.build(gains, every_n_steps=2500)
        module = self.FakeModule()
        for i in range(len(gains)):
            self.fire(probe, module, 2500 * (i + 1))
        return probe.report()

    def test_the_report_calls_a_rising_curve_rising(self):
        report = self.curve([0.001, 0.004, 0.009, 0.013, 0.016, 0.019, 0.022, 0.026])
        assert "RISING" in report

    def test_the_report_calls_a_falling_curve_falling(self):
        report = self.curve([0.020, 0.018, 0.016, 0.014, 0.012, 0.010, 0.008, 0.006])
        assert "FALLING" in report

    def test_the_report_calls_the_first_real_runs_curve_flat(self):
        """The regression this verdict exists for.

        These are the measured gains from the 20,000-step Stage A run of
        2026-10-26. Comparing only the endpoints called it "rising from
        +0.251% to +0.291%", which is true of two points and false of the
        run -- and is the sentence that would have justified buying more
        steps. The plateau is reached by step 2,500.
        """
        report = self.curve(
            [0.00251, 0.00273, 0.00300, 0.00251, 0.00293, 0.00288, 0.00281, 0.00291]
        )
        assert "FLAT" in report
        assert "RISING" not in report
        assert "More steps are not the missing ingredient" in report

    def test_a_noisy_plateau_is_not_called_a_direction_by_its_endpoints(self):
        """Endpoints that rise while the curve does not."""
        report = self.curve([0.010, 0.030, 0.005, 0.028, 0.008, 0.026, 0.009, 0.012])
        assert "FLAT" in report

    def test_a_short_curve_gets_no_verdict(self):
        """Three probes cannot distinguish a trend from jitter, so the report
        shows the numbers and declines to characterise them."""
        report = self.curve([0.001, 0.004, 0.009])
        for word in ("FLAT", "RISING", "FALLING"):
            assert word not in report

    def test_the_report_says_so_when_it_never_ran(self):
        probe, _ = self.build([])
        assert "never ran" in probe.report()
