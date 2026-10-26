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
        enabled=True, min_perplexity_gain=0.02, n_shuffles=5, fail_on_violation=True
    )
    base.update(overrides)
    return Gate2Params(**base)


class TestParams:
    def test_loads_from_shipped_config(self):
        loaded = Gate2Params.from_config(Config.load("train"))
        assert loaded.enabled is True
        assert loaded.fail_on_violation is True
        assert 0.0 < loaded.min_perplexity_gain < 1.0

    def test_gain_expressed_as_a_percent_is_rejected(self):
        """2 instead of 0.02 would demand a 200% improvement and never pass."""
        with pytest.raises(ValueError, match="relative improvement"):
            params(min_perplexity_gain=2.0)

    def test_zero_shuffles_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            params(n_shuffles=0)

    def test_zero_gain_is_allowed(self):
        """A bare 'any improvement at all' threshold is a legitimate choice."""
        assert params(min_perplexity_gain=0.0).min_perplexity_gain == 0.0


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
    def test_conditioning_that_helps_passes(self):
        result = evaluate_gate2(
            true_nll=100.0,
            true_tokens=100,
            shuffled_runs=[(110.0, 100)] * 5,
            params=params(min_perplexity_gain=0.02),
        )
        assert result.passed
        assert result.relative_gain > 0.02
        assert "GATE 2 PASSED" in result.summary()

    def test_conditioning_that_does_nothing_fails(self):
        result = evaluate_gate2(
            true_nll=100.0,
            true_tokens=100,
            shuffled_runs=[(100.0, 100)] * 5,
            params=params(),
        )
        assert not result.passed
        assert result.relative_gain == pytest.approx(0.0)
        assert "GATE 2 FAILED" in result.summary()

    def test_failure_message_says_what_to_check(self):
        result = evaluate_gate2(100.0, 100, [(100.0, 100)] * 5, params())
        summary = result.summary()
        assert "not using its growth-temperature token" in summary
        assert "label dropout" in summary

    def test_conditioning_that_hurts_fails_with_a_negative_gain(self):
        result = evaluate_gate2(
            true_nll=120.0, true_tokens=100, shuffled_runs=[(100.0, 100)] * 5, params=params()
        )
        assert not result.passed
        assert result.relative_gain < 0
        assert "-" in result.summary()

    def test_gain_just_below_the_threshold_fails(self):
        # 1% improvement against a 2% requirement.
        result = evaluate_gate2(
            true_nll=99.0, true_tokens=100, shuffled_runs=[(100.0, 100)] * 5,
            params=params(min_perplexity_gain=0.02),
        )
        assert result.relative_gain < 0.02
        assert not result.passed

    def test_threshold_is_inclusive(self):
        result = Gate2Result(
            true_perplexity=0.98, shuffled_perplexities=[1.0], min_gain_required=0.02
        )
        assert result.relative_gain == pytest.approx(0.02)
        assert result.passed

    def test_compares_against_the_mean_of_shuffles(self):
        """One unlucky shuffle must not decide the gate."""
        result = evaluate_gate2(
            true_nll=100.0,
            true_tokens=100,
            shuffled_runs=[(101.0, 100), (120.0, 100), (115.0, 100), (118.0, 100), (119.0, 100)],
            params=params(),
        )
        assert len(result.shuffled_perplexities) == 5
        assert result.mean_shuffled_perplexity > result.true_perplexity
        assert result.passed

    def test_beats_every_shuffle_is_reported_separately(self):
        """Passing on the mean while losing to some shuffles is worth seeing."""
        losing = evaluate_gate2(
            true_nll=100.0,
            true_tokens=100,
            shuffled_runs=[(90.0, 100), (130.0, 100), (130.0, 100), (130.0, 100), (130.0, 100)],
            params=params(),
        )
        assert losing.passed  # the mean is much worse than the true run
        assert not losing.beats_every_shuffle  # but one shuffle beat it
        assert "beats every shuffle: False" in losing.summary()

    def test_shuffle_count_must_match_the_config(self):
        """Silently running fewer shuffles than configured would weaken the gate."""
        with pytest.raises(ValueError, match="expected 5 shuffled runs"):
            evaluate_gate2(100.0, 100, [(110.0, 100)] * 2, params(n_shuffles=5))

    def test_no_shuffles_rejected(self):
        with pytest.raises(ValueError, match="at least one shuffled run"):
            evaluate_gate2(100.0, 100, [], params(n_shuffles=1))

    def test_differing_token_counts_are_handled(self):
        """Dropping over-length sequences can change token counts between runs."""
        result = evaluate_gate2(
            true_nll=200.0, true_tokens=200, shuffled_runs=[(230.0, 200)] * 5, params=params()
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

    def test_the_report_calls_a_rising_curve_rising(self):
        probe, _ = self.build([0.001, 0.004, 0.009])
        module = self.FakeModule()
        for step in (100, 200, 300):
            self.fire(probe, module, step)
        assert "rising" in probe.report()

    def test_the_report_calls_a_flat_curve_flat(self):
        """The distinction the probe exists to make: a flat curve means more
        steps are not the missing ingredient."""
        probe, _ = self.build([0.004, 0.004, 0.003])
        module = self.FakeModule()
        for step in (100, 200, 300):
            self.fire(probe, module, step)
        assert "flat or falling" in probe.report()

    def test_the_report_says_so_when_it_never_ran(self):
        probe, _ = self.build([])
        assert "never ran" in probe.report()
