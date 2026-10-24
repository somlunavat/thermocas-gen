"""Device and precision resolution across accelerators."""

from __future__ import annotations

import pytest

from thermocas.model.device import resolve_device, resolve_precision


@pytest.mark.parametrize("requested", ["16", "16-mixed", "bf16", "bf16-mixed"])
def test_mixed_precision_downgraded_on_mps(requested):
    """fp16/bf16 autocast is unsupported on mps.

    The prior repository hard-coded precision=16, which cannot run on this
    machine at all. An unsupported request is downgraded with a note rather
    than failing partway into a run.
    """
    precision, notes = resolve_precision("mps", requested)
    assert precision == "32-true"
    assert any("mps" in n for n in notes)


def test_auto_precision_is_fp32_on_mps_and_cpu():
    assert resolve_precision("mps", "auto") == ("32-true", ())
    assert resolve_precision("cpu", "auto") == ("32-true", ())


def test_fp16_on_cpu_downgraded():
    precision, notes = resolve_precision("cpu", "16-mixed")
    assert precision == "32-true"
    assert notes


def test_explicit_fp32_passes_through_unchanged():
    assert resolve_precision("mps", "32-true") == ("32-true", ())


def test_unknown_accelerator_rejected():
    with pytest.raises(ValueError, match="unknown accelerator"):
        resolve_device(accelerator="tpu")


def test_auto_resolves_to_something_runnable():
    spec = resolve_device("auto", "auto")
    assert spec.accelerator in {"cuda", "mps", "cpu"}
    assert spec.precision
    assert spec.torch_device.type == spec.accelerator


def test_unavailable_accelerator_falls_back_with_a_note():
    """Asking for cuda on this Mac must not hard-fail; it falls back and says so."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present; nothing to fall back from")
    spec = resolve_device(accelerator="cuda")
    assert spec.accelerator != "cuda"
    assert any("cuda" in n for n in spec.notes)


class TestWorkerCount:
    """Worker count is a device question, like precision.

    0 is right on mps, where worker processes hang rather than run slowly,
    and wrong on a cloud GPU container, where it leaves the GPU waiting on a
    single tokenizing process. The first A10 run was told by Lightning to
    use 16.
    """

    def test_cuda_gets_workers(self):
        from thermocas.model.device import resolve_num_workers

        count, notes = resolve_num_workers("cuda", "auto")
        assert count > 0
        assert count <= 8
        assert notes

    def test_mps_gets_none(self):
        from thermocas.model.device import resolve_num_workers

        assert resolve_num_workers("mps", "auto")[0] == 0

    def test_cpu_gets_none(self):
        from thermocas.model.device import resolve_num_workers

        assert resolve_num_workers("cpu", "auto")[0] == 0

    def test_an_explicit_count_is_honored(self):
        from thermocas.model.device import resolve_num_workers

        assert resolve_num_workers("cuda", "4")[0] == 4
        assert resolve_num_workers("cuda", 0)[0] == 0

    def test_an_explicit_count_is_refused_on_mps(self):
        """Downgraded with a note, the way an unsupported precision is."""
        from thermocas.model.device import resolve_num_workers

        count, notes = resolve_num_workers("mps", "8")
        assert count == 0
        assert any("mps" in note for note in notes)


class TestThroughputReport:
    """The callback that exists because a measurement was wrong."""

    @staticmethod
    def ran(seconds: float, steps: int, planned: int | None = None):
        from thermocas.model.throughput import ThroughputReport

        report = ThroughputReport(planned_steps=planned)
        report.elapsed = seconds
        report.steps = steps
        return report

    def test_it_reports_the_rate_it_measured(self):
        summary = self.ran(100.0, 50).summary()
        assert summary["seconds_per_step"] == 2.0
        assert summary["steps_per_second"] == 0.5

    def test_it_projects_the_planned_run(self):
        summary = self.ran(100.0, 50, planned=20000).summary()
        assert summary["projected_seconds"] == 40000

    def test_a_short_run_says_its_projection_is_rough(self):
        text = self.ran(100.0, 50, planned=20000).report()
        assert "warmup" in text

    def test_a_long_run_does_not(self):
        text = self.ran(4000.0, 2000, planned=20000).report()
        assert "warmup" not in text

    def test_nothing_is_claimed_when_training_did_not_run(self):
        from thermocas.model.throughput import ThroughputReport

        report = ThroughputReport(planned_steps=20000)
        assert report.summary() == {}
        assert "did not run" in report.report()

    def test_durations_are_readable(self):
        from thermocas.model.throughput import format_duration

        assert format_duration(45) == "45.0 s"
        assert format_duration(600) == "10.0 min"
        assert format_duration(33120) == "9.2 h"
        assert format_duration(4 * 86400) == "4.0 days"
