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
