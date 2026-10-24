"""Device and precision resolution.

The same training code runs on Apple Silicon (`mps`), plain CPU, and CUDA, with
the choice coming from config. Precision has to be resolved per accelerator
rather than set once: fp16 autocast is not supported on `mps`, which is why the
prior repository's hard-coded ``precision=16`` could not run locally at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSpec:
    accelerator: str  # "cuda" | "mps" | "cpu"
    precision: str  # a Lightning precision string
    notes: tuple[str, ...] = ()

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.accelerator)


def detect_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(
    accelerator: str, requested: str | int = "auto"
) -> tuple[str, tuple[str, ...]]:
    """Pick a precision that the accelerator actually supports.

    Returns the precision string and any notes worth logging. An unsupported
    request is downgraded with a note rather than left to fail mid-run.
    """
    notes: list[str] = []
    requested = str(requested)

    if requested == "auto":
        if accelerator == "cuda":
            if torch.cuda.is_bf16_supported():
                return "bf16-mixed", ()
            notes.append("bf16 unsupported on this GPU; using fp16-mixed")
            return "16-mixed", tuple(notes)
        # Both mps and cpu run fp32. MPS has no fp16/bf16 autocast support.
        return "32-true", ()

    if accelerator == "mps" and requested in {"16", "16-mixed", "bf16", "bf16-mixed"}:
        notes.append(
            f"precision {requested!r} is not supported on mps; using 32-true. "
            "Run mixed precision on a CUDA machine."
        )
        return "32-true", tuple(notes)

    if accelerator == "cpu" and requested in {"16", "16-mixed"}:
        notes.append(f"precision {requested!r} on cpu is slow and unstable; using 32-true")
        return "32-true", tuple(notes)

    if (
        accelerator == "cuda"
        and requested in {"bf16", "bf16-mixed"}
        and not torch.cuda.is_bf16_supported()
    ):
        notes.append("bf16 unsupported on this GPU; using fp16-mixed")
        return "16-mixed", tuple(notes)

    return requested, tuple(notes)


def resolve_num_workers(
    accelerator: str, requested: str | int = "auto"
) -> tuple[int, tuple[str, ...]]:
    """Pick a DataLoader worker count the accelerator can actually use.

    This is a device question for the same reason precision is. On `mps`,
    worker processes are a source of hangs, so the project runs single
    process. On a cloud GPU container that same 0 is a straight tax: the GPU
    waits while one process tokenizes, and the first A10 run was told by
    Lightning to use 16.

    Capped at 8 rather than the core count because the work per batch is
    small and each worker holds its own copy of the dataset's index.
    """
    notes: list[str] = []
    requested = str(requested)

    if requested != "auto":
        count = int(requested)
        if accelerator == "mps" and count > 0:
            notes.append(
                f"num_workers {count} requested on mps; using 0, where worker "
                "processes hang rather than run slowly"
            )
            return 0, tuple(notes)
        return count, ()

    if accelerator == "cuda":
        count = min(8, os.cpu_count() or 1)
        return count, (f"num_workers auto: {count} on cuda",)
    # mps hangs with workers; on cpu they compete with the training process.
    return 0, (f"num_workers auto: 0 on {accelerator}",)


def resolve_device(accelerator: str = "auto", precision: str | int = "auto") -> DeviceSpec:
    """Resolve the configured accelerator and precision against the machine."""
    notes: list[str] = []

    if accelerator == "auto":
        resolved = detect_accelerator()
    else:
        resolved = accelerator
        available = {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": True,
        }
        if resolved not in available:
            raise ValueError(
                f"unknown accelerator {accelerator!r}; expected cuda, mps, cpu or auto"
            )
        if not available[resolved]:
            fallback = detect_accelerator()
            notes.append(f"accelerator {resolved!r} unavailable; falling back to {fallback!r}")
            resolved = fallback

    if resolved == "mps" and not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        notes.append(
            "PYTORCH_ENABLE_MPS_FALLBACK is unset; unsupported ops will raise "
            "instead of falling back to CPU. Export it to 1 for local runs."
        )

    resolved_precision, precision_notes = resolve_precision(resolved, precision)
    notes.extend(precision_notes)

    return DeviceSpec(accelerator=resolved, precision=resolved_precision, notes=tuple(notes))
