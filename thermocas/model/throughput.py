"""Measuring how fast training actually runs.

This exists because of a measurement that was wrong in a way that mattered.
The first GPU smoke run was timed by wrapping the whole training script and
dividing by the step count, which reported 2.97 s/step -- but that wall time
also contained a 3 GB checkpoint download, the model build and a Gate 2
evaluation. Lightning's own progress bar put the training itself at about
nine times faster. A number assembled that way cannot be used to plan a run,
and quoting one off a progress bar is not better.

So the fit phase times itself, from inside, and reports the three figures a
schedule is actually built from: optimizer steps per second, how long the
configured ``max_steps`` will take at that rate, and peak GPU memory. Setup
cost is reported separately rather than averaged into the steps, because the
two scale differently -- setup is paid once however long the run is.
"""

from __future__ import annotations

import time

import lightning as L


def format_duration(seconds: float) -> str:
    """A duration a person can act on: 9.2 h rather than 33,120 s."""
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


class ThroughputReport(L.Callback):
    """Times the fit phase and prints what it implies for the full run.

    Attached to every run, not just smoke runs: the cost of a training step
    changes with batch size, sequence length and the attention path, so a
    figure measured once is a figure that goes stale silently.
    """

    def __init__(
        self, planned_steps: int | None = None, epoch_examples: int | None = None
    ) -> None:
        self.planned_steps = planned_steps
        self.epoch_examples = epoch_examples
        self.started: float | None = None
        self.elapsed: float | None = None
        self.steps = 0
        self.peak_memory_gb: float | None = None
        self.sequences = 0
        self.padded_tokens = 0
        self.real_tokens = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Count work done, not just steps taken.

        Steps per second cannot be compared across batch sizes: an optimizer
        step covers batch_size x accumulate_grad_batches sequences, so a
        larger batch makes each step slower and rarer at the same time. Rate
        per token is comparable, and splitting padded from real tokens makes
        the cost of a larger batch visible directly -- these sequences run
        from about 50 to 1,027 tokens, and every batch pads to its longest
        member.
        """
        mask = batch.get("attention_mask")
        if mask is None:
            return
        self.sequences += int(mask.shape[0])
        self.padded_tokens += int(mask.numel())
        self.real_tokens += int(mask.sum())

    def on_train_start(self, trainer, pl_module) -> None:
        import torch

        if torch.cuda.is_available():
            # Reset here rather than at process start so the figure describes
            # training, not the checkpoint load that preceded it.
            torch.cuda.reset_peak_memory_stats()
        self.started = time.monotonic()

    def on_train_end(self, trainer, pl_module) -> None:
        import torch

        if self.started is None:
            return
        self.elapsed = time.monotonic() - self.started
        self.steps = int(trainer.global_step)
        if torch.cuda.is_available():
            self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

    def summary(self) -> dict[str, float | int | None]:
        if not self.elapsed or not self.steps:
            return {}
        seconds_per_step = self.elapsed / self.steps
        result: dict[str, float | int | None] = {
            "steps": self.steps,
            "seconds": round(self.elapsed, 1),
            "seconds_per_step": round(seconds_per_step, 3),
            "steps_per_second": round(self.steps / self.elapsed, 3),
            "peak_memory_gb": (
                round(self.peak_memory_gb, 2) if self.peak_memory_gb else None
            ),
        }
        if self.sequences:
            result["sequences"] = self.sequences
            result["sequences_per_second"] = round(self.sequences / self.elapsed, 2)
            result["real_tokens_per_second"] = round(self.real_tokens / self.elapsed)
            result["padding_waste"] = round(
                1.0 - self.real_tokens / max(self.padded_tokens, 1), 3
            )
            # An epoch is the comparable unit across batch sizes, where a
            # step is not.
            if self.epoch_examples:
                result["projected_epoch_seconds"] = round(
                    self.epoch_examples * self.elapsed / self.sequences
                )
        if self.planned_steps:
            result["planned_steps"] = self.planned_steps
            result["projected_seconds"] = round(seconds_per_step * self.planned_steps)
        return result

    def report(self) -> str:
        summary = self.summary()
        if not summary:
            return "throughput: training did not run"

        lines = [
            f"throughput: {summary['steps']} optimizer steps in "
            f"{format_duration(float(summary['seconds']))} "
            f"({summary['seconds_per_step']} s/step)"
        ]
        if summary.get("sequences_per_second"):
            lines.append(
                f"  {summary['sequences_per_second']} sequences/s, "
                f"{summary['real_tokens_per_second']:,} real tokens/s, "
                f"{summary['padding_waste']:.1%} of the tokens processed were padding"
            )
        if summary.get("projected_epoch_seconds"):
            lines.append(
                f"  one epoch over {self.epoch_examples:,} examples: "
                f"{format_duration(float(summary['projected_epoch_seconds']))}"
            )
        if summary.get("peak_memory_gb"):
            lines.append(f"  peak GPU memory during training: "
                         f"{summary['peak_memory_gb']} GB")
        if summary.get("projected_seconds"):
            lines.append(
                f"  at this rate {summary['planned_steps']} steps would take "
                f"{format_duration(float(summary['projected_seconds']))}"
            )
            if summary["steps"] < 200:
                lines.append(
                    "  measured over a short run, so the projection carries the "
                    "warmup and the first-batch costs with it"
                )
        return "\n".join(lines)
