"""Gate 2: the conditioning check.

Validation perplexity under the correct growth-temperature token must beat
perplexity under a shuffled one. If it does not, the model is ignoring its
conditioning, and generating from it would produce sequences whose target
temperature had no influence on them — results about nothing.

The comparison logic lives here, apart from the training code, so it can be
tested without a GPU and so a bug in training cannot quietly weaken its own
gate.

Two details matter for the comparison to mean anything:

* **Shuffle, do not randomize.** Shuffling the bin assignment across examples
  preserves the class distribution, so the only thing that changes is whether
  each token matches its own sequence. Randomizing labels independently would
  also change the class prior, and a perplexity difference could then come
  from the prior rather than from conditioning.
* **Average in log space.** Perplexity is the exponential of mean token
  negative log-likelihood. Averaging per-sequence perplexities instead
  over-weights short sequences and can move the comparison across the
  threshold on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Gate2Params:
    """Gate 2 settings from ``configs/train.yaml``."""

    enabled: bool
    min_perplexity_gain: float
    n_shuffles: int
    fail_on_violation: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_perplexity_gain < 1.0:
            raise ValueError(
                "min_perplexity_gain is a relative improvement in [0, 1), got "
                f"{self.min_perplexity_gain}. Use 0.02 for 2%, not 2."
            )
        if self.n_shuffles < 1:
            raise ValueError(f"n_shuffles must be at least 1, got {self.n_shuffles}")

    @classmethod
    def from_config(cls, cfg, path: str = "conditioning_gate") -> Gate2Params:
        section = cfg.section(path)
        return cls(
            enabled=bool(section["enabled"]),
            min_perplexity_gain=float(section["min_perplexity_gain"]),
            n_shuffles=int(section["n_shuffles"]),
            fail_on_violation=bool(section["fail_on_violation"]),
        )


def perplexity(total_nll: float, total_tokens: int) -> float:
    """Perplexity from summed token negative log-likelihood.

    Summing NLL and token counts, then exponentiating once, is what makes this
    a token-level average. Averaging per-sequence perplexities would weight a
    20-residue sequence the same as a 1,400-residue one.
    """
    if total_tokens <= 0:
        raise ValueError("cannot compute perplexity over zero tokens")
    return math.exp(total_nll / total_tokens)


@dataclass
class Gate2Result:
    """Outcome of the conditioning check."""

    true_perplexity: float
    shuffled_perplexities: list[float] = field(default_factory=list)
    min_gain_required: float = 0.0

    @property
    def mean_shuffled_perplexity(self) -> float:
        if not self.shuffled_perplexities:
            raise ValueError("no shuffled runs recorded")
        return sum(self.shuffled_perplexities) / len(self.shuffled_perplexities)

    @property
    def relative_gain(self) -> float:
        """How much lower the true-token perplexity is, as a fraction.

        Positive means conditioning helps. Computed against the shuffled mean
        so a single unlucky shuffle cannot decide the gate.
        """
        shuffled = self.mean_shuffled_perplexity
        if shuffled <= 0:
            raise ValueError("shuffled perplexity must be positive")
        return (shuffled - self.true_perplexity) / shuffled

    @property
    def passed(self) -> bool:
        return self.relative_gain >= self.min_gain_required

    @property
    def beats_every_shuffle(self) -> bool:
        """Stronger signal than the mean: better than every shuffle drawn.

        Reported rather than enforced, because with few shuffles it is a noisy
        criterion, but a gate that passes on the mean while losing to some
        individual shuffles is worth seeing.
        """
        return all(self.true_perplexity < p for p in self.shuffled_perplexities)

    def summary(self) -> str:
        lines = [
            f"perplexity with the correct growth-temperature token: {self.true_perplexity:.4f}",
            f"perplexity with shuffled tokens: {self.mean_shuffled_perplexity:.4f} "
            f"(mean of {len(self.shuffled_perplexities)})",
        ]
        if self.shuffled_perplexities:
            lines.append(
                "  individual shuffles: "
                + ", ".join(f"{p:.4f}" for p in self.shuffled_perplexities)
            )
        lines.append(
            f"relative gain: {self.relative_gain:+.2%} "
            f"(required {self.min_gain_required:+.2%})"
        )
        lines.append(f"beats every shuffle: {self.beats_every_shuffle}")
        lines.append("GATE 2 PASSED" if self.passed else "GATE 2 FAILED")
        if not self.passed:
            lines.append(
                "The model is not using its growth-temperature token. Generating "
                "from it would produce sequences whose target temperature had no "
                "influence. Debug the conditioning before generating anything: "
                "check that the control token is not masked out of the loss, that "
                "label dropout is below 1.0, and that the training bins are not "
                "degenerate."
            )
        return "\n".join(lines)


def evaluate_gate2(
    true_nll: float,
    true_tokens: int,
    shuffled_runs: list[tuple[float, int]],
    params: Gate2Params,
) -> Gate2Result:
    """Build a Gate 2 result from summed NLLs and token counts.

    ``shuffled_runs`` is one ``(total_nll, total_tokens)`` pair per shuffle.
    """
    if not shuffled_runs:
        raise ValueError("Gate 2 needs at least one shuffled run to compare against")
    if len(shuffled_runs) != params.n_shuffles:
        raise ValueError(
            f"expected {params.n_shuffles} shuffled runs per the config, "
            f"got {len(shuffled_runs)}"
        )

    return Gate2Result(
        true_perplexity=perplexity(true_nll, true_tokens),
        shuffled_perplexities=[perplexity(nll, tokens) for nll, tokens in shuffled_runs],
        min_gain_required=params.min_perplexity_gain,
    )
