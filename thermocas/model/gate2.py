"""Gate 2: the conditioning check.

Validation perplexity under the correct growth-temperature token must beat
perplexity under a shuffled one. If it does not, the model is ignoring its
conditioning, and generating from it would produce sequences whose target
temperature had no influence on them -- results about nothing.

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

What this gate asks, and why it changed
---------------------------------------

Until 2026-10-26 the gate was a single absolute threshold: the relative
perplexity gain had to exceed 2%. That number was set by assertion and never
checked against the corpus, and it was wrong by more than an order of
magnitude. ``scripts/13_conditioning_ceiling.py`` computes the mutual
information between residue identity and growth-temperature bin, which bounds
what any model can gain from the token:

    I(residue; bin | family)   0.001720 nats
    best possible gain         +0.172%
    the gate demanded          +2.00%     -- 12x above the ceiling

So the gate was unreachable, and the first run that finished -- measuring
+0.264% on validation and +0.272% on held-out test, against a 0.172% unigram
ceiling -- was reported as a failure when it had in fact captured about one
and a half times the first-order signal available. A language model exceeding
the unigram ceiling is expected rather than suspicious: it also has context,
so it can use family- and position-dependent substitutions that a residue
histogram cannot see.

The lesson is that **effect size is a property of the corpus, not of the
model**, so thresholding it against a constant was a category error. This gate
now asks two questions that the data can actually answer:

1. **Is the effect real?** The true-token perplexity must beat every shuffle,
   and beat the shuffled mean by a margin large relative to the spread across
   shuffles (``z``). The null being tested is "it does not matter whether each
   token matches its own sequence", which is exactly what shuffling breaks.
   Both arms score the same sequences, so this is a paired comparison and the
   spread is small -- the measured run scored z = 80.
2. **Is the effect large, given what is available?** The gain is reported as a
   fraction of the computed ceiling. This is thresholded only when a ceiling
   has been supplied, because a fraction of an unknown quantity is not a
   criterion. With no ceiling in the config the gate reports the raw gain and
   says that the effect size is unjudged.

A gate that cannot be passed teaches nothing, and a gate whose threshold is
invented teaches something false. Both failure modes are the same mistake --
asserting a number instead of measuring one -- which is the mistake this
project keeps finding in its own work.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Gate2Params:
    """Gate 2 settings from ``configs/train.yaml``."""

    enabled: bool
    n_shuffles: int
    fail_on_violation: bool
    min_z: float = 3.0
    ceiling_gain: float | None = None
    min_ceiling_fraction: float = 0.5
    # Which corpus ceiling_gain was measured on, as a config name ("pretrain"
    # for stage A, "data" for the Cas9 stage). A ceiling is a property of one
    # corpus, so carrying it unlabelled is how stage B would silently come to
    # be judged against stage A's number. See check_corpus.
    ceiling_corpus: str | None = None

    def __post_init__(self) -> None:
        if self.n_shuffles < 2:
            raise ValueError(
                "n_shuffles must be at least 2: the gate measures the margin "
                f"against the spread across shuffles, and one shuffle has no "
                f"spread. Got {self.n_shuffles}"
            )
        if self.min_z < 0:
            raise ValueError(f"min_z must be non-negative, got {self.min_z}")
        if self.ceiling_gain is not None and not 0.0 < self.ceiling_gain < 0.1:
            # The upper bound is 10%, not 100%. A single conditioning token
            # worth more than a 10% perplexity gain would be a remarkable
            # result for any corpus of this kind, and the likelier reading of
            # such a value is the original units mistake in new clothes:
            # 0.172 written where 0.00172 was meant.
            raise ValueError(
                "ceiling_gain is a relative perplexity gain in (0, 0.1), got "
                f"{self.ceiling_gain}. Use 0.00172 for 0.172%, not 0.172. "
                "Compute it with scripts/13_conditioning_ceiling.py"
            )
        if not 0.0 <= self.min_ceiling_fraction <= 2.0:
            raise ValueError(
                "min_ceiling_fraction is a fraction of the computed ceiling; "
                f"got {self.min_ceiling_fraction}. Values above 1.0 are allowed "
                "because a language model has context and can exceed the unigram "
                "ceiling, but above 2.0 is almost certainly a units mistake."
            )
        if self.ceiling_gain is not None and not self.ceiling_corpus:
            raise ValueError(
                "ceiling_gain is set but ceiling_corpus is not. The ceiling is "
                "a property of one corpus and is meaningless against another, "
                "so it has to say which one it came from. Set "
                "conditioning_gate/ceiling_corpus to the config name the "
                "ceiling was measured on ('pretrain' or 'data')."
            )

    def check_corpus(self, corpus: str) -> None:
        """Refuse a ceiling measured on a different corpus than this run's.

        Nothing in the arithmetic notices this: a stale ceiling produces a
        confident ceiling-fraction, and a pass or fail that means nothing. The
        mistake is one line in a config and would survive any amount of
        staring at the gate's output, so it is checked before the GPU is
        touched rather than discovered after eighteen hours.
        """
        if self.ceiling_gain is None or self.ceiling_corpus == corpus:
            return
        raise ValueError(
            f"conditioning_gate/ceiling_gain ({self.ceiling_gain:+.3%}) was "
            f"measured on the '{self.ceiling_corpus}' corpus, but this run "
            f"trains on '{corpus}'. Effect size would be judged against the "
            "wrong ceiling. Measure this corpus's ceiling with\n"
            f"    python scripts/13_conditioning_ceiling.py --config {corpus}\n"
            "then set ceiling_gain and ceiling_corpus to match. To run with "
            "effect size unjudged instead, clear ceiling_gain."
        )

    @classmethod
    def from_config(cls, cfg, path: str = "conditioning_gate") -> Gate2Params:
        section = cfg.section(path)
        ceiling = None
        if "ceiling_gain" in section and section["ceiling_gain"]:
            ceiling = float(section["ceiling_gain"])
        return cls(
            enabled=bool(section["enabled"]),
            n_shuffles=int(section["n_shuffles"]),
            fail_on_violation=bool(section["fail_on_violation"]),
            min_z=float(section["min_z"]) if "min_z" in section else 3.0,
            ceiling_gain=ceiling,
            ceiling_corpus=(
                str(section["ceiling_corpus"])
                if "ceiling_corpus" in section and section["ceiling_corpus"]
                else None
            ),
            min_ceiling_fraction=(
                float(section["min_ceiling_fraction"])
                if "min_ceiling_fraction" in section
                else 0.5
            ),
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
    min_z: float = 3.0
    ceiling_gain: float | None = None
    min_ceiling_fraction: float = 0.5

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
    def shuffle_spread(self) -> float:
        """Sample standard deviation of perplexity across shuffle draws."""
        if len(self.shuffled_perplexities) < 2:
            return 0.0
        return statistics.stdev(self.shuffled_perplexities)

    @property
    def z(self) -> float:
        """Margin over the shuffled mean, in units of the shuffle spread.

        This tests "does it matter whether each token matches its own
        sequence", which is the null that shuffling breaks. It is not a
        confidence interval on the perplexity estimate itself: both arms score
        the same sequences in the same order, so sequence-sampling error is
        common to both and cancels, which is why the spread is small and z is
        large when the effect is real at all.

        Infinite when every shuffle landed on the same perplexity and the true
        token still beat them -- a degenerate but unambiguous pass. Returns
        0.0 when there is no margin to speak of, so a zero-spread tie does not
        read as infinitely significant.
        """
        margin = self.mean_shuffled_perplexity - self.true_perplexity
        spread = self.shuffle_spread
        if spread == 0.0:
            return math.inf if margin > 0 else 0.0
        return margin / spread

    @property
    def ceiling_fraction(self) -> float | None:
        """The gain as a fraction of what the corpus makes available."""
        if not self.ceiling_gain:
            return None
        return self.relative_gain / self.ceiling_gain

    @property
    def is_significant(self) -> bool:
        return self.beats_every_shuffle and self.z >= self.min_z

    @property
    def is_large_enough(self) -> bool:
        """True when the effect clears its share of the ceiling.

        Vacuously true with no ceiling supplied: an unjudged effect size must
        not silently fail the gate, and must not silently pass it either --
        ``summary`` says which of the two happened.
        """
        fraction = self.ceiling_fraction
        if fraction is None:
            return True
        return fraction >= self.min_ceiling_fraction

    @property
    def passed(self) -> bool:
        return self.is_significant and self.is_large_enough

    @property
    def beats_every_shuffle(self) -> bool:
        """Better than every shuffle drawn, not merely better than their mean.

        Part of the significance criterion rather than a side note: a result
        that beats the mean while losing to individual shuffles is within the
        noise of which shuffle was drawn.
        """
        if not self.shuffled_perplexities:
            return False
        return all(self.true_perplexity < p for p in self.shuffled_perplexities)

    def summary(self) -> str:
        lines = [
            "perplexity with the correct growth-temperature token: "
            f"{self.true_perplexity:.4f}",
            f"perplexity with shuffled tokens: {self.mean_shuffled_perplexity:.4f} "
            f"(mean of {len(self.shuffled_perplexities)})",
        ]
        if self.shuffled_perplexities:
            lines.append(
                "  individual shuffles: "
                + ", ".join(f"{p:.4f}" for p in self.shuffled_perplexities)
            )
        lines.append(f"relative gain: {self.relative_gain:+.3%}")

        lines.append("")
        lines.append("is the effect real?")
        lines.append(f"  beats every shuffle   {self.beats_every_shuffle}")
        z = self.z
        z_text = "inf (every shuffle scored alike)" if z == math.inf else f"{z:.1f}"
        lines.append(
            f"  z vs shuffle spread   {z_text}  (required {self.min_z:.1f}; "
            f"spread {self.shuffle_spread:.6f})"
        )
        lines.append(f"  -> {'SIGNIFICANT' if self.is_significant else 'NOT SIGNIFICANT'}")

        lines.append("")
        lines.append("is the effect large, given what the corpus allows?")
        fraction = self.ceiling_fraction
        if fraction is None:
            lines.append(
                "  no ceiling in the config, so effect size is UNJUDGED. Compute "
                "it with\n  scripts/13_conditioning_ceiling.py and set "
                "conditioning_gate/ceiling_gain."
            )
        else:
            lines.append(
                f"  corpus ceiling        {self.ceiling_gain:+.3%} "
                "(scripts/13_conditioning_ceiling.py)"
            )
            lines.append(
                f"  gain / ceiling        {fraction:.2f}x  "
                f"(required {self.min_ceiling_fraction:.2f}x)"
            )
            if fraction > 1.0:
                lines.append(
                    "  above the unigram ceiling, which is expected of a model "
                    "that also has\n  context and can use substitutions a residue "
                    "histogram cannot see."
                )
            lines.append(
                f"  -> {'LARGE ENOUGH' if self.is_large_enough else 'TOO SMALL'}"
            )

        lines.append("")
        lines.append("GATE 2 PASSED" if self.passed else "GATE 2 FAILED")
        if not self.passed:
            lines.append(self._diagnosis())
        return "\n".join(lines)

    def _diagnosis(self) -> str:
        """Why it failed, which decides what to do about it."""
        if not self.is_significant:
            return (
                "The model is not using its growth-temperature token: the gain is "
                "within the noise of which shuffle was drawn. Generating from it "
                "would produce sequences whose target temperature had no "
                "influence. Check that the control token is not masked out of the "
                "loss, that label dropout is below 1.0, and that the training bins "
                "are not degenerate."
            )
        return (
            "The conditioning is real but small: the effect is statistically "
            "unambiguous and still under its share of what this corpus makes "
            "available. More training steps are the wrong response if the "
            "conditioning curve has plateaued -- the question is whether the "
            "token has enough leverage on the model, not whether it has had "
            "enough time."
        )


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
        shuffled_perplexities=[
            perplexity(nll, tokens) for nll, tokens in shuffled_runs
        ],
        min_z=params.min_z,
        ceiling_gain=params.ceiling_gain,
        min_ceiling_fraction=params.min_ceiling_fraction,
    )
