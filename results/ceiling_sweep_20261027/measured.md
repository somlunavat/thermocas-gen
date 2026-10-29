# Predicted ceiling vs measured gain — five GPU runs

Date: 2026-10-28. Raw per-run reports in `runs/`.

The CPU half of this experiment (`sweep.txt`, `analysis.md`) computed a
corpus-only upper bound on how much a growth-temperature label can possibly
help, for eleven thresholds, two balance settings and four granularities. It
established that the bound varies by 4.2x across variants of the *same*
corpus. It did not establish that a trained model's measured gain follows it.
That is what these five runs test, and the test could fail.

## Design

Each run is a *relabeling*, not a second corpus: `temp_bin` is recomputed from
`ogt_c` in memory at startup, so all five train on byte-identical sequences,
clusters and splits. Only the meaning of the control token changes. Stage A,
ProGen2-base + LoRA, 3,000 steps, Gate 2 on 2,000 held-out sequences. Five
shuffles per gate.

## Result

Gains are relative perplexity improvement from the correct control token over
shuffled ones. The ceiling column is **corpus-wide**, taken from `sweep.txt`.

| run | edges (C) | ceiling | val gain | test gain | val/ceiling | gate 2 |
|---|---|---|---|---|---|---|
| `sweep_t30` | 30 | +0.095% | +0.037% | +0.033% | 0.39x | **FAILED** |
| `sweep_t45` | 45 | +0.154% | +0.260% | +0.251% | 1.69x | passed |
| `sweep_t55` | 55 | +0.257% | +0.255% | +0.238% | 0.99x | passed |
| `sweep_t75` | 75 | +0.118% | +0.121% | +0.119% | 1.03x | passed |
| `sweep_c4` | 32, 45, 56 | +0.374% | +0.249% | +0.218% | 0.67x | passed |

Test gains are on the cluster-disjoint test split, scored from the step-3,000
checkpoint; they track the validation gains within 0.03 pp everywhere, so
nothing here is an artifact of the validation split.

Every run's effect is significant: all beat all five of their shuffles, with
z between 6.7 and 20.8 against a required 3.0.

## What held

**The non-monotonic interior peak is real.** This was the sharp prediction, and
the one that could have failed. The ceiling rises from 30 C to a peak near
55 C and falls again by 75 C. Measured gain does the same: +0.037% at 30,
+0.255% at 55, +0.121% at 75. The margins are large — t55 beats t30 by
0.218 pp against a combined shuffle noise floor of 0.021 pp, and beats t75 by
0.134 pp against 0.027 pp.

This is not what intuition predicts. 30 C is the most lopsided threshold in the
sweep (89.2% of residues on one side) and the most extreme-sounding; it is also
the worst-conditioning variant, and the corpus-only bound called that in
advance, before any GPU time was spent.

**Rank correlation across the four thresholds is +0.80** (Spearman, val and
test alike; Pearson +0.79 val, +0.77 test). With n=4 a rho of 0.80 is not by
itself statistically distinguishable from chance — the evidence here is the
shape prediction above, not the correlation coefficient.

**Conditioning is not budget-limited.** `sweep_t45` reproduces the configured
binning exactly and reaches +0.260% at 3,000 steps against the 20,000-step
baseline's +0.265%. The comparison between variants is therefore a comparison
of converged conditioning, not of how far each got in 3,000 steps.

**The gate discriminates.** `sweep_t30` is the only Gate 2 failure, at 0.39x of
its own ceiling. The effect there is statistically unambiguous (z = 8.5) and
still small relative to what its corpus allows — which is exactly the
distinction the gate exists to draw.

## What did not hold

**The ceiling predicts ordering, not level.** `gain / ceiling` ranges from
0.39x to 1.69x, a 4.3x spread. A variant can land at 40% of its bound or at
170% of it. Exceeding it is permitted — it is a *unigram* bound and the model
has context — but a quantity that is undershot by 2.6x in one variant and
overshot by 1.7x in another is a predictor of rank, not a calibrated estimate
of achievable gain. Any claim framed as "the ceiling tells you what gain to
expect" is not supported by these five runs. "The ceiling tells you which
corpus variant will condition better" is.

**It does not rank across axes.** `sweep_c4` has the highest ceiling of the
five by a wide margin (+0.374%, 1.5x the 55 C threshold's) and a *lower*
measured gain than both `sweep_t45` and `sweep_t55`. Four classes raise the
bound without raising what the model extracts. So a single ceiling number does
not order variants that differ in *kind* of relabeling — moving a threshold and
adding classes are not interchangeable, and the measure as it stands cannot see
the difference. This is the most important negative result here and the obvious
next question: whether the granularity axis needs its own bound, or whether the
gap is a capacity or budget limit on the control token rather than a corpus
property.

**The peak's location is not resolved.** `sweep_t45` and `sweep_t55` are
+0.19 sigma apart on validation and +0.70 sigma on test — indistinguishable,
with t45 nominally ahead on both while the ceiling puts t55 1.7x higher. The
corpus limits the resolution available: `ogt_c` has only 47 distinct values
across 311,987 rows and an 8 C gap at 37–45 C, giving roughly nine effective
thresholds. The claim that survives is that measured gain peaks in the
*interior*, not that it peaks at 55 C specifically.

## Caveat on t45's printed gate

`sweep_t45` was launched with `--ceiling-gain 0.00172`, the within-cluster
figure the configured gate uses, while the other four carry corpus-wide
figures. `--ceiling-gain` sets only the denominator printed in that run's Gate 2
report; it does not touch training and does not touch the measured gain. Its
report therefore prints 1.51x where the corpus-wide column gives 1.69x. Every
number in the table above is computed against the corpus-wide column for all
five runs.

## Cost

Five A10 runs, concurrent: 2.6 h each, roughly $14 at $1.11/h. Probes fired
once per run at the configured step 2,500 — `--probe-every` was not passed, so
there is no per-variant probe curve in these runs.
