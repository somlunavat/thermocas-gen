# Eight thresholds: the corpus-wide ceiling does not survive, the within-cluster one does

Date: 2026-10-31. Raw per-run reports in `runs/`; call ids and settings in
`runs.txt`. Four new runs (35, 50, 65, 70 C) added to the four of 2026-10-28
(30, 45, 55, 75 C), identical settings throughout.

## Why this was run

`results/ceiling_sweep_20261027/measured.md` reported Spearman +0.80 between
the corpus-wide ceiling and measured conditioning gain across four thresholds,
and said in as many words that with n=4 the coefficient was not itself
significant and the evidence was the shape. These four runs double the sample.

## The headline number does not survive

| edge (C) | corpus-wide ceiling | within-cluster ceiling | val gain | test gain |
|---|---|---|---|---|
| 30 | +0.095% | +0.066% | +0.037% | +0.033% |
| 35 | +0.198% | +0.114% | +0.104% | +0.087% |
| 45 | +0.154% | +0.172% | +0.260% | +0.251% |
| 50 | +0.190% | +0.168% | **+0.296%** | **+0.270%** |
| 55 | +0.257% | +0.176% | +0.255% | +0.238% |
| 65 | +0.184% | +0.140% | +0.282% | +0.251% |
| 70 | +0.161% | +0.106% | +0.160% | +0.152% |
| 75 | +0.118% | +0.081% | +0.121% | +0.119% |

All four new runs passed Gate 2 with z between 8.3 and 19.1, so none of this is
noise in the measured column.

| predictor | rho (val) | exact p | rho (test) | exact p | Pearson (val) |
|---|---|---|---|---|---|
| corpus-wide ceiling | +0.381 | 0.180 | +0.359 | 0.190 | +0.615 |
| **within-cluster ceiling** | **+0.714** | **0.029** | **+0.743** | **0.021** | **+0.907** |
| `-abs(edge - 55)`, no corpus at all | +0.843 | 0.007 | +0.849 | 0.006 | +0.906 |

p-values are exact one-sided permutation tests over all 8! = 40,320
relabelings, not asymptotic approximations.

**The corpus-wide ceiling falls from +0.80 to +0.38 and is not significant.**
The same four thresholds reproduce +0.800 exactly when the other four are
dropped, so the earlier figure was not a mistake in computation; it was a
property of which four thresholds had been run. This is the result of these
four runs and it supersedes the headline of `measured.md`.

## What broke it, and why that is informative

The single largest miss is 35 C: seventh of eight by ceiling, second of eight
by measured gain. It is also the most balanced threshold in the corpus
(H(bin) = 0.679 of a possible 0.693, 58.4% of residues in the hot class).

That is the "it is only class balance" objection landing. `findings.md` Result 4
argued the objection fails because the ceiling's *peak* sits 15-20 C above the
most balanced threshold. That is still true, and it is not the whole story: the
objection does not need to move the peak to do damage, it only needs to inflate
the ceiling where the classes happen to be even. At 35 C the classes are nearly
even and the label is semantically mush -- the hot side is mostly 37 C
mesophiles -- so the residue histogram sees a large difference that the model
cannot turn into conditioning.

Conditioning on family removes most of this. The within-cluster ceiling asks
what the label adds *given the homologs*, which is the question the model
actually faces, and at 35 C it reads +0.114% against the corpus-wide +0.198% --
ranked third of eight, close to the measured rank of second.

## The within-cluster ceiling is the better predictor, and it was pre-specified

This matters for how much the result is worth: within-cluster is not a quantity
picked after seeing the data. It is what `configs/train.yaml`'s
`conditioning_gate/ceiling_gain` has been set to since 2026-10-26, and
`13_conditioning_ceiling.py` prints it as "the token's value given the family".
The corpus-wide column was used for the sweep comparisons because it is
available at every threshold without a cluster-count caveat, and that choice
now looks like the error.

At rho +0.71/+0.74 with p = 0.029/0.021 it clears significance at n=8 on both
splits. Pearson is +0.907, which is higher than the rank correlation and worth
noting: above 45 C the relationship is close to a fixed multiple.

| edge (C) | within-cluster | val gain | gain/ceiling |
|---|---|---|---|
| 30 | +0.066% | +0.037% | 0.56x |
| 35 | +0.114% | +0.104% | 0.91x |
| 45 | +0.172% | +0.260% | 1.51x |
| 50 | +0.168% | +0.296% | 1.76x |
| 55 | +0.176% | +0.255% | 1.45x |
| 65 | +0.140% | +0.282% | 2.01x |
| 70 | +0.106% | +0.160% | 1.51x |
| 75 | +0.081% | +0.121% | 1.49x |

The six thresholds at and above 45 C span 1.45x to 2.01x -- a factor of 1.4,
against 4.3x for the corpus-wide ratios in `measured.md`. The two cold
thresholds sit below 1x. Splitting the eight that way was decided after seeing
the table, so it is an observation and not a result.

## The problem this does not solve

`-abs(edge - 55)` -- a predictor that reads nothing from the corpus and is a
function of the threshold alone -- scores +0.84, above both ceilings. On this
corpus, a cheap corpus statistic does not beat "pick the middle of the
temperature range".

The defence is real but it is not yet evidence. That baseline has a constant in
it, 55, and the constant was chosen knowing the answer. Nothing computed from a
new corpus tells you where to centre it, and `findings.md` Result 2 already
shows the optimum is not always near 55: on the uncapped external corpus the
ceiling peaks at 35 C because of Swiss-Prot's 37 C pile-up. A corpus statistic
that moves with the corpus should beat a fixed-centre baseline exactly where
the optimum moves.

**That is now the most important open question in this project, and the five
external-corpus runs launched 2026-10-31 are the test.** If the within-cluster
ceiling tracks measured gain on a corpus whose optimum sits elsewhere, the
measure earns its keep. If `-abs(edge - 55)` wins there too, it does not.

## The actionable claim weakens a lot

`findings.md` Result 3 put the cost of the 45 C convention at 1.26x-1.67x in
available signal, from the ceiling. Measured, it is much smaller:

- measured optimum is **50 C at +0.296%**, against 45 C at +0.260%: **1.14x**
- the gap is roughly 2 sigma on the shuffle spread, so 50 C over 45 C is
  marginal, and 65 C at +0.282% is within noise of both
- measured gain is effectively a **plateau from 45 to 65 C** (+0.255% to
  +0.296%), falling away sharply outside it: +0.104% at 35 and +0.160% at 70

So the honest version of the practical claim is not "retune the threshold and
gain 1.3-1.7x". It is:

1. **Threshold choice matters a great deal.** Cold thresholds mean 3.9x less
   measured gain than the plateau (+0.070% at 30-35 C against +0.273% at
   45-65 C). That is the robust finding and n=8 strengthens it.
2. **The conventional 45 C sits at the low edge of the plateau**, and retuning
   within the plateau is worth about 1.14x, not 1.67x. The convention is nearly
   optimal on this corpus.
3. **The ceiling overstated the available improvement** by predicting a 1.67x
   gap between 45 and 55 C where the measured gap is 1.02x in the other
   direction.

## What this does to the paper

`paper/draft.md` sections 5.1, 5.4 and 6.1 are written against the n=4 result
and have to be rewritten. The claim "a corpus-only bound ranks label
definitions by measured conditioning gain" is supportable only for the
within-cluster aggregation, only at p ~ 0.02-0.03, and only until the
fixed-centre baseline is ruled out. The 1.3-1.7x number should not appear.

What survives unchanged: the interior peak and its size (now 8 thresholds, 3.9x
between the cold end and the plateau); the cross-corpus agreement of the
ceiling *curve* in `findings.md`; the sampling-design result; and the Cas9
negative case, which rests on Gate 1 and the leakage test rather than on the
ceiling's predictive power.
