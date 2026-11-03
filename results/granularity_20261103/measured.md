# The prediction was wrong: measured gain tracks the bound across granularity

Date: 2026-11-03. Four runs on the primary corpus (`configs/pretrain.yaml`) at
k = 2, 4, 6 and 9 classes, pre-registered in `paper/draft.md` section 5.2e at
commit `291d576`, launched at `f6757e3`. Edges maximise the smallest class's
share of residues. Reports in `runs/`, call ids in `runs.txt`.

**I predicted the primary outcome would be negative or within two standard
deviations of zero. It is positive at 9.4 sigma on test and 10.6 sigma on
validation.** The prediction is refuted, and this is the first pre-registered
test in the project the corpus statistic has won.

## The measurements

| run | k | edges (C) | min class | bound | val gain | test gain | gain/bound | Gate 2 |
|---|---|---|---|---|---|---|---|---|
| `gran_k2` | 2 | 37 | 41.6% | +0.198% | +0.103% | +0.110% | 0.55x | passed |
| `gran_k4` | 4 | 32,45,60 | 19.5% | +0.372% | +0.277% | +0.245% | 0.66x | passed |
| `gran_k6` | 6 | 30,32,45,55,67 | 10.8% | +0.488% | +0.320% | +0.248% | 0.51x | passed |
| `gran_k9` | 9 | 29,32,45,49,53,56,67,83 | 5.5% | +0.496% | +0.361% | +0.273% | 0.55x | passed |

All four cleared both Gate 2 criteria; z between 13.0 and 16.2 on validation,
8.6 and 23.5 on test. Error bars are gain / z, exact, because the reports print
the shuffle spread in absolute perplexity and the gain as a percentage.

Test is scored at the checkpoint validation selected, not the last step
(`scripts/11_train.py:843`). For `gran_k9` that was step 2,000 rather than
step 3,000, so the highest-gaining run is also the one that trained least --
which cuts against the result being a training-budget artifact, not for it.
All four scored 2,000 of the 16,952 test sequences, the same cap the
2026-10-27 batch used, so those runs are directly comparable.

## The pre-registered primary outcome

Section 5.2e fixed the sign of (test gain at k=9) - (test gain at k=2) before
launch. The bound rises 2.5x across that span.

| split | k=2 | k=9 | difference | sigma | sign |
|---|---|---|---|---|---|
| validation | +0.103% | +0.361% | **+0.258 pp** | 10.6 | **positive** |
| test | +0.110% | +0.273% | **+0.163 pp** | 9.4 | **positive** |

Neither void condition applies: every run cleared the significance criterion
with z >= 8.6, and every class holds at least 5.5% of the corpus's residues by
construction. The refuting outcome named in 5.2e -- "a positive difference
exceeding two standard deviations" -- is what happened, four times over.

**Measured gain rises 2.5x as the bound rises 2.5x.** `gain / bound` is
0.55x, 0.66x, 0.51x and 0.55x: close to constant across a 2.5x range of
bounds, which is the behaviour a usable scale should have and which no earlier
batch in this project produced.

## The secondary analysis

| predictor | rho (val) | exact p | rho (test) | exact p |
|---|---|---|---|---|
| reachable bound | **+1.000** | **0.042** | **+1.000** | **0.042** |

Exact one-sided permutation p over all 4! = 24 relabelings. 0.042 is the
smallest attainable at n=4 and only a perfect rank match reaches it. The bound
reaches it on both splits.

As pre-registered, this is reported for the sign and nothing more: n=4 cannot
establish agreement, and a perfect match on four points is what four points
can show at best.

## Where the gain actually saturates, and it is not where the bound does

| step | val | test |
|---|---|---|
| k=2 -> k=4 | +0.174 pp (8.8 sigma) | **+0.135 pp (5.8 sigma)** |
| k=4 -> k=6 | +0.043 pp (1.6 sigma) | +0.003 pp (0.1 sigma) |
| k=6 -> k=9 | +0.041 pp (1.4 sigma) | +0.025 pp (1.2 sigma) |
| k=4 -> k=9 | +0.084 pp (2.9 sigma) | +0.028 pp (1.2 sigma) |

**The only step that is large on both splits is two classes to four.**
Everything past four classes is within noise on test, and k=4 to k=9 is 2.9
sigma on validation against 1.2 on test -- a disagreement between splits,
which is a reason to quote the test figure and not the validation one.

So the recommendation section 5.1b derived from the bound's own saturation --
four to six classes rather than as many as the database records -- is now
measured rather than inferred, and it is the *lower* end of that range that
the measurement supports. Four classes captures +0.245% of the +0.273% that
nine classes reach, and the difference is 1.2 sigma.

## The result that matters more: the statistic ranks resolution, not placement

The two-class run here uses the mutual-information-optimal edge, 37 C, which
splits the corpus 41.6/58.4. `sweep_t45` on the same corpus and the same
splits uses the conventional 45 C edge.

| k=2 edge | how chosen | bound | test gain |
|---|---|---|---|
| 37 C | maximises the bound | +0.198% | +0.110% |
| 45 C | biological convention | +0.154% | **+0.251%** |

**The statistic's own preferred threshold measures 6.4 sigma worse than the
conventional one.** It prefers 37 C because mutual information is maximised at
an even marginal, and 37 C is where this corpus splits evenly -- the balance
artifact of section 5.1b and 5.2d, now measured directly rather than inferred
from a rank correlation.

Put beside the result above, this is a sharper statement than section 6.2's
current one:

- **Along the granularity axis, with the partition at each k chosen to
  maximise class balance, the statistic ranks perfectly** (rho +1.000, both
  splits).
- **Along the threshold axis, at fixed k=2, it ranks no better than chance and
  its own optimum is a bad choice** (rho +0.000 corpus-wide in section 5.2d;
  6.4 sigma worse than convention here).

The mechanism is the same in both halves. Moving a threshold changes class
balance and biological meaning together, and mutual information responds to
the balance; holding "best-balanced partition at this k" fixed removes balance
as the free variable, and what is left varying is resolution, which the
statistic reads correctly. Section 6.2 said the measure "does not rank across
granularities" and called the granularity axis open. It ranks granularity and
not placement, and that is the opposite of what we concluded from
`sweep_c4` alone.

## Why `sweep_c4` misled us

`sweep_c4` used equal-residue-mass quantile edges, 32,45,56, bound +0.374%,
and reached +0.218% on test -- below `sweep_t45`'s +0.251% at a 2.4x lower
bound. That single comparison is what section 6.2 was built on and what
motivated the prediction recorded in 5.2e.

It was a comparison against a *badly placed* two-class run that happened to be
well placed. `sweep_t45`'s 45 C edge is good for reasons the statistic cannot
see, so using it as the k=2 anchor made granularity look useless. With the
k=2 anchor chosen the same way as every other point in the series --
maximise the minimum class -- the series is monotone and the rank correlation
is perfect.

The lesson is narrow and worth recording: a granularity series whose members
are chosen by different rules is not a granularity series. Both the earlier
conclusion and the prediction built on it were wrong for that reason.

## What changes in the paper

1. **Section 5.2e reports a positive, pre-registered result.** The reachable
   bound predicts measured gain across granularity, rho +1.000 on both splits,
   with the direction fixed in advance and the prediction on record having
   been the opposite.
2. **Section 5.1b's "four to six classes" becomes a measured recommendation**,
   and narrows to four: the 2->4 step is 5.8 sigma on test and every later
   step is within noise.
3. **Section 6.2 must be rewritten.** Its claim is refuted by a better-designed
   version of its own experiment. The replacement claim -- resolution yes,
   placement no -- is stronger, has a mechanism, and has both halves measured.
4. **Section 5.2d stands exactly as it is.** The statistic still loses to
   `-abs(edge - 55)` on threshold placement, at rho +0.000 against +1.000.
   Nothing here rescues that, and the two results are about different axes.
5. **The abstract's framing holds.** The envelope is a scale; the reachable
   portion of it is 67-75%; a two-class threshold reaches 28-30%; and a model
   does move with the bound when the bound is varied along the one axis that
   does not confound balance with meaning.

## What this does not license

- Any claim that the statistic chooses thresholds. It does not, and section
  5.2d measured that against a trivial baseline.
- Any claim beyond k=9 or on another corpus. One corpus, one backbone, four
  points, 3,000 steps.
- Any claim that gain/bound near-constant at 0.51-0.66x is a law. It is four
  numbers on one corpus, and the same ratio spans 0.31x to 1.56x across the
  project's sixteen other runs.
- Any statement about downstream thermostability prediction. Nothing here is
  an assay.
