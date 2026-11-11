# The primary outcome replicates and the rank correlation does not

Date: 2026-11-10. Four runs on the uncapped corpus (`configs/external.yaml`,
Engqvist growth temperatures, no per-species cap and no species balancing) at
k = 2, 4, 6 and 9 with that corpus's own minimum-mass-maximising edges.
Pre-registered in `paper/draft.md` section 5.2g at commit `c138d31`, launched
at `67065a3`. Reports in `runs/`, call ids in `runs.txt`.

**Both pre-registered outcomes fired, and they point in opposite directions.**
The primary outcome is positive at 9.4 sigma, replicating 5.2e and 5.2f. The
secondary rank correlation collapses from +1.000 to **+0.200** on test, which
is the narrower refutation section 5.2g named in advance -- including the
specific mechanism it named, a k=4/k=6 inversion at bounds 0.053 pp apart.

So the run of perfect rank matches ends here, at four out of six
corpus-split combinations rather than four out of four.

## The measurements

| run | k | edges (C) | min mass | bound | val gain | test gain | gain/bound (test) |
|---|---|---|---|---|---|---|---|
| `gext_k2` | 2 | 35 | 49.4% | +0.148% | +0.069% | +0.072% | 0.49x |
| `gext_k4` | 4 | 29,32,37 | 18.5% | +0.204% | +0.106% | +0.070% | 0.34x |
| `gext_k6` | 6 | 27,29,31,35,37 | 10.1% | +0.257% | +0.100% | +0.062% | 0.24x |
| `gext_k9` | 9 | 26,28,29,30,31,35,37,38 | 7.4% | +0.338% | +0.119% | +0.191% | 0.57x |

Error bars are gain / z, exact. On test: 0.0055, 0.0063, 0.0100 and 0.0114 pp.
All four scored 2,000 of the 11,263 test sequences, the cap binding here as
`runs.txt` said in advance it would. All four selected step 2,500 by
validation, so the series is consistent in training budget.

**Gate 2 failed on three of the four runs, and this does not void the test.**
Every failure is on the *size* criterion, gain/ceiling under 0.50x; `gext_k4`
passed on validation and `gext_k9` on test. **All four runs are SIGNIFICANT on
both splits**, z from 6.2 to 27.2 against a required 3.0. Section 5.2g's void
condition (i) is z < 3 and no run comes near it; void condition (ii) is a
rarest class under 5% of residues and the minimum here is 7.4%. Section 5.2g
also stated in advance, following 5.2f, that a size failure does not void this
test, because the primary outcome is a difference between two runs and not a
ratio to a bound. That clause was written before these numbers existed and is
the reason the result is reportable.

**On `gext_k6`'s checkpoint file.** Its test report was scored from
`epoch=0-step=2500-v1.ckpt`, a suffixed duplicate, where the other three used
un-suffixed files. Checked rather than assumed benign: the provenance line
records it as the validation-selected checkpoint, and `gext_k6` has the
*shortest* wall clock of the four at 7,722s against 7,725 / 7,980 / 7,862, so
there is no extra training behind it. It matters because `gext_k6` is one of
the two runs whose inversion drives the rank result.

## The pre-registered primary outcome: positive

| split | k=2 | k=9 | difference | sigma | sign |
|---|---|---|---|---|---|
| validation | +0.069% | +0.119% | **+0.050 pp** | 5.6 | **positive** |
| test | +0.072% | +0.191% | **+0.119 pp** | 9.4 | **positive** |

The prediction on record -- positive beyond two standard deviations -- is met
on both splits. The bound rises 2.28x across that span and measured gain rises
2.65x on test. **The resolution claim survives on the corpus we did not
build**, and the null that would have made our sampling design load-bearing
for the paper's one positive pre-registered result did not happen.

## The pre-registered secondary outcome: the rank correlation fails

| split | rho vs bound | exact p | 5.2e/5.2f comparison |
|---|---|---|---|
| validation | +0.800 | 0.167 | +1.000, p = 0.042 |
| test | **+0.200** | **0.458** | +1.000, p = 0.042 |

Exact one-sided permutation p over all 24 relabelings. Neither split reaches
significance and the test split is barely above zero. Section 5.2g
pre-registered exactly this: "a positive primary outcome with a Spearman below
+1.000, which k=4 and k=6 could deliver at bounds 0.053 pp apart". That is
what happened. It did not require any outcome to be reinterpreted after the
fact, which is the only reason this paragraph is worth anything.

## Why: the bound's increments are anti-predictive here

| step | bound rise | test gain | sigma |
|---|---|---|---|
| k=2 -> k=4 | +0.056 pp | -0.002 pp | -0.2 |
| k=4 -> k=6 | +0.053 pp | -0.008 pp | -0.7 |
| k=2 -> k=6 | **+0.109 pp** | **-0.010 pp** | **-0.9** |
| k=6 -> k=9 | **+0.081 pp** | **+0.129 pp** | **8.5** |

**The only step that pays is k=6 to k=9, and it is the step with the smaller
bound increment.** The bound rises half again as much from k=2 to k=6 and buys
nothing measurable -- the point estimate goes the wrong way. Three of the four
runs are statistically indistinguishable from one another: every pairwise
difference among k=2, k=4 and k=6 is under 0.7 sigma.

That is the honest shape of the failure, and it is narrower than "the bound
mis-ranks". The bound separates the extremes correctly and carries **no**
information about the intermediate granularities, which are tied within error
while the bound rises 1.7x across them. A rank correlation over four points,
three of them tied, is close to uninformative, and +0.200 should be read as
"the ordering among the tied runs is noise" rather than as a measured
disagreement.

## A mechanism, offered as post-hoc and testable

Not pre-registered. Found by inspecting the class structure after the fact,
and reported as a hypothesis.

**k=9 is the first of these four partitions to separate the 37 C spike from
the thermophilic tail.**

| k | top class | holds |
|---|---|---|
| 2 | [35, inf) | 37 C spike **and** all 18,922 sequences above 38 C |
| 4 | [37, inf) | 37 C spike **and** all of them |
| 6 | [37, inf) | 37 C spike **and** all of them |
| 9 | [37, 38) and [38, inf) | **split**: 48,451 at the spike, 19,702 above it |

At k = 2, 4 and 6 every thermophile in the corpus shares a class with the
Swiss-Prot 37 C pile. At k = 9 the edge at 38 finally isolates them. So the
refinement that paid was not refinement in general but one specific cut, and
the bound did not know it was the important one: it scored the k=2-to-k=6
refinements higher in aggregate and those refinements only subdivided the
mesophilic pile.

This is consistent with section 5.2g's advance reasoning that resolution
concentrated inside one band is a different thing from resolution across the
range, and it sharpens it: on this corpus the bound's increments measure how
finely the *mesophiles* are cut, which is not what the model can use.

**It makes a cheap prediction, and the right next step is to test it rather
than to publish it.** The partition `{30, 37, 38}` has a bound of +0.2550%,
within 0.0016 pp of `gext_k6`'s +0.2566%, a smallest class of 9.9% of
residues, and only four classes instead of six. If the hypothesis is right it
should measure near `gext_k9`'s +0.191% rather than `gext_k6`'s +0.062%,
despite a bound matched to k=6's. That is one run, roughly $3.50, and it is a
matched-bound placement test: same bound, different cut, and the hypothesis
predicts a 3x difference in measured gain. Until it is run the mechanism stays
a reading of class tables.

## What changes in the paper

1. **Section 5.2g reports a split result**, both halves pre-registered: the
   primary outcome replicates at 9.4 sigma, the rank correlation does not.
2. **Section 6.2's claim must be qualified, not withdrawn.** "The statistic
   ranks resolution" holds for the coarse comparison on three corpora and
   three for three on the k=2-versus-k=9 contrast. The perfect-rank-match
   claim narrows to four of six corpus-split combinations, and gains the
   condition that it has only held where refinement is spread across the
   temperature range rather than concentrated in one band.
3. **Section 5.1b's procedure needs a caveat.** "Take the largest k whose
   smallest class stays trainable" still points the right way here -- k=9 is
   the best of the four. But the reachability table ranked k=4 and k=6 above
   k=2 and they measured no better, so the table orders granularities more
   confidently than the measurements support.
4. **`gain / bound` is confirmed as not a band, for the third time.** It is
   0.24x to 0.57x here, 0.51x-0.66x on `pretrain` and 1.13x-1.58x on
   `external_balanced`. The project-wide range is now 0.24x to 1.58x.
5. **The three-corpus framing stands.** Nothing here touches 5.1, 5.2b, 5.2c,
   5.2d or the Cas9 case.

## What this does not license

- Any claim that the bound ranks granularity *finely*. On this corpus it does
  not, and that is now measured rather than assumed.
- Any claim that the 38 C cut is the mechanism. It is a post-hoc reading of
  class tables with a named, unrun test.
- Any retreat to calling this corpus a special case. Section 5.2g gave up that
  move in advance and the primary outcome did not need it.
- Any efficiency claim from `gain / bound`.
- Any statement about downstream thermostability prediction. Nothing here is
  an assay.
