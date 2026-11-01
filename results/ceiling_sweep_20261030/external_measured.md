# The second corpus, and a second backbone

Date: 2026-10-31. Eight runs: five on the design-matched external corpus
(`configs/external_balanced.yaml`, screened and split 2026-10-30) at 30, 40,
55, 65 and 75 C, and three on progen2-small over the stage A corpus at 30, 55
and 75 C. Settings otherwise identical to the 2026-10-28 and 2026-10-30
batches: 3,000 steps, A10G, Gate 2 on the held-out split. Reports in `runs/`,
call ids in `runs.txt`.

All eight runs completed with exit code 0. The watcher attached to them
reported `ext_t30` and `ext_t40` as failed with "Deadline exceeded"; that was
the watcher's own poll deadline, not the runs, both of which had already
finished and returned normally.

## What this batch was for

`measured_n8.md` left one question doing all the work: `-abs(edge - 55)`, a
predictor that reads nothing from any corpus, outscored both corpus statistics
on stage A. The defence was that its constant was chosen knowing the answer and
that a corpus statistic should win where the optimum moves. These five runs
were pre-registered in `paper/draft.md` section 5.2b as the test of that, with
the three questions it had to answer fixed before the numbers landed. They are
answered below in order.

## The measurements

| edge (C) | thermo% | corpus-wide | within-cluster | val gain | test gain |
|---|---|---|---|---|---|
| 30 | 81.7% | +0.105% | +0.125% | +0.110% | +0.108% |
| 40 | 49.8% | +0.274% | +0.219% | +0.404% | **+0.428%** |
| 55 | 41.3% | +0.322% | +0.237% | +0.420% | +0.429% |
| 65 | 27.5% | +0.313% | +0.272% | **+0.449%** | +0.424% |
| 75 | 16.8% | +0.203% | +0.180% | +0.236% | +0.222% |

All five passed Gate 2, z between 8.4 and 41.2.

| predictor | rho (val) | exact p | rho (test) | exact p | Pearson (val) |
|---|---|---|---|---|---|
| corpus-wide ceiling | +0.900 | 0.042 | +0.900 | 0.042 | **+0.984** |
| within-cluster ceiling | **+1.000** | **0.008** | +0.700 | 0.117 | +0.971 |
| `-abs(edge - 55)`, no corpus at all | +0.900 | 0.042 | +0.900 | 0.042 | +0.830 |

p-values are exact one-sided permutation tests over all 5! = 120 relabelings.

## Question 1: does the within-cluster statistic predict here too?

On validation, yes, and perfectly: rho +1.000, p = 0.008, the only predictor of
the three that clears 0.05 at n=5. Its specific pre-registered prediction --
that 65 C ranks highest on this corpus, where stage A's measured optimum was
50 C -- is the measured val ordering exactly.

On test it falls to +0.700 (p = 0.117). The reason is visible in the table:
40, 55 and 65 C land at +0.428%, +0.429% and +0.424% on test, a spread of
0.005 pp against a shuffle spread of 0.0004-0.0009 pp per run. Those three are
a tie, and the rank correlation is being scored on the order within a tie. The
val/test disagreement is not evidence about the statistic; it is evidence that
n=5 with three tied points cannot resolve a ranking.

## Question 2: does the fixed-centre baseline degrade?

**No, and the test was weaker than section 5.2b claimed.** `-abs(edge - 55)`
scores +0.900 on both splits here, the same as on stage A in rank terms.

Two things went wrong with the test as designed:

The pre-registration said the two predictors "disagree on the top two and on
the ordering of 40 against 75". The first half is true; the second is not. Both
rank 40 C above 75 C. Over these five edges the two orderings differ by a
single adjacent swap, so five runs could separate them by at most one rank
unit, which is one tied pair away from nothing. That should have been checked
before the runs were launched rather than after.

More fundamentally, **the optimum did not move far enough to put the baseline
at risk.** Stage A's measured optimum is 50 C and this corpus's is 65 C. A
baseline centred at 55 sits between them and is wrong by 5 and 10 C
respectively. The corpus whose optimum would actually break it is the uncapped
external corpus, whose ceiling peaks at 35 C because of Swiss-Prot's 37 C
pile-up (`external_corpus_20261029/findings.md` Result 2). That corpus has not
been trained on.

What does separate them on this corpus is magnitude rather than rank. Pearson
is +0.984 for the corpus-wide statistic and +0.971 for the within-cluster one,
against +0.830 for the baseline. The baseline gets the ordering about as right
and the shape clearly worse: it is linear in distance from 55 by construction,
so it cannot reproduce the sharp fall at 30 C and the shallow one at 65 C. That
is a real difference and it is not the difference the test was designed around,
so it is an observation, not the pre-registered result.

**The honest summary is that the load-bearing test came back inconclusive on
its own terms.** It was built on the assumption that these two corpora's optima
are far apart, and they are not.

## Question 3: does the plateau reproduce, and does the cold end stay poor?

Yes, and this is the clearest result in the batch.

| | stage A | external |
|---|---|---|
| cold end (30 C) | +0.037% | +0.110% |
| plateau | +0.255% to +0.296% (45-65 C) | +0.404% to +0.449% (40-65 C) |
| hot end (75 C) | +0.121% | +0.236% |
| cold-to-plateau | 7.2x (55/30, val) | 3.8x (55/30, val) |

Two corpora, two growth-temperature databases, two sequence sources: an
interior plateau three to seven times the cold end, falling away at both ends.
The plateau's hot edge extends to 65 C here, where on stage A 65 C was already
inside it; its cold edge is 40 C rather than 45 C.

Measured gains are uniformly larger on this corpus -- +0.420% against +0.255%
at 55 C, 1.6x -- on a tenth of stage A's residues. The likely cause is in the
`thermo%` column: at 55 C the external corpus is 41.3% thermophilic against
stage A's 28.0%, because species-balancing a Swiss-Prot subset with sparse
thermophilic coverage leaves a hotter corpus than balancing Gosha's. So this is
not "a smaller corpus conditions better"; it is a different class balance at
the same threshold, which is the same confound section 5.3 is about.

Gain over ceiling is 1.05x to 1.47x corpus-wide here, against 0.56x to 2.01x on
stage A -- a tighter band, but on five points rather than eight.

## The second backbone

progen2-small (151M against 764M), stage A corpus, identical conditioning,
LoRA rank and token dropout; the only change is the backbone and the 1,024
token context it forces.

| edge (C) | base val | small val | ratio | base test | small test | ratio |
|---|---|---|---|---|---|---|
| 30 | +0.037% | +0.035% | 0.95x | +0.033% | +0.077% | 2.33x |
| 55 | +0.255% | +0.200% | 0.78x | +0.238% | +0.184% | 0.77x |
| 75 | +0.121% | +0.142% | 1.17x | +0.119% | +0.114% | 0.96x |

**What reproduces is the ordering: 55 > 75 > 30 on both backbones, on both
splits.** That is the threshold-choice result, and it is not a property of
progen2-base.

What does not reproduce cleanly is the size of the cold-to-plateau spread:
6.9x on base against 5.7x on small by validation, but 7.2x against 2.4x by
test, because `small_t30` measured +0.035% on validation and +0.077% on test.
That run is also the only Gate 2 failure in the batch -- it passed the
significance test at z 7.2 and failed the size test at 0.36x of its ceiling,
against a 0.50x floor, then passed on test at 0.81x. A gain that small is where
the val/test difference stops being noise around a stable quantity, so the
cold-end magnitude on this backbone should not be quoted as a number.

At the plateau the smaller model gets 0.78x the gain of the larger on both
splits. That is the expected direction -- less capacity, less use made of the
same label -- and it is two points, so it is a direction and not a slope.

**This is a capacity comparison, not an architecture-independence result.** Both
models are ProGen2 decoders with the same tokenizer and the same conditioning
scheme. A second architecture family is still untested.

## What this does to the paper

Section 5.2b can now be written, and it has to report the test as
inconclusive rather than passed:

- the within-cluster statistic predicts measured gain on a second corpus, and
  its pre-registered top-ranked threshold was right (rho +1.000 val, +0.700
  test with three tied points)
- the fixed-centre baseline does not degrade in rank, because the two corpora's
  optima are 15 C apart and both within 10 C of 55; on this evidence a cheap
  corpus statistic still does not beat a rule of thumb at ranking thresholds
- the corpus statistics do beat it on magnitude, Pearson +0.98/+0.97 against
  +0.83, which was not what the test was designed to measure
- the plateau and the cold-end penalty reproduce across corpora and across
  backbones, which is the robust finding and now rests on 13 runs

The experiment that would settle question 2 is the uncapped external corpus at
three or four thresholds spanning 30 to 60 C, where the within-cluster
statistic and `-abs(edge - 55)` genuinely disagree about the top. It needs the
Cas9 screen and a split assignment first, both CPU-only, and roughly $9-12 of
GPU. Until it is run, "no corpus statistic we tested beats a rule of thumb at
ranking label definitions" stands as the paper's own result.
