# The fixed-centre baseline wins, and the bound points the wrong way

Date: 2026-11-01. Four runs on the uncapped external corpus
(`configs/external.yaml`, screened and re-split 2026-11-01) at 30, 35, 45 and
60 C. 3,000 steps, A10G, Gate 2 on 5,000 validation sequences and the full
11,263-sequence test split. Reports in `runs/`, call ids in `runs.txt`.

This was the pre-registered test of `-abs(edge - 55)`, written into
`paper/draft.md` section 5.2d before the runs were launched. **It came back
negative, decisively, on both splits.**

## The measurements

| edge (C) | thermo% | H(bin) | corpus-wide bound | within-cluster bound | val gain | test gain | gain/bound | Gate 2 |
|---|---|---|---|---|---|---|---|---|
| 30 | 68.8% | 0.6205 | +0.113% | +0.100% | +0.035% | +0.038% | 0.31x | **failed (size)** |
| 35 | 49.4% | 0.6931 | **+0.148%** | **+0.119%** | +0.065% | +0.061% | 0.44x | **failed (size)** |
| 45 | 9.2% | 0.3080 | +0.100% | +0.110% | +0.120% | **+0.137%** | 1.20x | passed |
| 60 | 6.4% | 0.2386 | +0.115% | +0.107% | **+0.148%** | +0.119% | 1.29x | passed |

Every run cleared the significance criterion by a wide margin: z between 10.9
and 40.7 on validation, 10.9 and 20.5 on test. The two Gate 2 failures are
failures of the *size* criterion -- the model reached 0.31x and 0.44x of the
available bound against a 0.50x floor -- and that is itself part of the result.

## The pre-registered primary outcome

Section 5.2d fixed this in advance: the sign of (gain at 35) - (gain at 60).
Both corpus statistics rank 35 C first and 60 C third; the baseline ranks them
the other way.

| split | gain at 35 | gain at 60 | difference | sigma | sign |
|---|---|---|---|---|---|
| validation | +0.065% | +0.148% | **-0.083 pp** | 6.6 | negative |
| test | +0.061% | +0.119% | **-0.058 pp** | 7.1 | negative |

**Negative on both splits, at 6.6 and 7.1 sigma.** By the rule written down
before the runs: the baseline wins where it was most exposed, section 5.1c
stands as a negative result, and the paper must state that no corpus statistic
tested here chooses label definitions better than distance from the middle of
the temperature range.

### The void conditions do not apply, and the letter of them is what counts

Both were written in 5.2d before the numbers existed, and neither is met.

(i) *All four gains within two standard deviations of one another.* They span
0.035% to 0.148%, a range of 0.113 pp against a largest two-sigma of 0.025 pp.
Not void.

(ii) *Any run failing Gate 2's significance criterion.* None did; the smallest
z is 10.9. `unc_t30` and `unc_t35` failed Gate 2, but on the **size**
criterion, which 5.2d did not name. It would be easy to read "failed Gate 2"
as satisfying condition (ii) now that the result is unfavourable, and that
reading is not available: the condition says *significance*, and it says so
because a gain indistinguishable from zero cannot rank anything. These gains
are distinguishable from zero at 10 to 40 sigma. The test is valid and it is
negative.

## The secondary analysis

| predictor | rho (val) | exact p | rho (test) | exact p |
|---|---|---|---|---|
| corpus-wide bound | **+0.000** | 0.542 | -0.400 | 0.792 |
| within-cluster bound | +0.200 | 0.458 | +0.400 | 0.375 |
| `-abs(edge - 55)`, no corpus at all | **+1.000** | **0.042** | +0.800 | 0.167 |

Exact one-sided permutation p over all 4! = 24 relabelings; 0.042 is the
smallest attainable at n=4 and only a perfect rank match reaches it. The
baseline reaches it. The corpus-wide statistic scores exactly zero on
validation and *negative* on test. Family conditioning does not rescue it: at
+0.200 and +0.400 the within-cluster form is better than useless and nowhere
near the baseline.

As pre-registered, the secondary analysis cannot separate the two aggregations
from each other at n=4 and is not read as doing so.

## Why: the balance artifact survives family conditioning

This is the mechanism, and it is the same one section 5.1b identified on the
primary corpus -- but worse, because here it is not corrected by conditioning
on family.

| | Spearman with H(bin) |
|---|---|
| corpus-wide bound | +0.400 |
| within-cluster bound | +0.400 |
| **measured gain (val)** | **-0.800** |
| **measured gain (test)** | **-0.600** |

Pearson between H(bin) and measured validation gain is **-0.926**.

Mutual information is maximised at an even marginal, so both bounds rise with
balance. On this corpus measured gain does the opposite: the more balanced the
threshold, the *less* the model gains. The two quantities are anti-correlated
through class balance, and that is sufficient to explain a rank correlation of
zero.

The reason is the 37 C pile-up. At 30 and 35 C the corpus is 68.8% and 49.4%
thermophilic, but the hot class is overwhelmingly Swiss-Prot's clinical and
model organisms annotated at exactly 37 C. The label is near-perfectly
balanced and semantically empty: it separates "37 C" from "not 37 C" rather
than thermophiles from mesophiles. A residue histogram sees a large
composition difference between two arbitrary halves of a corpus; the model,
which also has context and homologs, finds almost nothing usable -- 0.31x and
0.44x of the bound, the two worst ratios in the project and the reason both
runs failed Gate 2.

At 45 and 60 C the hot class is 9.2% and 6.4% and is genuinely thermophilic.
The bound is *lower* there, because the marginal is lopsided, and measured
gain is higher -- 1.20x and 1.29x of the bound.

**Section 5.1b claimed that conditioning on family removes most of the balance
artifact. On this corpus it does not.** Within a cluster the class masses are
whatever that cluster's organisms happen to be, and when half the corpus is
37 C organisms, a great many clusters are internally balanced at a 35 C
threshold for the same empty reason. The claim in 5.1b has to be narrowed to
the two designed corpora, where per-species capping and species balancing
break up exactly this kind of pile-up.

## What this does to the paper

The open problem of section 5.1c closes against the measure. Specifically:

1. **"A corpus statistic ranks label definitions better than a rule of thumb"
   is now refuted**, not merely unsupported. On the one corpus built to
   separate them, the rule of thumb scored +1.000 and the corpus-wide
   statistic +0.000.
2. **The within-cluster rescue is corpus-dependent.** It works on the two
   species-balanced corpora and not on the natural one. Section 5.1b must say
   so, and the abstract's claim that the statistic "predicts this ordering
   when conditioned on sequence family" must be qualified by corpus design.
3. **The bound's failure mode is now characterised rather than suspected.** It
   is class balance, it is predictable in advance from H(bin), and it is
   strongest exactly where a temperature database piles up at one value --
   which is the normal condition of an unselected corpus, not an edge case.
4. **What survives is unchanged and still worth publishing.** Label definition
   is worth 3.9x on the primary corpus, 3.8x on the design-matched one and
   **4.2x here** (+0.035% at 30 C to +0.148% at 60 C), across three corpora and
   two backbones. The sampling-design result (5.3) is strengthened: the
   location of the measured optimum moved from 50 C to 65 C to 60 C with the
   corpus, and the cold end is poor everywhere.
5. **An honest reframing is available and should be taken.** The paper's
   question was whether a cheap corpus statistic can serve as the denominator
   this literature lacks. The answer these runs give is: *not this one, and
   here is exactly why, and here is the diagnostic -- H(bin) -- that tells you
   when it will fail.* A measurement paper that reports a measure failing a
   pre-registered test, with the mechanism identified, is a more useful paper
   than one that reports a +0.71 rank correlation on eight points.

What should not happen: quietly re-scoping the claim to "the statistic works
on species-balanced corpora". It does, on two of them, with n=8 and n=5 and a
baseline that beats it on both. That sentence would be true and misleading.
