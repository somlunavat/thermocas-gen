# Both predictions met: it is the function, not the temperature -- on test only

Date: 2026-11-15. Two runs on the uncapped external corpus
(`configs/external.yaml`), both holding k = 6 and moving `gext_k6`'s top edge
rather than adding one. Pre-registered in `paper/draft.md` section 5.2i at
commit `7cd763f`, launched at `091f8dc`. Reports in `runs/`, call ids in
`runs.txt`.

**Both pre-registered outcomes are met on test.** Moving one edge from 37 to
38 degrees C, with class count held at six, is worth +0.100 pp at 8.6 sigma --
75% of the entire k6 -> k7 jump that section 5.2h attributed to the same cut.
Moving it to 45 instead gives +0.103 pp, which differs from the 38 result by
0.2 sigma.

**On validation neither difference clears two standard deviations** (1.6 and
1.4 sigma). The primary outcome was fixed on test in advance and is met there,
but the split disagreement is real, it is the second time this corpus has
produced one, and it is reported in full below rather than left to a footnote.

## The measurements

| run | k | edges (C) | min class | bound | val gain | test gain | gain/bound (test) |
|---|---|---|---|---|---|---|---|
| `gext_k6` (reference) | 6 | 27,29,31,35,37 | 10.1% | +0.2566% | +0.100% | +0.062% | 0.24x |
| `gmod_e38` | 6 | 27,29,31,35,**38** | 9.93% | +0.2837% | +0.122% | **+0.162%** | 0.57x |
| `gmod_e45` | 6 | 27,29,31,35,**45** | 9.24% | +0.2853% | +0.120% | **+0.165%** | 0.58x |

Both new runs are significant on both splits -- z 11.5 and 11.1 on validation,
27.3 and 13.0 on test -- so void condition (i) does not apply, and both rarest
classes clear the 5% floor, so (ii) does not either. Error bars are gain / z.

Both new runs **failed Gate 2 on validation** (0.43x and 0.42x against the
0.50x size criterion) and **passed it on test** (0.57x and 0.58x). Section
5.2i had already recorded that a size failure does not void this test, both
outcomes being differences between runs. Exit code 3 on both calls is that
size failure, not a crash; all four reports were written.

Both selected `epoch=0-step=2500.ckpt` by validation, the same step `gext_k6`
selected, so the three-run comparison is at a common training budget. Wall
clocks were 7,849s and 7,787s against `gext_k6`'s 7,722s. Test was scored on
the same 2,000 of 11,263 sequences as every other run on this corpus.

## The pre-registered primary outcome

Section 5.2i fixed the sign of (`gmod_e38` test gain) - (`gext_k6` test gain)
before launch, predicting positive beyond two standard deviations and above
+0.09 pp.

| split | `gext_k6` | `gmod_e38` | difference | sigma | sign |
|---|---|---|---|---|---|
| test | +0.062% | +0.162% | **+0.100 pp** | **8.6** | **positive** |
| validation | +0.100% | +0.122% | +0.022 pp | 1.6 | positive |

**On test the prediction is met on both of its terms**: positive at 8.6 sigma,
and +0.100 pp against the +0.09 pp threshold named in advance. The predicted
figure was "two thirds of 5.2h's +0.133 pp"; the measured value is 75% of it.

**On validation it is not met.** +0.022 pp at 1.6 sigma is within noise.

## The pre-registered secondary outcome

| comparison | split | difference | sigma |
|---|---|---|---|
| `gmod_e45` - `gmod_e38` | test | +0.003 pp | **0.2** |
| `gmod_e45` - `gmod_e38` | validation | -0.002 pp | 0.1 |
| `gmod_e45` - `gext_k6` | test | **+0.103 pp** | 6.4 |
| `gmod_e45` - `gext_k6` | validation | +0.020 pp | 1.4 |

Both terms of the secondary prediction are met on test: the two runs agree to
0.2 sigma, and both exceed `gext_k6` by more than +0.09 pp. Seven degrees of
edge placement, spanning most of the distance to the conventional
mesophile-thermophile boundary, changes the measured gain by 0.003 pp.

**What this licenses, stated as 5.2i required.** Agreement within noise is not
equivalence. The claim is the negative one: **we found no evidence that the
specific value 38 degrees C matters**, and that is the claim 5.2h said it
could not make, because both of its runs used that edge. The positive
description of the mechanism is the function -- remove the modal temperature
from the top class -- and it now has a test behind it rather than a single
coincidence of edge value.

None of the three refuting outcomes named in 5.2i occurred. In particular
`gmod_e38` did not land within noise of `gext_k6` on test, so 5.2h is not
overturned and section 6.2's restructuring stands.

## The split disagreement, which is the real limitation here

Across all eight runs on this corpus the two splits behave differently, and
not only in these two runs:

| run | top class | modal fraction of top class | val gain | test gain |
|---|---|---|---|---|
| `gext_k2` | `[35, inf)` | 55.2% | +0.069% | +0.072% |
| `gext_k4` | `[37, inf)` | 73.3% | +0.106% | +0.070% |
| `gext_k6` | `[37, inf)` | 73.3% | +0.100% | +0.062% |
| `gext_k9` | `[38, inf)` | 7.4% | +0.119% | +0.191% |
| `gplc_nest` | `[38, inf)` | 7.4% | +0.144% | +0.195% |
| `gplc_match` | `[38, inf)` | 7.4% | +0.124% | +0.157% |
| `gmod_e38` | `[38, inf)` | 7.4% | +0.122% | +0.162% |
| `gmod_e45` | `[45, inf)` | 8.0% | +0.120% | +0.165% |

**Validation spans +0.069% to +0.144%, a factor of 2.1. Test spans +0.062% to
+0.195%, a factor of 3.1, and the ordering is far cleaner.** Every run whose
top class is free of the modal temperature sits at +0.157% or above on test;
the three whose top class is mode-dominated sit at +0.072% or below -- a gap
of 0.085 pp with nothing in it. On
validation the two groups overlap: `gext_k4`, whose top class is 73.3% one
temperature, measures +0.106% -- above `gext_k6`'s +0.100% and within two
standard deviations of `gmod_e45`'s +0.120%.

We do not know why the validation split compresses, and we are not going to
invent a reason. Three things are worth recording:

1. **It is not new.** Section 5.2g's rank correlation was +0.800 on validation
   and +0.200 on test, in the other direction. This corpus has now produced a
   split disagreement twice, and no other corpus in the project has produced
   one at all.
2. **Test is the figure we quote**, as it was in 5.2e and 5.2g, and for the
   same reason: it is the held-out split, it is cluster-disjoint from train,
   and the validation split is what selected the checkpoint being scored.
3. **The size of the primary effect does not depend on which split you take
   seriously -- only its significance does.** Both splits put `gmod_e38`
   above `gext_k6`, and both put the two new runs within noise of each other.
   The sign structure replicates across splits; the margins do not.

An honest summary of the primary outcome is therefore: **the effect is
positive on both splits, large and unambiguous on test, and within noise on
validation.**

## The mode/tail diagnostic, recomputed on residues

Section 5.1b's pre-check compares the modal temperature's own mass to the mass
above it. The figures previously recorded (0.47x, 0.28x, 2.46x) were computed
on sequence counts. On residues, the basis every other quantity in this paper
uses -- the 5% floor, the minimum class share, the bound itself -- they are:

| corpus | mode | mode's share of residues | share above the mode | mode/tail |
|---|---|---|---|---|
| `pretrain` | 30 C | 29.64% | 59.55% | **0.50x** |
| `external_balanced` | 37 C | 15.53% | 49.85% | **0.31x** |
| `external` | 37 C | 27.24% | 9.93% | **2.74x** |

Same ordering, same side of 1.0 in all three cases, so the diagnostic's 3/3
retrodiction is unaffected by the basis. The residue figures are the ones the
paper should quote, and section 5.1b is corrected accordingly.

The same recomputation moves `[37, inf)`'s modal fraction from 71.1% to
**73.3%**: the earlier number divided two rounded corpus shares, 26.4% by
37.2%. The exact residue figures are 27.24% and 37.17%. Nothing that depends
on it changes -- it was quoted to say "most of the top class is one recorded
temperature" -- but the paper's numbers should be right.

## What changes in the paper

1. **Section 5.2i reports both outcomes met on test**, with the validation
   disagreement stated in the same breath. First pre-registered test in the
   project where both a primary and a secondary prediction held.
2. **Section 6.2's mechanism paragraph can drop its "function not separated
   from temperature" caveat.** It gains the split disagreement instead.
3. **Section 5.2h's limitation is discharged.** Its gain was not the seventh
   class: with k held at six, moving one edge recovers 75% of it.
4. **Sections 5.1b and 5.2i take the residue-basis mode/tail figures**
   (0.50x, 0.31x, 2.74x) and the corrected 73.3% modal fraction.
5. **The limitations section gains the split disagreement on this corpus**, as
   a property of the corpus and not of the measure, and notes it has now
   happened twice here and nowhere else.

## What this does not license

- Any claim that 38 and 45 degrees C are equivalent. Two runs agreeing to
  0.2 sigma on one split is evidence against the value mattering, not evidence
  for equivalence, and the pre-registration said so before the numbers existed.
- Any claim that the mode/tail diagnostic is validated. It retrodicts three
  corpora and is realizable on one; validating it needs a fourth corpus chosen
  in advance for being mode-dominated.
- Any claim that the statistic chooses thresholds. Unchanged from 5.2d: it
  ranks placement at rho +0.000 against a trivial baseline's +1.000, and the
  bound is 0.0016 pp apart on two runs that differ by seven degrees of edge.
- Any claim about k above 9, a second architecture, or a fourth corpus.
- Any statement about downstream thermostability prediction. Nothing here is
  an assay.
