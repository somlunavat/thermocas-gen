# One cut, not more classes: the mechanism confirmed, and the bound is blind to it

Date: 2026-11-14. Two runs on the uncapped corpus (`configs/external.yaml`),
pre-registered in `paper/draft.md` section 5.2h at commit `da4b6a2`, launched
at `2c852e4`. Reports in `runs/`, call ids in `runs.txt`.

**Both pre-registered outcomes are met, and the post-hoc account from section
5.2g survives its test.** Adding a single edge at 38 C to `gext_k6`'s
partition raises measured test gain from +0.062% to +0.195%, recovering 103%
of what going from six classes to nine had bought. A four-class partition
containing that edge reaches +0.157% against `gext_k6`'s +0.062% **at a bound
matched to 0.0016 pp**.

## The measurements

| run | k | edges (C) | bound | val gain | test gain | gain/bound |
|---|---|---|---|---|---|---|
| `gplc_nest` | 7 | 27,29,31,35,37,**38** | +0.3154% | +0.144% | **+0.195%** | 0.62x |
| `gplc_match` | 4 | 30,37,**38** | +0.2550% | +0.124% | **+0.157%** | 0.62x |

Reference figures, same corpus, same splits, same 2,000-of-11,263 test cap,
from `results/granularity_20261110/`:

| run | k | edges (C) | bound | test gain | has a cut above the 37 C pile |
|---|---|---|---|---|---|
| `gext_k2` | 2 | 35 | +0.1481% | +0.072% | no |
| `gext_k4` | 4 | 29,32,37 | +0.2036% | +0.070% | no |
| `gext_k6` | 6 | 27,29,31,35,37 | +0.2566% | +0.062% | no |
| `gext_k9` | 9 | 26,...,35,37,**38** | +0.3377% | +0.191% | yes |

Error bars are gain / z, exact. Both new runs are significant on both splits
(test z 21.4 and 16.8; validation z 12.5 and 21.7) and **both passed Gate 2 on
test**, at 0.62x of their ceilings -- the two highest ratios in this corpus's
series. Both returned exit code 3 from the validation gate, which failed on
the size criterion at 0.46x and 0.49x; section 5.2h's void condition is z < 3
and neither run is near it, and 5.2h stated in advance that a size failure
does not void an outcome defined as a difference between runs. Both selected
step 2,500 by validation, as all four 5.2g runs did.

## The pre-registered outcomes

**Primary: `gplc_nest` minus `gext_k6`, one added edge.**

| split | gext_k6 | gplc_nest | difference | sigma |
|---|---|---|---|---|
| test | +0.062% | +0.195% | **+0.133 pp** | **9.8** |
| validation | +0.100% | +0.144% | +0.044 pp | 3.1 |

The prediction on record was positive beyond two standard deviations *and*
recovering more than half of the +0.129 pp that k=6 to k=9 bought, so at least
+0.065 pp. The measured difference is +0.133 pp, which is **103% of that
jump**. One edge does all of it.

**Secondary and discriminating: `gplc_match` minus `gext_k6`, at a matched
bound.**

| split | gext_k6 | gplc_match | difference | sigma |
|---|---|---|---|---|
| test | +0.062% | +0.157% | **+0.095 pp** | **6.9** |
| validation | +0.100% | +0.124% | +0.024 pp | 2.3 |

Both predictions are met on both splits. Validation agrees in sign throughout
but is compressed relative to test, as it was across the whole 5.2g batch;
5.2h fixed test as the pre-registered figure for that reason.

## Which explanation survives

Section 5.2h tabulated three and committed to the predictions each made.

| explanation | predicted `gplc_match` (k=4) | measured | verdict |
|---|---|---|---|
| **the cut above the 37 C pile** | near +0.19% | **+0.157%** | **survives** |
| class count | near `gext_k4`'s +0.07% | +0.157% | **refuted** |
| the bound | near `gext_k6`'s +0.06% | +0.157% | **refuted** |

The class-count explanation is refuted with class count held *exactly* fixed:
`gplc_match` and `gext_k4` are both four-class partitions and differ only in
where the edges sit. **+0.087 pp, 7.7 sigma.** The bound explanation is
refuted at a matched bound, +0.095 pp at 6.9 sigma -- and that is a *failure*
of the bound, not a success, exactly as 5.2h said in advance it would be.

## The decomposition, which is cleaner than the hypothesis predicted

All six runs on this corpus sort by whether the partition cuts above the 37 C
pile, and not by class count or by bound.

**Without such a cut, refinement buys nothing at all:**

| step | difference | sigma |
|---|---|---|
| k=2 -> k=4 | -0.002 pp | -0.2 |
| k=4 -> k=6 | -0.008 pp | -0.7 |
| k=2 -> k=6 | -0.010 pp | -0.9 |

**With the cut, refinement buys a little, then saturates:**

| step | difference | sigma |
|---|---|---|
| k=4 -> k=7 | +0.038 pp | +2.9 |
| k=7 -> k=9 | -0.004 pp | -0.3 |

So the honest statement is an interaction and not a single cause. **The cut is
worth +0.09 to +0.13 pp and class count is worth +0.038 pp, and the second
only pays once the first is present.** Going from two classes to six without
the cut is worth nothing measurable; going from four classes to seven with it
is worth 2.9 sigma. `gplc_nest` at seven classes is statistically
indistinguishable from `gext_k9` at nine (+0.004 pp, 0.3 sigma), so once the
cut is in, nine classes add nothing over seven.

Ranked by measured gain, the six runs separate completely by the cut:

| rank | run | k | test gain | cut |
|---|---|---|---|---|
| 1 | `gplc_nest` | 7 | +0.195% | yes |
| 2 | `gext_k9` | 9 | +0.191% | yes |
| 3 | `gplc_match` | 4 | +0.157% | yes |
| 4 | `gext_k2` | 2 | +0.072% | no |
| 5 | `gext_k4` | 4 | +0.070% | no |
| 6 | `gext_k6` | 6 | +0.062% | no |

## Why: the modal temperature was swamping the tail

This is what the cut does, and it is arithmetic from the label table rather
than a new measurement.

On this corpus 37 C alone holds 26.4% of sequences and only 10.8% lie above
it. So the top class of every partition before the cut was mostly one
temperature:

| top class | modal seqs | above-modal seqs | ratio |
|---|---|---|---|
| `[35, inf)` (`gext_k2`) | 53,941 | 46,008 | 1.17x |
| `[37, inf)` (`gext_k4`, `gext_k6`) | 53,941 | 21,941 | **2.46x** |
| `[38, inf)` (both new runs, `gext_k9`) | 0 | 21,941 | 0.00x |

A class that is 71% a single recorded temperature cannot carry information
about the range above it. The cut at 38 removes the mode from the top class
entirely, and that is the whole of what it does.

**This yields a diagnostic, and it retro-predicts all three corpora.** Compare
the modal temperature's mass to the mass above it:

| corpus | modal mass | mass above mode | mode/tail | granularity rho |
|---|---|---|---|---|
| `pretrain` | 28.9% (30 C) | 60.8% | 0.47x | **+1.000** |
| `external_balanced` | 14.7% (37 C) | 52.6% | 0.28x | **+1.000** |
| `external` | 26.4% (37 C) | 10.8% | **2.46x** | **+0.200** |

On the two corpora where the tail outweighs the mode, no partition's top class
contained the mode at all -- there was nothing for a cut to fix, and the bound
ranked granularity perfectly. On the one corpus where the mode outweighs the
tail, the bound's ranking failed. **Three for three.**

This also explains why the same numeric edge behaves differently in different
places: 37 C is nowhere near `pretrain`'s mode of 30 C, so `[37, inf)` is pure
tail there, while on this corpus 37 C *is* the mode.

**It is a hypothesis fitted to three corpora after the fact, and it is stated
as one.** Three points and a ratio is pattern-matching, not validation. What
would test it is a corpus selected in advance for being mode-dominated, with
the prediction that its granularity ranking fails unless the partition cuts
immediately above the mode -- or, cheaper and available now, re-running the
`pretrain` and `external_balanced` series with a deliberately mode-contaminated
top class and predicting that their perfect rank correlations break.

## What this does and does not license

**Licensed.** On this corpus, which cuts a partition contains dominates both
how many it contains and what the bound says about it, by 7 to 10 sigma. A
cheap statistic that ranks partitions by class count or by mutual information
is blind to the single most important property of a partition here. This is
the paper's first positive placement result and its sharpest statement about
what the bound misses.

**Not licensed, and 5.2h committed to this in advance.**

- **Not that 38 C is special.** Both runs use exactly that edge, so we cannot
  distinguish it from any cut isolating the upper tail, and we name the
  function -- separating the modal temperature from the range above it -- and
  not the temperature.
- **No rescue of the bound.** `gplc_match` beating `gext_k6` at a matched
  bound is a failure of the bound. The three cut-bearing runs reach 0.57x to
  0.62x of their ceilings and the three without reach 0.24x to 0.49x, so the
  bound does not even order these six correctly.
- **The mode/tail diagnostic is not validated.** Post-hoc on three corpora,
  with a named test that has not been run.
- **Nothing about downstream thermostability.** Nothing here is an assay.
- **One corpus and one backbone** for the mechanism itself.

## What changes in the paper

1. **Section 5.2h reports a positive pre-registered result on both outcomes**,
   and 5.2g's post-hoc paragraph becomes a measured mechanism.
2. **Section 6.2 restructures.** "It ranks resolution, not placement" was
   already qualified by 5.2g; it now has to say that on a mode-dominated
   corpus placement dominates resolution and the bound sees neither. The
   dichotomy survives for the two balanced corpora and inverts on the third.
3. **Section 5.1b's third caution becomes a measured one**, and gains the
   mode/tail diagnostic as a check to run before trusting a reachability
   table.
4. **Section 8 gains a recommendation**: compute the mode/tail ratio, and if
   the mode dominates, cut immediately above it before counting classes.
5. **The abstract gains the placement result.** It is the first thing in the
   paper that tells a practitioner what to *do* about a corpus statistic that
   fails.
