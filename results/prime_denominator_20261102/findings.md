# The denominator cannot be applied to PRIME's number, and that is the finding

Date: 2026-11-02. Envelope sweeps in `envelope_*.txt`, computed by
`scripts/18_label_envelope.py`. CPU-only.

## What was attempted and why it failed

The plan was the one flagged as future work since 2026-10-27: compute the
corpus-only bound for a published system's own corpus and report its gain as
a fraction of what that corpus allowed. PRIME was the target. Its OGT source
is Engqvist, the same table this project uses, so the bound transfers.

**It is not computable, and not because of missing data.** PRIME reports no
likelihood or perplexity quantity at all. Its only ablation of the growth
temperature label is in downstream Spearman correlation on zero-shot mutant
effect prediction, from the authors' own repository:

| model | TM datasets (avg) | activity datasets (avg) |
|---|---|---|
| Prime (no OGT task) | 0.333 | 0.418 |
| **Prime** | **0.439** | **0.414** |
| Prime (fine-tuned) | 0.488 | 0.421 |

So the label is worth **+0.106 Spearman on thermostability assays and -0.004
on activity assays**. There is no denominator that converts 0.106 Spearman on
20 assay datasets into a share of a corpus's mutual information, and nothing
public bridges them: the quantity depends on the assay set, the mutation
distribution, the scoring scheme and the fine-tuning, none of which are
properties of the corpus.

Claiming "PRIME captured X% of its available signal" would have required
inventing that bridge. It is not in the paper and it is not in this one.

## This is a stronger version of the paper's argument, not a weaker one

PRIME is the best-resourced system in this space -- 96M sequence-OGT pairs,
340x our corpus -- and it is also the only one of the six surveyed that
ablates its temperature label at all. Even there, a reader cannot tell whether
+0.106 Spearman is most of what the label could have given or a tenth of it,
because the number is reported in units that have no upper bound and no
corpus-level reference.

That is the no-denominator problem, stated with the field's own flagship
figures rather than with ours. The asymmetry is worth keeping too: the label
helps on thermostability assays and does nothing on activity assays, which is
what a growth-temperature label should do and which nobody can currently check
against a scale.

## What is computable: the envelope

Mutual information is monotone non-decreasing under refinement of the label
partition, so I(residue; OGT) at a database's native temperature resolution
upper-bounds every binning of that database's labels. That number is the
corpus's envelope, and it is the honest form of "how much can a
growth-temperature label be worth here".

| corpus | distinct OGTs | residues | best 2-class | envelope | 2-class share |
|---|---|---|---|---|---|
| primary (Gosha, capped + balanced) | 47 | 113.8M | +0.198% | **+0.661%** | 30.0% |
| external, design-matched | 63 | 11.1M | +0.274% | **+0.970%** | 28.3% |
| external, uncapped | 77 | 64.9M | +0.148% | **+0.488%** | 30.4% |

**The best two-class threshold -- which is the kind of label every system in
section 2 uses -- reaches 28% to 30% of what the corpus makes available.** That
is consistent across three corpora, two growth-temperature databases and a
twofold range of absolute bounds.

Miller-Madow bias correction, (|R|-1)(|B|-1)/2N nats, is reported in each sweep
and is negligible at these corpus sizes: 0.000 pp on the two large corpora and
0.005 pp on the 11.1M-residue one, against envelopes of 0.5-1.0 pp.
Establishing that was necessary rather than optional, because the correction
grows with exactly the refinement that raises the estimate, and the envelope
row is the most refined one.

## Correction, 2026-11-03: most of the gap is not trainable

The first version of this file reported the two-class share as 23.4% / 28.3% /
30.4% and a 3.3-4.3x gap to the envelope, using equal-residue-mass quantile
edges. Both halves of that needed fixing.

**The quantile scheme cannot deliver equal mass on these corpora.** A single
recorded temperature can hold more residues than 1/k of the corpus: 30 C alone
is 29.6% of the primary corpus's residues and 37 C is 18.5%. An atom cannot be
split, so from four classes upward the scheme degenerates -- at sixteen classes
its smallest class holds 0.1% of residues and 313 sequences. The intermediate
rows of the original sweep were arithmetic on partitions nobody could train on,
and the dips at six and eleven classes were that degeneration, not only the
non-nesting the file already noted.

`scripts/18_label_envelope.py` now also partitions to maximise the smallest
class's share of residues, by dynamic program over contiguous temperature
intervals, and reports which granularities clear a 5%-of-residues floor. Two
consequences:

1. **The best two-class edge is better than the quantile median split.** On the
   primary corpus the optimal edge is 37 C at +0.198%, against the quantile
   scheme's 45 C at +0.154%. The share rises from 23.4% to 30.0%, which makes
   the three corpora agree closely rather than span 23-30%.
2. **The reachable bound is well below the envelope.**

| corpus | largest trainable k | best reachable bound | share of envelope |
|---|---|---|---|
| primary | 9 | +0.496% | 75.1% |
| external, design-matched | 13 | +0.645% | 66.5% |
| external, uncapped | 11 | +0.341% | 69.9% |

So the gap worth quoting is **2.3x to 2.5x between the best two-class threshold
and the best trainable binning**, not 3.3-4.3x. A quarter to a third of the
envelope exists only in bins of a few hundred sequences.

**And it saturates early.** On the primary corpus five classes reach +0.462%
(69.9%) and nine reach +0.496% (75.1%): four more classes buy 0.034 pp. The
recommendation this licenses is *four to six classes*, which is both cheap and
concrete, and not *as many as the database records*.

This correction was found before any GPU time was spent on the granularity
series, by checking class populations for the k=8 and k=16 runs that series was
going to launch. The smallest class would have held 482 and 313 sequences, so a
weak measured gain there would have been control-token sparsity rather than a
property of the label. The series as pre-registered in section 5.2e uses the
minimum-mass-maximising edges instead, every class above the floor.

## What this licenses the paper to claim, and what it does not

**Does:**

- A growth-temperature label on an Engqvist- or Gosha-derived prokaryotic
  corpus is worth at most +0.5% to +1.0% in perplexity to a perfect unigram
  predictor, and a conventional two-class threshold leaves three to four times
  that on the table at the level of *available* signal.
- The field reports the value of its temperature labels in units that cannot
  be compared across papers or against any corpus property, and the one
  published ablation (PRIME's) demonstrates it.
- A cheap, concrete recommendation follows: report the likelihood-level
  conditioning gain -- one evaluation pass with the label shuffled -- alongside
  whatever downstream delta is being claimed, and report the corpus envelope
  beside it. Both are reproducible from the corpus and neither depends on an
  assay set.

**Does not:**

- Any statement about what fraction of its available signal PRIME captured.
- Any claim that the 2.3-2.5x reachable gap is realised by a trained model.
  Section 5.1 measures the two-class case and finds measured gain tracking the
  bound loosely at best; section 6.2 shows this measure cannot even rank
  granularities, so the envelope is a bound and explicitly not a target. The
  granularity runs we have (`sweep_c4`) gained +0.248% against a four-class
  bound of +0.374%, which is 0.66x, not 1x.
- Any claim that more classes are better in practice. That is the obvious
  next experiment and it is not run here.
