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

| corpus | distinct OGTs | residues | 2-class bound | envelope | 2-class share | envelope / 2-class |
|---|---|---|---|---|---|---|
| primary (Gosha, capped + balanced) | 47 | 113.8M | +0.154% | **+0.661%** | 23.4% | 4.3x |
| external, design-matched | 63 | 11.1M | +0.274% | **+0.970%** | 28.3% | 3.5x |
| external, uncapped | 77 | 64.9M | +0.148% | **+0.488%** | 30.4% | 3.3x |

**A two-class threshold -- which is what every system in section 2 uses --
reaches 23% to 30% of what the corpus makes available to a label.** That is
consistent across three corpora, two growth-temperature databases and a
threefold range of absolute bounds.

Miller-Madow bias correction, (|R|-1)(|B|-1)/2N nats, is reported in each
sweep and is negligible at these corpus sizes: 0.000 pp on the two large
corpora and 0.005 pp on the 11.1M-residue one, against envelopes of 0.5-1.0 pp.
Establishing that was necessary rather than optional, because the correction
grows with exactly the refinement that raises the estimate, and the envelope
row is the most refined one.

The intermediate rows are not nested -- equal-residue-mass edges for k and
k+1 classes are different partitions, not refinements -- so that column can
dip, as it does at 6 and 11 classes on the uncapped corpus. This is a property
of the quantile scheme and not a violation of monotonicity.

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
- Any claim that the 3.3-4.3x envelope gap is reachable by a trained model.
  Section 5.1 measures the two-class case and finds measured gain tracking the
  bound loosely at best; section 6.2 shows this measure cannot even rank
  granularities, so the envelope is a bound and explicitly not a target. The
  granularity runs we have (`sweep_c4`) gained +0.248% against a four-class
  bound of +0.374%, which is 0.66x, not 1x.
- Any claim that more classes are better in practice. That is the obvious
  next experiment and it is not run here.
