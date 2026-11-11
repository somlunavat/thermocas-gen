# No denominator: putting a scale on growth-temperature conditioning in protein language models

Four-page workshop version, extracted 2026-11-09 from `paper/draft.md`.
Target: an ICML computational-biology workshop (WCB, ML4LMS, AI4Science).
Every number here is carried from the full draft and is wired to a file under
`results/`; section numbers in brackets point back to the draft.

**Status note.** Complete as of 2026-11-10. Sections 4.4 and 4.5 were both
pre-registered in the draft before launch (5.2f, 5.2g) and both have landed.
4.5 is the one that cost a claim: the perfect rank correlation does not
survive on the corpus we did not build, and the abstract and section 6 say so.

**Length.** Now ~2,890 words as the file stands, against ~2,360 at extraction, which is over four
pages. If it has to come down, 4.5's post-hoc paragraph is the first cut and
section 3's envelope table the second; the three granularity results and the
placement failure all stay.

## Abstract

Six recent systems condition protein language models on growth temperature.
None reports how much its label could have been worth. The best-resourced,
trained on 96M sequence-temperature pairs, ablates its temperature objective
and reports +0.106 Spearman on thermostability assays and -0.004 on activity
assays: a careful experiment in units comparable neither across papers nor
against any property of its corpus. We supply the missing scale. Mutual
information between residue identity and label, computed from residue counts
in CPU-minutes, prices a label in perplexity units on the corpus the model
trains on, and at a database's native temperature resolution it bounds every
coarser binning of those labels. Across three prokaryotic corpora that
envelope is +0.49% to +0.97%, the best two-class threshold reaches 28-30% of
it, and a floor of 5% of residues per class caps any binning at 67-75%. In
twenty-nine runs that vary only the label definition within a corpus -- on
byte-identical sequences, clusters and splits -- label definition moves
measured conditioning gain by a factor of 3.9 to 4.2. Four pre-registered
tests divide what the scale can do from what it cannot. Across granularity,
nine classes beat two on all three corpora at 9.4 to 11.8 standard deviations,
but the finer ordering holds only on the two corpora we balanced by design and
fails on the one we left as the databases have it. Across threshold placement
the scale fails outright: it ranks no better than chance and its own preferred
threshold measures 6.4 standard deviations worse than the conventional one. So
it is a guide to how many classes rather than to where to put them, and on an
unbalanced corpus only to the coarse comparison.

## 1. There is no denominator

Conditioning a protein language model on the growth temperature of a
sequence's source organism is now a standard design. Six systems in the recent
literature do it, across Science Advances, NeurIPS, ICML, Scientific Reports
and Nature [draft 2]. They differ in backbone, corpus and label definition,
and they report gains in incompatible units: downstream assay correlations,
classifier accuracies, generated-sequence property distributions.

None of them reports how much its growth-temperature label could have been
worth. One ablates the label at all. PRIME, trained on ~96M sequence-OGT
pairs, removes its temperature objective and reports:

| | thermostability (20 datasets) | activity (11 datasets) |
|---|---|---|
| without the OGT task | 0.333 | 0.418 |
| with the OGT task | **0.439** | 0.414 |

So the label is worth +0.106 Spearman on thermostability and -0.004 on
activity. This is a well-designed ablation and a reader still cannot use it.
Is +0.106 most of what the label could have given on 96M pairs, or a tenth of
it? The quantity has no upper bound, depends on the assay set, the mutation
distribution and the fine-tuning, and is not a property of the corpus. We
tried to compute PRIME's gain as a share of its corpus's ceiling and found the
division is not available: no surveyed system reports a likelihood or
perplexity quantity for its label, and nothing public bridges downstream
Spearman to one [draft 8]. That failure is this paper's motivation stated as
an experiment rather than a complaint.

## 2. A scale that costs CPU-minutes

Let $R$ be residue identity and $B$ a growth-temperature label. The mutual
information $I(R; B)$, estimated from residue counts, converts to a relative
perplexity figure as $1 - e^{-I}$ -- the same units every measured gain in this
paper is reported in. It is computed from the corpus the model trains on, so
it says how strongly the label co-varies with composition *there*.

Two properties make it usable as a scale.

**It bounds every coarser binning.** Mutual information cannot fall when a
partition is refined, so $I(R; \text{OGT})$ at a database's native temperature
resolution upper-bounds $I(R; B)$ for every binning $B$ of those labels. We
call that the corpus's **envelope**. Miller-Madow bias correction,
$(|R|-1)(|B|-1)/2N$ nats, is reported beside every figure; it matters most on
the envelope row, because that row is the most refined, and it is 0.000-0.005
percentage points at these corpus sizes.

**It is a lower bound on what a model extracts, not a ceiling.** The estimate
is unigram: it sees composition and nothing else. A language model has context
and can use substitutions and motif placement no residue histogram represents.
**Ten of our sixteen threshold runs exceed their own ceiling, by up to 1.56x**,
with the ratio spanning 0.31x to 1.56x. We therefore never read a measured
gain as a fraction of capacity, and the envelope comparisons below are
information-to-information, which is what monotonicity licenses.

**Measurement design.** Every measured gain is a *relabeling*: `temp_bin` is
recomputed from the recorded growth temperature in memory at startup, so runs
that differ in label definition train on byte-identical sequences, clusters
and splits. Only the meaning of the control token changes. This isolates label
definition from corpus composition, which nothing in the related work does.
Gain is the relative perplexity improvement of the correct control token over
five shuffled ones, scored on a cluster-disjoint test split at the checkpoint
validation selected. Error bars are gain divided by the shuffle z-score.
Backbone: ProGen2-base with LoRA, stage A, 3,000 steps on an A10G.

Throughout, *growth temperature* is the label and *sampling temperature* is a
decoding setting; they are never abbreviated to the same word.

## 3. How much is available, and how much is reachable

| corpus | distinct OGTs | residues | best 2-class | envelope | 2-class share |
|---|---|---|---|---|---|
| primary (Gosha, capped + balanced) | 47 | 113.8M | +0.198% | **+0.661%** | 30.0% |
| external, design-matched (Engqvist) | 63 | 11.1M | +0.274% | **+0.970%** | 28.3% |
| external, uncapped | 77 | 64.9M | +0.148% | **+0.488%** | 30.4% |

**The best two-class threshold reaches 28-30% of what the corpus makes
available**, across three corpora, two growth-temperature databases and a
twofold range of absolute envelopes. These are the *best* two-class edges on
each corpus, found by maximising the smaller class's share of residues; a
threshold chosen by convention does worse.

**Most of the rest is not available to any trainable label.** Refinement
raises the bound by putting mass in narrow temperature classes, and a class
holding a few hundred sequences is a control token the model sees too rarely
to use. Equal-mass binning is not even available here: a single recorded
temperature can hold more residues than $1/k$ of the corpus -- 30 C alone is
29.6% of the primary corpus and 37 C is 18.5% -- and an atom cannot be split,
so an equal-frequency scheme degenerates, reaching a smallest class of 0.1% of
residues and 313 sequences at sixteen classes. We instead maximise the
smallest class's share, by dynamic program over contiguous temperature
intervals, and impose a floor of 5% of residues per class:

| corpus | largest trainable $k$ | best reachable bound | share of envelope |
|---|---|---|---|
| primary | 9 | +0.496% | 75.1% |
| external, design-matched | 13 | +0.645% | 66.5% |
| external, uncapped | 11 | +0.341% | 69.9% |

So the gap worth quoting is **2.3x to 2.5x between the best two-class
threshold and the best trainable binning**, not the 3.3-4.3x the envelope
alone suggests. A quarter to a third of the envelope exists only in bins
nobody could train on. It also saturates early: five classes reach 69.9% of
the primary corpus's envelope and nine reach 75.1%.

## 4. What a model actually extracts

### 4.1 Label definition is a factor of four

Across the threshold sweeps, the choice of threshold moves measured
conditioning gain by **3.9x on the primary corpus, 3.8x on the design-matched
one and 4.2x on the uncapped one**, with margins an order of magnitude above
the shuffle noise floor. Thresholds at or below 35 C return a small fraction
of what is available above 40 C, on all three corpora and both backbones.
Every system in section 1 fixes this parameter by convention. (Figure 3.)

### 4.2 Granularity: the bound reads it correctly

Four runs on the primary corpus at $k$ = 2, 4, 6, 9, with each partition
chosen to maximise the smallest class. Pre-registered, with the direction
fixed before launch [draft 5.2e].

| $k$ | edges (C) | min class | bound | test gain | gain/bound |
|---|---|---|---|---|---|
| 2 | 37 | 41.6% | +0.198% | +0.110% | 0.55x |
| 4 | 32,45,60 | 19.5% | +0.372% | +0.245% | 0.66x |
| 6 | 30,32,45,55,67 | 10.8% | +0.488% | +0.248% | 0.51x |
| 9 | 29,32,45,49,53,56,67,83 | 5.5% | +0.496% | +0.273% | 0.55x |

**We predicted this would fail and it did not.** The pre-registered outcome --
the sign of (gain at $k$=9) minus (gain at $k$=2) -- is positive at 9.4 sigma
on test and 10.6 on validation, against a prediction on record of
negative-or-within-two-sigma. Spearman between bound and measured gain is
**+1.000 on both splits**, exact one-sided $p$ = 0.042 over all 24
relabelings, the smallest attainable at $n$=4. Measured gain rises 2.5x as the
bound rises 2.5x, with `gain / bound` between 0.51x and 0.66x throughout.

**The gain saturates earlier than the bound.** Two classes to four is +0.135
pp at 5.8 sigma; four to six is 0.1 sigma, six to nine 1.2 sigma. Four classes
reach +0.245% of the +0.273% nine classes reach.

### 4.3 Placement: it fails, and its own optimum is a bad threshold

On a corpus chosen in advance so the predictors disagree, with the outcome
pre-registered, the corpus statistic ranked four thresholds no better than
chance (Spearman +0.000) while `-|edge - 55|` -- a baseline reading nothing
from any corpus -- ranked them perfectly (+1.000, $p$ = 0.042). The
pre-registered comparison was negative at 6.6 and 7.1 sigma [draft 5.2d].

Worse, the two-class edge that *maximises the bound* on the primary corpus is
37 C:

| $k$=2 edge | chosen by | bound | test gain |
|---|---|---|---|
| 37 C | maximises the bound | +0.198% | +0.110% |
| 45 C | convention | +0.154% | **+0.251%** |

**The statistic's preferred threshold measures 6.4 sigma worse, at a higher
bound.** One mechanism covers both halves: mutual information is maximised at
an even marginal, and on these corpora an even split separates organisms
annotated near 37 C from everything else rather than mesophiles from
thermophiles. Measured gain correlates -0.80 with label entropy on the
uncapped corpus (Pearson -0.926). Moving a threshold changes class balance and
biological meaning together; a granularity series holding the best-balanced
partition at each $k$ removes balance as the free variable and leaves
resolution, which the statistic reads correctly.

### 4.4 It replicates, and two secondary observations do not

Section 4.2 is one corpus and four points, so we repeated it on the
design-matched external corpus at the same $k$ with that corpus's own
partitions, pre-registered with the direction fixed [draft 5.2f].

| $k$ | bound | test gain | gain/bound |
|---|---|---|---|
| 2 | +0.274% | +0.432% | 1.58x |
| 4 | +0.403% | +0.484% | 1.20x |
| 6 | +0.517% | +0.582% | 1.13x |
| 9 | +0.576% | +0.670% | 1.16x |

The primary outcome is positive at 11.8 sigma on test and 11.5 on validation,
and Spearman is **+1.000 on both splits** again.

Two secondary observations do not replicate, both hedged in advance.
`gain / bound` is 0.51-0.66x on the primary corpus and 1.13-1.58x here, so it
is corpus-dependent and must not be read as an efficiency. And the early
saturation does not transfer: on the primary corpus only the two-to-four step
was reliable, while here every step is significant and nine classes beat four
by 8.8 sigma. The *ordering* replicates; the stopping point is corpus-specific
and has to be read off that corpus's reachability table.

### 4.5 On the corpus we did not build, only the coarse contrast survives

The third series runs the same four granularities on the uncapped corpus -- no
per-species cap, no balancing -- pre-registered with two distinct refutations
named in advance [draft 5.2g].

| $k$ | bound | test gain |
|---|---|---|
| 2 | +0.148% | +0.072% |
| 4 | +0.204% | +0.070% |
| 6 | +0.257% | +0.062% |
| 9 | +0.338% | **+0.191%** |

The primary outcome replicates: nine classes beat two by +0.119 pp at 9.4
sigma. **The rank correlation does not**, falling to +0.200 on test (exact
p = 0.458) and +0.800 on validation, neither significant -- the narrower of
the two refutations pre-registered for this series, arriving by the mechanism
named for it, a $k{=}4$/$k{=}6$ inversion at bounds 0.053 pp apart. Three of
the four runs are tied within 0.7 sigma while the bound rises 1.7x across
them: it separated the extremes and carried no information about the middle.
**Four perfect rank matches out of six corpus-split combinations, both
failures on the unbalanced corpus.** Gate 2 failed on three runs, all on its
size criterion, with every run significant at z from 6.2 to 27.2; 5.2g stated
before launch that a size failure does not void an outcome defined as a
difference between two runs.

Post-hoc and not a result: $k{=}9$ is the first of these partitions to cut
above the 37 degrees C pile, separating the 18,922 sequences above 38 degrees
C from it. The refinement that paid was one cut rather than refinement as
such, so placement re-enters inside a granularity series. The matched-bound
test that would settle it is specified and unrun.

## 5. Recommendations

1. **Report a likelihood-level conditioning gain, not only a downstream
   delta.** One evaluation pass with the label shuffled gives a number in the
   same units as the corpus statistic, costs a fraction of a training run, and
   is comparable across papers. Shuffling rather than removing holds sequence
   length and input distribution fixed. PRIME's ablation is why we ask.
2. **Report the corpus envelope and the label entropy beside it.** Both are
   CPU-minutes from the corpus and neither depends on an assay set.
3. **Use several classes rather than a threshold, and let your own corpus set
   how many.** Four classes instead of two raised measured gain from +0.110%
   to +0.245% on the primary corpus, 5.8 sigma; on the design-matched corpus
   nine classes beat four by 8.8 sigma; on the uncapped corpus nothing below
   nine beat two at all. All six surveyed systems use two. The count does not
   transfer, so compute the reachability table and take the largest $k$ whose
   smallest class stays trainable -- that was the best of the four on all
   three corpora. Trust the bound for "more classes, up to what you can
   train" and not for the ranking in between, which it got wrong on the
   unbalanced corpus. And read the edges, not just the row: check your
   partition separates the temperature range you care about, because on the
   uncapped corpus the step that paid was the first one to cut above the
   37 degrees C pile and the bound gave no sign that cut mattered.
4. **Do not use this statistic to choose the threshold.** Convention, or "well
   above the mode of your temperature distribution", does better.

## 6. Limitations

Prokaryotes only. One architecture family; the second backbone is a capacity
comparison, not an architecture-independence result. The granularity result is
four points per corpus on three corpora, and $n$=4 cannot establish agreement:
a perfect rank match on four points is what four points can show at best, and
it held on two corpora out of three. On the third, three of the four points
are tied within 0.7 sigma, so the rank correlation there is close to
uninformative in either direction. `gain / bound` does not transfer between
corpora -- 0.24x to 1.58x across the three -- and we draw no efficiency claim
from it. Where the gain stops paying is corpus-specific, so the class-count
advice is a procedure rather than a number. Our account of why the ordering
failed on the uncapped corpus, that one cut above the 37 degrees C pile did
the work, is post-hoc; the matched-bound test that would settle it is
specified and unrun. Growth temperature
is a property of an organism and not of a protein, so every label here is a
proxy. Nothing in this paper is an assay, and no claim is made about
thermostability prediction.
