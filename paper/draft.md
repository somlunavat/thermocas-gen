# Draft skeleton

Status: skeleton. The abstract and every table below are wired to numbers that
are committed under `results/`; the prose of sections 3-7 is an outline with
the claim each section has to make and the evidence it has to cite. Nothing
here depends on a GPU run that has not happened.

Provenance for every number: `results/ceiling_sweep_20261027/measured.md`,
`results/external_corpus_20261029/findings.md`,
`results/cas9_corpus_20261027/`, `results/related_work_20261027.md`.

## Title

**How much can a growth-temperature label help? A corpus-only bound on
conditioning gain in protein language models**

Alternates, in case the first reads too much like a survey:

- *The label is a free parameter: choosing growth-temperature bins before training*
- *No denominator: conditioning gains in protein language models are reported without a scale*

## Abstract

At least four recent methods claim control over the thermostability of
sequences sampled from protein language models, across Science Advances,
NeurIPS, ICML and Scientific Reports. The gains they report are small and are
compared against each other or against nothing at all: the field has no
reference scale for how much a growth-temperature label can help on a given
corpus. We ask whether a cheap corpus statistic can supply one. Across eight
GPU runs that differ only in how the label is defined -- identical sequences,
clusters and splits -- and five more on a second corpus and three on a second
backbone, we find that how a growth-temperature label is binned is
worth a factor of 3.9 in measured conditioning gain, from +0.070% at cold
thresholds to +0.273% on a plateau between 45 and 65 degrees C, with margins an
order of magnitude above the shuffle noise floor. Mutual information between
residue identity and label, computed in CPU-minutes from residue counts alone,
predicts this ordering only when it is conditioned on sequence family: the
family-conditioned statistic reaches Spearman +0.71 on validation and +0.74 on
a held-out test split (exact permutation p = 0.029 and 0.021), while the
corpus-wide statistic reaches only +0.38 and is not significant. The difference
is interpretable, and it is a warning about the corpus-wide form: its largest
error is at the most class-balanced threshold, where the residue histogram sees
a difference that the model cannot use. We also report two negative results.
The statistic does not rank label definitions that differ in granularity rather
than threshold, and normalizing by label entropy makes that worse rather than
better. And neither form beats a baseline that reads nothing from the corpus at
all -- distance from the middle of the temperature range -- on either of the two
corpora we train on, whose measured optima are 50 and 65 degrees C and so are
both close to that middle. The plateau itself is robust: it reproduces on a
second corpus built from a different growth-temperature database and a different
sequence source, and the ordering of thresholds reproduces on a second backbone
an order of magnitude smaller. Finally, a worked negative case shows two corpus
gates correctly ruling out a Cas9 conditioning experiment before any GPU time
is spent.

Word count target for submission: ~200. The draft above is ~290. The two
negative results compress; the p-values and the 3.9x do not.

NOTE 2026-10-31: rewritten after the n=8 runs. The previous version led with
"the conventional 45 C threshold leaves 1.3-1.7x of the available signal
unused", which the measured data does not support -- see
`results/ceiling_sweep_20261030/measured_n8.md`. The measured optimum is 50 C
at 1.14x over 45 C, roughly 2 sigma, so the convention is close to optimal and
that number must not reappear anywhere in the paper.

## 1. Introduction

Claim: **the gains in this literature are uninterpretable because there is no
denominator.**

Paragraphs:

1. Conditional generation of thermostable proteins is an active target, and
   control over sampled sequence by growth temperature is the usual mechanism.
2. The reported effects are small in absolute terms and reported without a
   reference scale. Cite the five works in the table below. A reader cannot
   tell whether a +0.3% perplexity improvement is most of what the corpus
   allows or a tenth of it.
3. A candidate denominator is a per-corpus budget for the control token,
   computable from residue counts for CPU-minutes. We test it rather than
   assume it, and the test is the paper.
4. Contributions, as a list: (i) a relabeling-only experimental design that
   isolates label definition from every other corpus property, which nothing in
   the related work does; (ii) the measurement that label definition is worth
   3.9x in conditioning gain; (iii) the finding that a corpus statistic
   predicts which definition is better only when conditioned on sequence
   family, with the failure of the corpus-wide form traced to class balance;
   (iv) three documented limits -- no ranking across granularities, no
   advantage over a fixed-centre rule of thumb on a single corpus, and a
   plateau wide enough that precision has little value; (v) a worked negative
   case in which two corpus gates rule out an experiment before any GPU spend.

Note to self, and this is now the central honesty requirement of the paper: the
introduction must not promise that the statistic estimates achievable gain, or
that it ranks robustly. On one corpus it is beaten by a rule of thumb. The
framing that survives is "we supply and test a denominator, it works in the
family-conditioned form, and here is precisely how far that goes". A reviewer
who reaches section 6 and finds it contradicting paragraph 3 will be right.

## 2. Related work

Claim: **temperature conditioning is crowded, and that is the argument rather
than the competition.**

| Work | Venue | Approach | Training scale |
|---|---|---|---|
| PRIME | Sci. Adv. 2024 | MLM + sequence-level OGT regression, coupled by a correlation loss | ~96M sequence-OGT pairs |
| ThermoFormer | NeurIPS 2024 workshop | Temperature-aware representation learning | ~96M OGT-annotated sequences |
| Steering PLMs | ICML 2025 | Activation-addition steering, >70 vs <50 degrees C | ESM2-650M, ESM3-open, ProLLaMA-7B |
| NOMELT | Sci. Rep. 2025 | Meso-to-thermo seq2seq translation | 4M pairs, 2.7k Pfam families |
| OpenCRISPR-1 | Nature 2025 | ProGen2-base fine-tuned on the CRISPR-Cas Atlas | 4M generated sequences |
| TemStaPro | Bioinformatics 2024 | ProtT5 embeddings to a thermostability classifier | predictor, not generator |

Three things this section must do:

- **Credit PRIME for the auxiliary OGT regression head.** Our configuration
  carries one and it is their idea, not ours.
- State plainly that none of these works, and nothing found in four searches
  across conditional protein generation, controllable text generation,
  information-theoretic limits on conditioning, and label informativeness,
  measures the achievable gain from a conditioning label. The nearest
  neighbours are variational MI bounds (Poole et al., ICML 2019), MI *capped as
  a training mechanism* (arXiv 2203.02055), entropy reduction as annotation
  informativeness in vision (arXiv 1812.10358), and label-aware generation in
  homologous families (arXiv 2507.15651, which reports gains plateauing past
  ~10^3 labels using discriminator AUC rather than entropy).
- Concede the scale gap without hedging: PRIME and ThermoFormer train on ~340x
  more labeled sequence than we do. No framing in which corpus scale is our
  contribution survives that, and we do not attempt one.

Cite *Off-Manifold Collapse in Guided Protein Language Models* (arXiv
2608.18597) where we state that structure prediction is a validity filter and
not a thermostability assay: they report that harder steering lowers ESMFold
pLDDT under both solubility and thermostability steering.

## 3. The measure

Claim: **a corpus-only quantity, what it bounds, and what it does not.**

Definition. For residues $r$ and a label $b$ assigned per sequence from its
organism's growth temperature,

$$ I(r; b) = H(r) - \sum_b P(b) H(r \mid b) $$

in nats over residue counts, with the relative perplexity gain available to a
perfect model written as $1 - e^{-I}$. Two aggregations are reported
throughout: **corpus-wide**, and **within-cluster**, which conditions on
sequence family and answers what the label adds given the homologs.

What has to be said here, in order of how badly a reviewer will punish its
absence:

0. **The two aggregations are not interchangeable and the paper's result turns
   on which one is used.** Within-cluster predicts measured gain; corpus-wide
   does not. Say this here, when both are defined, rather than letting a reader
   reach section 5.1b thinking the choice was cosmetic. The mechanism is class
   balance, and the fact that mutual information is maximized at an even
   marginal is the reason to expect it a priori -- stated in advance, it makes
   5.1b a confirmed prediction rather than a post-hoc rescue.
1. **This is a lower bound, not an upper bound on what a model can do.** It is
   unigram: a model has context and can exploit residue order the measure
   cannot see. Two of our five runs exceed their own bound. It therefore cannot
   support any claim that conditioning has saturated, and we make none.
2. It is nevertheless a *scale*. It is computed from the same corpus the model
   trains on, so it says how much the label co-varies with composition there.
3. Mutual information is monotone non-decreasing under refinement of the label
   partition, so adding classes mechanically raises it. This is why the measure
   cannot compare across granularities, and it is a fact about MI rather than a
   defect we might fix by implementation. Section 6.2.
4. Binning convention: upper-exclusive, $\text{ogt} < \text{edge}$ is the low
   class. 45 degrees C is a common exact value in these corpora, so the
   convention moves thousands of sequences.
5. **Growth temperature** is the label. **Sampling temperature** is a decoding
   setting and appears nowhere in this paper. State it once, here.

Cost: the full eleven-threshold sweep over 113.8M residues is CPU-minutes.

## 4. Experimental design

Claim: **a sweep variant is a relabeling, not a second corpus, and that is what
makes the comparison clean.**

This is a small methods contribution and should be stated as one. `temp_bin` is
recomputed from `ogt_c` in memory at startup, so every variant trains on
byte-identical sequences, clusters and splits. Only the meaning of the control
token changes. Nothing in the related work isolates label definition this way;
comparisons there confound it with corpus composition.

Setup: ProGen2-base with LoRA, stage A, 3,000 steps, A10. Thirteen runs in
all: eight thresholds on stage A (section 5.1), five on a second corpus
(5.2b) and three on a second backbone (5.2c), with 30, 55 and 75 degrees C
shared between the first and third so they are directly comparable.
Conditioning is
measured as the relative perplexity improvement from the correct control token
over shuffled ones, five shuffles per run, on 2,000 held-out sequences and
again on a cluster-disjoint test split.

Corpus (stage A): 311,987 sequences, 113.8M residues, 47 distinct growth
temperature values, 1,552 both-class clusters at the 45 degrees C threshold,
279,980 training sequences. Growth temperatures from Gosha; sequences from
UniProt, species-selected and capped at 500 per species.

Second corpus: the design-matched external corpus of section 5.2, screened for
Cas9 homology (11 of 35,361 sequences removed) and split cluster-disjoint at
90/5/5, giving train 31,968 / val 1,687 / test 1,695 with both classes present
in both held-out splits. Both corpora are clustered at the same identity and
coverage thresholds, which is what makes the within-cluster column comparable
between them.

Budget note that belongs in this section: `sweep_t45` reproduces the configured
binning and reaches +0.260% at 3,000 steps against a 20,000-step baseline's
+0.265%. Variant comparisons are therefore comparisons of converged
conditioning, not of how far each run got.

## 5. Results

### 5.1 Label definition is worth 3.9x in measured gain

Eight thresholds, one corpus, relabeling-only. Every run passed Gate 2 with z
between 6.7 and 20.8 against a required 3.0, so nothing in the measured column
is noise.

| edge (C) | corpus-wide | within-cluster | val gain | test gain |
|---|---|---|---|---|
| 30 | +0.095% | +0.066% | +0.037% | +0.033% |
| 35 | +0.198% | +0.114% | +0.104% | +0.087% |
| 45 | +0.154% | +0.172% | +0.260% | +0.251% |
| 50 | +0.190% | +0.168% | **+0.296%** | **+0.270%** |
| 55 | +0.257% | +0.176% | +0.255% | +0.238% |
| 65 | +0.184% | +0.140% | +0.282% | +0.251% |
| 70 | +0.161% | +0.106% | +0.160% | +0.152% |
| 75 | +0.118% | +0.081% | +0.121% | +0.119% |

The robust finding, and the one the rest of the paper rests on: measured gain
is a plateau from 45 to 65 degrees C (+0.255% to +0.296%) and falls away
sharply outside it, to +0.070% averaged over 30 and 35 degrees C. That is a
factor of 3.9 from label definition alone, on byte-identical sequences. Test
gains track validation within 0.03 pp at every threshold.

### 5.1b The statistic predicts it only when conditioned on family

| predictor | rho (val) | p | rho (test) | p | Pearson (val) |
|---|---|---|---|---|---|
| corpus-wide | +0.381 | 0.180 | +0.359 | 0.190 | +0.615 |
| **within-cluster** | **+0.714** | **0.029** | **+0.743** | **0.021** | **+0.907** |
| `-abs(edge-55)`, no corpus | +0.843 | 0.007 | +0.849 | 0.006 | +0.906 |

p-values are exact one-sided permutation tests over all 40,320 relabelings.

Two things must be said plainly here, in this order.

**The corpus-wide form fails, and an earlier four-threshold subset of this same
experiment gave +0.80 for it.** That subset reproduces +0.800 exactly when the
other four runs are dropped, so this is not a computational correction; it is
what doubling n did to a coefficient that was never significant. The paper
should say so, in one sentence, because it is the cleanest available
demonstration of why the field's small-n conditioning claims need a denominator
in the first place -- including ours.

**Its largest error is diagnostic.** 35 degrees C ranks seventh of eight by
corpus-wide statistic and second by measured gain. It is also the most
class-balanced threshold in the corpus, H(bin) = 0.679 of a possible 0.693. The
residue histogram sees a large difference because the classes are even, but the
hot side at 35 degrees C is mostly 37 degrees C mesophiles, so the label is
semantically mush and the model cannot use it. Conditioning on family removes
most of this: within-cluster reads +0.114% there, ranked third.

This is the within-cluster quantity's one real claim to credibility and it has
to be stated: it was not chosen after seeing these data. It is what
`conditioning_gate/ceiling_gain` has been configured on since 2026-10-26, and
the measure that computes it describes it as the token's value given the
family. The corpus-wide column was preferred for sweep comparisons only because
it carries no cluster-count caveat.

### 5.1c The baseline that reads nothing from the corpus

`-abs(edge - 55)` scores above both. On this corpus, a corpus statistic does not
beat "pick the middle of the temperature range", and the paper has to concede
that in the results rather than in the limitations.

The defence is that the baseline's constant was chosen knowing the answer, and
nothing computed from a new corpus tells you where to centre it -- while
section 5.3 shows the optimum is *not* always near 55. A corpus statistic should
win exactly where the optimum moves. Section 5.2b was pre-registered as that
test and came back inconclusive: the second corpus's measured optimum moved to
65 degrees C, which is not far enough from 55 to put the baseline at risk. So
this stands as an open problem, and the sentence the paper has to carry is that
no statistic tested here beats a rule of thumb at *ranking* label definitions.
The corpus statistics do beat it at predicting the *size* of the gain
(section 5.2b), which is a weaker and less useful claim.

### 5.2 The curve shape transfers to a corpus we did not build

Second corpus: UniProtKB/Swiss-Prot release 2026_03, 575,748 entries, joined to
Engqvist (BMC Microbiology 18:177, 2018; 13,816 prokaryote organisms after
plausibility bounds) rather than to Gosha. 204,039 sequences survive the join
and our length and alphabet filters, over 1,816 species and 64.9M residues,
with 77 distinct growth temperature values and 41,674 clusters. A
design-matched form -- capped at 500 per species and species-balanced, as stage
A is -- gives 35,361 sequences over 380 species and 11.1M residues.

| edge (C) | stage A | external, design-matched |
|---|---|---|
| 30 | +0.095% | +0.106% |
| 35 | +0.198% | +0.213% |
| 40 | +0.154% | +0.274% |
| 45 | +0.154% | +0.269% |
| 50 | +0.190% | +0.304% |
| 55 | **+0.257%** | +0.322% |
| 60 | +0.207% | **+0.339%** |
| 65 | +0.184% | +0.313% |
| 70 | +0.161% | +0.238% |
| 75 | +0.118% | +0.203% |
| 80 | +0.102% | +0.158% |

Spearman +0.809, Pearson +0.823 over eleven thresholds, both peaking in the
interior at adjacent sweep steps. A different growth-temperature database, a
different sequence set, a different curation level and a different set of
organisms give the same curve shape.

### 5.2b Measured gain on the second corpus

Five runs on the design-matched external corpus at 30, 40, 55, 65 and 75
degrees C, identical settings to section 5.1. All five passed Gate 2 at z
between 8.4 and 41.2. Full analysis in
`results/ceiling_sweep_20261030/external_measured.md`.

| edge (C) | thermo% | corpus-wide | within-cluster | val gain | test gain |
|---|---|---|---|---|---|
| 30 | 81.7% | +0.105% | +0.125% | +0.110% | +0.108% |
| 40 | 49.8% | +0.274% | +0.219% | +0.404% | **+0.428%** |
| 55 | 41.3% | +0.322% | +0.237% | +0.420% | +0.429% |
| 65 | 27.5% | +0.313% | **+0.272%** | **+0.449%** | +0.424% |
| 75 | 16.8% | +0.203% | +0.180% | +0.236% | +0.222% |

| predictor | rho (val) | p | rho (test) | p | Pearson (val) |
|---|---|---|---|---|---|
| corpus-wide ceiling | +0.900 | 0.042 | +0.900 | 0.042 | **+0.984** |
| within-cluster ceiling | **+1.000** | **0.008** | +0.700 | 0.117 | +0.971 |
| `-abs(edge - 55)` | +0.900 | 0.042 | +0.900 | 0.042 | +0.830 |

Exact permutation p over all 120 relabelings. This section was pre-registered
before the runs landed, with three questions fixed in advance; they are
answered here in order, including the one that came back against the design.

**The within-cluster statistic predicts measured gain here, and its
pre-registered prediction was right.** It ranked 65 degrees C highest on this
corpus, against 50 degrees C measured on stage A, and 65 degrees C is the
measured validation optimum. Spearman is +1.000 (p = 0.008), the only predictor
of the three significant at n=5. On test it falls to +0.700: 40, 55 and 65
degrees C land within 0.005 pp of each other there, against per-run shuffle
spreads of 0.0004 to 0.0009 pp, so the test ranking is scored on the order
inside a three-way tie. That disagreement is a statement about n=5, not about
the statistic.

**The fixed-centre baseline does not degrade, and the test was weaker than it
was advertised to be.** `-abs(edge - 55)` holds +0.900 on both splits. Two
reasons, both of which belong in the paper rather than in a footnote. First,
the pre-registration claimed the two predictors disagree on the ordering of 40
against 75 as well as on the top; they do not -- both rank 40 above 75, so over
these five edges the orderings differ by one adjacent swap and five runs could
separate them by at most one rank unit. Second and more seriously, **the
optimum did not move far enough**. Stage A's measured optimum is 50 degrees C
and this corpus's is 65; a baseline centred at 55 is wrong by 5 and 10 degrees
and is never exposed. The corpus that would expose it is the uncapped external
corpus of section 5.3, whose ceiling peaks at 35 degrees C, and it has not been
trained on.

What does separate them here is magnitude, not rank: Pearson +0.984 and +0.971
for the two corpus statistics against +0.830 for the baseline, which is linear
in distance from 55 by construction and so cannot reproduce the steep fall at
30 degrees C beside the shallow one at 65. That is a real difference, and it is
not the difference this test was built to measure, so it is reported as an
observation.

**The plateau reproduces, and this is the batch's clearest result.**

| | stage A | external |
|---|---|---|
| cold end (30 C) | +0.037% | +0.110% |
| plateau | +0.255-0.296% (45-65 C) | +0.404-0.449% (40-65 C) |
| hot end (75 C) | +0.121% | +0.236% |
| cold-to-plateau | 7.2x | 3.8x |

Two growth-temperature databases, two sequence sources, two curation levels:
an interior plateau several times the cold end, falling away at both ends.
Gains are uniformly larger on the external corpus -- 1.6x at 55 degrees C, on a
tenth of the residues -- and the `thermo%` column says why rather than letting
it stand as a mystery: at 55 degrees C that corpus is 41.3% thermophilic
against stage A's 28.0%, because species-balancing a Swiss-Prot subset with
sparse thermophilic coverage leaves a hotter corpus. Same threshold, different
class balance, which is section 5.3's confound again.

### 5.2c The threshold ordering reproduces on a second backbone

progen2-small (151M against progen2-base's 764M), stage A corpus, same
conditioning scheme, LoRA rank and token dropout; the backbone and the 1,024
token context it forces are the only changes.

| edge (C) | base val | small val | base test | small test |
|---|---|---|---|---|
| 30 | +0.037% | +0.035% | +0.033% | +0.077% |
| 55 | **+0.255%** | **+0.200%** | **+0.238%** | **+0.184%** |
| 75 | +0.121% | +0.142% | +0.119% | +0.114% |

**55 > 75 > 30 on both backbones and both splits.** The threshold-choice result
is not a property of progen2-base.

The size of the spread does not transfer as cleanly. Cold-to-plateau is 6.9x on
base against 5.7x on small by validation, but 7.2x against 2.4x by test,
because `small_t30` measured +0.035% on validation and +0.077% on test. That
run is also the batch's only Gate 2 failure: significant at z 7.2, then 0.36x
of its ceiling against a 0.50x floor, then 0.81x on test. At that size the
val/test gap stops being noise around a stable quantity, so the cold-end
magnitude on this backbone is not quotable as a number.

At the plateau the smaller model reaches 0.78x the larger model's gain on both
splits -- the expected direction, on two points, so a direction and not a slope.

This is a **capacity** comparison and not an architecture-independence result.
Both models are ProGen2 decoders sharing a tokenizer and a conditioning
scheme. A second architecture family is untested.

### 5.3 Sampling design is half the measurement

The uncapped, unbalanced external corpus peaks at 35 degrees C instead, and
agreement with stage A drops to Spearman +0.582. The cause is in the class
masses: between 35 and 40 degrees C the hot class loses four fifths of its
residues, 49.4% to 9.5%, because Swiss-Prot is dominated by clinical and model
organisms annotated at 37 degrees C and holds almost nothing between 37 and 45.
The 35 degrees C peak is that cliff.

So the bound's peak location is a joint property of the corpus and the
species-selection design, and a ceiling quoted without the design is not
reproducible. We prefer to own this as a result rather than have it found: it
is the main thing a practitioner needs to know before computing the measure on
their own data.

### 5.4 What the 45 degrees C convention actually costs: much less than the statistic predicts

The corpus statistic puts the convention well below the optimum in all three
corpora -- 1.67x, 1.48x and 1.26x. Measured, it is not:

| | predicted by statistic | measured |
|---|---|---|
| optimum | 55 C | 50 C |
| gain at the optimum | +0.257% | +0.296% |
| gain at 45 C | +0.154% | +0.260% |
| shortfall of the convention | 1.67x | **1.14x** |

And 1.14x is roughly 2 sigma on the shuffle spread, with 65 degrees C (+0.282%)
within noise of both. So the convention sits at the low edge of the measured
plateau and retuning inside the plateau buys little.

This is a correction the paper has to make in its own voice, because the
earlier ceiling-based version of it was the most quotable thing here. What
replaces it is narrower and still useful:

- **Getting the threshold badly wrong is expensive** -- 3.9x between a cold
  threshold and the plateau. That is the claim with margin to spare.
- **Getting it slightly wrong is cheap.** 45 degrees C is nearly optimal on
  this corpus, and anyone who has used the convention has lost about 14%.
- **The statistic was wrong about the size of the available improvement**, in
  the direction that would have motivated unnecessary work. A measure that
  overstates its own payoff has to report that.

### 5.5 The bound is not only class balance

Mutual information is largest when the marginal is even, so the obvious
objection is that a threshold sweep finds the median growth temperature by an
expensive route. The sweep reports $H(b)$ and $I/H$ so this is answerable from
its output.

| corpus | Spearman(H, ceiling) | most balanced edge | peak | balance explains it? |
|---|---|---|---|---|
| stage A | +0.545 | 35 C | 55 C | no |
| external, natural | +0.664 | 35 C | 35 C | **yes** |
| external, design-matched | +0.627 | 40 C | 60 C | no |

Balance correlates with the bound everywhere and determines the peak only in
the corpus with the 37 degrees C cliff. In both designed corpora the peak sits
15 to 20 degrees C above the most balanced threshold, so something other than
the marginal carries it: at hot thresholds the two classes' residue
compositions differ more, which is what the measure is for.

$I/H$ rises monotonically toward hot thresholds in all three corpora, peaking
at 70, 75 and 65 degrees C. That is consistent cross-corpus structure and it is
not the quantity that predicted measured gain, so it stays an observation.

## 6. Where the measure fails

This section is not a limitations paragraph bolted on at the end. For a
measurement paper it is a contribution, and it should be a numbered section
with its own results.

### 6.1 It ranks, loosely; it does not estimate

Corpus-wide, `gain / statistic` spans 0.39x to 1.69x, a 4.3x spread. A variant
can land at 40% of the figure or at 170% of it. Exceeding it is permitted --
the quantity is unigram and the model has context -- but a predictor undershot
by 2.6x in one variant and overshot by 1.7x in another estimates nothing.

Within-cluster is tighter. The six thresholds at and above 45 degrees C span
1.45x to 2.01x, a factor of 1.4, with the two cold thresholds below 1x. That
looks close to a fixed multiple of about 1.6 over the plateau, and it would be a
calibrated predictor if it held. It must be reported as an observation and not
a result, for a reason worth stating in the paper: the eight thresholds were
split into "plateau" and "cold" after the table was in hand, and a ratio that
tight across six post-hoc-selected points is exactly the shape a small sample
produces by chance. It is a hypothesis for the external corpus to test, nothing
more.

### 6.2 It does not rank across granularities, and the obvious fix is worse

`sweep_c4` has the highest bound of the five by a wide margin, +0.374%, 1.5x
the 55 degrees C threshold's, and a *lower* measured gain than both t45 and
t55. Four classes raise the bound without raising what the model extracts, so a
single number does not order label definitions that differ in kind.

The natural repair is to normalize by label entropy, since refinement raises
both $I$ and $H$. It makes things worse. Spearman against measured gain goes
from +0.600 raw to -0.300 normalized across all five runs, and from +0.800 to
-0.200 on the four thresholds. We report this as a dead end rather than omit
it. Whether the granularity axis needs its own bound, or whether the gap is a
capacity limit on a single control token rather than a corpus property, is open.

### 6.3 There is no peak to locate, only a plateau

At n=4 this section would have said the peak is interior but unresolved. At
n=8 the shape is clearer and different: 45, 50, 55 and 65 degrees C lie within
0.041 pp of each other, which is about 2 sigma end to end, so there is no
identifiable optimum. "Measured gain plateaus between 45 and 65 degrees C" is
the strongest true statement, and the corpus limits resolution further -- 47
distinct growth-temperature values across 311,987 rows with an 8 degrees C gap
between 37 and 45.

This has a consequence worth drawing out rather than burying, because it
weakens the practical value of any predictor: if the useful region is a wide
plateau, then the precision a corpus statistic could add over a rule of thumb
is small, and the thing worth predicting is which thresholds fall *off* the
plateau. Both forms of the statistic get the cold end right.

### 6.4 Scope

Prokaryotes only; Engqvist's 7,682 eukaryote rows are excluded, as are the
299,890 Swiss-Prot entries whose organism is absent from it. Two backbones,
both ProGen2 decoders with the same tokenizer and conditioning scheme, so
section 5.2c is a capacity comparison and not an architecture-independence
result. Measured gain is validated on two corpora (5.2b), both of which were
assembled by the same species-balancing and per-species capping design; the
uncapped corpus of 5.3, which is the one whose optimum sits elsewhere, has not
been trained on. The two growth-temperature databases both
scrape overlapping culture-collection catalogues (DSMZ, ATCC, JCM), so their
agreement is evidence the bound is not an artifact of one *corpus* and not
evidence that it survives an independent *measurement* of growth temperature.
The design-matched external corpus is a tenth of stage A's residue mass and
hotter at matched thresholds (41.3% thermophilic at 55 degrees C against stage
A's 28.0%), so its larger measured gains are not evidence that a smaller corpus
conditions better. Its within-cluster figures at 75 and 80 degrees C rest on 72
and 59 clusters; the within-cluster column is used for the 30-65 degrees C
comparisons in 5.2b, where cluster counts are 125 to 156, and the corpus-wide
column elsewhere.

## 7. A worked negative case: Cas9

Claim: **the measure's practical use is a go/no-go decision before the budget
is spent, and here it says no.**

A Cas9 corpus, collected for exactly the conditioning experiment above: 1,490
candidates from six Cas9-specific Pfam families and a protein-name query,
1,444 carrying both RuvC and HNH domains at i-Evalue <= 1e-3, 1,019 joined to a
growth temperature, 208 clusters at 45% identity and 80% coverage.

| gate | outcome |
|---|---|
| Gate 1, sequences per class | **fail** -- 23 thermophilic against a floor of 200 |
| Gate 2, measurable conditioning | **not evaluable** -- the test split holds zero thermophilic sequences, so shuffling the token across it is a no-op by construction |
| Leakage | **fail** -- 222 pairs over 21 held-out sequences above 45% identity and 80% coverage, despite cluster-disjoint splitting |
| Corpus-wide bound | +0.008%, against stage A's +0.154% |

Two points worth making rather than one. First, the bound is 19x smaller than
stage A's and the gates fail independently of it, so three cheap checks agree
and the experiment is correctly abandoned for the price of a CPU afternoon.
Second, the within-cluster figure for this corpus is +0.119%, computed from
three clusters and 57,385 residues, which looks almost usable and is not: MI at
small residue counts is upward-biased, three clusters is not a sample, and the
honest conclusion is that this corpus cannot measure its own ceiling. We
recorded it and refused to configure it, and a measurement paper should say
where its own measurement is untrustworthy.

Note also that the leakage failure is a general warning: MMseqs2 greedy
set-cover guarantees dissimilar cluster *representatives*, not dissimilar
members across clusters.

## 8. Discussion

Claim: **label definition is an untuned free parameter, and it is cheap to
tune.**

- For anyone training a temperature-conditioned model: compute the sweep first.
  It costs CPU-minutes and it moves a parameter worth 1.3x to 1.7x.
- Report gains against the bound. A gain of X% means nothing on its own; X% of
  the bound your corpus allows is comparable across papers.
- Hold the species-selection design fixed and report it, because the peak moves
  with it.
- The strongest follow-up, and it needs no new training: compute the bound for
  the published setups' own corpora. The Engqvist data behind PRIME's ~96M
  pairs is public. "Their reported gain is X% of what their corpus allows"
  would make the measure useful to everyone already in this space rather than
  only to us. Flagged as future work, not claimed.

## Figures

Two, both computable from committed results with no new runs.

**Figure 1** -- two panels side by side, same axes: corpus-wide statistic (x)
against measured gain (y), and within-cluster against measured gain. Eight
stage A points each, with the five external-corpus points overplotted in a
second marker so both corpora are visible on one scale, error bars from the
shuffle spread, rho and the exact p printed in each panel. This is the figure that carries 5.1b, and putting the failing
predictor beside the working one is the whole point -- a reader should be able
to see that the left panel is a cloud and the right is a line. Mark 35 degrees
C in both: it is the point that moves most between them and the mechanism in
5.1b is about it. `sweep_c4` goes in the left panel only, marked distinctly, for
6.2.

**Figure 2** -- bound against threshold, three corpora on shared axes, peaks
marked. A second row, or a twin axis, shows the `thermo%` column so the 37
degrees C cliff is visible where the natural corpus's peak sits. Carries 5.2,
5.3 and 5.4.

Figure 3, if a third is affordable: measured gain against threshold for the
three trained configurations -- stage A on progen2-base, the external corpus,
and stage A on progen2-small -- on one panel, which is the plateau result of
5.2b and 5.2c in one picture and the most quotable thing in the paper. The
granularity figure for 6.2 is below it in priority.

## Citations to collect

PRIME (Sci. Adv. 2024); ThermoFormer (NeurIPS 2024 workshop); Steering PLMs
(ICML 2025); NOMELT (Sci. Rep. 2025); OpenCRISPR-1 (Nature 2025); TemStaPro
(Bioinformatics 2024); ProGen2; LoRA; MMseqs2; HMMER; Pfam; UniProtKB;
Engqvist (BMC Microbiol. 18:177, 2018, doi 10.5281/zenodo.1175609, CC BY 4.0);
Gosha; Poole et al. (ICML 2019, arXiv 1812.01475); arXiv 2203.02055; arXiv
1812.10358; arXiv 2507.15651; Off-Manifold Collapse (arXiv 2608.18597);
Shannon-McMillan-Breiman for the entropy-rate framing.

Performance figures for PRIME, OpenCRISPR-1 and NOMELT are author-reported and
no independent replication was found; cite them as reported.

## Standing cautions for the writing

- The bound is a **lower** bound. No saturation claims, anywhere, in any
  phrasing.
- Structure prediction is a validity filter. It is not a thermostability assay
  and no figure title may imply otherwise.
- Growth temperature and sampling temperature stay separate in every name,
  caption and sentence.
- The auxiliary OGT head is PRIME's.
- `sweep_t45` was launched with `--ceiling-gain 0.00172`, the within-cluster
  figure, so its own Gate 2 report prints 1.51x where the corpus-wide column
  gives 1.69x. The flag sets only that printed denominator; it does not touch
  training or the measured gain. Every number in section 5.1 uses the
  corpus-wide column for all five runs. If the per-run reports go into an
  appendix, this needs a footnote there.

## What is missing before submission

Status 2026-10-31, after all thirteen runs. **No experiment in the plan is
outstanding, and no further GPU time is needed for the paper as scoped.** What
is left is writing and two figures, neither of which needs compute:

- **Prose for sections 1-4 and 7-8.** These are still skeletons.
- **Figures 1 and 2**, both computable from committed results.
- **A decision on the open problem of 5.1c.** One experiment would settle it:
  three or four runs on the *uncapped* external corpus, spanning 30 to 60
  degrees C, where the within-cluster statistic and `-abs(edge - 55)` genuinely
  disagree about which threshold is best -- unlike the design-matched corpus,
  where it turned out they barely disagree at all. Roughly $9-12 of A10 time
  plus a Cas9 screen and a split assignment, both CPU-only. Either run it or
  state the limitation and submit; do not restate the pre-registered test of
  5.2b as though it had resolved this.

Spend so far is about $35 of A10 time against the $31 estimated, the difference
being that the runs at 2.6 h each cost more than the round numbers suggested.

The experiment that would be worth designing beyond that is a third corpus with
a genuinely independent growth-temperature measurement rather than another
culture-collection scrape, and a second architecture family for 5.2c. Neither
is reachable on this budget. Note both as future work.

## Revision history

**2026-10-31, after the n=8 runs.** The four added thresholds (35, 50, 65, 70)
dropped the corpus-wide statistic's rank correlation with measured gain from
+0.80 to +0.38, p = 0.18. Sections 1, 2 (abstract), 3, 5.1, 5.4, 6.1, 6.3, the
figure spec and the submission checklist were rewritten. Three specific claims
were removed and must not return:

- "the conventional 45 C threshold leaves 1.3-1.7x of the available signal
  unused" -- measured, the convention costs 1.14x at roughly 2 sigma
- "measured gain follows the corpus-only bound across thresholds" -- it does
  not, in the corpus-wide form
- "the ceiling's non-monotonic interior peak is reproduced" -- there is no
  identifiable peak, only a plateau from 45 to 65 C

What survived the rewrite: the relabeling-only design, the 3.9x spread from
label definition, the cross-corpus agreement of the ceiling curve, the
sampling-design result, the granularity failure, and the Cas9 negative case.

The full analysis, including the exact permutation tests and the diagnosis of
the 35 C miss, is in `results/ceiling_sweep_20261030/measured_n8.md`.

**2026-10-31, after the external-corpus and backbone runs.** Sections 5.2b and
5.2c written, 5.1c and the abstract amended, the submission checklist closed
out. The pre-registered test of section 5.2b **did not pass**: the
within-cluster statistic predicted the second corpus's measured optimum
correctly (rho +1.000 on validation, p = 0.008) but the fixed-centre baseline
did not degrade, because the two corpora's measured optima are 50 and 65
degrees C and both sit within 10 degrees of the baseline's constant. Two
further claims must not appear:

- "a corpus statistic beats a fixed-centre rule of thumb at ranking label
  definitions" -- untested on any corpus where the two disagree substantially
- "the cold-to-plateau spread is backbone-independent in size" -- the ordering
  reproduces on progen2-small, the magnitude does not, and the cold-end run is
  the batch's one Gate 2 failure

What the batch added: the interior plateau reproduces on a second corpus and a
second backbone, and the corpus statistics beat the baseline on magnitude
(Pearson +0.98 and +0.97 against +0.83) if not on rank.

Analysis in `results/ceiling_sweep_20261030/external_measured.md`.
