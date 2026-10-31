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
clusters and splits -- we find that how a growth-temperature label is binned is
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
better. And on this corpus neither form beats a baseline that reads nothing
from the corpus at all -- distance from the middle of the temperature range --
so a corpus statistic earns its cost only where the optimum moves, which we
test on a second corpus whose optimum does move. Finally, a worked negative case
shows two corpus gates correctly ruling out a Cas9 conditioning experiment
before any GPU time is spent.

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

Setup: ProGen2-base with LoRA, stage A, 3,000 steps, A10. Conditioning is
measured as the relative perplexity improvement from the correct control token
over shuffled ones, five shuffles per run, on 2,000 held-out sequences and
again on a cluster-disjoint test split.

Corpus (stage A): 311,987 sequences, 113.8M residues, 47 distinct growth
temperature values, 1,552 both-class clusters at the 45 degrees C threshold,
279,980 training sequences. Growth temperatures from Gosha; sequences from
UniProt, species-selected and capped at 500 per species.

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
win exactly where the optimum moves. Section 5.2b is that test; until it
reports, this is an open problem and not a resolved one.

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

PENDING: five runs on the design-matched external corpus at 30, 40, 55, 65 and
75 degrees C, launched 2026-10-31, call ids in
`results/ceiling_sweep_20261030/runs.txt`. Edges were chosen to span the
within-cluster statistic's range on that corpus while matching four of stage
A's thresholds.

This section carries the paper's load-bearing test, so what it has to report is
fixed in advance here rather than chosen once the numbers are in:

1. Does the within-cluster statistic predict measured gain on this corpus too?
   It ranks 65 degrees C highest there; `-abs(edge - 55)` ranks 55 highest.
   They disagree on the top two and on the ordering of 40 against 75, so five
   runs can separate them.
2. Does the fixed-centre baseline degrade on a corpus whose ceiling peaks at
   60 degrees C rather than 55?
3. Does the plateau reproduce, and does the cold end stay poor?

If the statistic loses to the fixed-centre baseline here as well, the honest
paper is the negative one: label definition matters by 3.9x, and no corpus
statistic we tested predicts which definition is best better than a rule of
thumb. That version is worth publishing and the draft should be written so it
can absorb that outcome without restructuring.

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
299,890 Swiss-Prot entries whose organism is absent from it. One backbone.
Measured gain is validated on one corpus; section 5.2 validates the bound's
shape across two, not the gain. The two growth-temperature databases both
scrape overlapping culture-collection catalogues (DSMZ, ATCC, JCM), so their
agreement is evidence the bound is not an artifact of one *corpus* and not
evidence that it survives an independent *measurement* of growth temperature.
The design-matched external corpus is a tenth of stage A's residue mass. Its
within-cluster figures at 75 and 80 degrees C rest on 72 and 59 clusters, so
every comparison above uses the corpus-wide column.

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
points each, error bars from the shuffle spread, rho and the exact p printed in
each panel. This is the figure that carries 5.1b, and putting the failing
predictor beside the working one is the whole point -- a reader should be able
to see that the left panel is a cloud and the right is a line. Mark 35 degrees
C in both: it is the point that moves most between them and the mechanism in
5.1b is about it. `sweep_c4` goes in the left panel only, marked distinctly, for
6.2.

**Figure 2** -- bound against threshold, three corpora on shared axes, peaks
marked. A second row, or a twin axis, shows the `thermo%` column so the 37
degrees C cliff is visible where the natural corpus's peak sits. Carries 5.2,
5.3 and 5.4.

Figure 3, if a third is affordable: the sweep's four granularity settings
against measured gain, making 6.2 visual. Lower priority than the first two.

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

Status 2026-10-31. Threshold resolution is done: n=8, and it changed the
result. Two batches are in flight, call ids in
`results/ceiling_sweep_20261030/runs.txt`:

- **Five runs on the design-matched external corpus.** Section 5.2b. This is
  now the load-bearing experiment rather than a nice-to-have, because it is the
  only thing that can separate the within-cluster statistic from the
  fixed-centre baseline.
- **Three runs on progen2-small**, at 30, 55 and 75 degrees C. Tests whether
  the 3.9x spread is capacity-dependent. Same family and tokenizer, so it does
  not test architecture independence and section 6.4 must say so.

Prerequisites, done 2026-10-30: the external corpora now have cluster-disjoint
splits with both classes in both held-out sets, and the design-matched one has
been screened for Cas9 (11 of 35,361 sequences removed, ceilings unchanged to
0.001 pp).

Spend so far is about $35 of A10 time against the $31 estimated, the difference
being that eight runs at 2.6 h each cost more than the round numbers suggested.

**After these eight, stop and write.** The one experiment that would be worth
designing afterwards is a third corpus with a genuinely independent
growth-temperature measurement rather than another culture-collection scrape,
and it is not reachable on this budget. Note it as future work and submit.

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
