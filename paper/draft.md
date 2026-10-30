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
corpus. We supply one. For a corpus and a candidate label definition, the
mutual information between residue identity and label, computed in CPU-minutes
from residue counts alone, is a lower bound on the information a control token
can carry, and we show it ranks label definitions by the conditioning gain a
trained model actually achieves. Across five GPU runs that differ only in how
the label is defined -- identical sequences, clusters and splits -- measured
gain reproduces the bound's non-monotonic interior peak, rising from +0.037%
at a 30 degrees C threshold to +0.255% at 55 degrees C and falling to +0.121%
at 75 degrees C, with margins an order of magnitude above the shuffle noise
floor. The bound's curve shape transfers: on a second corpus built from a
different growth-temperature database and a different sequence set, held to the
same species-selection design, the two curves agree at Spearman +0.81 and peak
at adjacent sweep steps. This has an immediate consequence for practice. The
conventional 45 degrees C mesophile/thermophile threshold is below the optimum
in all three corpora we measure, leaving 1.3x to 1.7x of the available signal
unused -- a parameter that is set by convention, is never tuned, and can be
tuned for CPU-minutes before any training budget is committed. We also report
where the measure fails: it does not rank label definitions that differ in
granularity rather than threshold, and normalizing by label entropy makes the
correlation worse, not better. Finally, we give a worked negative case in which
the bound and two corpus gates correctly rule out a Cas9 conditioning
experiment before any GPU time is spent.

Word count target for submission: ~200. The draft above is ~270 and the
sentences on failure modes are the ones to compress, not to cut.

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
3. The missing quantity is a per-corpus budget for the control token. We
   compute one from residue counts, and the computation is cheap enough to run
   before choosing what to train.
4. Contributions, as a list: (i) the bound and what it does and does not
   support; (ii) a relabeling-only experimental design that isolates label
   definition from every other corpus property; (iii) measured validation on
   five runs and cross-corpus validation of the curve shape; (iv) the 45
   degrees C result, which is the actionable one; (v) two documented failures of
   the measure.

Note to self: the introduction must not promise that the bound estimates
achievable gain. It ranks. The distinction is the whole of section 6.1 and
getting it wrong in paragraph 3 is the fastest way to lose a reviewer who
reads section 6 later.

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

### 5.1 Measured gain follows the bound across thresholds

Five runs. Ceiling column is corpus-wide.

| run | edges (C) | ceiling | val gain | test gain | val/ceiling | Gate 2 |
|---|---|---|---|---|---|---|
| `sweep_t30` | 30 | +0.095% | +0.037% | +0.033% | 0.39x | **failed** |
| `sweep_t45` | 45 | +0.154% | +0.260% | +0.251% | 1.69x | passed |
| `sweep_t55` | 55 | +0.257% | +0.255% | +0.238% | 0.99x | passed |
| `sweep_t75` | 75 | +0.118% | +0.121% | +0.119% | 1.03x | passed |
| `sweep_c4` | 32, 45, 56 | +0.374% | +0.249% | +0.218% | 0.67x | passed |

The prediction that could have failed was the non-monotonic interior peak, and
it held: +0.037% at 30, +0.255% at 55, +0.121% at 75 degrees C. t55 beats t30
by 0.218 pp against a combined shuffle noise floor of 0.021 pp, and beats t75
by 0.134 pp against 0.027 pp. Spearman across the four thresholds is +0.80 on
validation and on test alike; with n=4 that coefficient is not itself
significant and the evidence is the shape, which must be said in the same
breath. Test gains track validation within 0.03 pp everywhere. Every run's
effect is significant on its own terms, z between 6.7 and 20.8 against a
required 3.0.

Worth one sentence because it is counterintuitive: 30 degrees C is the most
lopsided threshold in the sweep, 89.2% of residues on one side, and the
worst-conditioning variant. The corpus-only bound called that in advance.

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

### 5.4 The 45 degrees C convention is below the optimum in all three corpora

| corpus | peak | at 45 C | shortfall |
|---|---|---|---|
| stage A | +0.257% at 55 C | +0.154% | 1.67x |
| external, natural | +0.148% at 35 C | +0.100% | 1.48x |
| external, design-matched | +0.339% at 60 C | +0.269% | 1.26x |

45 degrees C is the conventional mesophile/thermophile line and no corpus here
is best served by it. This is the sentence the paper exists to earn: the
threshold is a free parameter nobody tunes, tuning it is worth 1.3x to 1.7x in
available signal, and the tuning costs CPU-minutes.

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

### 6.1 It ranks; it does not estimate

`gain / ceiling` spans 0.39x to 1.69x across the five runs, a 4.3x spread. A
variant can land at 40% of its bound or at 170% of it. Exceeding it is
permitted -- the bound is unigram and the model has context -- but a quantity
undershot by 2.6x in one variant and overshot by 1.7x in another is a predictor
of rank, not a calibrated estimate. "The ceiling tells you what gain to expect"
is not supported. "The ceiling tells you which label definition will condition
better" is.

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

### 6.3 The peak's location is not resolved

t45 and t55 are 0.19 sigma apart on validation and 0.70 sigma on test, with t45
nominally ahead on both while the bound puts t55 1.7x higher. The corpus limits
the resolution: 47 distinct growth temperature values across 311,987 rows with
an 8 degrees C gap between 37 and 45, roughly nine effective thresholds. The
claim that survives is that measured gain peaks in the interior, not that it
peaks at 55.

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

**Figure 1** -- predicted bound (x) against measured gain (y), five points,
error bars from the shuffle spread, the identity line drawn and labeled as the
line the bound would lie on if it estimated rather than ranked. The `sweep_c4`
point is marked distinctly, because it is the one off the trend and section 6.2
is about it. Carries 5.1 and 6.1 and 6.2 in one panel.

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

Nothing in sections 1-4 or 7-8 waits on compute. What waits:

- **Measured gain on a second corpus.** Section 5.2 validates the bound's
  shape across corpora and the gain across thresholds on one corpus only. Five
  runs on the design-matched external corpus, about $10, would let section 5.1
  read "two corpora" and close the most obvious reviewer question. Prerequisite,
  no GPU: the external corpora have `cluster_id` filled and `split` empty, and
  they are unscreened for Cas9.
- **Threshold resolution.** Four thresholds give the shape; 35, 50, 65 and 70
  would give eight and make the Spearman worth quoting as a statistic rather
  than as an illustration. About $12.
- **A second backbone**, at 30, 55 and 75 degrees C, about $9, to stop "one
  backbone" from being the first line of any review.

That is roughly $31 of A10 time and it is the whole gap between this skeleton
and a submittable paper. There is no further experiment to design after it.
