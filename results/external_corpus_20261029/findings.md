# The ceiling on a corpus we did not build

Date: 2026-10-29. Sweeps in `sweep_pretrain.txt`, `sweep_external.txt`,
`sweep_external_balanced.txt`; corpus built by `scripts/16_external_corpus.py`.

## The question

The five GPU runs showed measured conditioning gain tracking the corpus-only
ceiling across threshold choices, on one corpus: ours, assembled by our own
species balancing and per-species capping. Two explanations survived that
result. Either the ceiling is a property of corpora, or the curve is a
property of how stage A was built. This is the second observation that
separates them, and it is CPU-only.

## What was built

| | stage A (ours) | external, natural | external, balanced |
|---|---|---|---|
| OGT source | Gosha | Engqvist | Engqvist |
| sequences | UniProt, Gosha-selected species, capped at 500 | Swiss-Prot 2026_03, every species, uncapped | Swiss-Prot, capped at 500 + species-balanced |
| sequences | 311,987 | 204,039 | 35,361 |
| residues | 113.8M | 64.9M | 11.1M |
| species | — | 1,816 | 380 (190 per class) |
| thermophilic at 45 C | 39.9% | 9.2% | 49.5% |
| distinct OGT values | 47 | 77 | 63 |
| clusters | — | 41,674 | 12,687 |

Engqvist, BMC Microbiology 18:177 (2018), data at 10.5281/zenodo.1175609
(CC BY 4.0), 13,816 bacterial and archaeal organisms after plausibility
bounds. Sequences from UniProtKB/Swiss-Prot release 2026_03 (02-Sep-2026),
575,748 entries, of which 204,039 survive the organism join and stage A's own
length and alphabet filters.

Neither external corpus is training data and neither has been screened for
Cas9. They exist to have their residues counted.

## Result 1: under matched sampling design, the two corpora agree

| edge (C) | stage A | external balanced |
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

Spearman +0.809, Pearson +0.823 over eleven thresholds. Both peak in the
interior, at 55 and 60 C -- adjacent steps of the sweep -- and both fall away
at each end. A different growth-temperature database, a different sequence
set, a different curation level and a different set of 1,816 organisms produce
the same curve shape and nearly the same peak, provided the species-selection
design is held fixed.

That is the result worth having. The shape is not an artifact of our corpus.

## Result 2: without matched design, the peak moves to the database's mode

The uncapped, unbalanced external corpus peaks at **35 C**, not 55:

| edge (C) | thermo% | ceiling |
|---|---|---|
| 30 | 68.8% | +0.113% |
| 35 | 49.4% | **+0.148%** |
| 40 | 9.5% | +0.098% |
| 45 | 9.2% | +0.100% |
| 55 | 7.8% | +0.112% |
| 60 | 6.4% | +0.115% |
| 80 | 3.0% | +0.067% |

Agreement with stage A drops to Spearman +0.582. The cause is visible in the
`thermo%` column: between 35 and 40 C the hot class loses four fifths of its
mass, 49.4% to 9.5%. Swiss-Prot is dominated by clinical and model organisms
annotated at 37 C, so an unselected reviewed corpus has a pile-up there and
almost nothing between 37 and 45. The peak at 35 C is that cliff, not a
property of thermophilic sequence.

So the peak's location is a joint property of the corpus and the sampling
design. Sampling design is not a nuisance parameter here; it is half the
measurement, and a ceiling quoted without it is not reproducible.

## Result 3: the 45 C convention is below the optimum in all three

| corpus | peak | at 45 C | shortfall |
|---|---|---|---|
| stage A | +0.257% at 55 C | +0.154% | 1.67x |
| external natural | +0.148% at 35 C | +0.100% | 1.48x |
| external balanced | +0.339% at 60 C | +0.269% | 1.26x |

45 C is the conventional meso/thermo line and no corpus here is best served by
it. The actionable version of this work is not "condition on growth
temperature"; it is "the threshold is a free parameter nobody tunes, tuning it
is worth 1.3-1.7x in available signal, and the tuning costs CPU-minutes".

## Result 4: the balance objection fails on two of the three

Mutual information is largest when the marginal is even, so the obvious
objection is that a threshold sweep finds the median growth temperature by an
expensive route. `15_ceiling_sweep.py` now reports `H(bin)` and `I/H` so this
is answerable from the output.

| corpus | Spearman(H, ceiling) | most balanced edge | peak | balance explains it? |
|---|---|---|---|---|
| stage A | +0.545 | 35 C | 55 C | no |
| external natural | +0.664 | 35 C | 35 C | **yes** |
| external balanced | +0.627 | 40 C | 60 C | no |

Balance is correlated with the ceiling everywhere, and determines the peak
only in the corpus whose temperature distribution has the 37 C cliff. In both
designed corpora the peak sits 15-20 C above the most balanced threshold, so
something other than the marginal is carrying it: at hot thresholds the two
classes' residue compositions differ more, and that is what the measure is
for.

`I/H` -- information per nat of label -- rises monotonically toward hot
thresholds in all three corpora (peaking at 70, 75 and 65 C). That is a
consistent cross-corpus structure, and it is not the quantity that predicted
measured gain, so it stays an observation rather than a claim.

## What this does not establish

**No measured gain on this corpus.** Everything above is the ceiling. Whether
a trained model's gain follows the ceiling *here* is untested; that needs the
five-variant design re-run on the external corpus, which is about $14 and the
obvious next step. Until then the GPU evidence is still n=1 corpus.

**The two OGT sources are not independent.** Gosha and Engqvist both scrape
overlapping culture-collection catalogues (DSMZ, ATCC, JCM). Agreement is
evidence that the ceiling is not an artifact of one *corpus*; it is not
evidence that it survives an independent *measurement* of growth temperature.

**The balanced external corpus is small.** 380 species and 11.1M residues,
because balancing 190 thermophilic species against the mesophilic side after a
500-per-species cap is what Swiss-Prot's thermophilic coverage allows. Its
ceilings are computed over a tenth of stage A's residues.

**Two columns are noisy at the hot end.** The external corpus's within-cluster
figures at 75 and 80 C (+0.126%, +0.239%) rest on 72 and 59 clusters over 2.3M
and 1.0M residues. The corpus-wide column is the one used for every comparison
above.

**Eukaryotes are excluded** (7,682 of Engqvist's 21,498 rows), as are the
299,890 Swiss-Prot entries whose organism is absent from Engqvist. This is a
prokaryotic result.
