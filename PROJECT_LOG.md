# Project log

Append-only record of decisions, versions and costs. Newest entries at the
bottom of each section.

---

## 2026-10-19 — Audit of the prior repository

Audited `somlunavat/thermal-adaptive-protien-predictor` (local copy at
`~/Desktop/proj/thermal-adaptive-protien-predictor-clean-main`, identical to the
GitHub tree except for `README.md`). It is a 6-layer, d_model 256 masked LM with
scalar growth-temperature conditioning.

### What it does

`utils/processingData/data_collector.py` queries NCBI web BLAST for Cas9
homologs; `tempAndSequence.py` joins each hit to a species OGT table and fetches
full sequences from Entrez; `combineData.py` concatenates the CSV shards.
`model/dataloader.py` tokenizes into a 22-token amino-acid vocabulary and builds
15% masked-LM samples; `mlmModel.py` adds a projected scalar temperature
embedding to token and learned positional embeddings; `trainer.py` runs
Lightning with cosine warm restarts.

### What breaks at Cas9 length (~1,100–1,400 aa)

Cas9 is 2–3× the model's context. The failures compound:

1. **Full-length Cas9 is silently discarded, not truncated.**
   `ProteinDataset._filter_data` keeps a sequence only when
   `30 <= len(seq) <= max_length`, and `max_length` is capped at 512 in
   `setup_training` (`min(p95, 512)`). Every full-length Cas9 fails that
   predicate and is dropped with no warning. The model has therefore never seen
   a complete Cas9. This is the single most important finding: the prior results
   are not results about Cas9.
2. **Learned absolute positional embeddings cannot extrapolate.**
   `nn.Embedding(max_length, d_model)` has no values beyond index 511, so the
   context cannot be extended at inference. Needs rotary or ALiBi.
3. **Attention memory is quadratic and unmanaged.** `nn.MultiheadAttention`
   materializes the full L×L matrix. At L=1500, 8 heads, batch 16 that is
   ~288M floats (~1.1 GB fp32) per layer for the forward pass alone, times 6
   layers plus backward. Infeasible in 16 GB unified memory. Needs
   `scaled_dot_product_attention`.
4. **Generation is invalid for a masked LM.** `generate_sequence` samples
   left-to-right from a model trained only with bidirectional masked
   reconstruction and never trained with a causal mask, starting from one random
   residue. A masked LM requires iterative masked denoising.
5. **Conditioning cannot be strengthened or ablated.** Growth temperature enters
   as one `Linear(1, d_model)` vector added uniformly to every position, with no
   label dropout, so there is no classifier-free-guidance path and no clean
   no-label ablation. The brief's design (a discrete bin token with ~10%
   dropout) is not expressible.
6. **Normalization is inconsistent.** `TemperatureEmbedding` rescales by a
   hard-coded `(0, 100)` range while `ProteinDataset` computes `temp_mean` and
   `temp_std` that are never used.
7. **`precision=16` is unusable on `mps`.** fp16 autocast is not supported;
   needs bf16 or fp32 selected per device.
8. **No leakage control whatsoever.** Splits are arbitrary CSV files. Because the
   corpus is BLAST homologs of a handful of seeds, near-duplicate sequences
   certainly straddle the splits. Any reported metric is therefore
   uninterpretable. This is the fatal scientific flaw and the reason
   cluster-level splitting plus an enforced leakage test comes before training.
9. **Collection method conflicts with the brief.** NCBI web BLAST with 10-second
   sleeps and hand-edited row ranges is not reproducible and is the wrong tool
   for bulk search; the brief specifies local DIAMOND or MMseqs2.
10. **Reproducibility blockers.** Hard-coded Windows paths, a hard-coded Entrez
    email, hand-incremented output filenames, and a `.gitignore` with `*.csv`
    that would have excluded exactly the label tables and split files that need
    committing.

### Evidence from its own committed outputs

`model/test_sequences/` and `model/test_alphafold_sequences/` contain the prior
generated sequences. They corroborate the analysis above:

- Longest generated sequence is 299 aa; most are 50–200. None approaches Cas9
  length.
- FASTA headers misreport length: `temp80_seq3` is labeled `Length: 200` and
  contains 36 residues.
- `Fold_score` is identical across 4 °C, 42 °C and 80 °C at matched length
  (0.160 at 125 aa, 0.100 at 200 aa in every file), i.e. it is a function of
  length alone and carries no growth-temperature signal and no structural
  information.
- Composition is degenerate — glycine-dominated, e.g. 15/36 G in one sequence.

**Decision:** do not extend the prior codebase in place. Rebuild under the
staged layout, and keep the from-scratch transformer only as a clearly labeled
weak baseline, rewritten with rotary positions and SDPA attention at ≥1,536
context. Its purpose is to show what a small from-scratch model achieves, not to
be competitive.

---

### 2026-10-20 — Tools installed, Pfam profiles corrected by experiment

Installed the bioinformatics tools locally (Homebrew, native arm64 bottles):

| Tool | Version |
| --- | --- |
| DIAMOND | 2.2.8 |
| MMseqs2 | 18-8cc5c |
| HMMER | 3.4 (Aug 2023) |

Then ran hmmsearch against the eight reference seeds instead of trusting the
accessions written into the config. This found a real error.

**The config had the wrong RuvC model.** It pinned PF18541 (RuvC_III), a
generic RuvC subdomain. Querying InterPro for the Pfam domains actually
annotated on SpCas9 and St1Cas9 showed the Cas9-specific model is **PF22702
(Cas9_RuvC)**. With PF18541 alone, St1Cas9 — a canonical type II-A Cas9 —
produced no RuvC hit at all and would have been thrown out of the corpus.

Per-seed results with both models accepted (i-Evalue, domtblout column 13):

| Seed | Length | RuvC | best RuvC model | HNH | Verdict |
| --- | --- | --- | --- | --- | --- |
| SpCas9 | 1368 | 9.9e-70 | Cas9_RuvC | 2.4e-19 | pass |
| SaCas9 | 1053 | 8.7e-23 | Cas9_RuvC | 3.8e-24 | pass |
| GeoCas9 | 1087 | 2.8e-14 | RuvC_III | 9.5e-24 | pass |
| ThermoCas9 | 1082 | 1.1e-14 | RuvC_III | 1.9e-24 | pass |
| NmeCas9 | 1082 | 3.5e-18 | RuvC_III | 1.4e-25 | pass |
| CjCas9 | 984 | 1.2e-20 | RuvC_III | 3.4e-28 | pass |
| St1Cas9 | 1409 | 1.2e-66 | Cas9_RuvC | 1.4e-21 | pass |
| FnCas9 | 1629 | none | — | 6.7e-08 | **fail** |

**Both RuvC profiles are load-bearing.** Three seeds are best matched by
PF22702 and four by PF18541, so neither model alone covers the family. This is
why `hmm_profiles` takes a list of alternatives rather than one accession per
domain.

**Known limitation: type II-B.** FnCas9 matches no Pfam RuvC model, while
hitting the Csx12 / Cas9 C-terminal models at 2.6e-202. It is unambiguously a
Cas9 whose RuvC no current Pfam model captures, so divergent type II-B
orthologs will be under-recovered by this filter. Recorded as a scope
limitation for the paper rather than worked around by accepting a non-RuvC
domain as RuvC evidence, which would weaken the catalytic claim. A test
asserts the gap still exists so that a future Pfam release closing it shows up
as a failure rather than a stale claim.

HNH needs both models too, though less dramatically: every seed is best
matched by PF13395 (HNH_4), but PF01844 hits four of them and costs nothing to
keep.

Profile versions in use: PF22702.3, PF18541.8, PF13395.13, PF01844.30,
PF16595.11, PF16592.11, PF16593.12, PF21069.3, PF22385.1. Fetched from the
InterPro API on 2026-10-20; the HMM headers declare HMMER3/f format built
Nov 2019.

### 2026-10-20 — OGT source verified; two problems in the data

Downloaded and inspected the Gosha database rather than coding against the
brief's description of it.

**Source.** Gosha (Helena-Bueno, Brown & Melnikov, bioRxiv 2021.12.21.473645)
is distributed on figshare, DOI 10.6084/m9.figshare.23257940, CC BY 4.0,
version 1 posted 2023-05-29. Four CSVs; the one that matters is
`all_species_temperatures.csv` (1.36 MB, 25,324 species). Downloaded
2026-10-20. SHA-256 of the all-species file:
`12e3fc9fe2aeceffc51d9e324c4915a75c5a36ec14387aebae9c4f1e1e8c0a33`.

The brief said "CSV downloads updated monthly". The figshare snapshot is
version 1 from 2023-05-29 and has not moved, so the monthly refresh applies to
the web interface, not to this download. Treat the file as a fixed snapshot
and record its checksum with any run.

**Schema** is `Organism`, `Consensus optimal temperature`, then one column per
source culture collection: ARS, ATCC, BCCM, BCRC, CABRI, CCM, CCUG, CECT,
CICC, DSMZ, HAMBI, JCM, KEMB, NBIMCC, NBRC, NCIMB, NCMA, NCTC, NIES, VKM, CIP.
The consensus is the median across whichever collections reported a value.
This is better than expected: **the per-collection columns are the independent
cross-check the brief wanted a second database for**, so TEMPURA/BacDive is
not needed for the basic agreement check. Organism names are bare binomials,
while the NCBI-derived companion files use strain-level names, so joining
needs normalization.

**Problem 1: physically impossible values.** *Bacillus haloalkaliphilus* is
listed at 230 °C and *Halomonas aerodenitrificans* at 187 °C, each from a
single collection. Both are mesophiles. Nothing grows above about 122 °C, so
OGT is now bounded by `ogt.min_plausible_c` / `max_plausible_c` (-20 to 130),
and out-of-range rows are rejected and counted. Exactly 2 of 25,324 rows are
dropped.

**Problem 2: errors inflate temperature, so they land in the thermophilic
classes.** *Streptococcus massilliensis* is listed at 96 °C on the authority
of one collection, which would label a mesophile as a hyperthermophile — in
the very class this project has least data in and most needs to be right
about. Single-collection entries cannot be trusted.

**The cross-check criterion had to be redesigned against the real data.** An
absolute spread tolerance, which the config originally had at 5 °C, rejects
genuine thermophiles: *Pyrococcus furiosus* spans 20 °C across five
collections and *Geobacillus stearothermophilus* 15 °C across eighteen, and
both are beyond doubt thermophilic. Spread grows with temperature. Agreement
is therefore measured on the **bin** rather than on degrees: the fraction of
reporting collections that place the species in the same class as the
consensus. Effect at the species level:

| Criterion | meso | thermo | hyper |
| --- | --- | --- | --- |
| >=2 sources | 13945 | 507 | 173 |
| >=2 sources, spread <= 5 °C | 11677 | 434 | 164 |
| >=2 sources, bin agreement >= 0.67 | 13898 | 484 | 171 |

Bin agreement keeps more genuine thermophiles while still rejecting the
single-source errors, which the source count catches on its own. It also
correctly flags real ambiguity: *Thermus thermophilus* at 67 °C has sources on
both sides of the 70 °C boundary and scores 0.38, so its class is genuinely
uncertain and it is not used as a confident label.

**Gate 1 risk, flagged early.** At the species level, before any intersection
with Cas9 homologs, cross-checked counts are thermo 484 and hyper 171. These
are hard ceilings: the Cas9-labeled corpus can only be a subset. The configured
`min_per_bin: 200` is therefore almost certainly unreachable for `hyper` and
likely tight for `thermo`. My recommendation, for a decision before stage 1e:
plan on merging thermo and hyper into one "thermophilic" class, or on treating
OGT as a continuous regression target with the bins used only for conditioning
tokens. Deciding now is cheaper than discovering it at Gate 1.

Verified against all eight seed hosts: 8/8 resolve, 7/8 cross-check (FnCas9's
*Francisella tularensis* has a single source). GeoCas9 and ThermoCas9 both land
in `thermo` with agreement 0.944 and 1.000. *Streptococcus thermophilus* lands
in `meso` at 37 °C, which is correct biology and a reminder that organism names
are not labels.

### 2026-10-20 — Cluster-level splitting and the leakage gate, verified

Implemented stages 1e and 1f and tested them with real MMseqs2 against the
eight seeds. The small corpus produced a clean demonstration of why
cluster-level splitting is not optional.

**Clustering put GeoCas9 and ThermoCas9 in the same cluster.** Both are
*Geobacillus* type II-C Cas9s and they align at 86.9% identity over 100%
coverage. Cluster-level splitting therefore sent both to test. Had the split
been per-sequence, one could have landed in train, and since the two are
near-identical the thermophilic positive control would have been leaked into
training — which would make "generated sequences shift toward thermophilic
features" partly a measurement of recall. SpCas9 and St1Cas9 likewise cluster
together and both went to train.

8 seeds clustered to 6 clusters at 45% identity / 80% coverage. The II-C
subfamily holdout forced 3 clusters to test and the positive-control rule
forced 1.

**The leakage gate was verified in both directions.** On the real split it
passes, and the closest held-out-to-train pair is 39.7% identity over 3%
coverage — a conserved catalytic motif, correctly not a violation, which is
why the check requires high identity *and* high coverage rather than identity
alone. Then a deliberately leaked split was constructed with GeoCas9 in train
and ThermoCas9 in test; the gate reported the pair at 86.9% identity / 100%
coverage and exited 3. A gate that has only ever passed has not been tested.

Design note: the clustering and the leakage check are deliberately independent.
Clustering at 45% identity should make a cross-split violation impossible, so
the gate is verification rather than mechanism, and a mistake in the
clustering parameters cannot also switch off its own check.

The splitter stratifies cluster allocation by dominant growth-temperature bin.
Thermophilic clusters are scarce, and allocating clusters uniformly at random
would let train absorb all of them by chance, leaving no thermophiles in val or
test to evaluate against.

### 2026-10-20 — Gate 1 pre-flight: the thermophilic Cas9 corpus is 10 clusters

Before committing to a multi-gigabyte reference download and a cloud search,
estimated the thermophilic corpus from small UniProt queries
(`scripts/07_census_feasibility.py`). The answer changes the project.

**Census.** UniProtKB holds 1,479 Cas9 proteins at 950-1,700 aa. Joined to
Gosha, 1,046 (70.7%) carry an OGT label:

| Bin | Sequences | Cross-checked | Distinct organisms |
| --- | --- | --- | --- |
| meso | 1,023 | 785 | 968 |
| thermo | 23 | 16 | 21 |

**The decisive number.** Clustering those 23 thermophilic sequences at the
configured 45% identity / 80% coverage gives **10 clusters**, six of them
singletons. Cluster-level splitting at 80/10/10 would leave roughly 8 training
clusters, 1 val, 1 test. That is not a trainable conditioning signal, and
`min_per_bin: 200` is out of reach by more than an order of magnitude.

Sequence counts were misleading, as expected: the count that bounds both
learning and splitting is the cluster count.

**This is biology, not database coverage.** Checked whether a bigger database
fixes it. NCBI nr holds 48,959 Cas9 proteins, ~33x UniProtKB, so the corpus as
a whole would grow substantially. But the thermophilic end does not:

| Taxon | Cas9 proteins in nr (950-1,700 aa) |
| --- | --- |
| all | 48,959 |
| Bacteria | 34,888 |
| Archaea | **21** |
| *Geobacillus* | 90 |
| *Thermus* | **0** |
| *Thermotoga* | **0** |

Type II CRISPR-Cas9 is rare in thermophiles and nearly absent from archaea,
where most hyperthermophiles live. The classic thermophile genera have no Cas9
at all. Consistent with this, the hottest host of any Cas9 in the census is
60 °C — there is no hyperthermophilic Cas9 to find, which retrospectively
justifies merging the hyper bin but shows the problem runs deeper than
granularity.

Searching nr instead of UniProtKB is still worth doing for the mesophilic side
and would raise the thermophilic cluster count somewhat, but it cannot
manufacture diversity that does not exist.

**The fallback's premise checks out.** Gosha holds 1,081 thermophilic species
(672 cross-checked), of which 479 are at >=60 °C and 238 at >=70 °C. At a few
thousand proteins per proteome that is on the order of 2M thermophilic protein
sequences available for a broad temperature-labeled pre-training stage. The
scarcity is specific to Cas9, not to thermophilic proteins.

**Status: Gate 1 fails as configured.** Not changing the config or the
framing without a decision; options put to the user.

### 2026-10-20 — Decision: two-stage training. Stage A scaffolding added

Decided (user) to take the brief's fallback: learn temperature conditioning on
a broad corpus of temperature-labeled bacterial proteins, then fine-tune on
Cas9. Growth-temperature threshold stays at 45 °C; lowering it would have made
Gate 1 pass on paper by diluting the very contrast the thesis rests on.

**Why this is the right fallback, in one number.** Stage A species selection on
the real Gosha table yields 400 mesophilic and 400 thermophilic species, and
the thermophilic side reaches **103 °C** (*Pyrolobus fumarii*). The hottest
host of any known Cas9 is 60 °C. Stage A therefore contains a hyperthermophilic
signal that the Cas9 corpus does not and cannot contain, which is precisely the
information stage B needs and cannot learn for itself.

Corpus estimate: 800 species x up to 500 proteins = at most 400,000 sequences,
roughly 140 MB of FASTA. Affordable to download and to screen locally.

**Balance is enforced, not incidental.** Gosha holds 13,898 cross-checked
mesophilic species against 670 thermophilic ones. Sampling proportionally would
let the model minimize its loss by ignoring the conditioning token and always
predicting mesophilic features — exactly the failure Gate 2 is designed to
detect, arriving by way of the data rather than the model. The scarce class is
taken up to the cap and the abundant one matched to it, and when the abundant
class is trimmed the best-evidenced OGT labels survive rather than an arbitrary
sample.

**The hazard this stage introduces, and the guard against it.** A broad
bacterial corpus naturally contains Cas9 proteins. If a held-out Cas9 test
cluster appears in stage A, the stage B leakage gate is measuring nothing: the
model has already seen the test sequence, one stage earlier. `exclude_cas9` in
`configs/pretrain.yaml` screens by annotation (name patterns plus the nine Cas9
Pfam models) and by alignment against every held-out Cas9, at a stricter
threshold than stage B's leakage gate (30% identity over 50% coverage versus
45% over 80%), because a 400,000-sequence corpus hides a borderline homolog far
more easily than a 1,000-sequence one.

The two *Geobacillus* positive-control hosts are also excluded from stage A
species, so pre-training cannot teach the model GeoCas9's or ThermoCas9's own
genomic context.

Stage A has no test split by design. It is not evaluated on its own; it exists
to initialize stage B, and all held-out evaluation belongs on the Cas9 corpus
where the natural controls and the structural metrics live.

### 2026-10-20 — Pretrained checkpoint corrected: ProGen2-small cannot fit a Cas9

The brief recommended ProGen2-small and `model.yaml` had it. Checked the
published config instead of trusting the recommendation: **n_positions=1024**.
Cas9 is 984-1,629 aa, so SpCas9 at 1,368 does not fit. Only CjCas9 (984) would
have.

This is the prior repository's central bug arriving again with a different
number — a context too short for the protein, dropping or truncating it
silently. The `required_context` assertion in the config was already designed
to catch it at startup, but the checkpoint named was wrong.

| Checkpoint | Params | n_positions | Fits Cas9 |
| --- | --- | --- | --- |
| progen2-small | 151M | 1024 | no |
| **progen2-base** | **765M** | **2048** | **yes** |
| progen2-large | 2.7B | 1024 | no |
| ESM-2 650M | 650M | 1026 | no, and a masked LM |
| ProtT5-XL | 3B | 512 declared | relative positions, extensible |

**progen2-base has a longer context than progen2-large**; size does not track
context in this family, so picking the largest model that fits the GPU lands
on one that cannot see a whole Cas9.

Switched both `conditioned_lm` and `unconditioned_lm` to progen2-base, and set
`required_context: 1703` (filter.max_length of 1,700 residues plus BOS, the
growth-temperature control token and EOS). `train.yaml`'s `data.max_length`
and the scratch baseline's `max_length` were aligned to the same budget. Tests
now assert that every configured context covers the longest admissible
sequence, that the checkpoint is not one of the 1024-token variants, and that
the ablation uses the same checkpoint as the main model — otherwise the
ablation would measure the checkpoint rather than the conditioning token.

Open consequence: 765M parameters instead of 151M changes the compute story.
LoRA with bf16 and gradient checkpointing should fit a 16 GB cloud GPU at
batch 1, but that is unmeasured and should be smoke-tested before planning a
real training schedule.

### 2026-10-20 — Stage A collection, model layer, Gate 2, smoke test

**Compute question resolved.** The move from ProGen2-small (151M) to
ProGen2-base (765M) raised a budget concern. With LoRA it is not a problem:
estimated 1.5 GB with bf16 and gradient checkpointing, 2.0 GB with bf16 alone,
and 3.9 GB even in fp32, because LoRA leaves optimizer state covering well
under 1% of parameters. Caveat recorded in the script output: the estimate
counts hidden states and not attention matrices, which are
batch x heads x seq^2 and would add roughly 2.5 GB across 27 layers at 1,703
tokens if a backend materialized them. Flash/SDPA attention avoids that;
verify on the actual GPU before trusting the figures.

**Training path verified end to end** on this laptop with a small
randomly-initialized model at full Cas9 sequence length: 1,353 tokens on
`mps`, finite loss, perplexity 25.15 against a 28-token vocabulary. That is
the uniform value for an untrained model, which is the correct sanity result
rather than a suspiciously good one.

**Stage A collection bug worth recording.** The first full run died 74 species
in with a bare `TimeoutError`. `urlopen` raises it directly and it is not a
`URLError` subclass, so the handler missed it. Over 800 requests a transient
failure is close to certain, so fetches now retry with backoff, honour
`Retry-After` on 429, do not retry a 4xx that will not change, and no single
species can end the run. Shards are cached per species, so the relaunch
resumed from 74 rather than starting over.

**The Cas9 annotation screen is pulling its weight.** A six-species pilot
covering 3,000 sequences already contained 4 Cas9 proteins — two matched on
`cas9`, two on `crispr-associated endonuclease`. Extrapolated to the full
400,000-sequence corpus that is several hundred Cas9 proteins which would have
entered pre-training invisibly to the stage B leakage gate.

**Gate wiring.** `06_leakage_test.py` now writes a `leakage_passed` marker
recording the date, the number of alignments examined and the closest
held-out-to-train identity; `11_train.py` refuses to start stage B without it.
A stale marker is deleted when the gate fails, because a pass recorded for an
earlier clean split would otherwise authorize training on a split that now
leaks. Verified in three directions: clean split writes the marker, leaked
split exits 3 and writes none, and a planted stale marker is removed.

`11_train.py` also refuses to train when the validation split holds only one
growth-temperature class. Gate 2 compares the correct token against a shuffled
one, and on a single-class validation set shuffling changes nothing, so the
check would pass vacuously.

### 2026-10-20 — Stage A corpus built; Gate A passed; one silent screen failure

Collection finished: 800/800 species, 0 failures across 800 requests after the
retry fix. 330,626 sequences fetched, 311,993 kept after length and residue
filtering.

**The annotation screen removed 300 Cas9-family proteins** from the bacterial
proteomes — 254 matching "crispr-associated endonuclease", 46 matching "cas9".
A six-species pilot had predicted several hundred, and every one of these would
have entered pre-training invisibly to the stage B leakage gate, which only
ever compares the Cas9 corpus against itself.

**The alignment screen then caught 6 more that annotation missed**, at 30.3% to
35.1% identity to held-out Cas9 sequences. This is the justification for having
both screens rather than one.

**A silent failure worth recording in full.** The first run of
`09_screen_broad.py` reported those 6 exclusions and then wrote all 311,993
sequences, i.e. excluded nothing, and printed "stage A gate PASSED" on top. The
cause: the id canonicalization fix landed after collection had already started,
so the corpus was assembled by the then-running process with full
`tr|ACC|NAME` identifiers, while MMseqs2 reports results under the bare
accession. The exclusion comparison therefore never matched. I had patched the
same id mismatch at the cluster-join site earlier and missed this second
comparison.

The deeper failure was not the mismatch but that nothing checked the
script's own contradictory output: "6 excluded" followed by "wrote 311,993".
`09_screen_broad.py` now refuses to write when the number removed does not
equal the number flagged, names the unmatched ids, and exits 4. Identifiers are
canonicalized at read time for the corpus, the held-out set and the label
table, so every comparison downstream operates on one format.

**Corrected result, verified independently:** 311,987 sequences written, the 6
flagged ids absent from both the corpus and the splits, all ids canonical,
train and val disjoint and covering the corpus exactly.

| | clusters | sequences |
| --- | --- | --- |
| meso | 27,575 | 182,964 |
| thermo | 34,376 | 129,023 |
| **total** | **55,840** | **311,987** |

Train 295,035 sequences over 53,048 clusters; val 16,952 over 2,792. No test
split, by design.

Three observations that matter for stage B:

1. **The thermophilic class is the more diverse one.** 34,376 clusters from
   129k sequences against 27,575 from 183k mesophilic. The scarcer class is
   less redundant, which is favourable for learning the conditioning.
2. **About 6,100 clusters contain both classes** (27,575 + 34,376 = 61,951
   against 55,840 distinct clusters). Those are homologous proteins from
   organisms at different growth temperatures, which is exactly the contrast
   the conditioning token has to explain.
3. **Sequence balance is 1.42:1 rather than 1:1** despite balanced species
   selection, because thermophiles have smaller proteomes. Far better than the
   20:1 species imbalance, but worth watching at Gate 2.

### 2026-10-20 — Gate 2 pre-flight: the thermophilic signal is present

The stage A gate counts clusters and sequences; it does not say the corpus
contains the signal the conditioning token has to explain. Checked that
directly before spending GPU time (`scripts/12_signal_check.py`), using the
Zeldovich et al. 2007 composition indicators.

**Corpus-wide**, all three move as thermophily predicts:

| Group | meso | thermo | delta | expected |
| --- | --- | --- | --- | --- |
| IVYWREL | 39.256% | 42.681% | +3.425% | up, OK |
| charged (DEKR) | 23.539% | 25.495% | +1.957% | up, OK |
| polar (STNQ) | 18.076% | 15.725% | -2.352% | down, OK |

That alone is weak evidence, because it is confounded with phylogeny:
thermophiles are a different set of organisms with different proteome-wide
composition, so the difference could be taxonomic rather than thermal.

**Within-cluster, controlling for protein family**, the signal survives. Of
55,858 clusters, 6,111 (10.9%) contain both classes — homologous proteins from
organisms at different growth temperatures:

| Group | mean shift | median | families shifting as predicted |
| --- | --- | --- | --- |
| IVYWREL | +1.743% | +1.700% | **81.5%** |
| charged | +0.638% | +0.600% | 65.9% |
| polar | -1.657% | -1.587% | **81.4%** |

Chance is 50%. IVYWREL and polar at ~81% is a real effect independent of
family identity, which is what makes the conditioning learnable as thermal
adaptation rather than as family memorization.

Three caveats worth carrying into Gate 2:

1. **Roughly half the corpus-wide signal is phylogenetic.** Within-family
   IVYWREL shift is +1.74% against +3.43% corpus-wide, so about half the
   apparent effect is composition differences between organisms rather than
   thermal adaptation of individual proteins. The model will learn both; only
   the former is expected to transfer cleanly to a single family like Cas9.
2. **Effect sizes are modest.** A +1.74% within-family composition shift is a
   real but small signal, and Gate 2 asks for a 2% relative perplexity gain.
   Gate 2 could be marginal rather than decisive, and a narrow pass should not
   be over-read.
3. **Charged residues are the weakest of the three** at 65.9% consistency
   against ~81% for the other two. Salt-bridge content is a structural property
   that composition alone captures poorly, so this is expected, but it means
   charge-based evaluation metrics will be noisier than composition ones.

Because the cluster-level split puts validation clusters outside training, Gate
2 cannot be passed by memorizing cluster identity: the model has to generalize
from sequence features to unseen families. That makes the gate a real test.

### 2026-10-20 — Two correctness fixes before the first training run

**Training would have read the unscreened corpus.** `11_train.py` resolved
`paths/sequences`, which is the raw collected file still containing the 6 Cas9
homologs the alignment screen found; the screened output went to a different
path and nothing pointed training at it. `configs/pretrain.yaml` now names
`screened_sequences` explicitly, and training raises rather than falling back
when it is absent. The difference between those two files is exactly the
contamination the stage B leakage gate cannot detect, so a silent fallback was
the wrong behaviour.

**A redirected run overwrote project data — mine did.** While verifying the
exclusion guard I ran `09_screen_broad.py` with `--sequences`, `--out` and
`--splits-dir` pointed at scratch paths but left `--label-table` at its
configured default. A 21-sequence test corpus therefore rewrote 20 rows of the
real 311,987-row label table with splits derived from itself, which is how one
sequence (F8AMI4) came to be marked `val` in the label table while the split
file said `train`.

The one-row disagreement was trivial; the mechanism was not. The script now
refuses to run when any output path is redirected while `--label-table` stays
defaulted, since inputs and outputs have to travel together. Re-ran the screen
to regenerate the table, then verified independently: 0 split disagreements, 0
cluster_id mismatches, and the label table and split files each cover the
corpus exactly with no blanks and no train/val overlap.

Worth noting that the check which caught this was the same label-table/split
cross-check written for its own sake. Verifying two artifacts agree, rather
than trusting whichever was written last, is what made a silent one-row
corruption visible.

### 2026-10-21 — ProGen2 under transformers 5.x: four failures, one of them silent

Getting the checkpoint to load took four fixes. The order they were found in
matters, because each one hid the next.

**1. It did not load at all.** `AttributeError: 'ProGenForCausalLM' object has
no attribute 'all_tied_weights_keys'`. ProGen2's published code is remote code
written for transformers 4.40; it calls the 4.x `init_weights()` rather than
5.x's `post_init()`, which is where that mapping is populated.

**2. It loaded but could not run a forward pass.** `RuntimeError: Tensor on
device meta is not on the expected device cpu`. Its non-persistent
causal-mask buffer and its `scale_attn` tensor are built during 5.x's
meta-device initialization, and nothing in the checkpoint materializes them.

**3. `PreTrainedModel.get_head_mask` no longer exists** in 5.x, and the
forward pass calls it.

At this point the working fix was three runtime monkeypatches against a file
downloaded at import time. That was the wrong foundation, so the model code
was vendored into `thermocas/model/progen2/` at a pinned revision, with every
change marked `THERMOCAS:` and the provenance and licence recorded in the
package docstring. Pinning transformers 4.x was the alternative, and was
rejected because the 4.x line does not support Python 3.14 — the pin would
have extended to the interpreter.

**4. Then the weights were silently discarded.** The vendored copy loaded all
221 tensors, reported no missing keys, no mismatched keys, and no warnings —
and every parameter was random. Perplexity on an SpCas9 fragment was 44.5
against 46.3 on the same residues shuffled: a 765M-parameter model that could
not tell a protein from a bag of amino acids.

The cause is worth stating precisely, because it is a trap for anyone
vendoring a model. transformers 5.x applies weight initialization to every
module *after* loading the checkpoint, and relies on each initializer to check
the parameter's `_is_hf_initialized` flag and leave loaded weights alone. The
flag-aware initializers live in `transformers.initialization`. ProGen2 writes
in place through `module.weight.data.normal_()`, which ignores the flag.
transformers does carry a fallback that skips already-initialized modules
wholesale — but it is applied only when `is_remote_code()` is true, and that
is `cls._auto_class is not None`. **Vendoring the file made the fallback stop
applying.** The fix was to route `_init_weights` through
`transformers.initialization`.

After the fix, the same comparison reads 1.19 against 18.44, and greedy
continuation of the documented prompt `1MEVVIVTGMSGAGK` gives
`TVAVQSFEDLGYFCVDNLPPTLLPKFLELMKESGGKMNKV`. (The 1.19 is near-memorization:
SpCas9 is in ProGen2's training data.)

Two guards were added rather than relying on having noticed:

* `load_progen2` compares every loaded tensor against the safetensors file on
  disk and refuses to return a model if any differs. "The weights arrived" and
  "the weights are still there" turned out to be different claims, and only
  the second one matters.
* It also refuses a model with any tensor left on the meta device, which is
  failure 2 surfacing at load time instead of deep inside an attention layer.

**Two further defects found while vendoring**, neither of them version drift:

* `lm_head.weight` was listed in `_keys_to_ignore_on_load_missing`. ProGen2's
  head is an untied `nn.Linear` whose weight *is* in the checkpoint, so that
  entry meant a lost output layer would have loaded silently as a random one —
  and under LoRA, with the base weights frozen, it could never be relearned.
* `supports_gradient_checkpointing` was False and the forward pass only read
  `config.gradient_checkpointing`, so `gradient_checkpointing_enable()` raised
  `ValueError` and, had it not, would have had no effect. Stage A's 1.5 GB
  memory estimate assumes checkpointing is on. It is now a config key
  (`model.yaml: gradient_checkpointing`) that the module applies explicitly.

Verified: 769,295,431 parameters total, 4,482,596 trainable (0.58%) with LoRA
r=16 on `qkv_proj`/`out_proj`, the resized embedding and head, and the OGT
head. Gradient checkpointing confirmed active and differentiable.

### 2026-10-21 — The prompt layout is the tokenizer's to report

The dataset computed its masked prefix as "BOS, plus a control token if
conditioning is on". That is the from-scratch tokenizer's layout.
`PretrainedTokenizerAdapter` emits `[TEMP] 1 SEQUENCE [EOS]` — ProGen2's
direction marker in place of BOS — so the same arithmetic masked the right
*number* of positions when conditioning was on and the wrong number when it
was off, leaving the direction marker in the loss.

`Encoded` now carries `prompt_length` and each tokenizer exposes
`encoding_overhead()`, measured by encoding a single residue rather than
counted by hand so it cannot drift from `encode()`. The dataset reads both.
Separately, `11_train.py` now builds the adapter from the configured
checkpoint rather than the project's own vocabulary, and
`build_lightning_module` raises on anything else — the residue vocabulary
belongs to the checkpoint.

### 2026-10-21 — Stage A dry run against the real corpus

`11_train.py --stage a --dry-run` on `broad_proteins_screened.fasta`: 295,035
training examples and 16,952 validation examples admitted, longest 1,027
tokens, bins meso=172,558 / thermo=122,477 in train. Zero dropped for length,
no label, or missing sequence. Vocabulary 35 tokens (32 from the checkpoint
plus unconditional, meso, thermo).


### 2026-10-21 — The control tokens were all the same vector

The first training steps ever run produced a Gate 2 report that read:

    perplexity with the correct growth-temperature token: 4.3283
    perplexity with shuffled tokens: 4.3283 (mean of 5)
    individual shuffles: 4.3283, 4.3283, 4.3283, 4.3283, 4.3283

Identical to four decimals across five independent shuffles is not a weak
signal, it is no signal. The obvious explanation — two training steps is not
enough to learn anything — does not fit: an untrained-but-distinct token still
changes the forward pass. And the shuffles were not vacuous; the assignments
at the evaluated indices genuinely differed from the true labels, which was
checked before looking further.

Working inward: the dataset emits the right control token under override, the
attention mask covers it, the loss masks it. The model was the problem, and
specifically the resize:

    meso(33) vs thermo(34)    L2 = 0.0000    cosine = 1.000000

`resize_token_embeddings` defaults to `mean_resizing=True`, which draws new
rows from a multivariate normal fitted to the existing embeddings. ProGen2 has
32 rows of dimension 1536, so a 1536x1536 covariance is estimated from 32
samples — rank-deficient by a factor of 48 — and the draw collapses to the
mean. All three new rows came out bit-for-bit identical.

So `<temp:meso>`, `<temp:thermo>` and `<temp:any>` were one token wearing three
names. The growth-temperature label carried zero information, and label
dropout was a no-op because the unconditional token was the same vector as
both bin tokens.

What makes this the most dangerous bug in the project so far is that the rows
are trainable (`modules_to_save=["wte", "lm_head"]`), so they would have
slowly separated under gradient. It would not have crashed and it would not
have stayed at zero. It would have looked like conditioning that is somewhat
hard to learn — and that is exactly the finding the paper is about. A marginal
Gate 2 result would have been read as a fact about temperature conditioning in
protein language models rather than as a fact about an initializer.

The rows are now initialized explicitly: distinct, drawn from a seeded
generator, scaled to the mean norm of the pretrained rows (1.0545) so the
control tokens start on the residues' scale. Measured after the fix: pairwise
L2 ~1.50 at near-zero cosine, and switching the token moves the residue logits
by 0.62 against 2.3e-05 before.

`verify_control_tokens_are_distinct` now runs after every resize and raises
rather than warning, and a test asserts that the default resize path *is*
degenerate, so the fix cannot be mistaken for a stylistic preference and
quietly reverted.

### 2026-10-21 — Measured memory: 19 GB, not 1.5 GB

The 2-step stage A run was measured while it ran, on the 16 GB Mac:

| Measure | Value |
| --- | --- |
| Process footprint | 19 GB (peak 17 GB resident) |
| of which MPS allocator (IOAccelerator) | 15 GB |
| of which CPU side | 2.8 GB |
| Swap in use | 18.5 GB of 19.4 GB |

The estimate in `10_smoke_train.py` was 1.5 GB at bf16 with gradient
checkpointing and 3.9 GB at fp32 without. The measurement is ~5x the worse of
those, and the recorded caveat names the reason: the estimate counts hidden
states only and excludes attention matrices. Two multipliers apply here.
First, `mps` has no bf16 autocast, so `resolve_precision` correctly fell back
to `32-true` and every activation is fp32. Second, ProGen2's attention
materializes the full L x L matrix and makes several copies of it per layer
(fp32 cast, `torch.where`, softmax, dropout) — about 340 MB per layer at 1,027
tokens, and the MPS caching allocator does not return memory, so the
high-water mark accumulates.

Consequences, in order of value:

1. The earlier "~20 s/sequence, ~74 days locally" figure measured swap
   thrashing, not compute. It is not a compute estimate and should not be
   quoted as one.
2. Replacing ProGen2's hand-rolled attention with
   `scaled_dot_product_attention` in the vendored code is the single highest
   value change for fitting a 16 GB GPU, because it stops the L x L matrix
   being materialized at all.
3. Local runs are for plumbing only, at a capped sequence length. Full-length
   anything belongs on the cloud GPU, and the memory figures there must be
   measured rather than estimated before the training schedule is planned.


## 2026-10-22 — Fused attention, and a padding mask that was not needed

Replaced ProGen2's hand-rolled attention with
`torch.nn.functional.scaled_dot_product_attention` in the vendored code. The
eager implementation builds the full `(batch, heads, L, L)` score matrix in
fp32 and keeps several copies of it alive per layer. Measured on CPU at
L=1027 with the real base dimensions (1536 wide, 16 heads), one layer's
forward and backward costs:

| path | peak RSS for one layer |
| --- | --- |
| eager | 356 MB |
| fused, with a padding mask | 195 MB |
| fused, `is_causal` | 177 MB |

The flash backend is confirmed available and selected on this CPU (4x faster
than the math backend on the same inputs), so these are not math-backend
numbers. They are still CPU allocator high-water marks, though, and the
figure that decides the training schedule is GPU memory under bf16, which
remains unmeasured. What is now checked rather than estimated is the thing
that mattered: `tests/test_progen2.py` asserts the fused path never allocates
a tensor as large as `heads x L x L`, and the eager path always does.

Two things fell out of doing it.

**A float mask defeats the point.** Passing the additive padding mask to
`scaled_dot_product_attention` forces the mask-aware kernel and allocates an
`(L, L)` bias, where `is_causal=True` lets the kernel build the triangle
itself and allocate nothing. So it is worth not passing one.

**With right-aligned padding, causal attention does not need the padding mask
at all.** A query at a real position attends only to keys at positions at or
before its own, and every pad key sits after every real key, so the causal
triangle already excludes them. Pad positions' own outputs do differ, but a
pad position can only influence later positions, which are also pad, and both
the language-model loss and the OGT head's mean-pooling exclude those. So
`ProGenModel.forward` now drops the mask when the batch is right-padded --
one comparison per forward pass, not per layer -- and builds it only for the
left-padded case.

This is the kind of argument that is true until some collation change makes it
false, silently and only in the loss. Three tests hold it in place: the real
`collate` is asserted to produce right padding, a sequence scored alone is
asserted to give identical logits at its real positions when scored inside a
padded batch, and left padding is asserted to still agree with the eager
implementation.

The fused path deliberately differs from eager in one way. Eager upcasts the
scores to fp32; the fused kernels keep the input dtype and accumulate in fp32
internally, as every current transformers attention implementation does.
Under `32-true` the two are doing the same arithmetic, and the tests require
them to agree to 1e-5. Under bf16 autocast the fused path is less precise and
much smaller, which is the trade being made.

Also deleted `checkpoints/stage_a/last.ckpt`. It was 2.9 GB from the 2-step
run whose control tokens were all the same vector, so nothing in it is worth
keeping, and a stale checkpoint that loads cleanly is exactly the sort of
thing that gets picked up by accident later.


## Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-19 | Rebuild rather than extend the prior repo | Full-length Cas9 was never in its training set and it has no leakage control; both are structural, not incremental, fixes |
| 2026-10-19 | Primary model: pretrained autoregressive protein LM + LoRA, with a discrete growth-temperature bin token | Matches the brief's recommendation; a pretrained model is the only affordable route to 1,400-residue context on free/cheap GPU tiers |
| 2026-10-19 | Keep the from-scratch transformer as a labeled weak baseline only | Honest comparison point; not expected to be competitive |
| 2026-10-19 | Growth-temperature bins: meso < 45 °C, thermo 45–70 °C, hyper > 70 °C | Brief's defaults; set in `configs/data.yaml` and overridable |
| 2026-10-19 | Hold GeoCas9 and ThermoCas9 out of training as positive controls | Default pending review: a thermophilic Cas9 in the training set would make "shifted toward thermophilic" partly memorization. Flip `holdout_positive_controls` in `configs/data.yaml` to change |
| 2026-10-20 | ProGen2-base as the pretrained checkpoint, not -small or -large | Both of those declare n_positions=1024, which cannot fit SpCas9's 1,368 residues; base declares 2048 |
| 2026-10-20 | Two-stage training: broad temperature-labeled proteins, then Cas9 | ~10 independent thermophilic Cas9 clusters exist; conditioning cannot be learned from Cas9 alone, but stage A reaches 103 °C where Cas9 stops at 60 °C |
| 2026-10-20 | Keep the thermophilic threshold at 45 °C | Lowering it would pull 37-44 °C mesophiles into the thermophilic class, making Gate 1 pass while making the result meaningless |
| 2026-10-20 | Screen Cas9 out of stage A by annotation and alignment, stricter than stage B | A held-out Cas9 appearing in pre-training would void the stage B leakage gate silently |
| 2026-10-20 | Merge thermo and hyper into one thermophilic bin; keep continuous ogt_c and the auxiliary regression head | Gosha holds only 171 cross-checked hyperthermophilic species, so a separate bin could not reach min_per_bin and Gate 1 would have failed on a threshold rather than on the science |
| 2026-10-20 | Keep clustering and the leakage check independent | Clustering is the mechanism, the gate is the verification; a clustering mistake must not be able to disable its own check |
| 2026-10-20 | Cross-check OGT within Gosha, on bin agreement rather than absolute spread | The per-collection columns are independent sources, so no second database is needed; spread scales with temperature, so a fixed tolerance rejects genuine thermophiles |
| 2026-10-20 | Bound OGT to [-20, 130] °C | Gosha contains single-source values of 230 °C and 187 °C for known mesophiles |
| 2026-10-21 | Vendor ProGen2's model code at a pinned revision rather than pin transformers 4.x or monkeypatch remote code | The 4.x line does not support Python 3.14, so the pin would extend to the interpreter; three accumulating runtime patches against a file fetched at import time is worse than one copy in the repository |
| 2026-10-21 | Verify loaded weights against the checkpoint file on every load | A clean loading report is not evidence: transformers 5.x re-initialized all 221 tensors and reported nothing |
| 2026-10-21 | No GenerationMixin on the vendored model; sampling uses the project's own loop | ProGen2 carries the legacy tuple key/value cache, which 5.x's generate() does not produce. Classifier-free-guidance-style sampling needs two forward passes per step anyway, which generate() does not do |
| 2026-10-21 | Initialize control-token embeddings explicitly, and verify they are distinct | transformers' mean_resizing draw is rank-deficient for a 32x1536 embedding and produced three identical rows; the rows are trainable, so this would have read as "conditioning is hard to learn" rather than as a bug |
| 2026-10-22 | Fused attention by default, with the eager path kept and config-switchable | The eager path is the reference the fused one is tested against; keeping it means a numerical disagreement can be localized instead of argued about |
| 2026-10-22 | Drop the padding mask for right-padded batches rather than always building it | An additive mask forces the mask-aware attention kernel; under causal attention the mask is redundant for every position whose output is used |
| 2026-10-19 | Homolog search runs on a cloud machine, not locally | 16 GB unified memory measured on this Mac; a UniRef50 DIAMOND index does not fit comfortably alongside the OS. Local runs use a subset for smoke-testing only |

## Environment

| Date | Item | Value |
| --- | --- | --- |
| 2026-10-19 | Dev machine | MacBook, Apple M-series arm64, 16 GB unified memory, macOS 24.6.0 |
| 2026-10-19 | System python | 3.11 via conda env `thermocas` (system 3.14 not used) |
| 2026-10-19 | conda | not yet installed on this machine — miniforge needed before `environment-local.yml` can be created |

## Database versions and download dates

| Date | Database | Version | Size | Location |
| --- | --- | --- | --- | --- |
| 2026-10-19 | UniProtKB (REST) | live query, see seed_provenance.csv | 8 records | `data/raw/seeds.fasta` |
| 2026-10-19 | NCBI protein (E-utilities) | `WP_099233044.1` | 1 record | `data/raw/seeds.fasta` |

### 2026-10-19 — Seed collection

Eight reference Cas9s resolved and written to `data/raw/seeds.fasta`, with
per-seed provenance in `data/raw/seed_provenance.csv`. Lengths were checked
against published values:

| Seed | Accession | Source | Length | Type |
| --- | --- | --- | --- | --- |
| SpCas9 | Q99ZW2 | UniProt (reviewed) | 1368 | II-A |
| SaCas9 | J7RUA5 | UniProt | 1053 | II-A |
| GeoCas9 | A0ABF7PG96 | UniProt | 1087 | II-C |
| ThermoCas9 | WP_099233044.1 | NCBI RefSeq | 1082 | II-C |
| NmeCas9 | C9X1G5 | UniProt | 1082 | II-C |
| CjCas9 | Q0P897 | UniProt | 984 | II-C |
| St1Cas9 | G3ECR1 | UniProt | 1409 | II-A |
| FnCas9 | A0Q5Y3 | UniProt (reviewed) | 1629 | II-B |

Two findings worth recording:

1. **ThermoCas9 is not in UniProtKB.** The *Geobacillus thermodenitrificans*
   T12 strain appears to be GenBank/RefSeq-only, so the thermophilic positive
   control cannot be fetched from UniProt. `01_collect_seeds.py` therefore
   supports an `ncbi` source per seed, keyed on an explicit accession.
2. **The brief's 1,000-1,400 aa length estimate is too narrow at both ends.**
   CjCas9 is 984 aa and FnCas9 (type II-B) is 1,629 aa, both verified reviewed
   UniProt entries. `filter.max_length` was widened from 1,600 to 1,700 and
   `min_length` left at 950 so that no reference seed is rejected by the
   pipeline's own filter. A test asserts these bounds stay wide enough.

Also hit a local environment issue worth noting for anyone reproducing: the
python.org macOS build does not wire OpenSSL to the system keychain, so
`urllib` raises CERTIFICATE_VERIFY_FAILED against UniProt on a fresh machine.
`01_collect_seeds.py` prefers certifi's CA bundle rather than disabling
verification.

## Tool versions

| Date | Tool | Version |
| --- | --- | --- |
| 2026-10-19 | python (ad-hoc, pre-conda) | 3.14.3 |
| 2026-10-19 | pytorch | 2.12.0 |
| 2026-10-19 | pyyaml | 6.0.3 |
| 2026-10-19 | mps backend | available, fp32 only |
| 2026-10-21 | transformers | 5.10.2 |
| 2026-10-21 | peft | 0.21.2 |
| 2026-10-21 | lightning | 2.6.6 |
| 2026-10-21 | ProGen2 (vendored) | hugohrban/progen2-base @ 71228cbfca5960f9fab5775f378bba3673af9f00 |

## Seeds

| Date | Run | Seed |
| --- | --- | --- |
| | | |

## GPU hours

| Date | Machine | Hours | Purpose |
| --- | --- | --- | --- |
| | | | |
