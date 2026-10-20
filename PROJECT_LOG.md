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

## Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-19 | Rebuild rather than extend the prior repo | Full-length Cas9 was never in its training set and it has no leakage control; both are structural, not incremental, fixes |
| 2026-10-19 | Primary model: pretrained autoregressive protein LM + LoRA, with a discrete growth-temperature bin token | Matches the brief's recommendation; a pretrained model is the only affordable route to 1,400-residue context on free/cheap GPU tiers |
| 2026-10-19 | Keep the from-scratch transformer as a labeled weak baseline only | Honest comparison point; not expected to be competitive |
| 2026-10-19 | Growth-temperature bins: meso < 45 °C, thermo 45–70 °C, hyper > 70 °C | Brief's defaults; set in `configs/data.yaml` and overridable |
| 2026-10-19 | Hold GeoCas9 and ThermoCas9 out of training as positive controls | Default pending review: a thermophilic Cas9 in the training set would make "shifted toward thermophilic" partly memorization. Flip `holdout_positive_controls` in `configs/data.yaml` to change |
| 2026-10-20 | Two-stage training: broad temperature-labeled proteins, then Cas9 | ~10 independent thermophilic Cas9 clusters exist; conditioning cannot be learned from Cas9 alone, but stage A reaches 103 °C where Cas9 stops at 60 °C |
| 2026-10-20 | Keep the thermophilic threshold at 45 °C | Lowering it would pull 37-44 °C mesophiles into the thermophilic class, making Gate 1 pass while making the result meaningless |
| 2026-10-20 | Screen Cas9 out of stage A by annotation and alignment, stricter than stage B | A held-out Cas9 appearing in pre-training would void the stage B leakage gate silently |
| 2026-10-20 | Merge thermo and hyper into one thermophilic bin; keep continuous ogt_c and the auxiliary regression head | Gosha holds only 171 cross-checked hyperthermophilic species, so a separate bin could not reach min_per_bin and Gate 1 would have failed on a threshold rather than on the science |
| 2026-10-20 | Keep clustering and the leakage check independent | Clustering is the mechanism, the gate is the verification; a clustering mistake must not be able to disable its own check |
| 2026-10-20 | Cross-check OGT within Gosha, on bin agreement rather than absolute spread | The per-collection columns are independent sources, so no second database is needed; spread scales with temperature, so a fixed tolerance rejects genuine thermophiles |
| 2026-10-20 | Bound OGT to [-20, 130] °C | Gosha contains single-source values of 230 °C and 187 °C for known mesophiles |
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

## Seeds

| Date | Run | Seed |
| --- | --- | --- |
| | | |

## GPU hours

| Date | Machine | Hours | Purpose |
| --- | --- | --- | --- |
| | | | |
