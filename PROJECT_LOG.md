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

## Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-19 | Rebuild rather than extend the prior repo | Full-length Cas9 was never in its training set and it has no leakage control; both are structural, not incremental, fixes |
| 2026-10-19 | Primary model: pretrained autoregressive protein LM + LoRA, with a discrete growth-temperature bin token | Matches the brief's recommendation; a pretrained model is the only affordable route to 1,400-residue context on free/cheap GPU tiers |
| 2026-10-19 | Keep the from-scratch transformer as a labeled weak baseline only | Honest comparison point; not expected to be competitive |
| 2026-10-19 | Growth-temperature bins: meso < 45 °C, thermo 45–70 °C, hyper > 70 °C | Brief's defaults; set in `configs/data.yaml` and overridable |
| 2026-10-19 | Hold GeoCas9 and ThermoCas9 out of training as positive controls | Default pending review: a thermophilic Cas9 in the training set would make "shifted toward thermophilic" partly memorization. Flip `holdout_positive_controls` in `configs/data.yaml` to change |
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
