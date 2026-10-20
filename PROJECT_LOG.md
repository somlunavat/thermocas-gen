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
| | | | | |

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
