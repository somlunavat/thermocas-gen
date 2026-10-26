# thermocas-gen

Temperature-conditioned protein language modeling for Cas9-family sequences.

## What this is

A pipeline that collects Cas9 homologs, labels each with the optimal growth
temperature (OGT) of its source organism, fine-tunes a protein language model
conditioned on that label, and measures whether generated sequences shift
toward thermophilic sequence and structural features.

**The claim under test:** a temperature-conditioned protein language model
trained on homolog-derived Cas9 sequences, labeled with host OGT, can generate
Cas9-family sequences that shift toward thermophilic sequence and structural
features while preserving catalytic domains and predicted fold quality.

**What is explicitly not claimed:** that any generated protein is heat-resistant
or functional. There is no wet-lab validation. OGT is an organism-level proxy
for protein stability, not a measurement of it. A confident structure prediction
is a plausibility filter, not a thermostability assay. ESMFold and AlphaFold are
used here to reject implausible folds, nothing more.

## Pipeline

Four stages, each writing one artifact the next one reads. Evaluation code never
imports training code, so an evaluation result cannot depend on training state.

```
1  data      seeds -> homolog search -> domain filter + OGT label -> cluster + split
              └─ data/processed/cas9_labeled.fasta, data/splits/
2  train     baselines, conditioned LM, conditioning sanity gate
              └─ checkpoint
3  generate  sample at target growth temperatures, filter, novelty check
              └─ generated FASTA
4  evaluate  thermostability scoring, folding, comparison vs natural controls
              └─ tables + figures
```

Two gates stop the pipeline when its assumptions fail:

- **Gate 1 (data):** enough thermophilic Cas9 homologs per temperature bin.
- **Gate 2 (conditioning):** validation perplexity must be lower with the correct
  growth-temperature token than with a shuffled one. If conditioning does not
  measurably help, nothing downstream is worth generating.

## Terminology

Two distinct things are both called "temperature" in this domain. They are kept
apart everywhere in the code:

| Term | Meaning | Appears as |
| --- | --- | --- |
| growth temperature | the OGT label of the source organism, in °C | `ogt_c`, `temp_bin`, `growth_temp` |
| sampling temperature | the decoding softmax divisor | `sampling_temperature` |

## Setup

Local development (Apple Silicon, `mps`):

```bash
conda env create -f environment-local.yml
conda activate thermocas
export PYTORCH_ENABLE_MPS_FALLBACK=1
pytest
```

Cloud GPU (training and folding). Training runs on Modal, so the GPU type and
the package versions are in the repository rather than in someone's shell
history; `cloud/modal_app.py` documents the layout:

```bash
pip install modal
modal token new                            # opens a browser once
modal run cloud/modal_app.py::check        # GPU, corpus, dry run. Costs cents
modal run cloud/modal_app.py::upload       # pushes the corpus to the volume
modal run cloud/modal_app.py::smoke        # 50 steps; reports peak memory
modal run cloud/modal_app.py::train        # the real run; returns immediately
modal run cloud/modal_app.py::status --call-id fc-...   # has it finished?
modal run cloud/modal_app.py::ls           # what is on the volume
```

`train` spawns the job server-side and exits, so nothing local owns it and it
is safe to close the terminal. Do not reach for `--detach` or `nohup` here:
both were tried and both lost a run. A blocking `Function.remote()` call is
owned by the local client while it waits, and Modal cancels the in-flight
input when that client goes away — the first Stage A attempt died at step 314
and the second, relaunched with `--detach`, died at step 2,149 of 20,000.
`--detach` keeps the *app* alive, not the input. `Function.spawn()` is what
actually fixes it.

If a container does die, the run is resumable:

```bash
modal run cloud/modal_app.py::train --run-name default --resume
```

Checkpoints go to `checkpoints/stage_<x>/<run-name>/`, one directory per run.
Sharing one directory is what left 19 GB of `last-v1.ckpt` … `last-v5.ckpt`
behind: Lightning appends `-vN` rather than overwriting, so stale checkpoints
accumulate and load cleanly, indistinguishable from the real one.

Add `--gpu A100-40GB` (or `L4`, `H100`) to any of those. `check` first: it
fails for all the boring reasons that would otherwise fail ten minutes into a
paid run.

For an ssh-style machine instead, `environment-gpu.yml` builds the same
environment under conda:

```bash
conda env create -f environment-gpu.yml
```

Every threshold lives in `configs/*.yaml`. Nothing is hard-coded.

## Repository layout

```
configs/      data.yaml, model.yaml, train.yaml, generate.yaml, eval.yaml
data/
  raw/        seeds.fasta (committed); databases and dumps (gitignored)
  processed/  hits.tsv, cas9_labeled.fasta, label_table.csv (gitignored)
  splits/     cluster assignments (committed)
scripts/      numbered pipeline stages, run in order
              00_fetch_pfam.py fetches the HMM profiles (not committed)
cloud/        modal_app.py: GPU type, pinned versions and the run commands
thermocas/
  model/      tokenizer, dataset, LightningModule, train, generate
  eval/       thermostability scoring, folding, structure metrics, controls, report
tests/        data-integrity and model-shape tests; leakage test gates training
```

## Status

Stage 1 (data) is under construction. `PROJECT_LOG.md` records database versions
and download dates, tool versions, parameter choices and their reasons, random
seeds, and GPU hours.

## License

MIT
