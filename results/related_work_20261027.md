# Related-work check, 2026-10-27

Run to decide between three framings (see PROJECT_LOG 2026-10-27). The
question that gated the decision: has the conditioning-ceiling measurement
been published?

## The gating answer: no

Four searches from different directions — protein conditional generation,
controllable text generation, information-theoretic limits on conditioning,
and label informativeness — found nothing that measures the achievable gain
from a conditioning label. Nearest neighbours, none of them the same thing:

| Work | What it does | Why it is not this |
| --- | --- | --- |
| Poole et al., ICML 2019; arXiv 1812.01475 | Variational bounds on MI | Estimation machinery, no claim about achievable conditioning gain |
| arXiv 2203.02055 (text generation) | *Caps* conditional MI between latent and output to tune controllability | MI as a training mechanism, not a measurement of available headroom |
| arXiv 1812.10358 (vision) | Entropy reduction as label informativeness for object annotation | Closest in spirit; annotation selection, not generative conditioning |
| arXiv 2507.15651 (2025) | Label-aware generation in homologous protein families | Uses discriminator self-consistency and AUC, not entropy. Reports gains **plateau** past ~10^3 labels — adjacent to our saturation finding, different measurement |
| Shannon–McMillan–Breiman | Ties per-token log-likelihood to an entropy rate | Textbook; nobody applies it as a per-corpus budget for a control token |

So the ceiling, and "effect size is a property of the corpus", appear to be
open. That is the one contribution here that the literature does not already
contain.

## What the literature does already contain

This matters more than expected, and it closes off two of the three options.

### OpenCRISPR-1 (Nature, 2025) — Option 2 is dead

Profluent fine-tuned **ProGen2-base** — this project's exact backbone — on the
CRISPR-Cas Atlas, balanced for family and cluster size, and generated 4M
sequences. The result is >400 mutations from SpCas9, matches SpCas9 on-target
activity with reported reductions in off-target editing, is validated in human
cells, and the sequence plus the Atlas are publicly released.

An unvalidated 1,019-sequence Cas9 fine-tune with proxy composition metrics is
a strictly weaker version of a Nature paper using the same backbone. There is
no framing of Option 2 that survives this.

Note for any future Cas9 work: the Atlas is public and far larger than what
`14_collect_cas9.py` can collect. It would not fix this project's problem,
because metagenomic contigs carry no host OGT label, which is precisely the
label the conditioning needs.

### NOMELT (Sci. Rep., 2025) — Option 3 is largely occupied

NOMELT is neural machine translation from mesophilic to thermophilic sequence,
trained on **>4M homolog pairs across 2.7k Pfam families**, with a reported
+15.5 K melting temperature on an Engrailed Homeodomain variant.

That is the paired-ortholog-across-many-families design proposed as the strong
version of Option 3, already published at 2.7k families against the 1,552
both-class clusters in this project's corpus. It is seq2seq translation rather
than control-token conditioning, so not identical, but a reviewer would
reasonably ask why 1,552 clusters and a control token improve on 2.7k families
and a translation model.

### Temperature conditioning in PLMs is a crowded field

| Work | Venue | Approach | Scale |
| --- | --- | --- | --- |
| PRIME | Sci. Adv. 2024 | MLM + sequence-level OGT regression, correlation loss coupling them | 96M sequence–OGT pairs |
| ThermoFormer | NeurIPS 2024 workshop | Temperature-aware representation learning | 96M OGT-annotated sequences |
| Steering PLMs | **ICML 2025** | Activation-addition steering vectors, >70 °C vs <50 °C | ESM2-650M, ESM3-open, ProLLaMA-7B |
| TemStaPro | Bioinformatics 2024 | ProtT5 embeddings → thermostability classifier | Predictor, not generator |
| NOMELT | Sci. Rep. 2025 | Meso→thermo translation | 4M pairs, 2.7k families |

Two consequences for this project:

1. **The auxiliary OGT regression head in `model.yaml` is PRIME's idea.** It
   must be cited as such, not presented as ours.
2. **The data gap is 340x.** PRIME and ThermoFormer both train on ~96M
   sequence–OGT pairs; stage A has 279,980 training sequences. No framing in
   which corpus scale is the contribution survives that.

### A useful ally

`Off-Manifold Collapse in Guided Protein Language Models` (arXiv 2608.18597,
2026) reports that pushing steering hard makes representations less typical
and *lowers ESMFold pLDDT*, under both solubility and thermostability
steering. This independently supports the standing constraint in this
project's brief — that structure prediction is a validity filter, not a
thermostability test — and is a natural citation for it.

## What this means for the framing

The crowded field is the argument, not the competition. There are now at least
four methods claiming temperature control over protein sequence models —
across Science Advances, NeurIPS, ICML and Scientific Reports — and no common
denominator for how much control is attainable on a given corpus. Reported
effects are small and are compared against each other or against nothing.

So the paper is not "we conditioned a PLM on growth temperature", which is
crowded and 340x short on data. It is: **this literature has no denominator.**
A corpus-only, CPU-cheap upper bound on what any temperature-conditioning
method can achieve, validated by varying the corpus and watching measured gain
track the prediction, plus a worked negative case (Cas9) where the bound
correctly says the experiment is infeasible before any GPU is bought.

Strongest additional experiment, if it is reachable: compute the ceiling for
the published setups' own corpora. The Engqvist OGT data behind PRIME's 96M
pairs is public. "Their reported gain is X% of the ceiling their corpus
allows" would make the measure immediately useful to everyone already in this
space rather than only to us.

## Unverified

- **ICML 2027 deadline.** No official CFP found. A third-party tracker
  projects 22 January 2027 (abstracts 16 January), explicitly extrapolated
  from the 2026 cycle. Must be confirmed on the official site before any
  schedule rests on it.
- Performance figures for PRIME, OpenCRISPR-1 and NOMELT are author-reported;
  no independent replication was found.
- Whether the CRISPR-Cas Atlas carries any environmental temperature metadata
  could not be confirmed from the Atlas itself.
