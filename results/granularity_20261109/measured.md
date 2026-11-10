# It replicates: the bound ranks granularity on a second corpus too

Date: 2026-11-10. Four runs on the design-matched external corpus
(`configs/external_balanced.yaml`, Engqvist growth temperatures), at k = 2, 4,
6 and 9 with that corpus's own minimum-mass-maximising edges. Pre-registered
in `paper/draft.md` section 5.2f at commit `ce80eb9`, launched at `45ff318`.
Reports in `runs/`, call ids in `runs.txt`.

**The prediction on record was that section 5.2e would replicate. It does.**
The primary outcome is positive at 11.8 sigma on test and 11.5 on validation,
and the secondary rank correlation is +1.000 on both splits, as on `pretrain`.

Two of 5.2e's *secondary* observations do not replicate, and both were hedged
in advance. They are reported below in full.

## The measurements

| run | k | edges (C) | min class | bound | val gain | test gain | gain/bound |
|---|---|---|---|---|---|---|---|
| `gbal_k2` | 2 | 42 | 49.8% | +0.274% | +0.410% | +0.432% | 1.58x |
| `gbal_k4` | 4 | 31,42,67 | 24.8% | +0.403% | +0.531% | +0.484% | 1.20x |
| `gbal_k6` | 6 | 30,37,45,59,75 | 15.9% | +0.517% | +0.582% | +0.582% | 1.13x |
| `gbal_k9` | 9 | 28,31,37,42,56,63,73,90 | 8.5% | +0.576% | +0.774% | +0.670% | 1.16x |

All four passed both Gate 2 criteria. z is 25.1 to 32.4 on validation and 32.5
to 53.2 on test -- higher than any earlier batch, because this corpus's gains
are larger in absolute terms. The test split is 1,695 sequences and all of it
was scored, so the `--gate2-limit 2000` cap did not bind. Error bars are
gain / z.

All four runs selected `epoch=1-step=2000.ckpt` by validation, so the series is
consistent in training budget as well as in data.

**On the `-v1` checkpoint files.** `gbal_k9` wrote
`epoch=0-step=1500-v1.ckpt` and `last-v1.ckpt` alongside the un-suffixed
versions, which the other three did not. This was checked rather than assumed
benign: the four calls ran 7,748s, 7,572s, 7,625s and 7,635s, so `gbal_k9` is
mid-range and in particular *shorter* than `gbal_k2`, and the test report was
scored from an un-suffixed, validation-selected checkpoint. There is no extra
training behind the k=9 figure, which matters because k=9 is one end of the
primary outcome.

## The pre-registered primary outcome

Section 5.2f fixed the sign of (test gain at k=9) - (test gain at k=2) before
launch. The bound rises 2.1x across that span.

| split | k=2 | k=9 | difference | sigma | sign |
|---|---|---|---|---|---|
| validation | +0.410% | +0.774% | **+0.364 pp** | 11.5 | **positive** |
| test | +0.432% | +0.670% | **+0.238 pp** | 11.8 | **positive** |

Neither void condition applies: the smallest z is 25.1, and the rarest class
holds 8.5% of residues against the 5% floor. The prediction on record --
positive beyond two standard deviations -- is met by a factor of five.

## The secondary analysis

| predictor | rho (val) | exact p | rho (test) | exact p |
|---|---|---|---|---|
| reachable bound | **+1.000** | **0.042** | **+1.000** | **0.042** |

Exact one-sided permutation p over all 4! = 24 relabelings; 0.042 is the
smallest attainable at n=4 and only a perfect rank match reaches it.

**Across the two corpora this is four perfect rank matches out of four**
(2 corpora x 2 splits), on eight runs, two growth-temperature databases and
two independently computed reachability tables. That is the strongest form the
"it reads resolution" claim takes in this paper, and it is the claim section
6.2 rests on.

## What does not replicate

### `gain / bound` is corpus-dependent, not a band

| corpus | gain / bound across k = 2, 4, 6, 9 |
|---|---|
| `pretrain` | 0.55x, 0.66x, 0.51x, 0.55x |
| `external_balanced` | 1.58x, 1.20x, 1.13x, 1.16x |

On `pretrain` the ratio sat just above half the bound; here it sits above the
bound throughout. Section 5.2f pre-registered this as "reported, not tested",
with the explicit note that agreement would not be treated as a prediction
met -- which is fortunate, because it disagrees.

This is not a contradiction. The bound is a unigram estimate and a contextual
model can exceed it, which ten of the project's sixteen threshold runs already
do, and `external_balanced` is where the largest exceedances were measured
before (1.30x, 1.33x, 1.47x, 1.56x in section 5.2b). What it rules out is
reading `gain / bound` as a stable efficiency figure. **The near-constant
0.51-0.66x on `pretrain` is a property of that corpus, not a law**, and
`results/granularity_20261103/measured.md` said as much in its
"does not license" list. It stands.

### Early saturation does not replicate, and that is the more consequential one

| step | `pretrain` (test) | `external_balanced` (test) |
|---|---|---|
| k=2 -> k=4 | **+0.135 pp (5.8 sigma)** | +0.052 pp (2.6 sigma) |
| k=4 -> k=6 | +0.003 pp (0.1 sigma) | **+0.098 pp (5.3 sigma)** |
| k=6 -> k=9 | +0.025 pp (1.2 sigma) | **+0.088 pp (4.7 sigma)** |
| k=4 -> k=9 | +0.028 pp (1.2 sigma) | **+0.186 pp (8.8 sigma)** |

On `pretrain`, two classes to four was the only reliable step and everything
after it was noise, which is what section 5.1b's "four to six classes"
recommendation was built on. **Here every step is significant and nine classes
beat four by 8.8 sigma.** Four classes is not enough on this corpus.

So the recommendation has to be weakened from a number to a procedure. What
replicates is the *ordering*: more classes is better on both corpora, and the
bound ranks the granularities correctly on both. Where the paying stops is
corpus-specific and has to be read off that corpus's own reachability table.

An explanation consistent with both series, offered as such and not as a
tested claim: each corpus's bound saturates in a different place, and the
series were cut at the same k rather than at the same point relative to
saturation. `pretrain`'s reachable bound rises only +0.008 pp from k=6 to k=9
(+0.488% to +0.496%) and its k=9 bound is already 75.1% of its envelope.
`external_balanced`'s rises +0.060 pp over the same step (+0.517% to +0.576%)
and its k=9 bound is 59.3% of its envelope, with trainable granularities
available up to k=13. One series ran past its plateau; the other stopped short
of it.

**We checked whether the bound's step size predicts the measured step size and
it does not reliably.** Over the six steps above, Pearson is +0.491 and
Spearman +0.371, with `pretrain`'s k=4 -> k=6 a clear counterexample: the
bound rose +0.116 pp and the measured gain rose +0.003. So the explanation
above is a reading of two series, not a measured relationship, and it is not
claimed as one.

## What changes in the paper

1. **Section 5.2f reports a successful replication.** Primary outcome positive
   at 11.8 sigma; rho +1.000 on both splits. Four perfect rank matches out of
   four across the two corpora.
2. **Section 6.2's claim strengthens.** "The statistic ranks resolution and
   not placement" now rests on two corpora and two databases rather than one
   corpus and four points.
3. **Section 5.1b's recommendation must weaken.** "Four to six classes" was
   `pretrain`'s saturation point and does not transfer: on this corpus nine
   beats four by 8.8 sigma. The transferable advice is to compute the
   reachability table for your own corpus and use the largest k whose smallest
   class stays trainable, which is what the two series together support.
4. **The `gain / bound` band must not be quoted as a general figure**, in
   section 5.2e or anywhere else. It is 0.51-0.66x on one corpus and
   1.13-1.58x on the other.
5. **Section 8's recommendation 3 changes** from a class count to the
   procedure, and the limitations section gains the band's corpus-dependence.

## What this does not license

- Any claim that the statistic chooses thresholds. Unchanged: section 5.2d
  measured rho +0.000 against a trivial baseline's +1.000, and the
  bound-maximising two-class edge measures 6.4 sigma worse than convention.
- Any claim about k above 9, or about the uncapped third corpus, where the
  balance artifact is strongest and the series has not been run.
- Any efficiency claim from `gain / bound`, per above.
- Any statement about downstream thermostability prediction. Nothing here is
  an assay.
