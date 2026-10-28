# Ceiling sweep, 2026-10-27 — the CPU half says go

Full numbers in `sweep.txt`. One pass over 113,787,006 residues; every variant
computed from the same counts.

## The verdict

The ceiling moves, and by enough to test.

| Axis | Range (corpus-wide) | Spread |
| --- | --- | --- |
| Threshold, natural balance | +0.095% to +0.257% | 2.7x |
| Threshold, rebalanced | +0.161% to +0.398% | 2.5x |
| Granularity (2 to 5 classes) | +0.154% to +0.381% | 2.5x |
| **All variants** | **+0.095% to +0.398%** | **4.2x** |

For scale, the effect being predicted is +0.265% (stage A validation), and the
val/test agreement on that run was within 10% of itself. A 4.2x spread against
a measurement that reproduces to ~10% is a testable design.

## The two findings that make it a better experiment than expected

**The threshold curve is non-monotonic.** It rises from +0.095% at 30 C to a
peak of +0.257% at 55 C, then falls to +0.102% at 80 C.

This matters more than the spread. A monotonic trend is confoundable: "higher
threshold means a harder task" or "...means less data" would produce one too.
A curve with an interior peak is much harder to explain by anything other than
the quantity being claimed. The mechanism is plain and predicted in advance:
at 30 C the corpus is 89% thermophilic and at 80 C it is 7%, so at both ends a
near-constant label carries almost nothing, and the maximum sits where class
balance and per-class separation trade off best.

So the GPU half is not "does measured gain rise with predicted ceiling" but
"does measured gain peak where the ceiling peaks". That is a sharper claim and
a cheaper one to falsify.

**Granularity saturates, and the 2-class edge lands on 45 C by itself.**
Equal-residue-mass binning at 2 classes puts the edge at exactly 45 C, which
is the threshold `configs/data.yaml` has carried by convention since
2026-10-20. That is independent arithmetic agreeing with a choice made for
other reasons, which is reassuring about the main result.

Ceiling by class count: +0.154% (2), +0.303% (3), +0.374% (4), +0.381% (5).
The gain from 4 to 5 classes is 2%, so the first-order signal is essentially
exhausted at 4 classes. A paper can say where the knee is rather than
asserting a granularity.

## What the sweep also exposed about the corpus

**An 8 C gap at 37-45 C.** No species in the corpus has an OGT in (37, 45), so
thresholds of 40 and 45 bin identically -- the +0.154%/+0.172% rows for both
are the same calculation, not a coincidence. There are ~9 distinct effective
thresholds on this corpus, not 11. The 45 C convention sits in the middle of
an empty band rather than on a knife edge, which is good for the robustness of
the main result and bad for anyone wanting a fine threshold sweep.

**47 distinct OGT values in 311,987 rows.** Gosha labels are organism-level
and rounded, so the `ogt_c` the auxiliary regression head trains on is far
coarser than "continuous" suggests. Worth stating plainly wherever that head
is described.

## Design for the GPU half

Five runs, chosen so that four trace the non-monotonic threshold curve and the
fifth tests whether the ceiling predicts *across* axes rather than only along
one.

| Run | Variant | Predicted ceiling |
| --- | --- | --- |
| 1 | threshold 30 C | +0.095% |
| 2 | threshold 45 C (the measured baseline) | +0.154% |
| 3 | threshold 55 C | +0.257% |
| 4 | threshold 75 C | +0.118% |
| 5 | 4 classes at 32 / 45 / 56 C | +0.374% |

Runs 2 and 3 straddle the peak; 1 and 4 are the two tails. If measured gain
tracks the column, the claim holds. If run 3 does not beat runs 1 and 4, it
does not, and the paper does not exist -- which is the point of running this
before writing it.

**Cost.** Every variant is a pure relabeling: the dataset reads `temp_bin`
straight from the label table and the sequences, clusters and splits are
untouched, so all five runs train on identical residues and identical splits.
Conditioning saturated by step 2,500 on the full run, so 3,000 steps at the
measured 3.024 s/step is 2.52 h, plus ~0.43 h for a validation-only gate at
2,000 examples. About 2.95 h and ~$3.30 per run at the measured A10 rate:
**~$17 for all five**, against ~$79 left of the budget.

**One axis deliberately excluded.** The rebalanced variants are *not* pure
relabelings -- equalizing class mass requires resampling, which changes the
training data as well as the label, confounding the two. The threshold and
granularity axes change nothing but the label, so they carry the argument
alone. The balance numbers stay in `sweep.txt` as corpus description.

**A guard that now does real work.** `Gate2Params.check_corpus` refuses a
ceiling measured on a different corpus. Each variant has its own ceiling, so
each run must carry its own `ceiling_gain`, and a copy-pasted config will stop
at startup rather than judging run 3 against run 2's number. That failure was
hypothetical when the guard was written this morning; across five variants it
is close to inevitable.

## What this does not establish

That measured gain follows the ceiling. Nothing here is evidence for the
paper's claim -- it is evidence that the claim is cheap to test and that the
test can fail. The sweep could have come back with every variant inside
0.15-0.19%, and the honest conclusion would have been to drop the framing.
