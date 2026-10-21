"""Dataset: labeled FASTA plus label table to tokenized examples.

This is where the prior codebase failed. Its dataset filter kept sequences
only up to ``max_length``, which was capped at 512, so every full-length Cas9
was dropped inside a boolean predicate and the model trained on nothing it was
supposed to learn. Nobody noticed because nothing counted the drops.

So the two rules here are:

* an over-length sequence is **counted and reported**, never silently
  discarded, and the policy (drop or truncate) comes from config;
* the dataset refuses to be constructed empty, because an empty training set
  that trains for zero steps looks like a fast run rather than a bug.

Growth-temperature label dropout replaces the control token with the
unconditional token on a fraction of examples, which is what makes
classifier-free-guidance-style strengthening possible at generation time. The
dropout is seeded per example index rather than drawn globally, so two runs
with the same seed see the same examples unconditioned and the Gate 2
comparison is reproducible.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset

from thermocas.data.fasta import read_fasta
from thermocas.model.tokenizer import PretrainedTokenizerAdapter, ProteinTokenizer

IGNORE_INDEX = -100


@dataclass
class DatasetReport:
    """What the dataset accepted and rejected."""

    rows_in_table: int = 0
    sequences_in_fasta: int = 0
    kept: int = 0
    dropped_too_long: int = 0
    truncated: int = 0
    dropped_no_label: int = 0
    dropped_no_sequence: int = 0
    dropped_wrong_split: int = 0
    dropped_unverified: int = 0
    bin_counts: Counter[str] = field(default_factory=Counter)
    longest_kept: int = 0
    longest_seen: int = 0

    def summary(self) -> str:
        lines = [f"kept {self.kept} examples"]
        if self.kept:
            lines.append(f"  longest kept: {self.longest_kept} tokens")
            lines.append(
                "  bins: " + ", ".join(f"{k}={v}" for k, v in sorted(self.bin_counts.items()))
            )
        for label, count in (
            ("too long", self.dropped_too_long),
            ("truncated", self.truncated),
            ("no growth-temperature label", self.dropped_no_label),
            ("no sequence in FASTA", self.dropped_no_sequence),
            ("wrong split", self.dropped_wrong_split),
            ("OGT not cross-checked", self.dropped_unverified),
        ):
            if count:
                lines.append(f"  {label}: {count}")
        if self.dropped_too_long:
            lines.append(
                f"  WARNING {self.dropped_too_long} sequence(s) exceeded the context "
                f"(longest seen {self.longest_seen} tokens). Raise max_length or "
                "accept that these are excluded."
            )
        return "\n".join(lines)


class ConditionedProteinDataset(Dataset):
    """Causal-LM examples of the form ``[BOS] [TEMP_BIN] SEQUENCE [EOS]``.

    ``labels`` masks the prompt positions with ``IGNORE_INDEX`` so the loss is
    computed on residues only. Predicting the control token from BOS would let
    the model lower its loss by learning the bin prior rather than any
    sequence-level consequence of temperature, which is exactly the shortcut
    Gate 2 is meant to rule out.
    """

    def __init__(
        self,
        sequences: dict[str, str],
        labels: list[dict[str, str]],
        tokenizer: ProteinTokenizer | PretrainedTokenizerAdapter,
        max_length: int,
        split: str | None = None,
        on_too_long: str = "drop_and_report",
        growth_temp_token_dropout: float = 0.0,
        require_verified_ogt: bool = False,
        use_growth_temp_token: bool = True,
        seed: int = 0,
    ) -> None:
        if on_too_long not in {"drop_and_report", "truncate"}:
            raise ValueError(
                f"unknown on_too_long policy {on_too_long!r}; "
                "expected 'drop_and_report' or 'truncate'"
            )
        if not 0.0 <= growth_temp_token_dropout <= 1.0:
            raise ValueError(
                f"growth_temp_token_dropout must be in [0, 1], got {growth_temp_token_dropout}"
            )

        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.dropout = float(growth_temp_token_dropout)
        self.use_growth_temp_token = use_growth_temp_token
        self.seed = int(seed)
        self.overhead = tokenizer.encoding_overhead(use_growth_temp_token)
        self.report = DatasetReport(rows_in_table=len(labels), sequences_in_fasta=len(sequences))

        self.examples: list[dict[str, object]] = []

        for row in labels:
            seq_id = row.get("seq_id", "")
            if split is not None and (row.get("split") or "") != split:
                self.report.dropped_wrong_split += 1
                continue
            sequence = sequences.get(seq_id)
            if not sequence:
                self.report.dropped_no_sequence += 1
                continue
            temp_bin = row.get("temp_bin") or ""
            if not temp_bin or temp_bin not in tokenizer.growth_temp_bins:
                self.report.dropped_no_label += 1
                continue
            if require_verified_ogt and str(row.get("ogt_verified", "")).strip() not in {"1", "True", "true"}:
                self.report.dropped_unverified += 1
                continue

            # The tokenizer reports its own overhead. Hardcoding it here as
            # "BOS + control + EOS" was wrong for the pretrained adapter,
            # which emits [TEMP] and ProGen2's direction marker but no BOS.
            overhead = self.overhead
            budget = self.max_length - overhead
            if budget <= 0:
                raise ValueError(
                    f"max_length {self.max_length} leaves no room for residues after "
                    f"{overhead} special tokens"
                )

            token_length = len(sequence) + overhead
            self.report.longest_seen = max(self.report.longest_seen, token_length)

            if len(sequence) > budget:
                if on_too_long == "drop_and_report":
                    self.report.dropped_too_long += 1
                    continue
                sequence = sequence[:budget]
                self.report.truncated += 1

            try:
                ogt_c = float(row.get("ogt_c", "nan"))
            except (TypeError, ValueError):
                ogt_c = float("nan")

            self.examples.append(
                {
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "temp_bin": temp_bin,
                    "ogt_c": ogt_c,
                }
            )
            self.report.kept += 1
            self.report.bin_counts[temp_bin] += 1
            self.report.longest_kept = max(self.report.longest_kept, len(sequence) + overhead)

        if not self.examples:
            raise ValueError(
                "dataset is empty after filtering; an empty training set trains for "
                f"zero steps and looks like a fast run rather than a bug.\n{self.report.summary()}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def _drop_label(self, index: int, override: bool | None) -> bool:
        """Whether this example's control token is replaced by the unconditional one.

        Seeded on the example index so the choice is reproducible across runs
        and across the Gate 2 comparison, rather than drawn from global RNG
        state that a dataloader worker would reshuffle.
        """
        if override is not None:
            return override
        if self.dropout <= 0.0:
            return False
        # A tuple is not an accepted seed type on Python 3.14, so the pair is
        # combined into one deterministic integer.
        return random.Random(self.seed * 1_000_003 + index).random() < self.dropout

    def __getitem__(
        self, index: int, growth_temp_bin_override: str | None = None, drop_label: bool | None = None
    ) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        bin_name = growth_temp_bin_override or str(example["temp_bin"])

        if not self.use_growth_temp_token:
            encoded = self.tokenizer.encode(str(example["sequence"]), growth_temp_bin=None)
            control_index = None
        else:
            encoded = self.tokenizer.encode(str(example["sequence"]), growth_temp_bin=bin_name)
            control_index = encoded.control_index
            if self._drop_label(index, drop_label) and control_index is not None:
                encoded.ids[control_index] = self.tokenizer.unconditional_id

        ids = encoded.ids
        # Mask the prompt so the loss is computed on residues and EOS only.
        # The tokenizer reports how many leading tokens that is; the adapter
        # has two (control token, direction marker) where the from-scratch
        # tokenizer has BOS plus control.
        labels = list(ids)
        for position in range(min(encoded.prompt_length, len(labels))):
            labels[position] = IGNORE_INDEX

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ogt_c": torch.tensor(float(example["ogt_c"]), dtype=torch.float),
            "temp_bin": bin_name,
            "seq_id": str(example["seq_id"]),
        }

    def with_shuffled_labels(self, rng_seed: int) -> list[str]:
        """Bin assignment per example with the bins shuffled, for Gate 2.

        Gate 2 compares validation perplexity under the correct
        growth-temperature token against a shuffled one. Shuffling the
        assignment rather than randomizing each label independently preserves
        the bin distribution, so the comparison isolates whether the token
        matches its sequence instead of also changing the class prior.
        """
        bins = [str(e["temp_bin"]) for e in self.examples]
        shuffled = list(bins)
        rng = random.Random(rng_seed)
        rng.shuffle(shuffled)
        # If a shuffle leaves every label in place (possible when one class
        # dominates), the comparison would be vacuous. Force a difference.
        if len(set(bins)) > 1 and shuffled == bins:
            for i, b in enumerate(bins):
                if b != bins[0]:
                    shuffled[0], shuffled[i] = shuffled[i], shuffled[0]
                    break
        return shuffled


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor | list[str]]:
    """Right-pad a batch. Padded label positions are masked out of the loss."""
    width = max(item["input_ids"].numel() for item in batch)
    size = len(batch)

    input_ids = torch.full((size, width), pad_id, dtype=torch.long)
    labels = torch.full((size, width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((size, width), dtype=torch.long)

    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "ogt_c": torch.stack([item["ogt_c"] for item in batch]),
        "temp_bin": [item["temp_bin"] for item in batch],
        "seq_id": [item["seq_id"] for item in batch],
    }


def load_sequences(path: str | Path) -> dict[str, str]:
    return {record.seq_id: record.sequence for record in read_fasta(path)}


def load_label_table(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))
