"""Dataset construction, the over-length policy, and Gate 2 label shuffling.

The over-length tests are the direct regression against the prior
codebase's central bug, where full-length Cas9 vanished inside a filter
predicate and nothing counted the loss.
"""

from __future__ import annotations

import pytest
import torch

from thermocas.model.dataset import (
    IGNORE_INDEX,
    ConditionedProteinDataset,
    collate,
)
from thermocas.model.tokenizer import ProteinTokenizer

BINS = ["meso", "thermo"]


@pytest.fixture
def tok() -> ProteinTokenizer:
    return ProteinTokenizer(BINS)


def label_row(seq_id: str, temp_bin: str = "meso", **extra) -> dict[str, str]:
    row = {
        "seq_id": seq_id,
        "temp_bin": temp_bin,
        "ogt_c": "37" if temp_bin == "meso" else "60",
        "split": "train",
        "ogt_verified": "1",
    }
    row.update(extra)
    return row


def build(tok, sequences, labels, **kwargs):
    options = dict(max_length=64, split="train")
    options.update(kwargs)
    return ConditionedProteinDataset(sequences, labels, tok, **options)


class TestConstruction:
    def test_builds_and_encodes(self, tok):
        dataset = build(tok, {"a": "MGKP" * 5}, [label_row("a")])
        assert len(dataset) == 1
        item = dataset[0]
        assert item["input_ids"][0] == tok.bos_id
        assert item["input_ids"][1] == tok.control_id("meso")
        assert item["input_ids"][-1] == tok.eos_id

    def test_prompt_positions_are_masked_from_the_loss(self, tok):
        """Predicting the control token from BOS would let the model lower
        its loss by learning the bin prior, which is the shortcut Gate 2
        exists to rule out."""
        dataset = build(tok, {"a": "MGKP"}, [label_row("a")])
        labels = dataset[0]["labels"]
        assert labels[0].item() == IGNORE_INDEX  # BOS
        assert labels[1].item() == IGNORE_INDEX  # control token
        assert labels[2].item() != IGNORE_INDEX  # first residue

    def test_empty_dataset_raises_rather_than_training_on_nothing(self, tok):
        """An empty training set trains for zero steps and looks like a fast
        run rather than a bug."""
        with pytest.raises(ValueError, match="empty"):
            build(tok, {}, [label_row("missing")])

    def test_unknown_on_too_long_policy_rejected(self, tok):
        with pytest.raises(ValueError, match="on_too_long"):
            build(tok, {"a": "MGKP"}, [label_row("a")], on_too_long="silently_drop")

    def test_max_length_too_small_for_overhead_rejected(self, tok):
        with pytest.raises(ValueError, match="no room"):
            build(tok, {"a": "MGKP"}, [label_row("a")], max_length=2)


class TestOverLengthPolicy:
    """The prior repo's bug: over-length sequences disappeared uncounted."""

    def test_over_length_is_counted_not_silently_dropped(self, tok):
        sequences = {"short": "M" * 10, "long": "M" * 500}
        dataset = build(tok, sequences, [label_row("short"), label_row("long")], max_length=64)
        assert len(dataset) == 1
        assert dataset.report.dropped_too_long == 1
        assert dataset.report.longest_seen == 503
        assert "WARNING" in dataset.report.summary()
        assert "too long" in dataset.report.summary()

    def test_truncate_policy_keeps_and_counts(self, tok):
        dataset = build(
            tok, {"long": "M" * 500}, [label_row("long")], max_length=64, on_too_long="truncate"
        )
        assert len(dataset) == 1
        assert dataset.report.truncated == 1
        # BOS + control + 61 residues + EOS == 64
        assert dataset[0]["input_ids"].numel() == 64

    def test_boundary_sequence_fits_exactly(self, tok):
        """max_length 64 means 61 residues plus BOS, control token and EOS."""
        dataset = build(tok, {"a": "M" * 61}, [label_row("a")], max_length=64)
        assert dataset.report.dropped_too_long == 0
        assert dataset[0]["input_ids"].numel() == 64

    def test_one_residue_over_is_dropped(self, tok):
        """62 residues + BOS + control + EOS = 65 > 64, so it must not fit."""
        sequences = {"fits": "M" * 61, "over": "M" * 62}
        dataset = build(tok, sequences, [label_row("fits"), label_row("over")], max_length=64)
        assert [e["seq_id"] for e in dataset.examples] == ["fits"]
        assert dataset.report.dropped_too_long == 1

    def test_unconditioned_mode_has_one_token_more_room(self, tok):
        """Without a control token the residue budget grows by one."""
        dataset = build(
            tok, {"a": "M" * 62}, [label_row("a")], max_length=64, use_growth_temp_token=False
        )
        assert dataset.report.kept == 1
        assert dataset[0]["input_ids"].numel() == 64

    def test_a_real_cas9_fits_the_configured_context(self, tok):
        """Guards the production path: 1,703 tokens must hold a 1,368 aa Cas9."""
        from thermocas.config import Config

        budget = Config.load("train")["data/max_length"]
        dataset = build(tok, {"spcas9": "M" * 1368}, [label_row("spcas9")], max_length=budget)
        assert dataset.report.dropped_too_long == 0
        assert len(dataset) == 1


class TestFiltering:
    def test_split_filter(self, tok):
        labels = [label_row("a", split="train"), label_row("b", split="test")]
        dataset = build(tok, {"a": "MGKP", "b": "MGKP"}, labels, split="train")
        assert len(dataset) == 1
        assert dataset.report.dropped_wrong_split == 1

    def test_missing_sequence_counted(self, tok):
        dataset = build(tok, {"a": "MGKP"}, [label_row("a"), label_row("ghost")])
        assert dataset.report.dropped_no_sequence == 1

    def test_unknown_bin_counted_not_guessed(self, tok):
        labels = [label_row("a"), label_row("b", temp_bin="hyper")]
        dataset = build(tok, {"a": "MGKP", "b": "MGKP"}, labels)
        assert len(dataset) == 1
        assert dataset.report.dropped_no_label == 1

    def test_blank_bin_counted(self, tok):
        labels = [label_row("a"), label_row("b", temp_bin="")]
        dataset = build(tok, {"a": "MGKP", "b": "MGKP"}, labels)
        assert dataset.report.dropped_no_label == 1

    def test_require_verified_ogt(self, tok):
        labels = [label_row("a", ogt_verified="1"), label_row("b", ogt_verified="0")]
        dataset = build(
            tok, {"a": "MGKP", "b": "MGKP"}, labels, require_verified_ogt=True
        )
        assert len(dataset) == 1
        assert dataset.report.dropped_unverified == 1

    def test_unverified_kept_by_default(self, tok):
        labels = [label_row("a", ogt_verified="0")]
        assert len(build(tok, {"a": "MGKP"}, labels)) == 1


class TestLabelDropout:
    def test_no_dropout_keeps_every_control_token(self, tok):
        dataset = build(
            tok, {f"s{i}": "MGKP" for i in range(20)},
            [label_row(f"s{i}") for i in range(20)],
            growth_temp_token_dropout=0.0,
        )
        for i in range(len(dataset)):
            assert dataset[i]["input_ids"][1] == tok.control_id("meso")

    def test_dropout_replaces_some_tokens_with_the_unconditional_one(self, tok):
        dataset = build(
            tok, {f"s{i}": "MGKP" for i in range(200)},
            [label_row(f"s{i}") for i in range(200)],
            growth_temp_token_dropout=0.5,
        )
        dropped = sum(
            1 for i in range(len(dataset)) if dataset[i]["input_ids"][1] == tok.unconditional_id
        )
        assert 60 <= dropped <= 140, dropped

    def test_dropout_is_reproducible_for_a_fixed_seed(self, tok):
        """Gate 2 compares two passes; the same examples must be unconditioned
        in both, or the comparison measures dropout noise."""
        def pattern(seed):
            dataset = build(
                tok, {f"s{i}": "MGKP" for i in range(50)},
                [label_row(f"s{i}") for i in range(50)],
                growth_temp_token_dropout=0.3, seed=seed,
            )
            return [int(dataset[i]["input_ids"][1] == tok.unconditional_id) for i in range(50)]

        assert pattern(7) == pattern(7)
        assert pattern(7) != pattern(8)

    def test_dropout_rate_out_of_range_rejected(self, tok):
        with pytest.raises(ValueError, match="dropout"):
            build(tok, {"a": "MGKP"}, [label_row("a")], growth_temp_token_dropout=1.5)

    def test_override_forces_the_conditioning(self, tok):
        dataset = build(tok, {"a": "MGKP"}, [label_row("a", temp_bin="meso")])
        item = dataset.__getitem__(0, growth_temp_bin_override="thermo", drop_label=False)
        assert item["input_ids"][1] == tok.control_id("thermo")
        assert item["temp_bin"] == "thermo"


class TestGate2Shuffling:
    def test_shuffle_preserves_the_bin_distribution(self, tok):
        """Shuffling rather than randomizing isolates whether the token matches
        its sequence, instead of also changing the class prior."""
        labels = [label_row(f"m{i}", "meso") for i in range(30)]
        labels += [label_row(f"t{i}", "thermo") for i in range(10)]
        sequences = {row["seq_id"]: "MGKP" for row in labels}
        dataset = build(tok, sequences, labels)

        original = [str(e["temp_bin"]) for e in dataset.examples]
        shuffled = dataset.with_shuffled_labels(rng_seed=3)
        assert sorted(shuffled) == sorted(original)
        assert shuffled != original

    def test_shuffle_is_never_a_no_op_with_mixed_bins(self, tok):
        """A vacuous shuffle would make Gate 2 compare a model against itself."""
        labels = [label_row("a", "meso"), label_row("b", "thermo")]
        sequences = {"a": "MGKP", "b": "MGKP"}
        dataset = build(tok, sequences, labels)
        for seed in range(25):
            shuffled = dataset.with_shuffled_labels(rng_seed=seed)
            assert shuffled != [str(e["temp_bin"]) for e in dataset.examples], seed

    def test_shuffle_is_reproducible(self, tok):
        labels = [label_row(f"s{i}", "meso" if i % 2 else "thermo") for i in range(20)]
        dataset = build(tok, {row["seq_id"]: "MGKP" for row in labels}, labels)
        assert dataset.with_shuffled_labels(5) == dataset.with_shuffled_labels(5)


class TestCollate:
    def test_pads_and_masks(self, tok):
        labels = [label_row("a"), label_row("b")]
        dataset = build(tok, {"a": "M" * 10, "b": "M" * 20}, labels)
        batch = collate([dataset[0], dataset[1]], pad_id=tok.pad_id)

        assert batch["input_ids"].shape == batch["labels"].shape == (2, 23)
        # Padding must not contribute to the loss.
        padded = batch["attention_mask"] == 0
        assert torch.all(batch["labels"][padded] == IGNORE_INDEX)
        assert torch.all(batch["input_ids"][padded] == tok.pad_id)

    def test_attention_mask_marks_real_tokens(self, tok):
        dataset = build(tok, {"a": "M" * 10, "b": "M" * 20}, [label_row("a"), label_row("b")])
        batch = collate([dataset[0], dataset[1]], pad_id=tok.pad_id)
        assert batch["attention_mask"][0].sum().item() == 13
        assert batch["attention_mask"][1].sum().item() == 23

    def test_carries_ogt_and_metadata(self, tok):
        dataset = build(
            tok, {"a": "MGKP", "b": "MGKP"},
            [label_row("a", "meso"), label_row("b", "thermo")],
        )
        batch = collate([dataset[0], dataset[1]], pad_id=tok.pad_id)
        assert batch["ogt_c"].tolist() == [37.0, 60.0]
        assert batch["temp_bin"] == ["meso", "thermo"]
        assert batch["seq_id"] == ["a", "b"]


class TestPromptLayoutComesFromTheTokenizer:
    """The dataset must not assume where the residues start.

    It previously computed the masked prefix as "BOS plus maybe a control
    token", which is the from-scratch tokenizer's layout. The pretrained
    adapter emits ``[TEMP] 1 SEQUENCE [EOS]`` -- ProGen2's direction marker in
    place of BOS -- so the same arithmetic left the direction marker in the
    loss and, with conditioning off, trained the model to predict it from
    nothing. Both numbers now come from the tokenizer.
    """

    @staticmethod
    def adapter():
        from tests.test_pretrained_tokenizer import FakeBaseTokenizer
        from thermocas.model.tokenizer import PretrainedTokenizerAdapter

        return PretrainedTokenizerAdapter(FakeBaseTokenizer(), BINS, base_vocab_size=32)

    def test_adapter_prompt_is_masked_through_the_direction_marker(self):
        adapter = self.adapter()
        dataset = build(adapter, {"a": "MAKE"}, [label_row("a")])
        item = dataset[0]
        ids, labels = item["input_ids"], item["labels"]

        assert ids[0].item() == adapter.control_id("meso")
        assert ids[1].item() == adapter.base.token_to_id("1")
        assert labels[0].item() == IGNORE_INDEX  # control token
        assert labels[1].item() == IGNORE_INDEX  # direction marker
        assert labels[2].item() == ids[2].item() != IGNORE_INDEX  # first residue

    def test_unconditioned_adapter_masks_only_the_direction_marker(self):
        adapter = self.adapter()
        dataset = build(
            adapter, {"a": "MAKE"}, [label_row("a")], use_growth_temp_token=False
        )
        item = dataset[0]
        assert item["input_ids"][0].item() == adapter.base.token_to_id("1")
        assert item["labels"][0].item() == IGNORE_INDEX
        assert item["labels"][1].item() != IGNORE_INDEX

    def test_length_budget_uses_the_tokenizer_overhead(self):
        # [TEMP] 1 ... [EOS] is three non-residue tokens, so a max_length of
        # 8 admits five residues and rejects six.
        adapter = self.adapter()
        assert adapter.encoding_overhead() == 3
        dataset = build(
            adapter, {"a": "MAKER", "b": "MAKERS"}, [label_row("a"), label_row("b")],
            max_length=8,
        )
        assert dataset.report.kept == 1
        assert dataset.report.dropped_too_long == 1
        assert len(dataset[0]["input_ids"]) == 8

    def test_scratch_tokenizer_keeps_its_own_layout(self, tok):
        assert tok.encoding_overhead() == 3           # BOS + control + EOS
        assert tok.encoding_overhead(False) == 2      # BOS + EOS
        dataset = build(tok, {"a": "MAKE"}, [label_row("a")])
        labels = dataset[0]["labels"]
        assert labels[0].item() == IGNORE_INDEX       # BOS
        assert labels[1].item() == IGNORE_INDEX       # control token
        assert labels[2].item() != IGNORE_INDEX
