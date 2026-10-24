"""The pretrained tokenizer adapter.

Guards the mistake these tests were written for: feeding this project's own
token ids into a pretrained checkpoint whose vocabulary is ordered
differently, so every residue indexes the wrong embedding row. Training still
runs and the loss still falls, but the pretrained representation is noise and
LoRA cannot relearn it because the base weights are frozen.
"""

from __future__ import annotations

import os

import pytest

from thermocas.model.tokenizer import (
    STANDARD_RESIDUES,
    PretrainedTokenizerAdapter,
    ProteinTokenizer,
)

BINS = ["meso", "thermo"]
NETWORK = os.environ.get("THERMOCAS_NETWORK_TESTS") == "1"


class FakeBaseTokenizer:
    """Mimics a `tokenizers.Tokenizer` with ProGen2's actual vocabulary."""

    VOCAB = {
        "<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "1": 3, "2": 4,
        "A": 5, "B": 6, "C": 7, "D": 8, "E": 9, "F": 10, "G": 11, "H": 12,
        "I": 13, "K": 14, "L": 15, "M": 16, "N": 17, "O": 18, "P": 19,
        "Q": 20, "R": 21, "S": 22, "T": 23, "U": 24, "V": 25, "W": 26,
        "X": 27, "Y": 28, "Z": 29,
    }

    def get_vocab_size(self, with_added_tokens=True):
        return len(self.VOCAB)

    def token_to_id(self, token):
        return self.VOCAB.get(token)

    def id_to_token(self, index):
        for token, value in self.VOCAB.items():
            if value == index:
                return token
        return None

    def encode(self, text, add_special_tokens=False):
        class R:
            ids = [FakeBaseTokenizer.VOCAB[c] for c in text]
        return R()


@pytest.fixture
def adapter() -> PretrainedTokenizerAdapter:
    return PretrainedTokenizerAdapter(FakeBaseTokenizer(), BINS, base_vocab_size=32)


class TestVocabularyOwnership:
    def test_residues_use_the_checkpoints_ids_not_ours(self, adapter):
        """The core fix. ProGen2 maps alanine to 5; this project's own
        tokenizer maps 4 to alanine and 5 to cysteine."""
        ours = ProteinTokenizer(BINS)
        assert ours.stoi["A"] == 4
        assert adapter.base.token_to_id("A") == 5
        ids = adapter.encode("A").ids
        assert 5 in ids
        assert adapter.decode(ids) == "A"

    def test_every_standard_residue_resolves(self, adapter):
        assert adapter.decode(adapter.residue_ids()) == STANDARD_RESIDUES

    def test_control_tokens_sit_above_the_pretrained_vocabulary(self, adapter):
        """Appending above the embedding matrix means no pretrained row is
        reinterpreted or discarded."""
        assert adapter.unconditional_id == 32
        assert adapter.control_id("meso") == 33
        assert adapter.control_id("thermo") == 34
        assert len(adapter) == 35
        for token_id in (adapter.unconditional_id, 33, 34):
            assert token_id >= adapter.base_vocab_size

    def test_control_ids_do_not_collide_with_residues(self, adapter):
        residues = set(adapter.residue_ids())
        assert adapter.control_id("meso") not in residues
        assert adapter.control_id("thermo") not in residues
        assert adapter.unconditional_id not in residues

    def test_tokenizer_larger_than_embedding_matrix_is_rejected(self):
        with pytest.raises(ValueError, match="embedding matrix"):
            PretrainedTokenizerAdapter(FakeBaseTokenizer(), BINS, base_vocab_size=8)


class TestEncoding:
    def test_emits_the_direction_marker(self, adapter):
        """ProGen2's documented prompt is "1MEVVIVTGMSGAGK"; the model was
        trained with that marker and omitting it shifts the distribution."""
        encoded = adapter.encode("MEV", growth_temp_bin="thermo")
        assert encoded.ids[0] == adapter.control_id("thermo")
        assert encoded.ids[1] == adapter.base.token_to_id("1")
        assert encoded.ids[-1] == adapter.eos_id

    def test_control_index_points_at_the_control_token(self, adapter):
        encoded = adapter.encode("MEV", growth_temp_bin="meso")
        assert encoded.control_index == 0
        assert encoded.ids[encoded.control_index] == adapter.control_id("meso")
        # Swapping in place must keep the sequence decodable.
        encoded.ids[encoded.control_index] = adapter.unconditional_id
        assert adapter.decode(encoded.ids) == "MEV"

    def test_unconditioned_encoding_has_no_control_index(self, adapter):
        assert adapter.encode("MEV").control_index is None

    def test_roundtrip(self, adapter):
        seq = "MEVVIVTGMSGAGK"
        assert adapter.decode(adapter.encode(seq, growth_temp_bin="thermo").ids) == seq

    def test_no_bos_by_default(self, adapter):
        """ProGen2 prompts with the direction marker, not a BOS token."""
        assert adapter.encode("MEV").ids[0] != adapter.bos_id

    def test_nonstandard_residues_rejected(self, adapter):
        for bad in ("MEVX", "MEV-", "MEVU"):
            with pytest.raises(ValueError, match="non-standard"):
                adapter.encode(bad)

    def test_empty_rejected(self, adapter):
        with pytest.raises(ValueError, match="empty"):
            adapter.encode("  ")

    def test_unknown_bin_names_its_options(self, adapter):
        with pytest.raises(ValueError, match="meso"):
            adapter.control_id("hyper")


class TestSignature:
    def test_differs_from_the_scratch_tokenizer(self, adapter):
        """Loading a checkpoint against the wrong vocabulary must be
        detectable, and the two layouts are not interchangeable."""
        assert adapter.signature != ProteinTokenizer(BINS).signature

    def test_records_the_base_vocabulary_size(self, adapter):
        """Appending control tokens above a different base shifts every
        control id, so the base size belongs in the signature."""
        assert "base=32" in adapter.signature
        other = PretrainedTokenizerAdapter(FakeBaseTokenizer(), BINS, base_vocab_size=30)
        assert other.signature != adapter.signature

    def test_changes_when_bins_are_reordered(self, adapter):
        reversed_bins = PretrainedTokenizerAdapter(
            FakeBaseTokenizer(), list(reversed(BINS)), base_vocab_size=32
        )
        assert reversed_bins.signature != adapter.signature


@pytest.mark.skipif(not NETWORK, reason="set THERMOCAS_NETWORK_TESTS=1 to hit the HF Hub")
class TestAgainstRealCheckpoint:
    def test_matches_the_published_vocabulary(self):
        from thermocas.config import Config

        checkpoint = Config.load("model")["conditioned_lm/checkpoint"]
        adapter = PretrainedTokenizerAdapter.from_checkpoint(checkpoint, BINS)
        assert adapter.base_vocab_size == 32
        assert len(adapter) == 35
        assert adapter.base.token_to_id("A") == 5
        assert adapter.decode(adapter.encode("MEVVIVTGMSGAGK").ids) == "MEVVIVTGMSGAGK"

    def test_no_remote_code_is_fetched_or_executed(self, monkeypatch):
        """Building the tokenizer must not reach for the Hub's model code.

        It used to. `from_checkpoint` called
        `AutoConfig.from_pretrained(trust_remote_code=True)`, which resolves
        the auto_map in the published config.json and downloads
        configuration_progen.py from the Hub at call time. That is the
        moving target the whole vendored package exists to remove, and it
        was invisible until a clean container printed the download notice.

        Failing inside get_cached_module_file rather than asserting on a
        string makes this a test of behaviour: any future path back to
        remote code trips it, whatever calls it.
        """
        import transformers.dynamic_module_utils as dynamic

        from thermocas.config import Config

        def refuse(*args, **kwargs):
            raise AssertionError("remote code was requested from the Hub")

        monkeypatch.setattr(dynamic, "get_cached_module_file", refuse)
        monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", refuse)

        checkpoint = Config.load("model")["conditioned_lm/checkpoint"]
        adapter = PretrainedTokenizerAdapter.from_checkpoint(checkpoint, BINS)
        assert adapter.base_vocab_size == 32


class TestPromptLengthAndOverhead:
    """What the dataset needs in order not to guess the prompt layout."""

    def test_prompt_length_covers_control_token_and_direction_marker(self):
        adapter = PretrainedTokenizerAdapter(FakeBaseTokenizer(), BINS, base_vocab_size=32)
        assert adapter.encode("MAK", growth_temp_bin="thermo").prompt_length == 2
        assert adapter.encode("MAK").prompt_length == 1

    def test_overhead_is_measured_from_encode_not_counted_by_hand(self):
        adapter = PretrainedTokenizerAdapter(FakeBaseTokenizer(), BINS, base_vocab_size=32)
        for use_token in (True, False):
            bin_name = "meso" if use_token else None
            encoded = adapter.encode("MAKE", growth_temp_bin=bin_name)
            assert adapter.encoding_overhead(use_token) == len(encoded.ids) - 4

    def test_scratch_tokenizer_reports_its_own_layout(self):
        tokenizer = ProteinTokenizer(BINS)
        assert tokenizer.encode("MAK", growth_temp_bin="thermo").prompt_length == 2
        assert tokenizer.encode("MAK", add_bos=False).prompt_length == 0
        assert tokenizer.encoding_overhead() == 3
        assert tokenizer.encoding_overhead(False) == 2

    def test_the_two_tokenizers_disagree_about_the_layout(self):
        """Documents why the dataset cannot hardcode either one."""
        adapter = PretrainedTokenizerAdapter(FakeBaseTokenizer(), BINS, base_vocab_size=32)
        tokenizer = ProteinTokenizer(BINS)
        first_adapter = adapter.encode("MAK", growth_temp_bin="meso").ids[0]
        first_scratch = tokenizer.encode("MAK", growth_temp_bin="meso").ids[0]
        assert first_adapter == adapter.control_id("meso")   # control token leads
        assert first_scratch == tokenizer.bos_id             # BOS leads
