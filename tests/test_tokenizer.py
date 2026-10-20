"""Tokenizer vocabulary layout and encoding behavior."""

from __future__ import annotations

import pytest

from thermocas.model.tokenizer import (
    STANDARD_RESIDUES,
    ProteinTokenizer,
    control_token,
)

BINS = ["meso", "thermo", "hyper"]


@pytest.fixture
def tok() -> ProteinTokenizer:
    return ProteinTokenizer(BINS)


def test_vocab_size_accounts_for_every_token_class(tok):
    # 4 special + 20 residues + 1 unconditional + 3 control tokens
    assert tok.vocab_size == 4 + 20 + 1 + len(BINS) == 28
    assert len(set(tok.itos)) == tok.vocab_size


def test_control_tokens_exist_for_every_bin(tok):
    for name in BINS:
        assert control_token(name) in tok.stoi
        assert tok.control_id(name) == tok.stoi[control_token(name)]
    # Each bin maps to a distinct id, or conditioning would be ambiguous.
    assert len({tok.control_id(n) for n in BINS}) == len(BINS)


def test_unconditional_token_is_distinct_from_every_bin(tok):
    """CFG-style guidance needs an unconditional token separate from the bins."""
    assert tok.unconditional_id not in {tok.control_id(n) for n in BINS}


def test_unknown_bin_names_its_options(tok):
    with pytest.raises(ValueError, match="meso"):
        tok.control_id("tepid")


def test_duplicate_bins_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ProteinTokenizer(["meso", "meso"])


def test_roundtrip_preserves_sequence(tok):
    seq = "MGKPIPNPLLGLDSTRTGHHHHHH"
    encoded = tok.encode(seq, growth_temp_bin="thermo")
    assert tok.decode(encoded.ids) == seq


def test_control_index_points_at_the_control_token(tok):
    encoded = tok.encode("MGKP", growth_temp_bin="hyper")
    assert encoded.control_index is not None
    assert encoded.ids[encoded.control_index] == tok.control_id("hyper")
    # Swapping the token in place is how label dropout and the Gate 2
    # shuffled-token check work, so the index must stay valid.
    encoded.ids[encoded.control_index] = tok.unconditional_id
    assert tok.decode(encoded.ids) == "MGKP"


def test_no_control_index_when_unconditioned(tok):
    assert tok.encode("MGKP").control_index is None


def test_bos_eos_placement(tok):
    encoded = tok.encode("MG", growth_temp_bin="meso")
    assert encoded.ids[0] == tok.bos_id
    assert encoded.ids[1] == tok.control_id("meso")
    assert encoded.ids[-1] == tok.eos_id
    assert len(encoded.ids) == 2 + 2 + 1


def test_nonstandard_residues_raise_rather_than_mapping_to_x(tok):
    """Selenocysteine, ambiguity codes and gaps must be filtered upstream.

    Silently substituting them would let junk into the training set under a
    valid-looking token, which is the failure mode this project is guarding
    against throughout.
    """
    for bad in ("MGKXP", "MGK-P", "MGKUP", "MGKBP", "MGK*P"):
        with pytest.raises(ValueError, match="non-standard"):
            tok.encode(bad)


def test_empty_sequence_rejected(tok):
    with pytest.raises(ValueError, match="empty"):
        tok.encode("   ")


def test_lowercase_input_is_accepted(tok):
    assert tok.decode(tok.encode("mgkp").ids) == "MGKP"


def test_residue_ids_are_contiguous_and_complete(tok):
    ids = tok.residue_ids()
    assert len(ids) == 20
    assert tok.decode(ids) == STANDARD_RESIDUES


def test_signature_changes_when_bins_are_reordered():
    """A checkpoint's embedding matrix is indexed by token id.

    Reordering the bins silently reinterprets every control token, so the
    signature has to differ for the mismatch to be detectable at load time.
    """
    a = ProteinTokenizer(["meso", "thermo", "hyper"])
    b = ProteinTokenizer(["hyper", "thermo", "meso"])
    assert a.signature != b.signature
    assert a.vocab_size == b.vocab_size  # size alone cannot catch this


def test_signature_stable_for_identical_bins():
    assert ProteinTokenizer(BINS).signature == ProteinTokenizer(BINS).signature
