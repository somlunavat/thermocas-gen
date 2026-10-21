"""Amino-acid tokenizer with growth-temperature control tokens.

The vocabulary is the 20 standard residues plus special tokens, plus one
control token per growth-temperature bin and one unconditional token.

The unconditional token is what makes classifier-free-guidance-style
strengthening possible at generation time: training replaces the real control
token with ``<temp:any>`` on a fraction of examples, so the model learns both
the conditional and the unconditional distribution and the two can be
contrasted when sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

PAD = "<pad>"
MASK = "<mask>"
BOS = "<bos>"
EOS = "<eos>"
UNCONDITIONAL = "<temp:any>"

SPECIAL_TOKENS = (PAD, MASK, BOS, EOS)


def control_token(bin_name: str) -> str:
    """The control token for a growth-temperature bin, e.g. ``<temp:thermo>``."""
    return f"<temp:{bin_name}>"


@dataclass(frozen=True)
class Encoded:
    """One tokenized example.

    ``control_index`` is where the growth-temperature token sits, so training
    can swap it for the unconditional token (label dropout) or a different bin
    (the Gate 2 shuffled-token check) without re-tokenizing the sequence.
    """

    ids: list[int]
    control_index: int | None


class ProteinTokenizer:
    """Residue-level vocabulary with growth-temperature control tokens.

    Token ids are assigned deterministically: special tokens, then residues,
    then control tokens in the order the bins were configured. Order matters
    because a checkpoint's embedding matrix is indexed by it, so changing the
    bin list invalidates existing checkpoints. ``signature`` exists to detect
    that.
    """

    def __init__(self, growth_temp_bins: list[str]) -> None:
        if len(set(growth_temp_bins)) != len(growth_temp_bins):
            raise ValueError(f"duplicate growth-temperature bin names: {growth_temp_bins}")
        self.growth_temp_bins = list(growth_temp_bins)

        tokens = list(SPECIAL_TOKENS)
        tokens += list(STANDARD_RESIDUES)
        tokens += [UNCONDITIONAL]
        tokens += [control_token(name) for name in self.growth_temp_bins]

        self.itos: list[str] = tokens
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}

        self.pad_id = self.stoi[PAD]
        self.mask_id = self.stoi[MASK]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unconditional_id = self.stoi[UNCONDITIONAL]

        self._residue_ids = {self.stoi[r] for r in STANDARD_RESIDUES}
        self._control_ids = {
            name: self.stoi[control_token(name)] for name in self.growth_temp_bins
        }

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def signature(self) -> str:
        """Identifies the vocabulary layout, to be stored with a checkpoint.

        Comparing signatures catches the case where a model trained with bins
        (meso, thermo, hyper) is loaded against a config that has since
        reordered or renamed them, which would otherwise silently reinterpret
        every control token.
        """
        return "|".join(self.itos)

    def control_id(self, bin_name: str) -> int:
        try:
            return self._control_ids[bin_name]
        except KeyError:
            raise ValueError(
                f"unknown growth-temperature bin {bin_name!r}; "
                f"configured bins are {self.growth_temp_bins}"
            ) from None

    def is_residue(self, token_id: int) -> bool:
        return token_id in self._residue_ids

    def residue_ids(self) -> list[int]:
        """Sorted residue ids, for restricting generation to real residues."""
        return sorted(self._residue_ids)

    def encode(
        self,
        sequence: str,
        growth_temp_bin: str | None = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Encoded:
        """Tokenize a sequence, optionally prefixed with a control token.

        Raises on any non-standard residue. Upstream filtering is responsible
        for dropping sequences containing X, B, Z, U, O or gaps; reaching here
        with one means the filter was bypassed, and mapping it to a silent
        placeholder would hide that.
        """
        seq = sequence.strip().upper()
        if not seq:
            raise ValueError("cannot encode an empty sequence")

        unknown = sorted(set(seq) - set(STANDARD_RESIDUES))
        if unknown:
            raise ValueError(
                f"non-standard residues {unknown} in sequence; "
                "filter these upstream rather than substituting them"
            )

        ids: list[int] = []
        control_index: int | None = None

        if add_bos:
            ids.append(self.bos_id)
        if growth_temp_bin is not None:
            control_index = len(ids)
            ids.append(self.control_id(growth_temp_bin))

        ids.extend(self.stoi[residue] for residue in seq)

        if add_eos:
            ids.append(self.eos_id)

        return Encoded(ids=ids, control_index=control_index)

    def decode(self, ids: list[int], residues_only: bool = True) -> str:
        """Render token ids back to a sequence.

        With ``residues_only`` (the default) special and control tokens are
        dropped, giving a clean amino-acid string suitable for writing to
        FASTA.
        """
        if residues_only:
            return "".join(self.itos[i] for i in ids if self.is_residue(i))
        return "".join(self.itos[i] for i in ids)


class PretrainedTokenizerAdapter:
    """Wraps a pretrained checkpoint's tokenizer and appends control tokens.

    This exists because of a mistake worth describing. The project originally
    fed :class:`ProteinTokenizer` ids straight into ProGen2. The two
    vocabularies are unrelated -- ProGen2 maps id 4 to its direction marker
    ``2`` and id 5 to alanine, while ProteinTokenizer maps 4 to alanine -- so
    every residue indexed the wrong pretrained embedding row. Training would
    still have run and the loss would still have fallen, but the pretrained
    representation would have been noise, and with LoRA freezing the base
    weights the model could not have relearned it. Gate 2 would then have
    failed for a reason that has nothing to do with whether temperature
    conditioning transfers.

    The rule this encodes: when fine-tuning a pretrained model, the residue
    vocabulary belongs to the checkpoint. Only the control tokens are ours,
    and they are appended as new ids above the checkpoint's vocabulary so no
    pretrained row is reinterpreted or discarded.

    ProGen2 also expects a direction marker before the sequence (``"1"`` for
    N-to-C, as in its documented prompt ``"1MEVVIVTGMSGAGK"``), which is
    emitted here and masked out of the loss alongside the control token.
    """

    def __init__(
        self,
        base_tokenizer,
        growth_temp_bins: list[str],
        base_vocab_size: int,
        direction_token: str = "1",
        pad_token: str = "<|pad|>",
        eos_token: str = "<|eos|>",
        bos_token: str = "<|bos|>",
    ) -> None:
        if len(set(growth_temp_bins)) != len(growth_temp_bins):
            raise ValueError(f"duplicate growth-temperature bin names: {growth_temp_bins}")
        self.base = base_tokenizer
        self.growth_temp_bins = list(growth_temp_bins)
        self.base_vocab_size = int(base_vocab_size)

        named = base_tokenizer.get_vocab_size(with_added_tokens=False)
        if named > self.base_vocab_size:
            raise ValueError(
                f"tokenizer declares {named} tokens but the model's embedding matrix "
                f"holds {self.base_vocab_size} rows; control tokens cannot be appended "
                "safely"
            )

        # Control tokens occupy fresh ids above the checkpoint's embedding
        # matrix, so every pretrained row keeps its original meaning.
        self.unconditional_id = self.base_vocab_size
        self._control_ids = {
            name: self.base_vocab_size + 1 + index
            for index, name in enumerate(self.growth_temp_bins)
        }
        self.vocab_size = self.base_vocab_size + 1 + len(self.growth_temp_bins)

        self._direction_id = self._require_token(direction_token)
        self.pad_id = self._require_token(pad_token)
        self.eos_id = self._require_token(eos_token)
        self.bos_id = self._require_token(bos_token)

        self._residue_ids = {
            self.base.token_to_id(residue)
            for residue in STANDARD_RESIDUES
            if self.base.token_to_id(residue) is not None
        }
        if len(self._residue_ids) != len(STANDARD_RESIDUES):
            missing = [r for r in STANDARD_RESIDUES if self.base.token_to_id(r) is None]
            raise ValueError(f"checkpoint tokenizer lacks residues {missing}")

        self._id_to_token = {
            index: self.base.id_to_token(index) for index in range(named)
        }
        self._id_to_token[self.unconditional_id] = UNCONDITIONAL
        for name, token_id in self._control_ids.items():
            self._id_to_token[token_id] = control_token(name)

    def _require_token(self, token: str) -> int:
        token_id = self.base.token_to_id(token)
        if token_id is None:
            raise ValueError(f"checkpoint tokenizer has no {token!r} token")
        return token_id

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def signature(self) -> str:
        """Identifies the vocabulary layout, stored with a checkpoint.

        Includes the base vocabulary size, since appending control tokens above
        a different base would silently shift every control id.
        """
        controls = ",".join(
            f"{name}={self._control_ids[name]}" for name in self.growth_temp_bins
        )
        return f"pretrained:base={self.base_vocab_size};uncond={self.unconditional_id};{controls}"

    def control_id(self, bin_name: str) -> int:
        try:
            return self._control_ids[bin_name]
        except KeyError:
            raise ValueError(
                f"unknown growth-temperature bin {bin_name!r}; "
                f"configured bins are {self.growth_temp_bins}"
            ) from None

    def is_residue(self, token_id: int) -> bool:
        return token_id in self._residue_ids

    def residue_ids(self) -> list[int]:
        return sorted(self._residue_ids)

    def encode(
        self,
        sequence: str,
        growth_temp_bin: str | None = None,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> Encoded:
        """Encode as ``[TEMP] 1 SEQUENCE [EOS]``.

        ``add_bos`` defaults to False because ProGen2's documented prompt uses
        the direction marker rather than a BOS token.
        """
        seq = sequence.strip().upper()
        if not seq:
            raise ValueError("cannot encode an empty sequence")
        unknown = sorted(set(seq) - set(STANDARD_RESIDUES))
        if unknown:
            raise ValueError(
                f"non-standard residues {unknown} in sequence; "
                "filter these upstream rather than substituting them"
            )

        ids: list[int] = []
        control_index: int | None = None
        if add_bos:
            ids.append(self.bos_id)
        if growth_temp_bin is not None:
            control_index = len(ids)
            ids.append(self.control_id(growth_temp_bin))

        ids.append(self._direction_id)
        ids.extend(self.base.encode(seq, add_special_tokens=False).ids)

        if add_eos:
            ids.append(self.eos_id)
        return Encoded(ids=ids, control_index=control_index)

    def decode(self, ids: list[int], residues_only: bool = True) -> str:
        if residues_only:
            return "".join(
                self._id_to_token.get(i, "") for i in ids if self.is_residue(i)
            )
        return "".join(self._id_to_token.get(i, "") for i in ids)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str, growth_temp_bins: list[str], base_vocab_size: int | None = None
    ) -> PretrainedTokenizerAdapter:
        """Load the checkpoint's own tokenizer and append control tokens."""
        from tokenizers import Tokenizer

        base = Tokenizer.from_pretrained(checkpoint)
        if base_vocab_size is None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
            base_vocab_size = int(
                getattr(config, "vocab_size_lm_head", None)
                or getattr(config, "vocab_size", 0)
            )
        return cls(base, growth_temp_bins, base_vocab_size)
