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
