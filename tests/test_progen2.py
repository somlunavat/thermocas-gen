"""Tests for the vendored ProGen2 code and its loader.

Most of these need the real checkpoint, because what they check is whether
loading it produces a working model -- and the failures this file exists for
all looked fine until the weights were compared or the model was asked to
score a sequence. They are gated on THERMOCAS_NETWORK_TESTS=1 so the suite
still runs offline, and the gated ones are the reason the ungated count is
small rather than reassuring.
"""

from __future__ import annotations

import math
import os
import random

import pytest

from thermocas.model.compat import (
    _is_expected_missing,
    unloaded_parameter_keys,
)

NETWORK = os.environ.get("THERMOCAS_NETWORK_TESTS") == "1"
CHECKPOINT = "hugohrban/progen2-base"

# A fragment of SpCas9 (Q99ZW2), which is in ProGen2's training data.
SPCAS9_N_TERMINUS = (
    "MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNRICYLQEI"
    "FSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGNIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEG"
)


class TestUnloadedParameterReporting:
    def test_mask_buffers_are_expected_to_be_absent(self):
        assert _is_expected_missing("transformer.h.3.attn.bias")
        assert _is_expected_missing("transformer.h.0.attn.masked_bias")

    def test_real_weights_are_not(self):
        assert not _is_expected_missing("lm_head.weight")
        assert not _is_expected_missing("transformer.wte.weight")

    def test_missing_and_mismatched_keys_are_both_reported(self):
        info = {
            "missing_keys": ["lm_head.weight", "transformer.h.0.attn.bias"],
            "mismatched_keys": [("transformer.wte.weight", (32, 1536), (35, 1536))],
        }
        assert unloaded_parameter_keys(info) == ["lm_head.weight", "transformer.wte.weight"]

    def test_a_clean_load_reports_nothing(self):
        assert unloaded_parameter_keys({"missing_keys": [], "mismatched_keys": []}) == []


class TestLoadedWeightsSurviveInitialization:
    """The bug that mattered most, tested on a model small enough to build here.

    transformers 5.x applies weight initialization to every module *after* the
    checkpoint has been loaded, and relies on each initializer to check the
    ``_is_hf_initialized`` flag and leave loaded weights alone. Upstream
    ProGen2 wrote through ``.data.normal_()``, which ignores the flag.
    Transformers has a fallback that skips already-initialized modules, but it
    applies only to remote code -- and vendoring this file makes it local. The
    first vendored copy therefore loaded all 221 tensors, reported no missing
    keys, and overwrote every one with random values.
    """

    def test_initialization_leaves_loaded_weights_alone(self):
        import torch

        model = TestTinyModelOffline.tiny()
        # Stand in for a checkpoint load: set recognizable values and mark
        # them the way the loader does.
        for parameter in model.parameters():
            with torch.no_grad():
                parameter.fill_(0.5)
            parameter._is_hf_initialized = True
        for buffer in model.buffers():
            buffer._is_hf_initialized = True

        model.initialize_weights()

        unchanged = all(
            bool((parameter == 0.5).all()) for parameter in model.parameters()
        )
        assert unchanged, "post-load initialization overwrote loaded weights"

    def test_unmarked_weights_are_still_initialized(self):
        """The flip side: initialization must not become a no-op.

        A fix that simply stopped initializing anything would pass the test
        above, and would leave genuinely new parameters -- the control-token
        embedding rows -- as whatever torch left in memory.
        """
        import torch

        model = TestTinyModelOffline.tiny()
        with torch.no_grad():
            model.lm_head.weight.zero_()
        # Unmark it, which is the state a parameter absent from the checkpoint
        # is in after loading.
        model.lm_head._is_hf_initialized = False
        model.lm_head.weight._is_hf_initialized = False
        model.lm_head.bias._is_hf_initialized = False

        model.initialize_weights()

        assert not bool((model.lm_head.weight == 0).all())


class TestVendoredSourceInvariants:
    """Declarations that a careless re-vendoring would quietly undo."""

    def test_no_lm_head_weight_in_the_missing_key_ignore_list(self):
        # ProGen2's head is untied and its weight is in the checkpoint, so
        # ignoring it when missing would hide a lost output layer.
        from thermocas.model.progen2 import ProGenForCausalLM

        ignored = list(ProGenForCausalLM._keys_to_ignore_on_load_missing or [])
        assert not any("lm_head" in pattern for pattern in ignored), ignored

    def test_gradient_checkpointing_is_supported(self):
        from thermocas.model.progen2 import ProGenForCausalLM

        assert ProGenForCausalLM.supports_gradient_checkpointing is True

    def test_no_generation_mixin(self):
        # The model carries the legacy tuple key/value cache, which
        # transformers 5.x's generate() does not produce. Sampling uses this
        # project's own loop; see the class comment.
        from transformers.generation import GenerationMixin

        from thermocas.model.progen2 import ProGenForCausalLM

        assert not issubclass(ProGenForCausalLM, GenerationMixin)

    def test_config_exposes_the_standard_attribute_names(self):
        from thermocas.model.progen2 import ProGenConfig

        config = ProGenConfig(n_layer=3, n_head=4, embed_dim=16, n_positions=64)
        assert config.num_hidden_layers == 3
        assert config.num_attention_heads == 4
        assert config.hidden_size == 16
        assert config.max_position_embeddings == 64


class TestTinyModelOffline:
    """Construction and a forward pass on a model small enough to build here."""

    @staticmethod
    def tiny():
        from thermocas.model.progen2 import ProGenConfig, ProGenForCausalLM

        return ProGenForCausalLM(
            ProGenConfig(
                vocab_size_emb=32,
                vocab_size_lm_head=32,
                n_positions=64,
                embed_dim=32,
                n_layer=2,
                n_head=4,
                rotary_dim=8,
            )
        )

    def test_nothing_is_left_on_the_meta_device(self):
        import torch  # noqa: F401

        from thermocas.model.compat import meta_device_parameters

        assert meta_device_parameters(self.tiny()) == []

    def test_forward_runs_and_shapes_match(self):
        import torch

        model = self.tiny().eval()
        with torch.no_grad():
            logits = model(torch.tensor([[3, 16, 9, 25]])).logits
        assert tuple(logits.shape) == (1, 4, 32)

    def test_resizing_updates_both_vocabularies(self):
        model = self.tiny()
        model.resize_token_embeddings(35)
        assert model.get_input_embeddings().num_embeddings == 35
        assert model.lm_head.out_features == 35
        # The config has to follow, or a reloaded checkpoint builds the old
        # size and the control-token rows vanish.
        assert model.config.vocab_size_emb == 35
        assert model.config.vocab_size_lm_head == 35

    def test_gradient_checkpointing_takes_effect(self):
        import torch

        model = self.tiny()
        model.gradient_checkpointing_enable()
        assert model.transformer.gradient_checkpointing is True
        model.train()
        model(torch.tensor([[3, 16, 9, 25]])).logits.sum().backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_incremental_decoding_with_the_tuple_cache(self):
        import torch

        model = self.tiny().eval()
        with torch.no_grad():
            first = model(torch.tensor([[3, 16, 9]]), use_cache=True)
            second = model(
                torch.tensor([[25]]), past_key_values=first.past_key_values, use_cache=True
            )
        assert len(second.past_key_values) == 2
        assert tuple(second.logits.shape) == (1, 1, 32)

    def test_head_mask_is_refused_rather_than_ignored(self):
        import torch

        model = self.tiny().eval()
        with pytest.raises(NotImplementedError):
            model(torch.tensor([[3, 16]]), head_mask=torch.ones(2))


class TestFusedAttention:
    """The fused path has to be arithmetically the old path, not a new model.

    Replacing the hand-rolled score matrix with
    ``scaled_dot_product_attention`` is a memory change: at a 1,027-token
    context the eager path holds roughly 340 MB of fp32 scores per layer,
    times 27 layers. A mask built with the wrong row offset, or a scale
    applied twice, would still train -- to a different model, quietly. So
    these compare the two implementations rather than checking the fused one
    runs.
    """

    @staticmethod
    def pair():
        """The same weights, one model fused and one eager."""
        import copy

        fused = TestTinyModelOffline.tiny().eval()
        eager = copy.deepcopy(fused).eval()
        for block in eager.transformer.h:
            block.attn.use_sdpa = False
        return fused, eager

    def test_the_fused_path_is_on_by_default(self):
        model = TestTinyModelOffline.tiny()
        assert all(block.attn.use_sdpa for block in model.transformer.h)

    def test_it_can_be_turned_off_from_the_config(self):
        from thermocas.model.progen2 import ProGenConfig, ProGenForCausalLM

        model = ProGenForCausalLM(
            ProGenConfig(
                vocab_size_emb=32,
                vocab_size_lm_head=32,
                n_positions=64,
                embed_dim=32,
                n_layer=2,
                n_head=4,
                rotary_dim=8,
                use_sdpa_attention=False,
            )
        )
        assert not any(block.attn.use_sdpa for block in model.transformer.h)

    def test_unpadded_logits_match_the_eager_path(self):
        import torch

        fused, eager = self.pair()
        tokens = torch.tensor([[3, 16, 9, 25, 7, 11]])
        with torch.no_grad():
            assert torch.allclose(
                fused(tokens).logits, eager(tokens).logits, atol=1e-5
            )

    def test_padded_logits_match_the_eager_path(self):
        """Padding is where a mask built in the wrong shape shows up."""
        import torch

        fused, eager = self.pair()
        tokens = torch.tensor([[3, 16, 9, 25, 0, 0], [7, 11, 4, 19, 22, 6]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            fused_logits = fused(tokens, attention_mask=mask).logits
            eager_logits = eager(tokens, attention_mask=mask).logits
        assert torch.allclose(fused_logits, eager_logits, atol=1e-5)

    def test_padding_does_not_change_the_real_positions(self):
        """The check that does not depend on the eager path being right.

        A shorter sequence scored on its own and scored inside a padded batch
        must give identical logits at its real positions. This fails if the
        causal triangle and the padding mask are combined in the wrong shape,
        or if the pad keys leak in.
        """
        import torch

        fused, _ = self.pair()
        alone = torch.tensor([[3, 16, 9, 25]])
        padded = torch.tensor([[3, 16, 9, 25, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
        with torch.no_grad():
            solo = fused(alone).logits
            batched = fused(padded, attention_mask=mask).logits[:, :4]
        assert torch.allclose(solo, batched, atol=1e-5)

    def test_incremental_decoding_matches_a_full_forward(self):
        """The offset case: query length 1 against a longer key.

        ``is_causal=True`` would be wrong here -- it aligns the triangle to
        the top left, so a single query would attend only to position 0.
        """
        import torch

        fused, _ = self.pair()
        tokens = torch.tensor([[3, 16, 9, 25]])
        with torch.no_grad():
            full = fused(tokens).logits[:, -1]
            prefix = fused(tokens[:, :-1], use_cache=True)
            stepped = fused(
                tokens[:, -1:], past_key_values=prefix.past_key_values
            ).logits[:, -1]
        assert torch.allclose(full, stepped, atol=1e-5)

    def test_left_padded_logits_match_the_eager_path(self):
        """The case the padding mask is still built for.

        Left padding breaks the argument that the causal triangle already
        excludes the pad keys, so the additive mask is kept -- and it has to
        agree with the eager implementation.
        """
        import torch

        fused, eager = self.pair()
        tokens = torch.tensor([[0, 0, 3, 16, 9, 25], [7, 11, 4, 19, 22, 6]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            assert torch.allclose(
                fused(tokens, attention_mask=mask).logits,
                eager(tokens, attention_mask=mask).logits,
                atol=1e-5,
            )

    def test_right_padding_is_recognized(self):
        import torch

        from thermocas.model.progen2.modeling_progen import _is_right_padded

        assert _is_right_padded(torch.tensor([[1, 1, 1, 0, 0]]))
        assert _is_right_padded(torch.tensor([[1, 1, 1, 1, 1]]))
        assert _is_right_padded(torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]))
        assert not _is_right_padded(torch.tensor([[0, 0, 1, 1]]))
        assert not _is_right_padded(torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]]))
        # Not a 2-D batch mask; the caller has already prepared it.
        assert not _is_right_padded(torch.ones(1, 1, 1, 4))

    def test_the_collate_function_produces_right_padding(self):
        """The assumption above, checked against the batch the trainer builds.

        If collation ever switched to left padding, the mask would start
        being built again -- correct, but slower and larger, and the comment
        in ProGenModel.forward would be describing something that no longer
        happens.
        """
        import torch

        from thermocas.model.dataset import collate
        from thermocas.model.progen2.modeling_progen import _is_right_padded
        from thermocas.model.tokenizer import ProteinTokenizer

        tokenizer = ProteinTokenizer(["meso", "thermo"])

        def item(sequence: str) -> dict:
            ids = torch.tensor(tokenizer.encode(sequence).ids)
            return {
                "input_ids": ids,
                "labels": ids.clone(),
                "ogt_c": torch.tensor(40.0),
                "temp_bin": "meso",
                "seq_id": sequence,
            }

        batch = collate(
            [item(s) for s in ("MKVL", "MKVLAAGG", "MK")],
            pad_id=tokenizer.pad_id,
        )
        assert _is_right_padded(batch["attention_mask"])

    def test_output_attentions_falls_back_rather_than_returning_nothing(self):
        """The fused kernel has no score matrix to hand back."""
        import torch

        fused, _ = self.pair()
        tokens = torch.tensor([[3, 16, 9, 25]])
        with torch.no_grad():
            attentions = fused(tokens, output_attentions=True).attentions
        assert len(attentions) == 2
        assert attentions[0] is not None
        assert tuple(attentions[0].shape) == (1, 4, 4, 4)

    def test_the_fused_path_is_differentiable_under_checkpointing(self):
        import torch

        fused, _ = self.pair()
        fused.gradient_checkpointing_enable()
        fused.train()
        fused(torch.tensor([[3, 16, 9, 25]])).logits.sum().backward()
        assert fused.transformer.wte.weight.grad is not None

    def test_the_fused_path_does_not_allocate_the_score_matrix(self):
        """The reason for the change, measured rather than asserted.

        Tracked with a tensor-size counter rather than a memory profiler so
        it works on CPU in CI. The eager path allocates at least one
        (batch, heads, L, L) fp32 tensor per layer; the fused path allocates
        the combined mask once per layer at (1, 1, L, L) and no per-head
        scores.
        """
        import torch

        length = 64
        tokens = torch.arange(1, length + 1)[None] % 32
        per_head = 4 * length * length  # heads x L x L

        class Watch(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.largest = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if isinstance(out, torch.Tensor):
                    self.largest = max(self.largest, out.numel())
                return out

        def largest_allocation(model):
            watch = Watch()
            with torch.no_grad(), watch:
                model(tokens)
            return watch.largest

        fused, eager = self.pair()
        assert largest_allocation(eager) >= per_head
        assert largest_allocation(fused) < per_head


@pytest.mark.skipif(not NETWORK, reason="set THERMOCAS_NETWORK_TESTS=1 to hit the HF Hub")
class TestAgainstRealCheckpoint:
    """The checks that actually caught the bugs."""

    @pytest.fixture(scope="class")
    def model(self):
        from thermocas.model.compat import load_progen2

        return load_progen2(CHECKPOINT).eval()

    def test_every_weight_matches_the_checkpoint_file(self, model):
        from thermocas.model.compat import weights_differing_from_checkpoint

        assert weights_differing_from_checkpoint(model, CHECKPOINT) == []

    def test_parameter_count(self, model):
        assert 760e6 < sum(p.numel() for p in model.parameters()) < 770e6

    def test_context_is_long_enough_for_cas9(self, model):
        # progen2-small and progen2-large both declare 1024, which cannot hold
        # SpCas9's 1,368 residues. This is the invariant the config depends on.
        assert model.config.n_positions >= 1703

    def test_a_real_protein_scores_far_better_than_a_shuffle(self, model):
        """The end-to-end check that random weights cannot pass.

        A randomly initialized model scores both sequences at roughly the
        uniform-vocabulary loss and separates them by ~0.04 nats. Requiring a
        wide margin is therefore a test of whether the pretrained weights are
        present and being used, not a test of ProGen2's quality.
        """
        import torch
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(CHECKPOINT)

        def nll_per_token(sequence: str) -> float:
            ids = [tokenizer.token_to_id("1")] + tokenizer.encode(
                sequence, add_special_tokens=False
            ).ids
            tokens = torch.tensor([ids])
            with torch.no_grad():
                logits = model(tokens).logits[0].float()
            log_probs = torch.log_softmax(logits[:-1], dim=-1)
            return float(-log_probs.gather(-1, tokens[0, 1:, None]).mean())

        shuffled = "".join(
            random.Random(0).sample(SPCAS9_N_TERMINUS, len(SPCAS9_N_TERMINUS))
        )
        real_nll = nll_per_token(SPCAS9_N_TERMINUS)
        shuffled_nll = nll_per_token(shuffled)

        assert real_nll < 1.0, f"real sequence scored {real_nll:.3f} nats/token"
        assert shuffled_nll - real_nll > 1.0
        # Below the uniform-vocabulary loss, which a random model sits at.
        assert real_nll < math.log(32)
