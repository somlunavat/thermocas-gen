# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified forward-pass implementation based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py

# ---------------------------------------------------------------------------
# VENDORED into thermocas by this project, with modifications. Every change is
# marked with a "THERMOCAS:" comment. See thermocas/model/progen2/__init__.py
# for provenance (source repo, revision, date) and the reasons.
# ---------------------------------------------------------------------------

from typing import Tuple

import math

import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.initialization import normal_, ones_, zeros_
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_progen import ProGenConfig


logger = logging.get_logger(__name__)


def _is_right_padded(attention_mask) -> bool:
    """True if every row is ones followed by zeros.

    THERMOCAS: added. See the call site in ProGenModel.forward.
    """
    if attention_mask.dim() != 2:
        return False
    return bool((attention_mask[:, :-1] >= attention_mask[:, 1:]).all())


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq)
        .to(x.device)
        .float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x: torch.Tensor):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)

def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(
            2, 3
        ),
        sincos,
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class ProGenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.max_positions = config.n_positions
        # THERMOCAS: upstream registered a (1, 1, n_positions, n_positions)
        # causal-mask buffer and a scalar masked-bias buffer here, both
        # non-persistent. See the module docstring: those are built on the
        # meta device under transformers 5.x and never materialized. The mask
        # is now built per forward from the query/key lengths instead.
        self.masked_bias = -1e9     # approx. -inf

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # THERMOCAS: see _attn_sdpa. Read once here so turning it off is a
        # config change rather than an edit to this file.
        self.use_sdpa = bool(getattr(config, "use_sdpa_attention", True))

        self.embed_dim = config.embed_dim
        self.num_attention_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and `num_attention_heads`: {self.num_attention_heads})."
            )
        # THERMOCAS: a float, not a tensor -- a tensor built in __init__ lands
        # on the meta device under transformers 5.x. Dividing fp32 attention
        # weights by a Python float is numerically identical.
        self.scale_attn = float(math.sqrt(self.head_dim))
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def _split_heads(self, x: torch.Tensor, n_head, dim_head) -> torch.Tensor:
        x = x.reshape(x.shape[:-2] + (-1,))                             # (B, T, 8 * E // 8)
        x = x.reshape(x.shape[:-1] + (n_head, dim_head))             # (B, T, n_heads, dim_head)
        return x

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size) -> torch.Tensor:
        """
        Merges attn_head_size dim and num_attn_heads dim into n_positions
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _causal_mask(self, query, key):
        # THERMOCAS: causal mask built here rather than sliced from a buffer.
        # Shape (1, 1, query_length, key_length), True where attention is
        # allowed. The row offset matters: during incremental decoding the
        # query is shorter than the key, and the queries are the *last*
        # query_length positions, so the mask is the bottom slice of the
        # lower-triangular matrix rather than the top-left corner.
        query_length, key_length = query.size(-2), key.size(-2)
        return torch.ones(
            (key_length, key_length), dtype=torch.bool, device=query.device
        ).tril()[None, None, key_length - query_length : key_length, :key_length]

    def _attn_sdpa(self, query, key, value, attention_mask=None):
        """Fused attention via ``scaled_dot_product_attention``.

        This exists for memory, not speed. The eager path below materializes
        the (batch, heads, L, L) score matrix in fp32, several copies of it
        per layer: at L=1027 with 16 heads that is roughly 340 MB per layer
        and 27 layers of it, which is what made a 1.5 GB estimate measure as
        19 GB. The fused kernels never hold the full matrix.

        The dtype differs from the eager path deliberately. Eager upcasts the
        scores to fp32 to avoid overflow; the fused kernels keep the input
        dtype and accumulate in fp32 internally, which is what every current
        transformers attention implementation does. Under ``32-true`` the two
        paths are doing the same arithmetic; under bf16 autocast this one is
        less precise and much smaller. ``tests/test_progen2.py`` checks the
        two agree.
        """
        # A boolean mask forces the mask-aware kernel. Where there is no
        # padding and the query covers the whole key, is_causal lets the
        # flash kernel build the triangle itself and allocate nothing.
        if attention_mask is None and query.size(-2) == key.size(-2):
            attn_mask, is_causal = None, True
        else:
            # The additive padding mask arrives as (batch, 1, 1, key_length)
            # of 0.0 and -10000.0 (see ProGenModel.forward). Combining it
            # with the causal triangle additively keeps one (batch, 1, L, L)
            # fp32 mask -- one copy, not one per head.
            causal_mask = self._causal_mask(query, key)
            # fp16 cannot hold -1e9; it would become -inf, and an all-masked
            # row would then produce NaN rather than a uniform distribution.
            floor = max(self.masked_bias, torch.finfo(query.dtype).min / 2)
            attn_mask = torch.zeros(
                causal_mask.shape, dtype=query.dtype, device=query.device
            ).masked_fill(~causal_mask, floor)
            if attention_mask is not None:
                attn_mask = attn_mask + attention_mask.to(query.dtype)
            is_causal = False

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            # ProGen2 divides by sqrt(head_dim) after the matmul; this is the
            # same constant, passed rather than defaulted so a change to
            # scale_attn cannot silently diverge between the two paths.
            scale=1.0 / self.scale_attn,
        )
        return attn_output, None

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        causal_mask = self._causal_mask(query, key)

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = query @ key.transpose(-1, -2)                 # (B, n_heads, T, T)

        attn_weights = attn_weights / self.scale_attn

        # attend only to previous positions
        attn_weights = torch.where(
            causal_mask,
            attn_weights,
            torch.tensor(
                self.masked_bias, dtype=attn_weights.dtype, device=attn_weights.device
            ),
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = attn_weights @ value                # (B, n_heads, T, dim_head)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        qkv = self.qkv_proj(hidden_states)                                         # (B, T, 3 * E)

        mp_num = 8
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))                     # (B, T, 8, 3 * E // 8)

        query, value, key = torch.split(qkv_split, self.embed_dim // mp_num, dim=-1) # 3 * (B, T, 8, E // 8)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim)  # (B, T, n_heads, dim_head)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim)      # (B, T, n_heads, dim_head)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim)  # (B, T, n_heads, dim_head)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: softmax((Q @ K.T) / sqrt(dim_head)) @ V
        # THERMOCAS: the fused kernel cannot return the score matrix and
        # cannot apply a per-head mask, so those two requests fall back to
        # the eager path rather than being silently ignored.
        if self.use_sdpa and not output_attentions and head_mask is None:
            attn_output, attn_weights = self._attn_sdpa(
                query, key, value, attention_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

        attn_output = self._merge_heads(                               # (B, T, E)  
            attn_output, self.num_attention_heads, self.head_dim
        )

        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class ProGenMLP(nn.Module):
    def __init__(
        self, intermediate_size, config
    ):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.embed_dim

        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ProGenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.embed_dim
        self.ln_1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_epsilon)
        self.attn = ProGenAttention(config)
        self.mlp = ProGenMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)            # (B, T, E)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class ProGenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProGenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    # THERMOCAS: upstream left this False, so transformers'
    # gradient_checkpointing_enable() refused to turn checkpointing on and the
    # forward pass below only honoured config.gradient_checkpointing. Stage A
    # relies on checkpointing to fit, and silently not getting it is the kind
    # of failure that shows up as an OOM three hours in.
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights.

        THERMOCAS: the initializers come from ``transformers.initialization``
        rather than being in-place ``.data`` writes. This is not cosmetic. In
        transformers 5.x, ``_initialize_weights`` is applied to every module
        after the checkpoint has been loaded, and it relies on the
        initializers to check each parameter's ``_is_hf_initialized`` flag and
        leave loaded weights alone. Upstream's in-place ``.data.normal_()``
        ignores that flag.

        Transformers has a fallback that skips already-initialized modules,
        but it is applied only when ``is_remote_code()`` is true -- and
        vendoring this file makes it false. The first vendored version of this
        file therefore loaded all 221 tensors, reported no missing keys, and
        then overwrote every one of them with random values. The model
        produced garbage and nothing warned. ``load_progen2`` now also
        compares the loaded tensors against the checkpoint file, because
        "the weights arrived" and "the weights are still there" turned out to
        be different claims.
        """
        if isinstance(module, (nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None and not getattr(
                module.weight, "_is_hf_initialized", False
            ):
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)


class ProGenModel(ProGenPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size_emb = config.vocab_size_emb
        self.embed_dim = config.embed_dim
        self.wte = nn.Embedding(config.vocab_size_emb, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([ProGenBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(
            config.rotary_dim, config.n_positions // config.n_head
        )
        # THERMOCAS: transformers' gradient_checkpointing_enable() requires a
        # module to carry this attribute, and refuses outright without one.
        self.gradient_checkpointing = False
        self.post_init()

    # THERMOCAS: upstream defined neither accessor, so transformers warned
    # that "gradients cannot flow back to the token embeddings when using
    # adapters or gradient checkpointing" -- and, worse,
    # resize_token_embeddings() had nothing to resize. This project appends
    # growth-temperature control tokens above the checkpoint's vocabulary, so
    # resizing has to work.
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
        self.vocab_size_emb = new_embeddings.num_embeddings
        self.config.vocab_size_emb = new_embeddings.num_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)

        # THERMOCAS: under causal attention, right-aligned padding needs no
        # mask. A query at a real position attends to keys at positions <=
        # its own, and every pad key sits at a position after every real one,
        # so the causal triangle already excludes them. The pad positions'
        # own outputs do differ, but a pad position can only influence later
        # positions, which are also pad, and both the language-model loss and
        # the OGT head's pooling exclude them (see
        # test_padding_does_not_change_the_real_positions).
        #
        # Dropping it is worth the check: an additive mask makes attention
        # take the mask-aware kernel and allocate an (L, L) bias, where
        # is_causal lets the fused kernel build the triangle itself and
        # allocate nothing. One comparison per forward pass, not per layer.
        if attention_mask is not None and _is_right_padded(attention_mask):
            attention_mask = None

        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        # THERMOCAS: transformers 5.x removed PreTrainedModel.get_head_mask.
        # Nothing in this project passes a head mask.
        if head_mask is not None:
            raise NotImplementedError("head_mask is not supported")
        head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # THERMOCAS: also honour the attribute set by
            # PreTrainedModel.gradient_checkpointing_enable(), which is how
            # callers actually request this.
            checkpointing = (
                getattr(self, "gradient_checkpointing", False)
                or getattr(self.config, "gradient_checkpointing", False)
            )
            if checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. "
                        "Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    # THERMOCAS: explicit, because the default flipped between
                    # torch versions and the reentrant path does not propagate
                    # gradients to inputs that LoRA adapters need.
                    use_reentrant=False,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ProGenForCausalLM(ProGenPreTrainedModel):
    # THERMOCAS: deliberately NOT a GenerationMixin. ProGen2 carries its
    # key/value cache as the legacy tuple-of-tuples, which transformers 5.x's
    # generate() no longer produces -- it constructs a Cache object and hands
    # it to forward, which would fail inside the attention. Inheriting the
    # mixin would therefore advertise a .generate() that breaks deep in the
    # stack.
    #
    # This costs little, because sampling here needs a custom loop regardless:
    # classifier-free-guidance-style strengthening requires two forward passes
    # per step, one with the growth-temperature token and one with the
    # unconditional token, with the logits combined. That lives in
    # thermocas/eval/ and drives this forward pass directly, using the tuple
    # cache the model already supports.
    _keys_to_ignore_on_load_missing = [
        # THERMOCAS: upstream also ignored a missing r"lm_head\.weight" here.
        # ProGen2's head is an untied nn.Linear whose weight IS in the
        # checkpoint, so a missing one means the pretrained output layer was
        # silently replaced by a random one. The buffer entries below are
        # vestigial now that the buffers are gone, and are kept only so an
        # older checkpoint listing them still loads cleanly.
        r"h\.\d+\.attn\.masked_bias",
        r"h\.\d+\.attn\.bias",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ProGenModel(config)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size_lm_head)
        self.post_init()

    # THERMOCAS: see the note on ProGenModel.get_input_embeddings. ProGen2's
    # head is untied, so the output embeddings are resized independently and
    # config.vocab_size_lm_head has to follow.
    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.config.vocab_size_lm_head = new_embeddings.out_features

    # THERMOCAS: rewritten. Upstream took the transformers 4.x `past=` keyword,
    # which nothing passes any more. The method is kept because PEFT requires
    # it on a CAUSAL_LM base model and raises AttributeError without it, not
    # because this class supports .generate() -- it does not; see the class
    # comment. The signature follows the current convention so that a caller
    # driving the forward pass with the tuple cache gets the right inputs.
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            # Only the newest token needs to be fed through once the cache
            # holds the prefix.
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # THERMOCAS: upstream's _reorder_cache (beam-search bookkeeping for
    # GenerationMixin) was removed for the same reason.
