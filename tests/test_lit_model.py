"""Loss arithmetic, the context assertion, and the OGT head.

These run without a GPU and without loading a 765M-parameter checkpoint,
because the arithmetic is where a silent error would survive a smoke test.
"""

from __future__ import annotations

import math

import pytest
import torch

from thermocas.config import Config
from thermocas.model.dataset import IGNORE_INDEX
from thermocas.model.lit_model import (
    ConditionedProteinLM,
    ContextTooShortError,
    LoraSettings,
    OgtHead,
    assert_context_fits,
)


class TestContextAssertion:
    def test_progen2_small_is_rejected_for_cas9(self):
        """1024 positions cannot hold SpCas9's 1,368 residues."""
        with pytest.raises(ContextTooShortError, match="1024"):
            assert_context_fits(1024, 1703, "hugohrban/progen2-small")

    def test_error_names_a_working_alternative(self):
        with pytest.raises(ContextTooShortError, match="progen2-base"):
            assert_context_fits(1024, 1703, "hugohrban/progen2-small")

    def test_progen2_base_is_accepted(self):
        assert_context_fits(2048, 1703, "hugohrban/progen2-base")

    def test_exact_fit_is_accepted(self):
        assert_context_fits(1703, 1703, "whatever")

    def test_shipped_config_requirement_is_satisfiable(self):
        """The configured checkpoint must actually clear its own requirement."""
        model = Config.load("model")
        required = model["conditioned_lm/required_context"]
        # progen2-base declares 2048; verified against the published config.
        assert_context_fits(2048, required, model["conditioned_lm/checkpoint"])


class TestCausalLmLoss:
    def test_sums_nll_and_counts_unmasked_tokens(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 7)
        labels = torch.tensor(
            [
                [IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5],
                [IGNORE_INDEX, IGNORE_INDEX, 1, 2, IGNORE_INDEX],
            ]
        )
        total_nll, n_tokens = ConditionedProteinLM.causal_lm_loss(logits, labels)
        # Shifting drops the first label column; 3 + 2 targets remain.
        assert n_tokens.item() == 5
        assert total_nll.item() > 0

    def test_labels_are_shifted_so_position_t_predicts_t_plus_one(self):
        """A one-position-off shift trains the model to copy its input."""
        vocab = 6
        # Build logits that are confident about the NEXT token at each step.
        labels = torch.tensor([[IGNORE_INDEX, 2, 3, 4]])
        logits = torch.full((1, 4, vocab), -20.0)
        logits[0, 0, 2] = 20.0  # position 0 predicts label at position 1
        logits[0, 1, 3] = 20.0
        logits[0, 2, 4] = 20.0
        total_nll, n_tokens = ConditionedProteinLM.causal_lm_loss(logits, labels)
        assert n_tokens.item() == 3
        # Near-zero loss only if the shift is correct.
        assert total_nll.item() < 1e-3

    def test_masked_positions_do_not_contribute(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 4, 5)
        all_masked = torch.full((1, 4), IGNORE_INDEX)
        total_nll, n_tokens = ConditionedProteinLM.causal_lm_loss(logits, all_masked)
        assert n_tokens.item() == 0
        assert total_nll.item() == 0.0

    def test_mean_loss_guards_an_all_masked_batch(self):
        """Returning NaN here would poison the optimizer from one bad batch."""
        total = torch.tensor(0.0)
        mean = ConditionedProteinLM.mean_loss(total, torch.tensor(0))
        assert mean.item() == 0.0
        assert not math.isnan(mean.item())

    def test_mean_loss_is_per_token(self):
        mean = ConditionedProteinLM.mean_loss(torch.tensor(40.0), torch.tensor(10))
        assert mean.item() == pytest.approx(4.0)

    def test_perplexity_is_consistent_with_gate2(self):
        """The module and the gate must agree, or Gate 2 compares two scales."""
        from thermocas.model.gate2 import perplexity

        assert ConditionedProteinLM.perplexity_from(40.0, 10) == pytest.approx(
            perplexity(40.0, 10)
        )


class TestOgtHead:
    def test_pools_only_real_tokens(self):
        """Padding must not drag the pooled representation toward zero."""
        head = OgtHead(hidden_size=4, hidden=8, dropout=0.0, mean_c=45.0, std_c=20.0)
        hidden = torch.ones(1, 6, 4)
        full = torch.ones(1, 6, dtype=torch.long)
        padded = torch.tensor([[1, 1, 1, 0, 0, 0]])
        # Rows are identical, so masking changes nothing if pooling is correct.
        hidden_with_junk = hidden.clone()
        hidden_with_junk[0, 3:] = 999.0
        head.eval()
        with torch.no_grad():
            a = head(hidden, full)
            b = head(hidden_with_junk, padded)
        assert torch.allclose(a, b, atol=1e-5)

    def test_output_is_in_celsius(self):
        head = OgtHead(hidden_size=4, hidden=8, dropout=0.0, mean_c=45.0, std_c=20.0)
        head.eval()
        with torch.no_grad():
            value = head(torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.long))
        # A zero-ish normalized prediction lands near the mean, not near 0 C.
        assert 0.0 < value.item() < 120.0

    def test_normalize_is_the_inverse_scale(self):
        head = OgtHead(hidden_size=2, hidden=4, dropout=0.0, mean_c=45.0, std_c=20.0)
        celsius = torch.tensor([45.0, 65.0, 25.0])
        assert torch.allclose(head.normalize(celsius), torch.tensor([0.0, 1.0, -1.0]))

    def test_zero_std_rejected(self):
        with pytest.raises(ValueError, match="std_c"):
            OgtHead(hidden_size=2, hidden=4, dropout=0.0, mean_c=45.0, std_c=0.0)

    def test_loss_is_computed_in_normalized_units(self):
        """In Celsius, a 20-degree error squared is 400, which would swamp a
        per-token cross-entropy of order 1."""
        head = OgtHead(hidden_size=2, hidden=4, dropout=0.0, mean_c=45.0, std_c=20.0)
        predicted = torch.tensor([65.0])
        target = torch.tensor([45.0])
        raw = torch.nn.functional.mse_loss(predicted, target)
        normalized = torch.nn.functional.mse_loss(
            head.normalize(predicted), head.normalize(target)
        )
        assert raw.item() == pytest.approx(400.0)
        assert normalized.item() == pytest.approx(1.0)


class TestLoraSettings:
    def test_loads_from_shipped_config(self):
        model = Config.load("model")
        lora = LoraSettings.from_config(model.section("conditioned_lm"))
        assert lora.enabled is True
        assert lora.r > 0
        assert lora.target_modules

    def test_ablation_lora_matches_the_main_model(self):
        """Otherwise the ablation varies two things at once."""
        model = Config.load("model")
        main = LoraSettings.from_config(model.section("conditioned_lm"))
        ablation = LoraSettings.from_config(model.section("unconditioned_lm"))
        assert (main.r, main.alpha, main.target_modules) == (
            ablation.r,
            ablation.alpha,
            ablation.target_modules,
        )
