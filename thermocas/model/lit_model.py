"""LightningModule: LoRA-adapted causal protein LM with a growth-temperature head.

Wraps a pretrained autoregressive protein LM. Two things are added to it:

* the growth-temperature control token, which is a new vocabulary entry and so
  requires the embedding matrix to be resized and the new rows to stay
  trainable even when the rest of the model is frozen by LoRA;
* an auxiliary head regressing OGT in degrees Celsius, which carries the
  resolution lost when the hyperthermophilic bin was merged into thermophilic.

The context assertion at construction is deliberate. The prior codebase's
central bug was a context too short for Cas9, and the checkpoint originally
named in this project's own config (ProGen2-small, 1024 positions) could not
hold SpCas9's 1,368 residues either. Discovering that at step 0 rather than
after a training run is worth a hard failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from thermocas.model.dataset import IGNORE_INDEX
from thermocas.model.tokenizer import PretrainedTokenizerAdapter, ProteinTokenizer


class ContextTooShortError(RuntimeError):
    """The checkpoint cannot represent a full-length sequence."""


@dataclass(frozen=True)
class LoraSettings:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg, path: str = "lora") -> LoraSettings:
        section = cfg.section(path)
        return cls(
            enabled=bool(section["enabled"]),
            r=int(section["r"]),
            alpha=int(section["alpha"]),
            dropout=float(section["dropout"]),
            target_modules=tuple(str(m) for m in section["target_modules"]),
        )


class OgtHead(nn.Module):
    """Regresses growth temperature in degrees Celsius from pooled states.

    Predicting in normalized units and denormalizing keeps the loss scale
    independent of the Celsius range, which otherwise dominates the language
    modeling loss: an error of 20 degrees squared is 400, against a per-token
    cross-entropy of order 1.
    """

    def __init__(self, hidden_size: int, hidden: int, dropout: float, mean_c: float, std_c: float):
        super().__init__()
        if std_c <= 0:
            raise ValueError(f"std_c must be positive, got {std_c}")
        self.mean_c = float(mean_c)
        self.std_c = float(std_c)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool over real tokens only, then regress."""
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        normalized = self.net(pooled).squeeze(-1)
        return normalized * self.std_c + self.mean_c

    def normalize(self, celsius: torch.Tensor) -> torch.Tensor:
        return (celsius - self.mean_c) / self.std_c


def assert_context_fits(declared_context: int, required_context: int, checkpoint: str) -> None:
    """Fail loudly when the checkpoint cannot hold a full-length sequence.

    ProGen2-small and ProGen2-large both declare 1024 positions, which cannot
    hold SpCas9 at 1,368 residues; ProGen2-base declares 2048 and can. Context
    does not track parameter count in that family, so this is easy to get
    wrong by picking the largest model that fits the GPU.
    """
    if declared_context < required_context:
        raise ContextTooShortError(
            f"{checkpoint} declares a context of {declared_context} tokens but this "
            f"project requires {required_context} (the longest admissible sequence "
            f"plus BOS, the growth-temperature token and EOS). Training would "
            f"silently drop or truncate full-length sequences, which is the bug "
            f"this project exists downstream of. Use a longer-context checkpoint: "
            f"hugohrban/progen2-base declares 2048."
        )


def load_stage_checkpoint(module, state_dict: dict) -> tuple[int, int]:
    """Load a stage A state dict into a stage B module, loudly.

    Returns ``(loaded, total)`` parameter-tensor counts.

    ``load_state_dict(..., strict=False)`` was used here, which is a silent
    no-op when the key names do not line up -- and they are easy to misalign,
    because PEFT renames every module it wraps. Stage B would then have
    started from the raw pretrained checkpoint while reporting that it had
    initialized from stage A, and the whole two-stage design would have been
    untested. Any missing or unexpected key is therefore an error, not a
    warning.
    """
    result = module.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            "stage A checkpoint does not match this module: "
            f"{len(missing)} missing key(s) {missing[:5]}, "
            f"{len(unexpected)} unexpected key(s) {unexpected[:5]}. "
            "Loading it anyway would start stage B from the pretrained "
            "checkpoint while reporting otherwise."
        )
    total = len(module.state_dict())
    return total - len(missing), total


def check_validation_cadence(
    val_interval_batches: int,
    accumulate_grad_batches: int,
    checkpoint_every_n_steps: int,
    monitor: str,
) -> int:
    """Validate that the checkpoint monitor will have a fresh metric.

    Returns the validation cadence in optimizer steps, or raises.

    This exists because of a silent failure rather than a loud one. A stage A
    epoch is roughly 18,400 optimizer steps and training is bounded by
    ``max_steps``, so with Lightning's default of validating once per epoch,
    validation can run nearly never. ``ModelCheckpoint`` then monitors a
    metric that was never logged -- and it does not raise. Verified on a tiny
    model: ``save_top_k=3, monitor="val_loss"`` with no validation produced a
    single unmonitored ``last.ckpt`` and a best score of ``None``. A 20,000
    step run would have ended with no best checkpoint and no error anywhere.

    ``val_interval_batches`` counts training batches, which is Lightning's
    unit for ``val_check_interval`` and is not the same as optimizer steps
    whenever gradients are accumulated.
    """
    if accumulate_grad_batches <= 0:
        raise ValueError(
            f"accumulate_grad_batches must be positive, got {accumulate_grad_batches}"
        )
    if val_interval_batches <= 0:
        raise ValueError(
            f"validation check interval must be positive, got {val_interval_batches}"
        )
    if val_interval_batches % accumulate_grad_batches != 0:
        raise ValueError(
            f"validation/check_interval_batches ({val_interval_batches}) must be a "
            f"multiple of batch/accumulate_grad_batches ({accumulate_grad_batches}); "
            "it counts training batches, not optimizer steps"
        )
    cadence = val_interval_batches // accumulate_grad_batches
    if cadence != checkpoint_every_n_steps:
        raise ValueError(
            f"validation runs every {cadence} optimizer steps but "
            f"checkpoint/every_n_steps is {checkpoint_every_n_steps}. "
            f"ModelCheckpoint(monitor={monitor!r}) would fire with a stale or absent "
            "metric and silently save nothing."
        )
    return cadence


def extend_embeddings_for_control_tokens(
    model, tokenizer: ProteinTokenizer | PretrainedTokenizerAdapter, base_vocab_size: int
) -> int:
    """Resize the token embeddings to fit the control tokens.

    Returns the number of rows added. The new rows must remain trainable even
    under LoRA: a frozen, randomly initialized control-token embedding cannot
    learn to mean anything, and Gate 2 would then fail for a reason that has
    nothing to do with whether temperature conditioning is learnable.
    """
    target = len(tokenizer)
    if target <= base_vocab_size:
        return 0
    model.resize_token_embeddings(target)
    return target - base_vocab_size


class ConditionedProteinLM:
    """Factory and loss helpers, kept importable without Lightning installed.

    The Lightning wrapper lives in :func:`build_lightning_module` so that the
    loss arithmetic below can be unit-tested on a machine with no GPU and no
    Lightning, which is where most of this project's development happens.
    """

    @staticmethod
    def causal_lm_loss(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shifted cross-entropy, returning summed NLL and the token count.

        Summing rather than averaging is what lets Gate 2 compute a
        token-level perplexity across a whole validation set; averaging per
        batch would weight a batch of short sequences the same as a batch of
        long ones.
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
        total_nll = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        n_tokens = (shift_labels != IGNORE_INDEX).sum()
        return total_nll, n_tokens

    @staticmethod
    def mean_loss(total_nll: torch.Tensor, n_tokens: torch.Tensor) -> torch.Tensor:
        """Per-token mean, guarded against an all-masked batch.

        An all-masked batch is possible when every sequence in it was
        truncated to nothing; returning 0 rather than NaN keeps one bad batch
        from poisoning the optimizer state.
        """
        if n_tokens.item() == 0:
            return total_nll * 0.0
        return total_nll / n_tokens

    @staticmethod
    def perplexity_from(total_nll: float, n_tokens: int) -> float:
        if n_tokens <= 0:
            raise ValueError("cannot compute perplexity over zero tokens")
        return math.exp(total_nll / n_tokens)


def build_lightning_module(
    model_cfg,
    train_cfg,
    tokenizer: ProteinTokenizer | PretrainedTokenizerAdapter,
    ogt_mean_c: float = 45.0,
    ogt_std_c: float = 20.0,
    model_name: str = "conditioned_lm",
):
    """Construct the LightningModule for a configured model.

    Imported lazily so that the rest of the package, and most of the test
    suite, does not require Lightning, transformers or peft to be installed.
    """
    import lightning as L  # noqa: N812

    from thermocas.model.compat import load_progen2
    from thermocas.model.progen2 import ProGenConfig

    section = model_cfg.section(model_name)
    checkpoint = str(section["checkpoint"])
    required_context = int(section["required_context"])

    hf_config = ProGenConfig.from_pretrained(checkpoint)
    declared = int(hf_config.n_positions)
    assert_context_fits(declared, required_context, checkpoint)

    if not isinstance(tokenizer, PretrainedTokenizerAdapter):
        raise TypeError(
            "fine-tuning a pretrained checkpoint requires PretrainedTokenizerAdapter, "
            f"got {type(tokenizer).__name__}. The residue vocabulary belongs to the "
            "checkpoint: feeding this project's own ids into ProGen2 points every "
            "residue at the wrong embedding row, and LoRA freezes the base weights so "
            "the model cannot recover."
        )
    if tokenizer.base_vocab_size != int(hf_config.vocab_size_emb):
        raise ValueError(
            f"tokenizer was built against a base vocabulary of "
            f"{tokenizer.base_vocab_size} but {checkpoint} has "
            f"{hf_config.vocab_size_emb} embedding rows; the control-token ids would "
            "land on pretrained rows"
        )

    class LitConditionedProteinLM(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.save_hyperparameters(
                {
                    "checkpoint": checkpoint,
                    "required_context": required_context,
                    "declared_context": int(declared),
                    "model_name": model_name,
                    # Pinned so a checkpoint cannot be reloaded against a
                    # reordered vocabulary, which would silently reinterpret
                    # every control token.
                    "tokenizer_signature": tokenizer.signature,
                }
            )

            # load_progen2 verifies that every pretrained weight actually
            # arrived and was not re-initialized; see thermocas/model/compat.py.
            self.backbone = load_progen2(checkpoint)
            base_vocab = self.backbone.get_input_embeddings().weight.shape[0]
            self.added_rows = extend_embeddings_for_control_tokens(
                self.backbone, tokenizer, base_vocab
            )

            lora = LoraSettings.from_config(section)
            if lora.enabled:
                from peft import LoraConfig, get_peft_model

                self.backbone = get_peft_model(
                    self.backbone,
                    LoraConfig(
                        r=lora.r,
                        lora_alpha=lora.alpha,
                        lora_dropout=lora.dropout,
                        target_modules=list(lora.target_modules),
                        task_type="CAUSAL_LM",
                        # The resized embedding rows for the control tokens
                        # must stay trainable; a frozen random embedding
                        # cannot learn to mean anything.
                        modules_to_save=["wte", "lm_head"] if self.added_rows else None,
                    ),
                )

            if bool(section["gradient_checkpointing"]):
                # Stage A's memory budget assumes this is on. The vendored
                # model had to be changed for it to take effect at all.
                self.backbone.gradient_checkpointing_enable()

            hidden_size = int(hf_config.embed_dim)
            self.ogt_head = None
            if bool(section["ogt_head/enabled"]):
                self.ogt_head = OgtHead(
                    hidden_size=int(hidden_size),
                    hidden=int(section["ogt_head/hidden"]),
                    dropout=float(section["ogt_head/dropout"]),
                    mean_c=ogt_mean_c,
                    std_c=ogt_std_c,
                )
            self.ogt_weight = float(train_cfg["loss/ogt_head_weight"])

        def forward(self, input_ids, attention_mask):
            return self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.ogt_head is not None,
            )

        def _step(self, batch, stage: str):
            outputs = self(batch["input_ids"], batch["attention_mask"])
            total_nll, n_tokens = ConditionedProteinLM.causal_lm_loss(
                outputs.logits, batch["labels"]
            )
            lm_loss = ConditionedProteinLM.mean_loss(total_nll, n_tokens)
            loss = lm_loss

            if self.ogt_head is not None and self.ogt_weight > 0:
                hidden = outputs.hidden_states[-1]
                predicted = self.ogt_head(hidden, batch["attention_mask"])
                target = batch["ogt_c"]
                finite = torch.isfinite(target)
                if finite.any():
                    ogt_loss = nn.functional.mse_loss(
                        self.ogt_head.normalize(predicted[finite]),
                        self.ogt_head.normalize(target[finite]),
                    )
                    loss = loss + self.ogt_weight * ogt_loss
                    self.log(f"{stage}_ogt_mse", ogt_loss, prog_bar=False)

            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_lm_loss", lm_loss, prog_bar=False)
            self.log(f"{stage}_tokens", n_tokens.float(), prog_bar=False, reduce_fx="sum")
            return loss

        def training_step(self, batch, batch_idx):
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx):
            return self._step(batch, "val")

        def configure_optimizers(self):
            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable,
                lr=float(train_cfg["optim/lr"]),
                weight_decay=float(train_cfg["optim/weight_decay"]),
                betas=tuple(float(b) for b in train_cfg["optim/betas"]),
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(train_cfg["optim/lr"]),
                total_steps=int(train_cfg["optim/max_steps"]),
                pct_start=min(
                    0.3,
                    int(train_cfg["optim/warmup_steps"]) / max(1, int(train_cfg["optim/max_steps"])),
                ),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return LitConditionedProteinLM()
