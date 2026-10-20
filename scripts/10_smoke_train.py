#!/usr/bin/env python3
"""Smoke-test the training path and estimate its memory cost.

Two modes, because they answer different questions.

``--tiny`` builds a small randomly-initialized model and runs a real
forward/backward at full Cas9 sequence length. It answers "does the code path
work end to end" and runs on this laptop in seconds, with no download.

``--real`` loads the configured checkpoint (765M parameters) and does the same.
It answers "does this fit the GPU". A 16 GB machine will not train it
comfortably at full length without LoRA, bf16 and gradient checkpointing, so
the estimate is printed either way and should be checked against the cloud GPU
before a real schedule is planned.

Usage:
    python scripts/10_smoke_train.py --tiny
    python scripts/10_smoke_train.py --estimate-only
    python scripts/10_smoke_train.py --real          # downloads ~1.5 GB
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from thermocas.config import Config, GrowthTemperatureBinner  # noqa: E402
from thermocas.model.dataset import (  # noqa: E402
    ConditionedProteinDataset,
    collate,
)
from thermocas.model.device import resolve_device  # noqa: E402
from thermocas.model.lit_model import ConditionedProteinLM, OgtHead  # noqa: E402
from thermocas.model.tokenizer import ProteinTokenizer  # noqa: E402

BYTES_PER_GB = 1024**3


def estimate_memory(
    n_params: int, seq_len: int, batch: int, n_layers: int, hidden: int,
    lora: bool, precision_bytes: int, gradient_checkpointing: bool,
) -> dict[str, float]:
    """Rough training memory estimate in GB.

    Deliberately coarse: it is meant to distinguish "comfortable" from
    "hopeless" before renting a GPU, not to predict the allocator's behaviour.
    """
    weights = n_params * precision_bytes
    # LoRA trains well under 1% of parameters, so optimizer state is tiny;
    # a full fine-tune carries two fp32 moments per parameter.
    trainable = n_params * 0.005 if lora else n_params
    optimizer = trainable * 8  # AdamW: two fp32 moments
    gradients = trainable * precision_bytes
    # Activations dominate at long sequence length. Checkpointing trades
    # recomputation for storing only layer boundaries.
    stored_layers = 1 if gradient_checkpointing else n_layers
    activations = batch * seq_len * hidden * stored_layers * precision_bytes * 4

    total = weights + optimizer + gradients + activations
    return {
        "weights": weights / BYTES_PER_GB,
        "optimizer": optimizer / BYTES_PER_GB,
        "gradients": gradients / BYTES_PER_GB,
        "activations": activations / BYTES_PER_GB,
        "total": total / BYTES_PER_GB,
    }


class TinyConditionedLM(nn.Module):
    """A small causal LM with the same interface as the real backbone.

    Exists so the training path, the loss shift and the OGT head can be
    exercised at full Cas9 sequence length without a 1.5 GB download.
    """

    def __init__(self, vocab_size: int, hidden: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.hidden_size = hidden

    def forward(self, input_ids, attention_mask):
        length = input_ids.shape[1]
        # Causal mask: position t may not attend to anything after t.
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        hidden = self.embed(input_ids)
        hidden = self.blocks(
            hidden, mask=causal, src_key_padding_mask=(attention_mask == 0)
        )
        hidden = self.norm(hidden)
        return self.lm_head(hidden), hidden


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true", help="small random model, no download")
    parser.add_argument("--real", action="store_true", help="load the configured checkpoint")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    model_cfg = Config.load("model")
    train_cfg = Config.load("train")
    data_cfg = Config.load("data")
    binner = GrowthTemperatureBinner.from_config(data_cfg)

    seq_len = args.seq_len or int(train_cfg["data/max_length"])
    batch = args.batch or int(train_cfg["batch/size"])
    accum = int(train_cfg["batch/accumulate_grad_batches"])

    spec = resolve_device(
        str(train_cfg["device/accelerator"]), str(train_cfg["device/precision"])
    )
    print(f"device:    {spec.accelerator} / {spec.precision}")
    for note in spec.notes:
        print(f"  note: {note}")
    print(f"seq_len:   {seq_len} tokens (batch {batch} x accumulate {accum})")

    # Analytic estimate for the real model, printed in every mode.
    checkpoint = str(model_cfg["conditioned_lm/checkpoint"])
    lora_on = bool(model_cfg["conditioned_lm/lora/enabled"])
    print(f"\nestimate for {checkpoint} (765M params, 27 layers, hidden 1536)")
    for label, (ckpt, prec) in {
        "fp32, no checkpointing": (False, 4),
        "bf16, no checkpointing": (False, 2),
        "bf16 + gradient checkpointing": (True, 2),
    }.items():
        est = estimate_memory(
            n_params=765_000_000, seq_len=seq_len, batch=batch, n_layers=27,
            hidden=1536, lora=lora_on, precision_bytes=prec,
            gradient_checkpointing=ckpt,
        )
        verdict = "fits 16 GB" if est["total"] < 14 else "too large for 16 GB"
        print(
            f"  {label:<32} {est['total']:>6.1f} GB  "
            f"(w {est['weights']:.1f} / opt {est['optimizer']:.2f} / "
            f"act {est['activations']:.2f})  {verdict}"
        )
    print("  LoRA assumed: optimizer state covers ~0.5% of parameters")
    print(
        "  CAVEAT: the activation term counts hidden states only. It does NOT\n"
        "  count attention matrices, which are batch x heads x seq^2 and would add\n"
        "  ~2.5 GB across 27 layers at this length if materialized. Flash/SDPA\n"
        "  attention avoids materializing them; a backend that falls back to the\n"
        "  naive path will use far more than the figures above. Verify on the\n"
        "  actual GPU before trusting them."
    )

    if args.estimate_only:
        return 0

    if args.real:
        print(f"\nloading {checkpoint} (this downloads ~1.5 GB)")
        from thermocas.model.lit_model import build_lightning_module

        tokenizer = ProteinTokenizer(binner.names)
        try:
            module = build_lightning_module(model_cfg, train_cfg, tokenizer)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        print(f"  parameters: {total:,} total, {trainable:,} trainable ({trainable/total:.2%})")
        print("  real forward/backward at full length is for the GPU, not this laptop")
        return 0

    if not args.tiny:
        print("\npass --tiny, --real or --estimate-only")
        return 1

    # Tiny end-to-end run at full sequence length.
    print("\ntiny model, real sequence length")
    tokenizer = ProteinTokenizer(binner.names)
    sequences = {f"s{i}": "MGKPIPNPLLGLDST" * 90 for i in range(4)}  # 1,350 residues
    labels = [
        {
            "seq_id": f"s{i}",
            "temp_bin": "thermo" if i % 2 else "meso",
            "ogt_c": "60" if i % 2 else "37",
            "split": "train",
            "ogt_verified": "1",
        }
        for i in range(4)
    ]
    dataset = ConditionedProteinDataset(
        sequences, labels, tokenizer, max_length=seq_len, split="train",
        growth_temp_token_dropout=float(model_cfg["conditioned_lm/growth_temp_token_dropout"]),
        seed=int(train_cfg["seed"]),
    )
    print(dataset.report.summary())

    device = spec.torch_device
    model = TinyConditionedLM(vocab_size=len(tokenizer)).to(device)
    head = OgtHead(model.hidden_size, 64, 0.0, mean_c=45.0, std_c=20.0).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=1e-4
    )

    batch_data = collate([dataset[i] for i in range(min(batch, len(dataset)))], tokenizer.pad_id)
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_data.items()
    }
    print(f"batch shape: {tuple(inputs['input_ids'].shape)}")

    start = time.time()
    logits, hidden = model(inputs["input_ids"], inputs["attention_mask"])
    total_nll, n_tokens = ConditionedProteinLM.causal_lm_loss(logits, inputs["labels"])
    lm_loss = ConditionedProteinLM.mean_loss(total_nll, n_tokens)
    predicted = head(hidden, inputs["attention_mask"])
    ogt_loss = nn.functional.mse_loss(
        head.normalize(predicted), head.normalize(inputs["ogt_c"])
    )
    loss = lm_loss + float(train_cfg["loss/ogt_head_weight"]) * ogt_loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["optim/grad_clip"]))
    optimizer.step()
    optimizer.zero_grad()
    elapsed = time.time() - start

    print(f"\nforward + backward + step: {elapsed:.2f}s")
    print(f"  loss        {loss.item():.4f}  (lm {lm_loss.item():.4f}, ogt {ogt_loss.item():.4f})")
    print(f"  tokens      {int(n_tokens.item())} contributed to the loss")
    print(f"  perplexity  {ConditionedProteinLM.perplexity_from(total_nll.item(), int(n_tokens.item())):.2f}")
    print(f"  grad norm   {grad_norm:.4f}")

    if not torch.isfinite(loss):
        print("  loss is not finite", file=sys.stderr)
        return 1
    print("\ntraining path works end to end at full Cas9 sequence length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
