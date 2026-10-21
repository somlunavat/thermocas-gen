#!/usr/bin/env python3
"""Stage 2: train the conditioned LM, then run Gate 2.

Refuses to start until the leakage gate has passed on the split it is about to
train on, and refuses to report success until Gate 2 has passed. Both are hard
stops rather than warnings: training on a leaked split produces numbers that
mean nothing, and generating from a model that ignores its conditioning
produces sequences whose target temperature had no influence on them.

Supports both stages of the two-stage design:

    --stage a    broad temperature-labeled proteins (configs/pretrain.yaml)
    --stage b    Cas9 fine-tuning from a stage A checkpoint

Usage:
    python scripts/11_train.py --stage a --max-steps 50 --dry-run
    python scripts/11_train.py --stage a
    python scripts/11_train.py --stage b --init-from checkpoints/stage_a/last.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from thermocas.config import Config, GrowthTemperatureBinner  # noqa: E402
from thermocas.model.dataset import (  # noqa: E402
    ConditionedProteinDataset,
    collate,
    load_label_table,
    load_sequences,
)
from thermocas.model.device import resolve_device  # noqa: E402
from thermocas.model.gate2 import Gate2Params, evaluate_gate2  # noqa: E402
from thermocas.model.lit_model import ConditionedProteinLM  # noqa: E402
from thermocas.model.tokenizer import PretrainedTokenizerAdapter  # noqa: E402


def leakage_gate_passed(splits_dir: Path) -> tuple[bool, str]:
    """Check that the leakage gate has been run and recorded a pass.

    Training must not be the step that discovers a leak. The marker is written
    by 06_leakage_test.py; its absence is treated as failure rather than as
    permission, because "not yet checked" and "checked and clean" must not be
    the same state.
    """
    marker = splits_dir / "leakage_passed"
    if not marker.is_file():
        return False, (
            f"no leakage-gate marker at {marker}. Run scripts/06_leakage_test.py "
            "first; an unchecked split is not a clean split."
        )
    return True, marker.read_text().strip()


def resolve_training_corpus(cfg) -> Path:
    """The FASTA training is allowed to read.

    Stage A must read the screened corpus, not the raw one. The difference
    between the two files is the Cas9 homologs that the alignment screen
    removed, and training on the raw corpus would put held-out Cas9 into
    pre-training where the stage B leakage gate cannot see it. A missing
    screened file is therefore an error, not a reason to fall back.
    """
    if "paths/screened_sequences" in cfg:
        screened = cfg.resolve_path("paths/screened_sequences")
        if not screened.is_file():
            raise FileNotFoundError(
                f"screened corpus not found: {screened}\n"
                "Run scripts/09_screen_broad.py first. Training on the unscreened "
                "corpus would place Cas9 homologs in pre-training, which the stage B "
                "leakage gate cannot detect."
            )
        return screened
    if "paths/labeled_fasta" in cfg:
        return cfg.resolve_path("paths/labeled_fasta")
    return cfg.resolve_path("paths/sequences")


def build_datasets(cfg, data_cfg, tokenizer, model_cfg, model_name, split_names=("train", "val")):
    corpus = resolve_training_corpus(cfg)
    print(f"corpus:    {corpus.name}")
    sequences = load_sequences(corpus)
    labels = load_label_table(cfg.resolve_path("paths/label_table"))
    datasets = {}
    for split in split_names:
        datasets[split] = ConditionedProteinDataset(
            sequences,
            labels,
            tokenizer,
            max_length=int(data_cfg["data/max_length"]),
            split=split,
            on_too_long=str(data_cfg["data/on_too_long"]),
            growth_temp_token_dropout=(
                float(model_cfg[f"{model_name}/growth_temp_token_dropout"])
                if split == "train"
                else 0.0
            ),
            use_growth_temp_token=bool(model_cfg[f"{model_name}/use_growth_temp_token"]),
            seed=int(data_cfg["seed"]),
        )
    return datasets


@torch.no_grad()
def evaluate_perplexity(
    module, dataset, tokenizer, device, bin_override=None, batch_size=2, limit=None
):
    """Summed NLL and token count over a dataset, optionally re-binned.

    ``bin_override`` is a per-example bin assignment, which is how the Gate 2
    shuffled comparison is run without re-tokenizing anything.
    """
    module.eval()
    total_nll = 0.0
    total_tokens = 0
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for start in range(0, n, batch_size):
        items = []
        for index in range(start, min(start + batch_size, n)):
            override = bin_override[index] if bin_override is not None else None
            items.append(dataset.__getitem__(index, growth_temp_bin_override=override, drop_label=False))
        batch = collate(items, tokenizer.pad_id)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = module(input_ids, attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        nll, n_tokens = ConditionedProteinLM.causal_lm_loss(logits, labels)
        total_nll += float(nll.item())
        total_tokens += int(n_tokens.item())
    return total_nll, total_tokens


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=["a", "b"], required=True)
    parser.add_argument("--model", default="conditioned_lm")
    parser.add_argument("--init-from", type=Path, default=None, help="stage A checkpoint")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-leakage-check", action="store_true",
                        help="only for stage A, which has no Cas9 test split")
    parser.add_argument("--dry-run", action="store_true", help="build everything, train nothing")
    parser.add_argument(
        "--gate2-limit",
        type=int,
        default=None,
        help="evaluate Gate 2 on at most this many validation examples. For smoke "
             "runs only: the real gate uses the whole validation split.",
    )
    args = parser.parse_args()

    stage_cfg = Config.load("pretrain" if args.stage == "a" else "data")
    data_cfg = Config.load("train")
    model_cfg = Config.load("model")
    binner = GrowthTemperatureBinner.from_config(Config.load("data"))
    # The residue vocabulary comes from the checkpoint, not from this project.
    # Only the growth-temperature control tokens are ours, appended above the
    # checkpoint's rows. See PretrainedTokenizerAdapter for what went wrong
    # when the project used its own ids here.
    checkpoint = str(model_cfg[f"{args.model}/checkpoint"])
    tokenizer = PretrainedTokenizerAdapter.from_checkpoint(checkpoint, binner.names)

    spec = resolve_device(
        str(data_cfg["device/accelerator"]), str(data_cfg["device/precision"])
    )
    print(f"stage:     {args.stage.upper()}")
    print(f"model:     {args.model} ({checkpoint})")
    print(f"device:    {spec.accelerator} / {spec.precision}")
    for note in spec.notes:
        print(f"  note: {note}")
    print(
        f"vocab:     {len(tokenizer)} tokens "
        f"({tokenizer.base_vocab_size} from the checkpoint + "
        f"{len(tokenizer) - tokenizer.base_vocab_size} control), bins {binner.names}"
    )

    # Gate: leakage must have been checked before stage B training.
    if args.stage == "b" and not args.skip_leakage_check:
        splits_dir = Config.load("data").resolve_path("paths/splits_dir")
        passed, detail = leakage_gate_passed(splits_dir)
        print(f"\nleakage gate: {'PASSED' if passed else 'NOT VERIFIED'}")
        if not passed:
            print(f"  {detail}", file=sys.stderr)
            return 3
        print(f"  {detail}")

    # Data.
    try:
        datasets = build_datasets(stage_cfg, data_cfg, tokenizer, model_cfg, args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\ncould not build datasets: {exc}", file=sys.stderr)
        return 1

    print()
    for split, dataset in datasets.items():
        print(f"{split}:")
        for line in dataset.report.summary().splitlines():
            print(f"  {line}")

    # A degenerate bin distribution makes Gate 2 untestable: if validation is
    # single-class, shuffling its labels changes nothing.
    val_bins = datasets["val"].report.bin_counts
    if len(val_bins) < 2:
        print(
            f"\nvalidation set has only {len(val_bins)} growth-temperature class "
            f"({dict(val_bins)}). Gate 2 cannot distinguish a correct token from a "
            "shuffled one on a single-class set, so the conditioning check would be "
            "vacuous. Re-split with stratification before training.",
            file=sys.stderr,
        )
        return 3

    gate2 = Gate2Params.from_config(data_cfg)
    print(f"\nGate 2: enabled={gate2.enabled}, "
          f"required gain {gate2.min_perplexity_gain:.1%} over {gate2.n_shuffles} shuffles")

    if args.dry_run:
        print("\ndry run: datasets built, nothing trained")
        return 0

    # Model and trainer.
    import lightning as L  # noqa: N812
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from thermocas.model.lit_model import (
        build_lightning_module,
        check_validation_cadence,
    )

    ogt_values = [
        float(e["ogt_c"]) for e in datasets["train"].examples if e["ogt_c"] == e["ogt_c"]
    ]
    ogt_mean = sum(ogt_values) / len(ogt_values) if ogt_values else 45.0
    ogt_std = (
        max((sum((v - ogt_mean) ** 2 for v in ogt_values) / len(ogt_values)) ** 0.5, 1.0)
        if ogt_values
        else 20.0
    )
    print(f"OGT normalization from the training split: mean {ogt_mean:.1f} C, std {ogt_std:.1f} C")

    module = build_lightning_module(
        model_cfg, data_cfg, tokenizer, ogt_mean_c=ogt_mean, ogt_std_c=ogt_std,
        model_name=args.model,
    )

    if args.init_from is not None:
        if not args.init_from.is_file():
            print(f"stage A checkpoint not found: {args.init_from}", file=sys.stderr)
            return 1
        state = torch.load(args.init_from, map_location="cpu")
        signature = state.get("hyper_parameters", {}).get("tokenizer_signature")
        if signature and signature != tokenizer.signature:
            print(
                "stage A checkpoint was trained with a different vocabulary layout. "
                "Loading it would silently reinterpret every control token.",
                file=sys.stderr,
            )
            return 3
        module.load_state_dict(state["state_dict"], strict=False)
        print(f"initialized from {args.init_from}")

    def loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            datasets[split],
            batch_size=int(data_cfg["batch/size" if split == "train" else "batch/val_size"]),
            shuffle=shuffle,
            num_workers=int(data_cfg["data/num_workers"]),
            collate_fn=lambda b: collate(b, tokenizer.pad_id),
        )

    checkpoint_dir = Path(str(data_cfg["checkpoint/dir"])) / f"stage_{args.stage}"

    # The checkpoint callback monitors a validation metric, so validation has
    # to have run by the time it fires. Lightning does not complain when it
    # has not: it saves nothing and reports no error. An epoch here is ~18,400
    # steps, so relying on the per-epoch default means a whole run can finish
    # with no monitored checkpoint.
    accumulate = int(data_cfg["batch/accumulate_grad_batches"])
    val_interval_batches = int(data_cfg["validation/check_interval_batches"])
    checkpoint_every_n_steps = int(data_cfg["checkpoint/every_n_steps"])
    try:
        cadence = check_validation_cadence(
            val_interval_batches,
            accumulate,
            checkpoint_every_n_steps,
            str(data_cfg["checkpoint/monitor"]),
        )
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    print(f"validation: every {cadence} steps on at most "
          f"{int(data_cfg['validation/limit_batches'])} batches")

    trainer = L.Trainer(
        accelerator=spec.accelerator,
        precision=spec.precision,
        max_steps=args.max_steps or int(data_cfg["optim/max_steps"]),
        accumulate_grad_batches=int(data_cfg["batch/accumulate_grad_batches"]),
        gradient_clip_val=float(data_cfg["optim/grad_clip"]),
        log_every_n_steps=int(data_cfg["logging/log_every_n_steps"]),
        val_check_interval=val_interval_batches,
        limit_val_batches=int(data_cfg["validation/limit_batches"]),
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                every_n_train_steps=int(data_cfg["checkpoint/every_n_steps"]),
                save_top_k=int(data_cfg["checkpoint/save_top_k"]),
                monitor=str(data_cfg["checkpoint/monitor"]),
                mode=str(data_cfg["checkpoint/mode"]),
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=CSVLogger("logs", name=f"stage_{args.stage}"),
        enable_progress_bar=True,
    )

    L.seed_everything(int(data_cfg["seed"]), workers=True)
    trainer.fit(module, loader("train", True), loader("val", False))

    # Gate 2.
    if not gate2.enabled:
        print("\nGate 2 disabled in config; skipping the conditioning check")
        return 0

    print("\nrunning Gate 2 on the validation split")
    device = spec.torch_device
    module.to(device)
    val = datasets["val"]

    limit = args.gate2_limit
    if limit is not None and limit < len(val):
        print(f"  WARNING evaluating on {limit} of {len(val)} validation examples "
              "(--gate2-limit). This is a smoke check, not the real gate.")

    true_nll, true_tokens = evaluate_perplexity(module, val, tokenizer, device, limit=limit)
    shuffled_runs = []
    for index in range(gate2.n_shuffles):
        assignment = val.with_shuffled_labels(rng_seed=int(data_cfg["seed"]) + index)
        shuffled_runs.append(
            evaluate_perplexity(
                module, val, tokenizer, device, bin_override=assignment, limit=limit
            )
        )

    result = evaluate_gate2(true_nll, true_tokens, shuffled_runs, gate2)
    print()
    print(result.summary())

    report = checkpoint_dir / "gate2_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(result.summary() + "\n")
    print(f"\nwrote {report}")

    if not result.passed and gate2.fail_on_violation:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
