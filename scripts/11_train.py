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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

# Lightning is already a module-level dependency through
# thermocas.model.throughput, so importing the callback base here costs
# nothing that was not already being paid.
from lightning.pytorch.callbacks import Callback  # noqa: E402

from thermocas.config import Config, GrowthTemperatureBinner  # noqa: E402
from thermocas.model.dataset import (  # noqa: E402
    ConditionedProteinDataset,
    collate,
    load_label_table,
    load_sequences,
)
from thermocas.model.device import resolve_device, resolve_num_workers  # noqa: E402
from thermocas.model.gate2 import Gate2Params, evaluate_gate2  # noqa: E402
from thermocas.model.lit_model import ConditionedProteinLM  # noqa: E402
from thermocas.model.throughput import (  # noqa: E402
    ThroughputReport,
    format_duration,
)
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

    Deliberately fp32, not the bf16-mixed that training runs in. The quantity
    being measured is a perplexity *difference* of order 0.01-2%, and bf16
    carries about three decimal digits per op -- enough noise to be the same
    size as the effect. Training in bf16 and scoring in fp32 is the right way
    round; the 2x it costs is the price of the gate meaning something.
    """
    module.eval()
    # Gradient checkpointing trades compute for memory during a backward
    # pass. There is no backward pass here, so under no_grad it buys nothing
    # and still pays: it was the reason evaluation ran at 7.8 sequences/s,
    # slower than training managed *with* a backward pass. Restored
    # afterwards, because training is not finished when the probe runs.
    backbone = getattr(module, "backbone", None)
    was_checkpointing = bool(getattr(backbone, "is_gradient_checkpointing", False))
    if was_checkpointing:
        backbone.gradient_checkpointing_disable()
    # Training leaves the allocator holding several GB reserved but
    # unallocated, and evaluation's allocation pattern is a different shape,
    # so the first large eval batch OOMs against fragmentation rather than
    # against real use: 4.38 GB reserved-unallocated with 29 MB free was how
    # batch 16 failed.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return _perplexity_loop(
            module, dataset, tokenizer, device, bin_override, batch_size, limit
        )
    finally:
        if was_checkpointing:
            backbone.gradient_checkpointing_enable()


@torch.no_grad()
def _perplexity_loop(
    module, dataset, tokenizer, device, bin_override, batch_size, limit
):
    total_nll = 0.0
    total_tokens = 0
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for start in range(0, n, batch_size):
        items = []
        for index in range(start, min(start + batch_size, n)):
            override = bin_override[index] if bin_override is not None else None
            items.append(
                dataset.__getitem__(
                    index, growth_temp_bin_override=override, drop_label=False
                )
            )
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


class ConditioningProbe(Callback):
    """Gate 2's measurement, taken periodically during training.

    Without this, a 20,000-step run yields one pass/fail at the end and no way
    to tell a model that cannot learn the conditioning from one that had not
    learned it yet. The gain curve separates those two: flat from step 2,500
    to 20,000 is a different finding, and a different paper, from rising and
    short of the threshold.

    Cheap on purpose. A fixed subsample, the same one every time so the
    numbers are comparable, and six forward passes over it -- minutes, against
    the hour the full split costs. It is a trend line, not the gate; the gate
    still runs once, on everything, at the end.
    """

    def __init__(self, check, dataset, limit: int, every_n_steps: int) -> None:
        self.check = check
        self.dataset = dataset
        self.limit = limit
        self.every_n_steps = every_n_steps
        self.history: list[tuple[int, float]] = []
        self._last_probed = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.every_n_steps or step == self._last_probed:
            return
        self._last_probed = step
        was_training = pl_module.training
        result = self.check(self.dataset, "probe", self.limit, quiet=True)
        if was_training:
            pl_module.train()
        self.history.append((step, result.relative_gain))
        pl_module.log("cond_gain", result.relative_gain)
        print(f"\n  conditioning probe at step {step}: "
              f"gain {result.relative_gain:+.2%} on {self.limit} val examples")

    def report(self) -> str:
        """The curve, and a verdict that compares the change to the scatter.

        The first version of this compared only the first probe to the last,
        which called the first real run "rising from +0.251% to +0.291%". The
        curve behind that was 0.251, 0.273, 0.300, 0.251, 0.293, 0.288,
        0.281, 0.291 -- a plateau reached by step 2,500 with the last point
        below the third. "Rising" was true of the two endpoints and false of
        the run, and it is exactly the sentence that would have justified
        buying more steps.
        """
        if not self.history:
            return "conditioning probe: never ran"
        lines = ["conditioning gain over training (probe subsample, not the gate):"]
        lines += [f"  step {step:>6}  {gain:+.3%}" for step, gain in self.history]
        if len(self.history) < 4:
            return "\n".join(lines)

        gains = [gain for _, gain in self.history]
        half = len(gains) // 2
        early, late = gains[:half], gains[half:]
        change = sum(late) / len(late) - sum(early) / len(early)
        # Jitter, not total range. Range is the wrong noise estimate: on a
        # smoothly rising curve the range *is* the rise, so comparing the two
        # calls every real trend flat. The mean absolute step-to-step change
        # is large on a noisy plateau and small on a smooth trend, which is
        # the distinction being drawn.
        jitter = sum(
            abs(b - a) for a, b in zip(gains, gains[1:], strict=False)
        ) / max(len(gains) - 1, 1)
        if abs(change) < jitter:
            lines.append(
                f"  FLAT: halves differ by {change:+.3%}, within the "
                f"{jitter:.3%} step-to-step jitter."
            )
            lines.append(
                "  More steps are not the missing ingredient -- whatever the "
                "control\n  token can contribute, it contributed early."
            )
        else:
            lines.append(
                f"  {'RISING' if change > 0 else 'FALLING'}: halves differ by "
                f"{change:+.3%}, against {jitter:.3%} step-to-step jitter."
            )
        return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=["a", "b"], required=True)
    parser.add_argument("--model", default="conditioned_lm")
    parser.add_argument("--init-from", type=Path, default=None, help="stage A checkpoint")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override batch/size. For finding the largest batch a GPU fits; "
             "the value that is actually used belongs in configs/train.yaml",
    )
    parser.add_argument(
        "--run-name",
        default="default",
        help="checkpoints go to <checkpoint dir>/stage_<x>/<run-name>. Give smoke "
             "runs and sweeps their own name: Lightning appends -v1, -v2 ... when a "
             "last.ckpt is already there rather than overwriting it, so runs sharing "
             "a directory leave stale checkpoints that load cleanly and are "
             "indistinguishable from the real one.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from last.ckpt in this run's checkpoint directory, if one is "
             "there. Starts from scratch with a note when it is not, so a relaunch "
             "after a container dies is the same command either way. NOTE: the "
             "dataloader is not resumable, so a resumed run re-reads the epoch from "
             "the start -- it is a recovery mechanism, not equivalent to an "
             "uninterrupted run, and a run that used it has to say so.",
    )
    parser.add_argument("--skip-leakage-check", action="store_true",
                        help="only for stage A, which has no Cas9 test split")
    parser.add_argument("--dry-run", action="store_true", help="build everything, train nothing")
    parser.add_argument(
        "--probe-every",
        type=int,
        default=None,
        help="override conditioning_gate/probe_every_n_steps. Exists so a smoke "
             "run can exercise the probe without 2,500 steps of training.",
    )
    parser.add_argument(
        "--probe-examples",
        type=int,
        default=None,
        help="override conditioning_gate/probe_examples",
    )
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
    num_workers, worker_notes = resolve_num_workers(
        spec.accelerator, str(data_cfg["data/num_workers"])
    )
    print(f"stage:     {args.stage.upper()}")
    print(f"model:     {args.model} ({checkpoint})")
    print(f"device:    {spec.accelerator} / {spec.precision}")
    for note in (*spec.notes, *worker_notes):
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
        datasets = build_datasets(
            stage_cfg, data_cfg, tokenizer, model_cfg, args.model,
            split_names=("train", "val", "test"),
        )
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
    ceiling = (
        f"ceiling {gate2.ceiling_gain:+.3%}, need "
        f"{gate2.min_ceiling_fraction:.2f}x of it"
        if gate2.ceiling_gain
        else "no corpus ceiling set, so effect size will be unjudged"
    )
    print(f"\nGate 2: enabled={gate2.enabled}, {gate2.n_shuffles} shuffles, "
          f"z >= {gate2.min_z:.1f}; {ceiling}")

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
        load_stage_checkpoint,
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
        try:
            loaded, total = load_stage_checkpoint(module, state["state_dict"])
        except ValueError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 3
        print(f"initialized from {args.init_from} ({loaded}/{total} tensors)")

    train_batch_size = args.batch_size or int(data_cfg["batch/size"])
    if args.batch_size:
        print(f"batch size:  {train_batch_size} (overridden from "
              f"{int(data_cfg['batch/size'])} on the command line)")

    def loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            datasets[split],
            batch_size=(
                train_batch_size
                if split == "train"
                else int(data_cfg["batch/val_size"])
            ),
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=lambda b: collate(b, tokenizer.pad_id),
        )

    checkpoint_dir = (
        Path(str(data_cfg["checkpoint/dir"])) / f"stage_{args.stage}" / args.run_name
    )

    # Resume is resolved before the trainer is built so that a missing
    # checkpoint is a printed note here rather than an exception thrown from
    # inside fit after the model is already on the GPU.
    resume_from = None
    if args.resume:
        candidate = checkpoint_dir / "last.ckpt"
        if candidate.is_file():
            resume_from = candidate
            print(f"resuming from {candidate}")
            # Said at the top of every resumed run, because the consequence is
            # a property of the results and not of the infrastructure.
            print(
                "  NOTE the dataloader is not resumable: this run restarts the\n"
                "  epoch from the beginning, so it will see early examples twice\n"
                "  and some later ones not at all. Step count is preserved;\n"
                "  data coverage is not. Report this run as resumed."
            )
        else:
            print(f"--resume given but no {candidate}; starting from step 0")

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

    # Timed from inside the fit phase. Timing the whole script instead puts
    # the checkpoint download and Gate 2 into the per-step figure, which is
    # how the first GPU run came back at 2.97 s/step against a real rate
    # roughly nine times faster.
    throughput = ThroughputReport(
        planned_steps=int(data_cfg["optim/max_steps"]),
        epoch_examples=len(datasets["train"]),
    )

    device = spec.torch_device
    eval_batch_size = int(data_cfg["batch/eval_size"])

    def conditioning_check(dataset, label: str, limit: int | None, quiet: bool = False):
        """Perplexity with the correct control token against shuffled ones.

        The shuffled runs are the control the whole claim rests on: a model
        that ignores its growth-temperature token scores the same either way,
        and a model that has merely learned the corpus scores well on both.
        """
        if not quiet and limit is not None and limit < len(dataset):
            if args.gate2_limit:
                print(f"  WARNING evaluating on {limit} of {len(dataset)} {label} "
                      "examples (--gate2-limit). This is a smoke check, not the "
                      "real gate.")
            else:
                # A budgeted subsample is a different statement from a smoke
                # check, and saying "not the real gate" about it would be
                # wrong in the other direction.
                print(f"  evaluating a {limit:,}-example subsample of the "
                      f"{len(dataset):,} {label} examples "
                      "(conditioning_gate/max_examples)")
        # Timed and reported, because the cost of these passes was guessed at
        # once and the guess was out by a factor that would have decided
        # whether the run finished. n_shuffles + 1 passes over the split is
        # the single largest block of GPU time after training itself.
        started = time.monotonic()
        scored = (min(limit, len(dataset)) if limit else len(dataset)) * (
            gate2.n_shuffles + 1
        )
        true_pair = evaluate_perplexity(
            module, dataset, tokenizer, device,
            batch_size=eval_batch_size, limit=limit,
        )
        runs = []
        for index in range(gate2.n_shuffles):
            assignment = dataset.with_shuffled_labels(
                rng_seed=int(data_cfg["seed"]) + index
            )
            runs.append(
                evaluate_perplexity(
                    module, dataset, tokenizer, device, bin_override=assignment,
                    batch_size=eval_batch_size, limit=limit,
                )
            )
        elapsed = time.monotonic() - started
        if not quiet:
            print(f"  {scored:,} sequences scored in {format_duration(elapsed)} "
                  f"({scored / max(elapsed, 1e-6):.1f} sequences/s at batch "
                  f"{eval_batch_size})")
        return evaluate_gate2(*true_pair, runs, gate2)

    probe = ConditioningProbe(
        conditioning_check,
        datasets["val"],
        limit=args.probe_examples or int(data_cfg["conditioning_gate/probe_examples"]),
        every_n_steps=(
            args.probe_every or int(data_cfg["conditioning_gate/probe_every_n_steps"])
        ),
    )

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
            throughput,
            probe,
        ],
        logger=CSVLogger("logs", name=f"stage_{args.stage}"),
        enable_progress_bar=True,
    )

    L.seed_everything(int(data_cfg["seed"]), workers=True)
    trainer.fit(
        module,
        loader("train", True),
        loader("val", False),
        ckpt_path=str(resume_from) if resume_from else None,
    )

    # Printed before Gate 2, which is itself minutes of GPU time and would
    # otherwise push the throughput figures out of sight above the gate's
    # verdict.
    print()
    print(throughput.report())

    # Gate 2.
    if not gate2.enabled:
        print("\nGate 2 disabled in config; skipping the conditioning check")
        return 0

    print()
    print(probe.report())

    print("\nrunning Gate 2 on the validation split")
    module.to(device)

    # --gate2-limit is the smoke-check override and wins when given;
    # max_examples is the standing budget cap, and 0 means the whole split.
    limit = args.gate2_limit or int(data_cfg["conditioning_gate/max_examples"]) or None
    result = conditioning_check(datasets["val"], "validation", limit)
    print()
    print(result.summary())

    report = checkpoint_dir / "gate2_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(result.summary() + "\n")
    print(f"\nwrote {report}")

    # The held-out number. Validation chose the checkpoint and validation is
    # what Gate 2 just decided on, so validation cannot also be the figure the
    # paper reports -- selection has seen it. Test is scored once, here, with
    # the checkpoint validation picked rather than the last step's weights.
    test = datasets.get("test")
    if test is None or not len(test):
        print("\nno test split for this stage; nothing held out to report")
    else:
        best = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
        if best and Path(best).is_file():
            state = torch.load(best, map_location="cpu")
            module.load_state_dict(state["state_dict"])
            module.to(device)
            score = getattr(trainer.checkpoint_callback, "best_model_score", None)
            print(f"\nloaded the checkpoint validation selected: {best}")
            if score is not None:
                print(f"  its {data_cfg['checkpoint/monitor']} was {float(score):.4f}")
        else:
            print("\nWARNING no monitored checkpoint was saved, so the test split is "
                  "being scored with the weights training ended on. That is the "
                  "last step, not a selected model; do not report it as one.")

        scored_n = min(limit, len(test)) if limit else len(test)
        print(f"\nheld-out test report on {scored_n:,} of {len(test):,} "
              "sequences, cluster-disjoint from train")
        test_result = conditioning_check(test, "test", limit)
        print()
        print(test_result.summary())

        # Records what was actually scored, not the split size. The first
        # version of this file said "test sequences: 16952" after scoring
        # 1,200 of them, which is the kind of line that gets quoted later by
        # someone who did not run it.
        provenance = best or (
            "LAST STEP, not selected -- these weights were not chosen by validation"
        )
        test_report = checkpoint_dir / "test_report.txt"
        test_report.write_text(
            f"checkpoint: {provenance}\n"
            f"test sequences scored: {scored_n} of {len(test)} in the split\n"
            "cluster-disjoint from train: yes\n\n"
            + test_result.summary() + "\n"
        )
        print(f"\nwrote {test_report}")

    if not result.passed and gate2.fail_on_violation:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
