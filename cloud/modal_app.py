"""Training on a cloud GPU, via Modal.

Why this file exists rather than an ssh session: the GPU figures that decide
the training schedule have to be measured on the GPU, and the measurement has
to be repeatable. Keeping the GPU type, the package versions and the command
in the repository means the numbers in PROJECT_LOG.md can be traced to
something, and that a run can be repeated after a package upgrade without
anyone remembering what was typed.

It also costs nothing while idle, which matters for the actual pattern here:
a short run, read the numbers, change something, run again.

Layout. Modal gives the container a read-only mount of this repository and a
persistent Volume. The volume holds exactly what git cannot: the 171 MB stage
A corpus, the checkpoints, the training logs and the Hugging Face cache for
the 3 GB ProGen2 checkpoint. Everything else -- code, configs and the split
files, which are committed -- comes from the repository, so the only thing
that can drift between a local run and a GPU run is the corpus, and that is
uploaded by ``upload`` from the same local files the local run reads.

First use::

    pip install modal
    modal token new                     # opens a browser
    modal run cloud/modal_app.py::check     # ~30 s, confirms the plumbing
    modal run cloud/modal_app.py::upload    # ~2 min, pushes the corpus
    modal run cloud/modal_app.py::smoke     # 50 steps, reports memory

A long run goes through ``train``, which spawns::

    modal run cloud/modal_app.py::train --stage a
    modal run cloud/modal_app.py::status --call-id fc-...

No ``--detach``, and no backgrounding. Two Stage A runs were lost before this
was understood. A blocking ``Function.remote()`` call is owned by the local
client for as long as it waits, and Modal cancels the in-flight input when
that client goes away -- so the run dies with the terminal, the laptop or the
client's expiring auth. The first attempt died at step 314 ("Stopping app -
local client disconnected"), and the second, relaunched with ``--detach`` on
the theory that this was the fix, died at step 2,149 of 20,000 with "Received
a cancellation signal". ``--detach`` keeps the *app* alive; it does not stop
the input being cancelled, and the launch output says so in its first line:
detached mode "only keeps the last triggered Modal function alive".

``Function.spawn()`` is the fix. The call is created server-side and returns
an id; no client is in the loop, so nothing local can cancel it. ``_train``
also prints its own summary rather than only returning one, because with
nobody waiting there is no local process left to print the returned dict --
read it back with ``status`` or ``modal app logs``.

``check`` first. It is cheap and it fails for all the boring reasons -- no
volume, no corpus, wrong CUDA -- that would otherwise fail ten minutes into a
paid training run.
"""

from __future__ import annotations

import modal

APP_NAME = "thermocas-gen"

# The default GPU for a first measurement, not a considered choice: 24 GB is
# enough to run and cheap enough to be wrong about. The real choice comes from
# what `smoke` reports about peak memory, which is the number this project
# does not have yet. Override with --gpu on any entrypoint below.
DEFAULT_GPU = "A10G"

# Pinned to the versions the test suite passes against locally, so a GPU run
# is not also a package upgrade. Python is 3.12 here against 3.14 locally,
# because torch wheels for 3.14 are not dependably available; the project
# targets >=3.11 and the ruff target is py311, so no 3.14 syntax is in play.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.12.0",
        "lightning==2.6.6",
        "transformers==5.10.2",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "tokenizers==0.22.2",
        "safetensors==0.7.0",
        "huggingface-hub==1.18.0",
        "numpy==2.4.4",
        "pyyaml==6.0.3",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(
        ".",
        remote_path="/src",
        ignore=[
            ".git",
            ".claude",
            "docs",
            "data/raw/broad_shards",
            "data/processed",
            "checkpoints",
            "logs",
            "notebooks",
            "**/__pycache__",
            "**/.DS_Store",
            "**/.pytest_cache",
            "**/.ruff_cache",
        ],
    )
)

volume = modal.Volume.from_name(f"{APP_NAME}-vol", create_if_missing=True)
app = modal.App(APP_NAME, image=image)

# What the volume has to hold for stage A, as repository-relative paths. The
# split files are not here on purpose: they are committed, so they arrive with
# the source and git stays the single source of truth for them.
STAGE_A_CORPUS = (
    "data/processed/broad_proteins_screened.fasta",
    "data/processed/broad_label_table.csv",
)
STAGE_B_CORPUS = (
    "data/processed/cas9_labeled.fasta",
    "data/processed/label_table.csv",
)

# Directories that live on the volume rather than in the repository copy.
# Everything large, everything written, and nothing git tracks.
LINKED = ("data/processed", "checkpoints", "logs")

REMOTE_REPO = "/root/paper"


def _prepare_repo() -> str:
    """Stage a writable copy of the repository with the volume linked in.

    The source mount is read-only and config paths resolve against the
    repository root (see ``Config.resolve_path``), so the volume is linked in
    at those relative paths rather than passed as arguments. Copying first is
    what makes the links possible; the copy is code and configs only, a few
    megabytes, because the mount excludes the corpus.
    """
    import os
    import shutil

    # Refreshed on every call rather than once per container: Modal reuses
    # warm containers, and a stale copy of the code would be the worst
    # possible kind of confusing. The source mount excludes every linked
    # directory below, so this never overwrites a symlink with a real one.
    shutil.copytree("/src", REMOTE_REPO, dirs_exist_ok=True)

    for relative in LINKED:
        target = f"/vol/{relative}"
        os.makedirs(target, exist_ok=True)
        link = os.path.join(REMOTE_REPO, relative)
        if os.path.islink(link):
            continue
        if os.path.exists(link):
            shutil.rmtree(link)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(target, link)

    os.chdir(REMOTE_REPO)
    return REMOTE_REPO


@app.function(volumes={"/vol": volume}, gpu=DEFAULT_GPU, timeout=600)
def _check() -> dict:
    """Report what the container can see. Nothing is trained."""
    import os
    import subprocess

    import torch

    _prepare_repo()

    report: dict = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "total_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            if torch.cuda.is_available()
            else None
        ),
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip(),
    }

    # The flash backend is the whole reason the vendored attention was
    # changed; if it is unavailable here, the memory figures from `smoke`
    # would be math-backend figures and would mislead the schedule.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query = torch.randn(1, 16, 512, 96, device="cuda", dtype=torch.bfloat16)
    backends = {}
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(
                    query, query, query, is_causal=True
                )
            backends[backend.name] = True
        except Exception as exc:
            backends[backend.name] = str(exc)[:80]
    report["sdpa_backends"] = backends

    report["corpus"] = {
        path: (os.path.getsize(path) if os.path.exists(path) else None)
        for path in STAGE_A_CORPUS
    }
    report["splits"] = sorted(os.listdir("data/splits/broad")) if os.path.isdir(
        "data/splits/broad"
    ) else []

    # A dry run exercises config loading, the tokenizer, dataset construction
    # and the model build without a single optimizer step. It is the cheapest
    # thing that fails for a real reason.
    dry = subprocess.run(
        ["python", "scripts/11_train.py", "--stage", "a",
         "--skip-leakage-check", "--dry-run"],
        capture_output=True, text=True, cwd=REMOTE_REPO,
    )
    report["dry_run_exit"] = dry.returncode
    report["dry_run_tail"] = (dry.stdout + dry.stderr)[-3000:]
    return report


@app.function(volumes={"/vol": volume}, gpu=DEFAULT_GPU, timeout=24 * 3600)
def _train(
    stage: str,
    max_steps: int | None,
    gate2_limit: int | None,
    init_from: str | None,
    batch_size: int | None,
    extra: list[str],
    run_name: str = "default",
    resume: bool = False,
) -> dict:
    """Run scripts/11_train.py and report what it cost.

    The script is imported and called rather than subprocessed so that CUDA's
    peak-memory counters can be read from the same process. Its return value
    is its exit code, and the codes keep their meanings: 2 for a validation
    cadence mismatch, 3 for a failed gate or a checkpoint whose keys do not
    match.

    The wall time measured here is the whole call: the checkpoint download,
    the model build, training and Gate 2. It is the right number for "what
    did this cost" and the wrong one for "how fast is a step", which is why
    dividing it by the step count is no longer done here. ThroughputReport
    times the fit phase from inside and prints that figure in the output
    above.
    """
    import importlib.util
    import sys
    import time

    import torch

    _prepare_repo()

    argv = ["11_train.py", "--stage", stage, "--run-name", run_name]
    if stage == "a":
        argv.append("--skip-leakage-check")
    if resume:
        argv.append("--resume")
    if max_steps is not None:
        argv += ["--max-steps", str(max_steps)]
    if gate2_limit is not None:
        argv += ["--gate2-limit", str(gate2_limit)]
    if init_from is not None:
        argv += ["--init-from", init_from]
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    argv += list(extra)
    print(f"argv: {' '.join(argv)}", flush=True)

    spec = importlib.util.spec_from_file_location(
        "thermocas_train", f"{REMOTE_REPO}/scripts/11_train.py"
    )
    module = importlib.util.module_from_spec(spec)

    torch.cuda.reset_peak_memory_stats()
    sys.argv = argv
    started = time.monotonic()
    spec.loader.exec_module(module)
    exit_code = module.main()
    elapsed = time.monotonic() - started

    volume.commit()
    result = {
        "argv": argv,
        "exit_code": exit_code,
        "seconds": round(elapsed, 1),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        ),
        "gpu": torch.cuda.get_device_name(0),
    }
    # Printed here as well as returned. Under `modal run --detach` the local
    # entrypoint dies when the client disconnects and never prints the
    # returned dict, so a long run's summary has to be in the container's own
    # output or it is lost with the connection.
    print()
    print(_summary_lines(result))
    return result


@app.function(volumes={"/vol": volume}, timeout=3600)
def _list_volume(prefix: str = "") -> list[str]:
    import os

    root = os.path.join("/vol", prefix)
    if not os.path.exists(root):
        return []
    found = []
    for directory, _, files in os.walk(root):
        for name in files:
            path = os.path.join(directory, name)
            found.append(f"{os.path.getsize(path) / 1e6:10.1f} MB  {path}")
    return sorted(found)


@app.local_entrypoint()
def check(gpu: str = DEFAULT_GPU):
    """Confirm the GPU, the corpus and a dry run, before paying for training."""
    report = _check.with_options(gpu=gpu).remote()

    print(f"\ngpu          {report['gpu']}  ({report['total_memory_gb']} GB)")
    print(f"driver       {report['driver']}")
    print(f"torch        {report['torch']} / cuda {report['cuda']}")
    print(f"bf16         {report['bf16']}")
    for name, state in report["sdpa_backends"].items():
        print(f"sdpa         {name}: {state}")

    print("\ncorpus on the volume:")
    for path, size in report["corpus"].items():
        print(f"  {'MISSING  ' if size is None else f'{size / 1e6:8.1f} MB'}  {path}")
    print(f"splits from git: {report['splits'] or 'MISSING'}")

    print(f"\ndry run exit {report['dry_run_exit']}")
    print(report["dry_run_tail"])
    if report["dry_run_exit"] != 0 or any(
        size is None for size in report["corpus"].values()
    ):
        print("\nnot ready. Run `modal run cloud/modal_app.py::upload` if the "
              "corpus is missing; otherwise read the dry run output above.")


@app.local_entrypoint()
def upload(stage: str = "a"):
    """Push the corpus to the volume from the local files training reads.

    Idempotent and worth repeating: re-running after the corpus is rebuilt is
    how the volume is kept honest. Uploading is the only step where a GPU run
    can diverge from a local one, because everything else comes from git.
    """
    import os

    paths = STAGE_A_CORPUS if stage == "a" else STAGE_B_CORPUS
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            "not present locally, so there is nothing to upload: "
            + ", ".join(missing)
        )

    total = sum(os.path.getsize(path) for path in paths)
    print(f"uploading {total / 1e6:.0f} MB to the {APP_NAME}-vol volume")
    with volume.batch_upload(force=True) as batch:
        for path in paths:
            print(f"  {os.path.getsize(path) / 1e6:8.1f} MB  {path}")
            batch.put_file(path, f"/{path}")
    print("done")


@app.local_entrypoint()
def smoke(
    gpu: str = DEFAULT_GPU,
    steps: int = 50,
    gate2_limit: int = 64,
    batch_size: int = 0,
):
    """50 steps on the GPU: the first real memory and throughput measurement.

    Gate 2 runs on a deliberately small slice here. At 50 steps it cannot
    pass honestly, and it is not meant to -- what is being tested is that the
    gate computes and reports, and that the control tokens are no longer
    identical. The real gate runs on the whole validation split.

    What this does NOT exercise: validation and checkpointing. Those are on a
    500-optimizer-step cadence (configs/train.yaml), so 50 steps reaches
    neither. That is the point of the cadence rather than a gap in the smoke
    run -- but it does mean the first run to prove a checkpoint gets written
    is the first run past step 500.

    The conditioning probe and the held-out test report are exercised, with
    the probe interval and both evaluation sizes cut to something a smoke run
    can afford. They are the two newest code paths, and both only otherwise
    run after hours of training -- the worst place to discover a typo.
    """
    _report(
        _train.with_options(gpu=gpu).remote(
            "a", steps, gate2_limit, None, batch_size or None,
            ["--probe-every", "20", "--probe-examples", "32"],
            "smoke", False,
        )
    )


@app.local_entrypoint()
def train(
    stage: str = "a",
    gpu: str = DEFAULT_GPU,
    max_steps: int = 0,
    gate2_limit: int = 0,
    init_from: str = "",
    batch_size: int = 0,
    run_name: str = "default",
    resume: bool = False,
):
    """A full run, started server-side. max_steps 0 means configs/train.yaml.

    ``spawn`` rather than ``remote``, and this is the whole reason this
    function is not a one-liner.

    ``remote`` blocks. The local client owns the in-flight input for as long
    as it waits, and when it goes away Modal cancels the input -- so the job
    dies with the laptop, the ssh session or the client's expiring auth.
    ``--detach`` does not fix that: it keeps the *app* alive, not the input.
    Two Stage A runs were lost this way, the first at step 314 and the second
    at step 2,149 of 20,000, both with "Received a cancellation signal" and
    nothing wrong in the training code.

    ``spawn`` creates the call server-side and returns an id. There is no
    client in the loop to disconnect, so nothing local can cancel it. Follow
    it with ``status`` below, or ``modal app logs``.
    """
    call = _train.with_options(gpu=gpu).spawn(
        stage,
        max_steps or None,
        gate2_limit or None,
        init_from or None,
        batch_size or None,
        [],
        run_name,
        resume,
    )
    print(f"spawned {call.object_id}")
    print("nothing local holds this job; it is safe to close the terminal.\n")
    print("follow it:")
    print(f"  modal run cloud/modal_app.py::status --call-id {call.object_id}")
    print(f"  modal run cloud/modal_app.py::ls --prefix checkpoints/stage_{stage}")
    print("\nif the container dies, the same command with --resume continues "
          "from the last checkpoint:")
    print(f"  modal run cloud/modal_app.py::train --stage {stage} "
          f"--run-name {run_name} --resume")


@app.local_entrypoint()
def status(call_id: str):
    """Has the spawned run finished, and what did it report?

    Polls once and exits rather than waiting, so this is safe to run from
    anywhere at any time without becoming the client that owns the job.
    """
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        print(f"{call_id} is still running")
        print("  logs:  modal app logs " + APP_NAME)
        return
    print(f"{call_id} finished")
    _report(result)


@app.local_entrypoint()
def ls(prefix: str = ""):
    """What is on the volume. Checkpoints land in checkpoints/."""
    for line in _list_volume.remote(prefix):
        print(line)


def _summary_lines(result: dict) -> str:
    headroom = result["total_memory_gb"] / max(result["peak_reserved_gb"], 0.01)
    return "\n".join([
        f"command      {' '.join(result['argv'])}",
        f"gpu          {result['gpu']}  ({result['total_memory_gb']} GB)",
        f"exit code    {result['exit_code']}",
        f"wall time    {result['seconds']} s, setup and Gate 2 included "
        "(the per-step rate is in the throughput line above)",
        f"peak memory  {result['peak_allocated_gb']} GB allocated, "
        f"{result['peak_reserved_gb']} GB reserved, of "
        f"{result['total_memory_gb']} GB ({headroom:.1f}x headroom)",
    ])


def _report(result: dict) -> None:
    print()
    print(_summary_lines(result))
    if result["exit_code"] != 0:
        print("\nthe run did not finish cleanly; exit 2 is a validation cadence "
              "mismatch, exit 3 is a failed gate")
