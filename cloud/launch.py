#!/usr/bin/env python3
"""Start and check a long training run against the *deployed* app.

Run with plain ``python3``, not ``modal run``. That is the whole point of the
file, and it took three dead runs to arrive at.

Why not ``modal run``. ``modal run`` creates an *ephemeral* app, whose
lifetime is the entrypoint's. Three things follow, and each one looks like a
different bug:

1. A blocking ``Function.remote()`` call is owned by the local client while it
   waits, and Modal cancels the in-flight input when that client goes away.
   This killed the first Stage A run at step 314.
2. ``--detach`` keeps the ephemeral *app* alive past the disconnect, but does
   nothing about the input being cancelled. This killed the second run at step
   2,149 of 20,000 -- "Received a cancellation signal", with nothing wrong in
   the training code.
3. ``Function.spawn()`` fixes the ownership, but under ``modal run`` the app
   still stops when the entrypoint returns, and takes the spawned call with
   it. This produced a run that emitted no output whatsoever: the only line in
   its logs was "Stopping app - local entrypoint completed".

A deployed app has no entrypoint and no client. It persists until it is
stopped, so a call spawned against it survives anything that happens locally:

    modal deploy cloud/modal_app.py
    python3 cloud/launch.py train --max-steps 20000
    python3 cloud/launch.py status --call-id fc-...

``modal run`` is still right for the short things -- ``check``, ``upload``,
``smoke``, ``ls`` -- where blocking is what you want.
"""

from __future__ import annotations

import argparse
import sys

import modal

APP_NAME = "thermocas-gen"
FUNCTION = "_train"


def _function(gpu: str):
    try:
        fn = modal.Function.from_name(APP_NAME, FUNCTION)
    except Exception as exc:  # noqa: BLE001 - message matters more than type
        print(
            f"could not find {FUNCTION} in the deployed app {APP_NAME!r}: {exc}\n"
            "Deploy it first:\n"
            "    modal deploy cloud/modal_app.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return fn.with_options(gpu=gpu) if gpu else fn


def train(args) -> int:
    # The ceiling sweep's variants travel as extra argv rather than as new
    # parameters on _train, because they are arguments to 11_train.py and
    # adding a parameter to the remote function means redeploying before any
    # of them can be used.
    extra: list[str] = []
    if args.bin_edges:
        extra += ["--bin-edges", str(args.bin_edges)]
    if args.ceiling_gain:
        extra += ["--ceiling-gain", str(args.ceiling_gain)]
    if args.bin_edges and not args.ceiling_gain:
        # 11_train.py refuses this too, but it refuses it on the GPU, after
        # the container has started and the checkpoint has downloaded. A
        # variant's ceiling is not the configured one.
        print(
            "--bin-edges needs --ceiling-gain: the variant's ceiling is not "
            "the configured one.\nGet it from "
            "results/ceiling_sweep_20261027/sweep.txt.",
            file=sys.stderr,
        )
        return 2
    if extra:
        print(f"variant:  {' '.join(extra)}")

    call = _function(args.gpu).spawn(
        args.stage,
        args.max_steps or None,
        args.gate2_limit or None,
        args.init_from or None,
        args.batch_size or None,
        extra,
        args.run_name,
        args.resume,
    )
    print(f"spawned {call.object_id}")
    print("against the deployed app, so nothing local holds it: closing this "
          "terminal, losing the network or letting auth expire will not stop it.\n")
    print("follow it:")
    print(f"  python3 cloud/launch.py status --call-id {call.object_id}")
    print(f"  modal app logs {APP_NAME}")
    print(f"  modal run cloud/modal_app.py::ls --prefix checkpoints/stage_{args.stage}")
    print("\nif the container dies, continue from the last checkpoint with:")
    print(f"  python3 cloud/launch.py train --stage {args.stage} "
          f"--run-name {args.run_name} --resume")
    return 0


def status(args) -> int:
    call = modal.FunctionCall.from_id(args.call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        print(f"{args.call_id} is still running")
        print(f"  logs: modal app logs {APP_NAME}")
        return 0
    except Exception as exc:  # noqa: BLE001
        # A terminated call reports an empty RemoteError, which is
        # indistinguishable from a crash without looking at the logs -- so say
        # that rather than guess.
        print(f"{args.call_id} did not finish: {type(exc).__name__}: {exc}")
        print(f"  check why: modal app logs {APP_NAME}")
        return 1
    print(f"{args.call_id} finished")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0 if result.get("exit_code") == 0 else int(result.get("exit_code", 1))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("train", help="spawn a run on the deployed app")
    run.add_argument("--stage", default="a", choices=["a", "b"])
    run.add_argument("--gpu", default="A10G")
    run.add_argument("--max-steps", type=int, default=0)
    run.add_argument("--gate2-limit", type=int, default=0)
    run.add_argument("--init-from", default="")
    run.add_argument("--batch-size", type=int, default=0)
    run.add_argument("--run-name", default="default")
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--bin-edges",
        default="",
        help="growth-temperature edges in C for a ceiling-sweep variant, "
             "e.g. 55 or 32,45,56. Relabels the corpus; splits are unchanged.",
    )
    run.add_argument(
        "--ceiling-gain",
        type=float,
        default=0.0,
        help="the variant's own ceiling as a fraction (0.00257 for +0.257%%), "
             "from results/ceiling_sweep_20261027/sweep.txt",
    )
    run.set_defaults(func=train)

    check = sub.add_parser("status", help="has a spawned run finished?")
    check.add_argument("--call-id", required=True)
    check.set_defaults(func=status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
