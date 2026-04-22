#!/usr/bin/env python
"""Run OLMES evaluations with OLMo3-oriented presets.

This is a thin wrapper around the official `olmes` CLI. It keeps hearth-OLMo's
commands stable while leaving task definitions and metrics in upstream OLMES.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


OLMO3_SUITES = {
    "base_easy": [
        "olmo3:base_easy:code_bpb",
        "olmo3:base_easy:math_bpb",
        "olmo3:base_easy:qa_rc",
        "olmo3:base_easy:qa_bpb",
    ],
    "base": [
        "olmo3:base:stem_qa_mc",
        "olmo3:base:nonstem_qa_mc",
        "olmo3:base:gen",
        "olmo3:base:math",
        "olmo3:base:code",
        "olmo3:base:code_fim",
    ],
    "heldout": ["olmo3:heldout"],
}


def parse_model_args(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--model-args must be valid JSON: {exc}") from exc
    return json.dumps(parsed, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF model id or local HF model directory")
    parser.add_argument(
        "--suite",
        choices=sorted(OLMO3_SUITES),
        default="base_easy",
        help="Official OLMo3 OLMES suite preset.",
    )
    parser.add_argument(
        "--task",
        nargs="*",
        default=None,
        help="Explicit OLMES task names. Overrides --suite when provided.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/olmes"))
    parser.add_argument("--model-type", default=None, help="Forwarded to OLMES, e.g. hf or vllm.")
    parser.add_argument("--model-args", default=None, help="JSON string forwarded to OLMES.")
    parser.add_argument("--limit", default=None, help="Forwarded to OLMES for smoke runs.")
    parser.add_argument("--batch-size", default=None, help="Forwarded to OLMES.")
    parser.add_argument("--gpus", default=None, help="Forwarded to OLMES.")
    parser.add_argument("--dry-run", action="store_true", help="Print OLMES launch command without running tasks.")
    parser.add_argument("--inspect", action="store_true", help="Forward OLMES --inspect.")
    parser.add_argument("--list-tasks", nargs="?", const="", default=None, help="Forward OLMES --list-tasks.")
    parser.add_argument("--list-task-suites", nargs="?", const="", default=None, help="Forward OLMES --list-task-suites.")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument(
        "--hf-datasets-cache",
        type=Path,
        default=None,
        help="Set HF_DATASETS_CACHE for OLMES dataset downloads.",
    )
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra raw argument forwarded to OLMES.")
    args = parser.parse_args()

    olmes_bin = shutil.which("olmes")
    local_olmes = Path(__file__).resolve().parents[1] / "src" / "olmes"
    if olmes_bin is not None:
        cmd = [olmes_bin]
        python_path = None
    elif local_olmes.exists():
        cmd = [sys.executable, "-m", "oe_eval.launch"]
        python_path = str(local_olmes)
    else:
        raise SystemExit(
            "Could not find OLMES. Install it first, for example:\n"
            "  git clone https://github.com/allenai/olmes.git src/olmes\n"
            "or:\n"
            "  python -m pip install -e 'git+https://github.com/allenai/olmes.git#egg=ai2-olmes'"
        )

    if args.list_tasks is not None:
        cmd.extend(["--list-tasks", args.list_tasks])
        env = os.environ.copy()
        if python_path:
            env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")
        raise SystemExit(subprocess.call(cmd, env=env))

    if args.list_task_suites is not None:
        cmd.extend(["--list-task-suites", args.list_task_suites])
        env = os.environ.copy()
        if python_path:
            env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")
        raise SystemExit(subprocess.call(cmd, env=env))

    if not args.model:
        raise SystemExit("--model is required unless --list-tasks or --list-task-suites is used.")

    tasks = args.task if args.task else OLMO3_SUITES[args.suite]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd.extend(["--model", args.model, "--task", *tasks, "--output-dir", str(args.output_dir)])
    if args.model_type:
        cmd.extend(["--model-type", args.model_type])
    model_args = parse_model_args(args.model_args)
    if model_args:
        cmd.extend(["--model-args", model_args])
    if args.limit:
        cmd.extend(["--limit", args.limit])
    if args.batch_size:
        cmd.extend(["--batch-size", args.batch_size])
    if args.gpus:
        cmd.extend(["--gpus", args.gpus])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.inspect:
        cmd.append("--inspect")
    cmd.extend(args.extra_arg)

    env = os.environ.copy()
    if args.hf_endpoint:
        env.setdefault("HF_ENDPOINT", args.hf_endpoint)
    if args.hf_datasets_cache:
        args.hf_datasets_cache.mkdir(parents=True, exist_ok=True)
        env["HF_DATASETS_CACHE"] = str(args.hf_datasets_cache.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if python_path:
        env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")

    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
