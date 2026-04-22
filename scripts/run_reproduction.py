#!/usr/bin/env python
"""Run the hearth-OLMo reproduction workflow.

By default this script prints the commands it would run. Pass --execute to
launch them.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    # 370M is kept as a legacy pipeline smoke target. The main reproduction
    # line focuses on 1B and 3B.
    "370m": REPO_ROOT / "configs" / "train_olmo3_370m_wikitext2.yaml",
    "1b": REPO_ROOT / "configs" / "train_olmo3_1b_wikitext2_probe.yaml",
    "3b": REPO_ROOT / "configs" / "train_olmo3_3b_wikitext2_probe.yaml",
}

OFFICIAL_BASELINES = {
    "official_7b": "allenai/Olmo-3-1025-7B",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def step_number(path: Path) -> int:
    match = re.fullmatch(r"step(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def latest_checkpoint(save_folder: Path) -> Path | None:
    if not save_folder.exists():
        return None
    steps = [path for path in save_folder.iterdir() if path.is_dir() and step_number(path) >= 0]
    if not steps:
        return None
    return max(steps, key=step_number)


def planned_checkpoint(save_folder: Path, cfg: dict) -> Path:
    max_steps = int(cfg.get("train", {}).get("max_steps", 0))
    return save_folder / f"step{max_steps}"


def command_str(cmd: Iterable[str], env: dict[str, str] | None = None) -> str:
    prefix = ""
    if env:
        prefix = " ".join(f"{key}={value}" for key, value in sorted(env.items())) + " "
    return prefix + " ".join(cmd)


def run_or_print(cmd: list[str], *, env: dict[str, str], execute: bool) -> None:
    shown_env = {
        key: value
        for key, value in env.items()
        if key in {"CUDA_VISIBLE_DEVICES", "HF_ENDPOINT", "TOKENIZERS_PARALLELISM"}
    }
    print(command_str(cmd, shown_env))
    if execute:
        merged_env = os.environ.copy()
        merged_env.update(env)
        subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)


def train_cmd(config: Path, nproc_per_node: int, train_single: bool) -> list[str]:
    if train_single:
        return [sys.executable, "scripts/train_olmo3.py", "--config", str(config), "--train-single"]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        "scripts/train_olmo3.py",
        "--config",
        str(config),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-size",
        choices=sorted([*CONFIGS, *OFFICIAL_BASELINES]),
        default="1b",
        help="Model ladder point or official baseline.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "train", "convert", "ppl", "olmes", "collect"],
        action="append",
        default=None,
        help="Workflow stage. Repeat to run multiple stages. Default: all for local models, ppl+olmes for official baselines.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Override local training config.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override OLMo-core checkpoint directory.")
    parser.add_argument("--hf-output", type=Path, default=None, help="Override converted HF output directory.")
    parser.add_argument("--eval-data", type=Path, nargs="+", default=None, help="Token-id eval files for PPL.")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=128)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument("--data-dtype", default="uint32", choices=["uint16", "uint32", "uint64"])
    parser.add_argument(
        "--olmes",
        choices=["none", "smoke", "base_easy", "base", "heldout"],
        default="smoke",
        help="OLMES stage type. smoke runs a single ARC-Easy task.",
    )
    parser.add_argument("--olmes-limit", default="8", help="Limit for OLMES smoke/evaluation runs.")
    parser.add_argument("--olmes-batch-size", default="1")
    parser.add_argument(
        "--olmes-hf-datasets-cache",
        type=Path,
        default=REPO_ROOT / ".cache" / "hf_datasets_olmes",
        help="HF datasets cache used by OLMES. Keep this project-local to avoid stale global cache metadata.",
    )
    parser.add_argument(
        "--skip-conversion-validation",
        dest="skip_conversion_validation",
        action="store_true",
        default=True,
        help="Forward --skip-validation to the OLMo-core HF conversion script. This is the default for 1B/3B.",
    )
    parser.add_argument(
        "--validate-conversion",
        dest="skip_conversion_validation",
        action="store_false",
        help="Run OLMo-core's HF conversion validation instead of skipping it.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--train-single", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--execute", action="store_true", help="Actually run commands instead of printing them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    is_official = args.model_size in OFFICIAL_BASELINES

    config = args.config or CONFIGS.get(args.model_size)
    cfg = load_yaml(config) if config else {}
    run_name = cfg.get("run", {}).get("name", args.model_size)
    save_folder = Path(cfg.get("run", {}).get("save_folder", REPO_ROOT / "outputs" / run_name))
    checkpoint = args.checkpoint or latest_checkpoint(save_folder)
    if checkpoint is None and not is_official:
        checkpoint = planned_checkpoint(save_folder, cfg)

    hf_model: str | Path
    if is_official:
        hf_model = OFFICIAL_BASELINES[args.model_size]
    else:
        if args.hf_output:
            hf_model = args.hf_output
        elif checkpoint is not None:
            hf_model = REPO_ROOT / "outputs" / "hf" / f"{run_name}_{checkpoint.name}"
        else:
            hf_model = REPO_ROOT / "outputs" / "hf" / run_name

    eval_data = args.eval_data or [Path(path) for path in cfg.get("data", {}).get("eval_paths", [])]
    sequence_length = args.sequence_length or int(cfg.get("data", {}).get("sequence_length", 1024))
    stages = args.stage
    if stages is None:
        stages = ["ppl", "olmes"] if is_official else ["train", "convert", "ppl", "olmes", "collect"]
    if "all" in stages:
        stages = ["train", "convert", "ppl", "olmes", "collect"]

    env = {
        "HF_ENDPOINT": args.hf_endpoint,
        "TOKENIZERS_PARALLELISM": "false",
    }

    print(f"# hearth-OLMo workflow: {args.model_size}")
    print(f"# execute: {args.execute}")

    if "train" in stages:
        if is_official:
            raise SystemExit("The train stage is only valid for local ladder models.")
        env_train = dict(env)
        env_train["CUDA_VISIBLE_DEVICES"] = args.gpus
        run_or_print(train_cmd(config, args.nproc_per_node, args.train_single), env=env_train, execute=args.execute)
        if args.execute and args.checkpoint is None:
            checkpoint = latest_checkpoint(save_folder)
            if checkpoint is None:
                raise SystemExit(f"Training finished but no checkpoint was found under {save_folder}.")
            if args.hf_output is None:
                hf_model = REPO_ROOT / "outputs" / "hf" / f"{run_name}_{checkpoint.name}"

    if "convert" in stages:
        if is_official:
            raise SystemExit("The convert stage is only valid for local ladder models.")
        if checkpoint is None:
            raise SystemExit(f"No checkpoint found under {save_folder}; pass --checkpoint.")
        convert_cmd = [
            sys.executable,
            "scripts/convert_core_to_hf.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(hf_model),
        ]
        if args.skip_conversion_validation:
            convert_cmd.append("--extra-arg=--skip-validation")
        run_or_print(
            convert_cmd,
            env=env,
            execute=args.execute,
        )

    ppl_output = REPO_ROOT / "reports" / f"{args.model_size}_{Path(str(hf_model)).name}_ppl.json"
    if "ppl" in stages:
        if not eval_data:
            raise SystemExit("No eval data paths found; pass --eval-data.")
        env_ppl = dict(env)
        env_ppl["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[0]
        run_or_print(
            [
                sys.executable,
                "scripts/eval_hf_model.py",
                "--model",
                str(hf_model),
                "--data",
                *[str(path) for path in eval_data],
                "--output",
                str(ppl_output),
                "--sequence-length",
                str(sequence_length),
                "--max-windows",
                str(args.max_windows),
                "--batch-size",
                str(args.ppl_batch_size),
                "--data-dtype",
                args.data_dtype,
            ],
            env=env_ppl,
            execute=args.execute,
        )

    olmes_output = REPO_ROOT / "reports" / "olmes" / f"{args.model_size}_{Path(str(hf_model)).name}_{args.olmes}"
    if "olmes" in stages and args.olmes != "none":
        env_olmes = dict(env)
        env_olmes["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[0]
        if args.olmes == "smoke":
            olmes_args = ["--task", "arc_easy:rc::olmes:full", "--limit", args.olmes_limit]
        else:
            olmes_args = ["--suite", args.olmes]
            if args.olmes_limit:
                olmes_args.extend(["--limit", args.olmes_limit])
        run_or_print(
            [
                sys.executable,
                "scripts/run_olmes.py",
                "--model",
                str(hf_model),
                *olmes_args,
                "--batch-size",
                args.olmes_batch_size,
                "--hf-datasets-cache",
                str(args.olmes_hf_datasets_cache),
                "--output-dir",
                str(olmes_output),
            ],
            env=env_olmes,
            execute=args.execute,
        )

    if "collect" in stages:
        inputs = sorted(REPO_ROOT.glob("reports/*_ppl.json"))
        if inputs:
            run_or_print(
                [
                    sys.executable,
                    "scripts/collect_results.py",
                    "--inputs",
                    *[str(path) for path in inputs],
                    "--csv",
                    str(REPO_ROOT / "reports" / "olmo3_ladder_ppl.csv"),
                    "--md",
                    str(REPO_ROOT / "reports" / "olmo3_ladder_ppl.md"),
                ],
                env=env,
                execute=args.execute,
            )
        if olmes_output.exists() or not args.execute:
            run_or_print(
                [
                    sys.executable,
                    "scripts/collect_olmes_results.py",
                    str(olmes_output),
                    "--csv",
                    str(olmes_output.with_suffix(".csv")),
                    "--md",
                    str(olmes_output.with_suffix(".md")),
                ],
                env=env,
                execute=args.execute,
            )


if __name__ == "__main__":
    main()
