#!/usr/bin/env python
"""Train small OLMo3 ladder models with OLMo-core.

This script is intentionally thin: it keeps OLMo-core as the training engine and
only adds YAML configuration, smoke-data generation, and local defaults suitable
for a 4x4090 workstation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rich
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OLMO_CORE_SRC = Path(os.environ.get("OLMO_CORE_SRC", WORKSPACE_ROOT / "OLMo-core" / "src"))
if OLMO_CORE_SRC.exists():
    sys.path.insert(0, str(OLMO_CORE_SRC.resolve()))

from olmo_core.config import DType  # noqa: E402
from olmo_core.data import (  # noqa: E402
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.distributed.utils import get_rank, is_distributed  # noqa: E402
from olmo_core.nn.attention import AttentionBackendName  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode  # noqa: E402
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride  # noqa: E402
from olmo_core.train import (  # noqa: E402
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (  # noqa: E402
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
)
from olmo_core.train.train_module import (  # noqa: E402
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all  # noqa: E402


log = logging.getLogger("hearth_olmo.train")


MODEL_FACTORIES = {
    "370M": "olmo3_370M",
    "1B": "olmo3_1B",
    "3B": "olmo3_3B",
    "olmo3_370M": "olmo3_370M",
    "olmo3_1B": "olmo3_1B",
    "olmo3_3B": "olmo3_3B",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return data


def _as_path(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def _dtype(name: str) -> DType:
    return {
        "float32": DType.float32,
        "fp32": DType.float32,
        "bfloat16": DType.bfloat16,
        "bf16": DType.bfloat16,
        "float16": DType.float16,
        "fp16": DType.float16,
    }[name.lower()]


def _attention_backend(name: str) -> AttentionBackendName:
    return {
        "sdpa": AttentionBackendName.torch,
        "torch": AttentionBackendName.torch,
        "flash_2": AttentionBackendName.flash_2,
        "flash-attn": AttentionBackendName.flash_2,
        "flash_attention_2": AttentionBackendName.flash_2,
    }[name.lower()]


def _tokenizer_config(identifier: str) -> TokenizerConfig:
    path = Path(identifier).expanduser()
    if path.exists():
        config_path = path / "config.json"
        if not config_path.exists():
            config_path = path / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json or tokenizer_config.json found under {path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return TokenizerConfig(
            vocab_size=config["vocab_size"],
            eos_token_id=config["eos_token_id"],
            pad_token_id=config.get("pad_token_id", config["eos_token_id"]),
            bos_token_id=config.get("bos_token_id"),
            identifier=str(path.resolve()),
        )

    name = identifier.lower()
    if name in {"dolma2", "allenai/dolma2-tokenizer"}:
        return TokenizerConfig.dolma2()
    if name in {"dolma2_sigdig", "allenai/dolma2-tokenizer-sigdig"}:
        return TokenizerConfig.dolma2_sigdig()
    if name in {"gpt_neox_olmo_dolma_v1_5", "allenai/gpt-neox-olmo-dolma-v1_5"}:
        return TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    if name == "gpt2":
        return TokenizerConfig.gpt2()
    return TokenizerConfig.from_hf(identifier)


def _maybe_generate_smoke_data(cfg: dict[str, Any], tokenizer: TokenizerConfig) -> tuple[list[str], list[str]]:
    synthetic = cfg.get("synthetic", {})
    if not synthetic.get("enabled", False):
        return list(cfg["train_paths"]), list(cfg["eval_paths"])

    out_dir = Path(synthetic.get("output_dir", REPO_ROOT / "data" / "smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(synthetic.get("seed", 17))
    num_train_tokens = int(synthetic.get("num_train_tokens", 1_048_576))
    num_eval_tokens = int(synthetic.get("num_eval_tokens", 131_072))
    vocab_limit = min(int(synthetic.get("vocab_limit", 32_000)), tokenizer.vocab_size - 1)
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32

    rng = np.random.default_rng(seed)
    train_path = out_dir / "train.npy"
    eval_path = out_dir / "eval.npy"

    def needs_write(path: Path) -> bool:
        if not path.exists():
            return True
        with path.open("rb") as f:
            return f.read(6) == b"\x93NUMPY"

    if needs_write(train_path):
        arr = rng.integers(0, vocab_limit, size=num_train_tokens, dtype=dtype)
        arr[::257] = tokenizer.eos_token_id
        arr.tofile(train_path)
    if needs_write(eval_path):
        arr = rng.integers(0, vocab_limit, size=num_eval_tokens, dtype=dtype)
        arr[::251] = tokenizer.eos_token_id
        arr.tofile(eval_path)
    return [_as_path(train_path)], [_as_path(eval_path)]


def build_components(raw: dict[str, Any]):
    run = raw.get("run", {})
    model_cfg = raw.get("model", {})
    data_cfg = raw.get("data", {})
    optim_cfg = raw.get("optim", {})
    train_cfg = raw.get("train", {})
    parallel_cfg = raw.get("parallel", {})
    eval_cfg = raw.get("eval", {})
    ckpt_cfg = raw.get("checkpoint", {})

    tokenizer = _tokenizer_config(str(data_cfg.get("tokenizer", "allenai/dolma2-tokenizer")))
    factory_name = MODEL_FACTORIES.get(str(model_cfg.get("size", "370M")), str(model_cfg.get("size", "370M")))
    try:
        factory = getattr(TransformerConfig, factory_name)
    except AttributeError as ex:
        raise ValueError(f"Unknown OLMo3 model factory: {factory_name}") from ex

    sequence_length = int(data_cfg.get("sequence_length", 1024))
    attn_backend = _attention_backend(str(model_cfg.get("attn_backend", "sdpa")))
    param_dtype = _dtype(str(model_cfg.get("param_dtype", "bfloat16")))
    model = factory(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=attn_backend,
        dtype=param_dtype,
    )

    train_paths, eval_paths = _maybe_generate_smoke_data(data_cfg, tokenizer)
    work_dir = _as_path(run.get("work_dir", REPO_ROOT / "outputs" / "work"))

    dataset = NumpyFSLDatasetConfig(
        paths=train_paths,
        metadata=[{"label": "train"} for _ in train_paths],
        sequence_length=sequence_length,
        max_target_sequence_length=int(data_cfg.get("max_target_sequence_length", sequence_length)),
        tokenizer=tokenizer,
        work_dir=work_dir,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=int(train_cfg.get("global_batch_tokens", sequence_length)),
        seed=int(run.get("seed", 1337)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    ac_config = None
    ac_mode = str(parallel_cfg.get("activation_checkpointing", "full")).lower()
    if ac_mode not in ("none", "false", "off"):
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
            if ac_mode == "full"
            else TransformerActivationCheckpointingMode.selected_blocks,
            block_interval=parallel_cfg.get("activation_checkpoint_block_interval"),
        )

    dp_config = None
    if bool(parallel_cfg.get("fsdp", True)):
        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=param_dtype,
            reduce_dtype=_dtype(str(parallel_cfg.get("reduce_dtype", "float32"))),
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            prefetch_factor=int(parallel_cfg.get("prefetch_factor", 0)),
        )

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=int(train_cfg.get("rank_microbatch_tokens", sequence_length)),
        max_sequence_length=sequence_length,
        optim=AdamWConfig(
            lr=float(optim_cfg.get("lr", 3.0e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.1)),
            betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
            eps=float(optim_cfg.get("eps", 1.0e-8)),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=int(optim_cfg.get("warmup_steps", 100))),
        compile_model=bool(parallel_cfg.get("compile_model", False)),
        dp_config=dp_config,
        ac_config=ac_config,
        z_loss_multiplier=float(train_cfg["z_loss_multiplier"])
        if "z_loss_multiplier" in train_cfg
        else None,
        max_grad_norm=float(optim_cfg.get("max_grad_norm", 1.0)),
    )

    save_folder = _as_path(run.get("save_folder", REPO_ROOT / "outputs" / run.get("name", "olmo3")))
    trainer = (
        TrainerConfig(
            save_folder=save_folder,
            work_dir=work_dir,
            save_overwrite=bool(run.get("save_overwrite", True)),
            metrics_collect_interval=int(run.get("metrics_collect_interval", 1)),
            cancel_check_interval=int(run.get("cancel_check_interval", 25)),
            max_duration=Duration.steps(int(train_cfg.get("max_steps", 20))),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=int(ckpt_cfg.get("save_interval", 100)),
                ephemeral_save_interval=ckpt_cfg.get("ephemeral_save_interval"),
                save_async=bool(ckpt_cfg.get("save_async", False)),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=bool(run.get("profile", False))))
    )

    if bool(eval_cfg.get("enabled", True)):
        trainer = trainer.with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    paths=eval_paths,
                    metadata=[{"label": eval_cfg.get("label", "validation")} for _ in eval_paths],
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    work_dir=work_dir,
                ),
                eval_interval=int(eval_cfg.get("interval", 100)),
                eval_duration=Duration.steps(int(eval_cfg.get("steps", 10))),
            ),
        )

    return model, dataset, data_loader, train_module, trainer, int(run.get("seed", 1337))


def _experiment_config(
    raw_cfg: dict[str, Any],
    model_config: TransformerConfig,
    dataset_config: NumpyFSLDatasetConfig,
    data_loader_config: NumpyDataLoaderConfig,
    train_module_config: TransformerTrainModuleConfig,
    trainer_config: TrainerConfig,
) -> dict[str, Any]:
    return {
        "model": model_config.as_config_dict(),
        "dataset": dataset_config.as_config_dict(),
        "data_loader": data_loader_config.as_config_dict(),
        "train_module": train_module_config.as_config_dict(),
        "trainer": trainer_config.as_config_dict(),
        "hearth": raw_cfg,
    }


def train(raw_cfg: dict[str, Any], dry_run: bool = False, train_single: bool = False) -> None:
    model_config, dataset_config, data_loader_config, train_module_config, trainer_config, seed = (
        build_components(raw_cfg)
    )

    if train_single and train_module_config.dp_config is not None:
        log.warning("--train-single requested; disabling FSDP dp_config")
        train_module_config.dp_config = None

    if dry_run:
        rich.print(
            {
                "model": model_config,
                "dataset": dataset_config,
                "data_loader": data_loader_config,
                "train_module": train_module_config,
                "trainer": trainer_config,
            }
        )
        return

    backend = None if train_single else ("cpu:gloo,cuda:nccl" if torch.cuda.is_available() else None)
    prepare_training_environment(shared_filesystem=True, backend=backend)
    try:
        seed_all(seed)
        model = model_config.build(init_device="meta")
        train_module = train_module_config.build(model)
        dataset = dataset_config.build()
        data_loader = data_loader_config.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = trainer_config.build(train_module, data_loader)

        for callback in trainer.callbacks.values():
            if isinstance(callback, ConfigSaverCallback):
                callback.config = _experiment_config(
                    raw_cfg,
                    model_config,
                    dataset_config,
                    data_loader_config,
                    train_module_config,
                    trainer_config,
                )
                break

        if get_rank() == 0:
            rich.print(raw_cfg)
        trainer.fit()
    finally:
        teardown_training_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-single", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw_cfg = _load_yaml(args.config)
    train(raw_cfg, dry_run=args.dry_run, train_single=args.train_single)


if __name__ == "__main__":
    main()
