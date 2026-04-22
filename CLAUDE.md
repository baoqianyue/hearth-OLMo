# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

hearth-OLMo is a thin small-GPU (4x4090) training/evaluation harness wrapped around Allen AI's `OLMo-core` engine. The first milestone is reproducing the OLMo3 training + OLMES evaluation loop for 1B and 3B models. The 370M configs are retained only as legacy pipeline smoke tests.

**Do not vendor OLMo-core into this repo.** Treat `ai2-olmo-core` as an upstream dependency; put narrow fixes under `patches/olmo-core/`. See `docs/dependency_strategy.md`.

## Environment

All commands run under the `olmo` conda env (PyTorch 2.6.0 / CUDA 12.4):

```bash
conda activate olmo        # or prefix commands with: conda run -n olmo ...
```

OLMo-core source resolution order inside `scripts/train_olmo3.py`:
1. `$OLMO_CORE_SRC` (set this for development against a local OLMo-core checkout)
2. Sibling `../OLMo-core/src`
3. Installed `ai2-olmo-core` package

For China mainland networking when pulling HF models/datasets, set `HF_ENDPOINT=https://hf-mirror.com` and `HF_HOME=/home/ly/lowgpu_train/.cache/huggingface`. OLMES runs should also use a project-local datasets cache (`.cache/hf_datasets_olmes`) to avoid stale MMLU-style schema failures.

Known compat issue: upstream OLMo-core calls `DefaultSavePlanner(enable_plan_caching=...)`, which torch 2.6.0 doesn't accept. `patches/olmo-core/torch26-default-save-planner.patch` is the workaround applied to the sibling checkout.

## Core Workflows

The end-to-end pipeline is: **train (OLMo-core) → convert to HF → PPL eval → OLMES eval → collect results**. Use `scripts/run_reproduction.py` to orchestrate this consistently across `1b` / `3b` / `370m` / `official_7b`; it prints commands by default and only runs them with `--execute`.

```bash
# 1B Wikitext2 probe, full pipeline
conda run -n olmo python scripts/run_reproduction.py \
  --model-size 1b --skip-conversion-validation --olmes smoke --execute
```

### Training (scripts/train_olmo3.py)

Key flags: `--config <yaml>` (required), `--dry-run` (build config only), `--train-single` (single-GPU debug; disables FSDP `dp_config`). Multi-GPU uses `torch.distributed.run --nproc_per_node=N`.

### Data (scripts/prepare_text_data.py)

Generates pre-tokenized `.npy` files in OLMo-core's **flat raw-binary token format** (NOT a NumPy header `.npy`). Tokenizer defaults to Dolma2; pass `--tokenizer allenai/Olmo-3-1025-7B` when preparing eval tokens for the official 7B baseline. Never compare PPL across different tokenizers.

### Evaluation

- `scripts/convert_core_to_hf.py` — convert OLMo-core checkpoint dir to HF format.
- `scripts/eval_hf_model.py` — local Wikitext-2 PPL sanity. `--max-windows 0` = use all windows.
- `scripts/run_olmes.py` — wraps upstream OLMES. Falls back to `src/olmes` checkout when the `olmes` console entry point isn't installed. Suites: `smoke`, `base_easy`, `base`, `heldout`. `base_easy` expands to many subtasks and is substantially heavier than smoke.
- `scripts/collect_olmes_results.py` / `scripts/collect_results.py` — aggregate JSON outputs into CSV/Markdown under `reports/`.

## Repository Layout

- `configs/` — training YAMLs (`train_olmo3_{370m,1b,3b}*.yaml`) and OLMES task lists (`olmes_olmo3_*.txt`). `eval_common.yaml` holds shared eval defaults.
- `scripts/` — all CLIs. `train_olmo3.py` is intentionally thin; real logic lives in OLMo-core.
- `src/olmes`, `src/ai2-olmes` — OLMES checkouts used as editable installs.
- `patches/olmo-core/` — small compatibility patches for pinned OLMo-core versions.
- `outputs/` — OLMo-core checkpoint dirs and converted HF models (`outputs/hf/...`).
- `reports/` — PPL JSON, OLMES result trees, and aggregated CSV/Markdown tables.
- `data/` — pre-tokenized token-id binaries; placeholder paths in 1B/3B configs must be pointed at real Dolma/OLMo shards for non-probe runs.

## Conventions

- Don't hand-run the train/convert/eval scripts for reproduction work — extend `run_reproduction.py` instead so the 1B and 3B paths stay aligned.
- The 370M track is legacy smoke only; add new features to the 1B/3B path first.
- When editing training config YAMLs, remember OLMo-core consumes them via the `TransformerConfig` / `TrainerConfig` / `TransformerTrainModuleConfig` builders imported at the top of `scripts/train_olmo3.py`.
