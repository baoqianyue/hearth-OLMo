# hearth-OLMo

Local OLMo3 1B/3B training and evaluation harness for a 4x4090 workstation.

This repo intentionally keeps `OLMo-core` as the training engine. The first
milestone is to reproduce the OLMo3 training/evaluation loop for 1B and 3B
models before adding a CPU-master adapter. The earlier 370M configs remain as
legacy pipeline smoke tests, but they are not part of the main reproduction
track.

## Environment

The target environment is `olmo`:

```bash
conda activate olmo
```

This workspace has been verified with PyTorch `2.6.0` CUDA `12.4` in the
`olmo` conda environment, plus `accelerate` for Hugging Face `device_map`
loading. The scripts first look for `OLMO_CORE_SRC`, then for a sibling
`../OLMo-core/src` checkout, and otherwise use the installed `ai2-olmo-core`
package:

```bash
conda run -n olmo python -c "import torch, olmo_core; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), olmo_core.__file__)"
```

For development against a local OLMo-core checkout:

```bash
export OLMO_CORE_SRC=/home/ly/lowgpu_train/OLMo-core/src
```

For China mainland connectivity, prefer these environment variables when
downloading Hugging Face models:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/ly/lowgpu_train/.cache/huggingface
```

## Legacy Smoke Training

The 370M configs are retained only for quick pipeline checks. The main
reproduction track starts from 1B. The synthetic 370M config generates token-id
`.npy` files automatically, so it does not need external data:

```bash
cd /home/ly/lowgpu_train/hearth-OLMo
conda run -n olmo python scripts/train_olmo3.py \
  --config configs/train_olmo3_370m.yaml \
  --dry-run
```

Single-process debug run:

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run -n olmo python scripts/train_olmo3.py \
  --config configs/train_olmo3_370m.yaml \
  --train-single
```

Short 2-GPU FSDP verification run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
conda run -n olmo python -m torch.distributed.run --nproc_per_node=2 \
  scripts/train_olmo3.py \
  --config configs/train_olmo3_370m_fsdp_smoke.yaml
```

4-GPU FSDP run:

```bash
conda run -n olmo python -m torch.distributed.run --nproc_per_node=4 \
  scripts/train_olmo3.py \
  --config configs/train_olmo3_370m.yaml
```

Verified smoke outputs:

- Single GPU: `outputs/olmo3_370m_smoke_corecfg/step0`, `outputs/olmo3_370m_smoke_corecfg/step20`
- 2-GPU FSDP: `outputs/olmo3_370m_fsdp_smoke_corecfg/step0`, `outputs/olmo3_370m_fsdp_smoke_corecfg/step4`

Note: current `OLMo-core` calls a newer `DefaultSavePlanner(enable_plan_caching=...)`
argument that is not present in torch `2.6.0`. The sibling `OLMo-core` checkout
contains a small compatibility guard so checkpoint saving works on this machine.

## Dependency Strategy

Do not vendor OLMo-core into this repository for the first release. hearth-OLMo
is intended to be the small-GPU training/evaluation layer, while OLMo-core stays
the upstream engine. See `docs/dependency_strategy.md` for the decision record.

## Real Data

Prepare a small real-text sanity corpus with Dolma2 token IDs. The generated
files use OLMo-core's expected flat raw-binary token format, even though the
default filenames end in `.npy`.

```bash
HF_ENDPOINT=https://hf-mirror.com \
conda run -n olmo python scripts/prepare_text_data.py \
  --preset wikitext2 \
  --output-dir data/wikitext2_raw
```

The earlier 370M real-text pilot can still be run for debugging, but it is no
longer part of the recommended mainline:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
conda run -n olmo python -m torch.distributed.run --nproc_per_node=2 \
  scripts/train_olmo3.py \
  --config configs/train_olmo3_370m_wikitext2.yaml
```

For larger `1B` and `3B` runs, use the official Dolma 3 150B sample mix as the
first practical pilot corpus. This dataset keeps the same broad source mix
strategy as Dolma 3 while being much smaller than the full 6T-token mix.

Prepare a local token-id pilot corpus:

```bash
HF_ENDPOINT=https://hf-mirror.com \
conda run -n olmo python scripts/prepare_text_data.py \
  --preset dolma3_150b_pilot \
  --output-dir data/dolma3_150b_pilot \
  --tokenizer allenai/Olmo-3-1025-7B \
  --max-train-tokens 1000000000 \
  --max-eval-tokens 8388608
```

This preset reads Dolma 3 JSONL shards directly from Hugging Face, parses only
the `text` field, tokenizes with the OLMo3 tokenizer, and writes flat token-id
binary files. It records the selected source shards in
`data/dolma3_150b_pilot/manifest.json`. By default it excludes paths containing
`adult_content` and samples files round-robin across source directories.

The mainline configs point at these generated files:

- `configs/train_olmo3_1b.yaml`
- `configs/train_olmo3_3b.yaml`

The single-source OLMo data manifest is retained only for low-cost format smoke
tests:

```bash
conda run -n olmo python scripts/download_files.py \
  --manifest data/manifests/olmo3_dolma_smoke.json
```

Training files are flat raw binary arrays of token IDs produced with the
matching tokenizer, not NumPy `.npy` files with headers.

Official OLMo data examples are hosted under `https://olmo-data.org/`; if direct
download is slow, mirror the files manually into `data/` and point the configs at
the local paths.

Example resumable download:

```bash
conda run -n olmo python scripts/download_files.py \
  https://olmo-data.org/preprocessed/dolma3-0625/v0.1-official/allenai/dolma3-tokenizer/olmocr_science_pdfs/science_math_and_technology/000000.npy \
  --output data/olmo3_sample_000000.npy
```

## Evaluation

Convert a local OLMo-core checkpoint to HF format:

```bash
conda run -n olmo python scripts/convert_core_to_hf.py \
  --checkpoint outputs/olmo3_370m_wikitext2_raw/step100 \
  --output outputs/hf/olmo3_370m_wikitext2_step100
```

Evaluate an HF-format model. Use `--max-windows 0` for every available token
window:

```bash
conda run -n olmo python scripts/eval_hf_model.py \
  --model outputs/hf/olmo3_370m_wikitext2_step100 \
  --data data/wikitext2_raw/eval.npy \
  --output reports/olmo3_370m_wikitext2_step100_eval_full.json \
  --sequence-length 512 \
  --max-windows 0 \
  --batch-size 8
```

For the official 7B baseline, generate eval tokens with the official model
tokenizer first:

```bash
HF_ENDPOINT=https://hf-mirror.com \
conda run -n olmo python scripts/prepare_text_data.py \
  --preset wikitext2 \
  --output-dir data/wikitext2_olmo3_7b_tokenizer \
  --tokenizer allenai/Olmo-3-1025-7B

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
conda run -n olmo python scripts/eval_hf_model.py \
  --model allenai/Olmo-3-1025-7B \
  --data data/wikitext2_olmo3_7b_tokenizer/eval.npy \
  --output reports/olmo3_7b_official_wikitext2_eval_full.json \
  --sequence-length 512 \
  --max-windows 0 \
  --batch-size 1
```

Do not directly compare perplexity across different tokenizers. Use Wikitext-2
PPL as a local pipeline sanity check, and use OLMES task metrics for the main
1B/3B vs official 7B quality gap.

### Official OLMES Evaluation

OLMo3's official downstream evaluation stack is OLMES. hearth-OLMo wraps the
upstream OLMES task definitions instead of reimplementing task logic locally:

```bash
git clone https://github.com/allenai/olmes.git src/olmes
conda run -n olmo python -m pip install -e "src/olmes" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If installing the full package tries to upgrade PyTorch beyond the training
environment, keep the `src/olmes` checkout and install only the missing runtime
dependencies as needed. `scripts/run_olmes.py` automatically falls back to
`src/olmes` when an `olmes` console entry point is not installed.

Use a project-local datasets cache for OLMES runs. This avoids stale global
Hugging Face dataset metadata causing failures such as MMLU feature schema
mismatches:

```bash
mkdir -p .cache/hf_datasets_olmes
```

Available OLMo3 presets:

- `base_easy`: `olmo3:base_easy:code_bpb`, `math_bpb`, `qa_rc`, `qa_bpb`
- `base`: `olmo3:base:stem_qa_mc`, `nonstem_qa_mc`, `gen`, `math`, `code`, `code_fim`
- `heldout`: `olmo3:heldout`

Dry-run the official `base_easy` preset expansion:

```bash
HF_ENDPOINT=https://hf-mirror.com \
conda run -n olmo python scripts/run_olmes.py \
  --model outputs/hf/olmo3_1b_wikitext2_probe_step50 \
  --suite base_easy \
  --limit 2 \
  --batch-size 1 \
  --hf-datasets-cache .cache/hf_datasets_olmes \
  --dry-run
```

Run a single quick OLMES task for connectivity:

```bash
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
conda run -n olmo python scripts/run_olmes.py \
  --model outputs/hf/olmo3_370m_wikitext2_step100 \
  --task arc_easy:rc::olmes:full \
  --limit 8 \
  --batch-size 1 \
  --hf-datasets-cache .cache/hf_datasets_olmes \
  --output-dir reports/olmes/1b_olmo3_1b_wikitext2_probe_step50_smoke
```

Run the official OLMo3 `base_easy` suite. This is much heavier than the
single-task smoke because each suite expands to many OLMES subtasks:

```bash
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
conda run -n olmo python scripts/run_olmes.py \
  --model outputs/hf/olmo3_1b_wikitext2_probe_step50 \
  --suite base_easy \
  --batch-size 1 \
  --hf-datasets-cache .cache/hf_datasets_olmes \
  --output-dir reports/olmes/1b_base_easy
```

Collect OLMES task metrics:

```bash
conda run -n olmo python scripts/collect_olmes_results.py \
  reports/olmes/1b_base_easy \
  --csv reports/olmes/1b_base_easy.csv \
  --md reports/olmes/1b_base_easy.md
```

Collect JSON eval outputs into a table:

```bash
conda run -n olmo python scripts/collect_results.py \
  --inputs reports/*.json \
  --csv reports/olmo3_eval.csv \
  --md reports/olmo3_eval.md
```

Current verified outputs:

- 1B Wikitext2 probe: `outputs/olmo3_1b_wikitext2_probe/step50`
- 1B converted HF: `outputs/hf/olmo3_1b_wikitext2_probe_step50`
- 3B Wikitext2 probe: `outputs/olmo3_3b_wikitext2_probe/step20`
- 3B converted HF: `outputs/hf/olmo3_3b_wikitext2_probe_step20`
- 1B OLMES base_easy limit=8 table:
  `reports/olmes/1b_olmo3_1b_wikitext2_probe_step50_base_easy_l8_combined.md`
- PPL table: `reports/olmo3_ladder_ppl.md`
- OLMES smoke tables under `reports/olmes/`

## Recommended Order

1. Generate `data/dolma3_150b_pilot/train.npy` and `eval.npy`.
2. Run the `1B` mainline config on GPU0/1.
3. Convert the checkpoint and run PPL plus OLMES smoke.
4. Run the `3B` mainline config after the 1B path is stable.
5. Use `base_easy` selectively for checkpoints worth comparing.
6. Evaluate official `allenai/Olmo-3-1025-7B` after local 1B/3B runs stabilize.

## Reproduction Runner

Use `scripts/run_reproduction.py` to keep the training, conversion, PPL eval,
OLMES eval, and result collection commands consistent across `1b` and `3b`.
The script prints commands by default; add `--execute` to run them. `370m`
remains available as a legacy option for debugging.

Run the 1B mainline workflow:

```bash
conda run -n olmo python scripts/run_reproduction.py \
  --model-size 1b \
  --skip-conversion-validation \
  --olmes smoke \
  --execute
```

Run the 3B mainline workflow with conservative defaults:

```bash
conda run -n olmo python scripts/run_reproduction.py \
  --model-size 3b \
  --skip-conversion-validation \
  --olmes smoke \
  --execute
```

The completed Wikitext2 probe configs remain available as `1b_probe` and
`3b_probe`:

```bash
conda run -n olmo python scripts/run_reproduction.py \
  --model-size 1b_probe \
  --skip-conversion-validation \
  --olmes smoke
```

Plan the official 7B baseline against the same eval workflow:

```bash
conda run -n olmo python scripts/run_reproduction.py \
  --model-size official_7b \
  --stage ppl \
  --stage olmes \
  --eval-data data/wikitext2_olmo3_7b_tokenizer/eval.npy \
  --sequence-length 512 \
  --olmes smoke
```

For official OLMo3 task groups, replace `--olmes smoke` with
`--olmes base_easy`, `--olmes base`, or `--olmes heldout`. Start with
`base_easy`; it expands to many subtasks and is much heavier than the smoke.
