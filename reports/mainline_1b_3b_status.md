# hearth-OLMo Mainline Status

Main reproduction targets: OLMo3 1B and 3B.

370M is retained as a legacy pipeline smoke target only.

## Completed Probe Runs

| target | checkpoint | HF model | train GPUs | steps | eval tokens | PPL | OLMES smoke |
|---|---|---|---:|---:|---:|---:|---:|
| 1B | `outputs/olmo3_1b_wikitext2_probe/step50` | `outputs/hf/olmo3_1b_wikitext2_probe_step50` | 0,1 | 50 | 65,536 | 2862.78 | 0.375 |
| 3B | `outputs/olmo3_3b_wikitext2_probe/step20` | `outputs/hf/olmo3_3b_wikitext2_probe_step20` | 0,1 | 20 | 65,536 | 8842.17 | 0.25 |

OLMES smoke task: `arc_easy:rc::olmes:full`, `--limit 8`, primary metric
`acc_per_char`.

## OLMES Base-Easy Status

1B `base_easy --limit 8` completed for
`outputs/hf/olmo3_1b_wikitext2_probe_step50`.

Output directories:

- `reports/olmes/1b_olmo3_1b_wikitext2_probe_step50_base_easy_l8_clean`
- `reports/olmes/1b_olmo3_1b_wikitext2_probe_step50_base_easy_l8_resume1`
- `reports/olmes/1b_olmo3_1b_wikitext2_probe_step50_base_easy_l8_resume2`

Completeness check: `192` metric files, `192` unique task aliases, `0`
duplicate aliases.

Combined outputs:

- `reports/olmes/1b_olmo3_1b_wikitext2_probe_step50_base_easy_l8_combined.csv`
- `reports/olmes/1b_olmo3_1b_wikitext2_probe_step50_base_easy_l8_combined.md`

Combined mean primary score: `2.63009`.

## Next Mainline Steps

1. Generate the Dolma 3 150B pilot token files:
   `data/dolma3_150b_pilot/train.npy` and
   `data/dolma3_150b_pilot/eval.npy`. Use the `dolma3_150b_pilot` preset in
   `scripts/prepare_text_data.py`; it reads only the `text` field and records
   the selected shards in `data/dolma3_150b_pilot/manifest.json`. For unstable
   network runs, use `scripts/run_prepare_dolma3_150b_pilot.sh`; it uses the
   pinned shard list in `data/manifests/dolma3_150b_pilot_files.json` and
   writes resumable state files.
2. Run the 1B mainline config:
   `configs/train_olmo3_1b.yaml`.
3. Increase 1B/3B training budgets with the same runner and collect comparable
   OLMES tables.
4. Keep 3B `base_easy` eval paused until the 1B reporting path is finalized.
5. Run the official `allenai/Olmo-3-1025-7B` comparison after the local
   1B/3B training path stabilizes.

## Commands

1B probe:

```bash
conda run -n olmo python scripts/run_reproduction.py \
  --model-size 1b \
  --nproc-per-node 2 \
  --gpus 0,1 \
  --skip-conversion-validation \
  --olmes smoke \
  --execute
```

3B probe:

```bash
conda run -n olmo python scripts/run_reproduction.py \
  --model-size 3b \
  --nproc-per-node 2 \
  --gpus 0,1 \
  --skip-conversion-validation \
  --olmes smoke \
  --execute
```
