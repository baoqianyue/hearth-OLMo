#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${CLEAR_PROXY:-1}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

mkdir -p logs
LOG_FILE="${LOG_FILE:-logs/prepare_dolma3_150b_pilot.log}"
MAX_RETRIES="${MAX_RETRIES:-100}"
RETRY_SLEEP="${RETRY_SLEEP:-120}"
HF_FILE_MANIFEST="${HF_FILE_MANIFEST:-data/manifests/dolma3_150b_pilot_files.json}"
TOKENIZER="${TOKENIZER:-data/tokenizers/olmo3_1025_7b}"

exec >> "${LOG_FILE}" 2>&1

echo "[$(date -Is)] prepare_dolma3_150b_pilot start"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1000000000}"
echo "MAX_EVAL_TOKENS=${MAX_EVAL_TOKENS:-8388608}"
echo "HF_FILE_MANIFEST=${HF_FILE_MANIFEST}"
echo "TOKENIZER=${TOKENIZER}"
echo "CLEAR_PROXY=${CLEAR_PROXY:-1}"

attempt=1
while true; do
  echo "[$(date -Is)] attempt ${attempt}/${MAX_RETRIES}"
  if conda run --no-capture-output -n olmo python scripts/prepare_text_data.py \
    --preset dolma3_150b_pilot \
    --output-dir data/dolma3_150b_pilot \
    --tokenizer "${TOKENIZER}" \
    --hf-file-manifest "${HF_FILE_MANIFEST}" \
    --max-train-tokens "${MAX_TRAIN_TOKENS:-1000000000}" \
    --max-eval-tokens "${MAX_EVAL_TOKENS:-8388608}" \
    --hf-max-train-files "${HF_MAX_TRAIN_FILES:-256}" \
    --hf-max-eval-files "${HF_MAX_EVAL_FILES:-32}"; then
    echo "[$(date -Is)] prepare_dolma3_150b_pilot completed"
    exit 0
  fi

  if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
    echo "[$(date -Is)] prepare_dolma3_150b_pilot failed after ${MAX_RETRIES} attempts"
    exit 1
  fi

  echo "[$(date -Is)] attempt ${attempt} failed; retrying in ${RETRY_SLEEP}s"
  attempt=$((attempt + 1))
  sleep "${RETRY_SLEEP}"
done
