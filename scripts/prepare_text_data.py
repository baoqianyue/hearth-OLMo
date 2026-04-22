#!/usr/bin/env python
"""Prepare text corpora as flat Dolma2 token-id numpy files.

The output format matches OLMo-core's NumpyFSLDatasetConfig: a flat .npy array
of token IDs with EOS separators between documents.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from transformers import AutoTokenizer


WIKITEXT2_FILES = {
    "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "eval": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
}


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with output.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def prepare_wikitext2(raw_dir: Path, hf_endpoint: str) -> tuple[list[Path], list[Path]]:
    base = hf_endpoint.rstrip("/") + "/datasets/Salesforce/wikitext/resolve/main"
    paths: dict[str, Path] = {}
    for split, rel_path in WIKITEXT2_FILES.items():
        out = raw_dir / rel_path
        download(f"{base}/{rel_path}", out)
        paths[split] = out
    return [paths["train"]], [paths["eval"]]


def iter_texts(path: Path, field: str) -> Iterable[str]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=[field])
        for value in df[field].dropna():
            text = str(value).strip()
            if text:
                yield text
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    text = str(json.loads(line).get(field, "")).strip()
                    if text:
                        yield text
    elif path.suffixes[-2:] == [".jsonl", ".gz"]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    text = str(json.loads(line).get(field, "")).strip()
                    if text:
                        yield text
    else:
        with path.open("r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                yield text


def tokenize_files(
    paths: list[Path],
    output: Path,
    tokenizer_id: str,
    field: str,
    max_docs: int | None,
    max_tokens: int | None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    eos_id = int(tokenizer.eos_token_id)
    token_chunks: list[np.ndarray] = []
    total_tokens = 0
    docs = 0

    for path in paths:
        for text in iter_texts(path, field):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            ids.append(eos_id)
            arr = np.asarray(ids, dtype=np.uint32)
            if max_tokens is not None and total_tokens + len(arr) > max_tokens:
                keep = max_tokens - total_tokens
                if keep <= 0:
                    break
                arr = arr[:keep]
            token_chunks.append(arr)
            total_tokens += int(arr.size)
            docs += 1
            if (max_docs is not None and docs >= max_docs) or (
                max_tokens is not None and total_tokens >= max_tokens
            ):
                break
        if (max_docs is not None and docs >= max_docs) or (
            max_tokens is not None and total_tokens >= max_tokens
        ):
            break

    if not token_chunks:
        raise RuntimeError(f"No text tokens produced for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(token_chunks).tofile(output)
    return {"output": str(output), "docs": docs, "tokens": total_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["wikitext2"], default=None)
    parser.add_argument("--train-input", type=Path, nargs="*", default=[])
    parser.add_argument("--eval-input", type=Path, nargs="*", default=[])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/wikitext2_raw"))
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    parser.add_argument("--field", default="text")
    parser.add_argument("--max-train-docs", type=int, default=None)
    parser.add_argument("--max-eval-docs", type=int, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-eval-tokens", type=int, default=None)
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    train_inputs = args.train_input
    eval_inputs = args.eval_input
    if args.preset == "wikitext2":
        train_inputs, eval_inputs = prepare_wikitext2(args.raw_dir, args.hf_endpoint)

    if not train_inputs or not eval_inputs:
        raise ValueError("Provide --preset or both --train-input and --eval-input")

    train_stats = tokenize_files(
        train_inputs,
        args.output_dir / "train.npy",
        args.tokenizer,
        args.field,
        args.max_train_docs,
        args.max_train_tokens,
    )
    eval_stats = tokenize_files(
        eval_inputs,
        args.output_dir / "eval.npy",
        args.tokenizer,
        args.field,
        args.max_eval_docs,
        args.max_eval_tokens,
    )
    manifest = {"train": train_stats, "eval": eval_stats, "tokenizer": args.tokenizer}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
