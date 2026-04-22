#!/usr/bin/env python
"""Evaluate a Hugging Face causal LM on token-id numpy files.

Use this for official OLMo3 checkpoints and for OLMo-core checkpoints after
conversion to Hugging Face format.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def load_token_array(path: Path, dtype: str):
    with path.open("rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, mode="r", dtype=np.dtype(dtype))


def iter_token_windows(
    paths: list[Path], sequence_length: int, max_windows: int | None, data_dtype: str
):
    emitted = 0
    for path in paths:
        arr = load_token_array(path, data_dtype)
        usable = (len(arr) - 1) // sequence_length
        for idx in range(usable):
            if max_windows is not None and emitted >= max_windows:
                return
            start = idx * sequence_length
            x = np.asarray(arr[start : start + sequence_length + 1], dtype=np.int64)
            yield torch.from_numpy(x[:-1].copy()), torch.from_numpy(x[1:].copy())
            emitted += 1


def iter_batches(windows, batch_size: int):
    batch_inputs = []
    batch_labels = []
    for input_ids, labels in windows:
        batch_inputs.append(input_ids)
        batch_labels.append(labels)
        if len(batch_inputs) == batch_size:
            yield torch.stack(batch_inputs), torch.stack(batch_labels)
            batch_inputs = []
            batch_labels = []
    if batch_inputs:
        yield torch.stack(batch_inputs), torch.stack(batch_labels)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    windows = 0
    max_windows = None if args.max_windows <= 0 else args.max_windows
    token_windows = iter_token_windows(
        args.data, args.sequence_length, max_windows, args.data_dtype
    )
    for input_ids, labels in iter_batches(
        token_windows, batch_size=max(1, args.batch_size)
    ):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        out = model(input_ids=input_ids, labels=labels)
        n_tokens = labels.numel()
        total_loss += float(out.loss.detach().cpu()) * n_tokens
        total_tokens += n_tokens
        windows += labels.shape[0]

    if total_tokens == 0:
        raise RuntimeError("No evaluation tokens found. Check --data and --sequence-length.")

    ce = total_loss / total_tokens
    return {
        "model": args.model,
        "data": [str(p) for p in args.data],
        "sequence_length": args.sequence_length,
        "windows": windows,
        "tokens": total_tokens,
        "ce_loss": ce,
        "ppl": math.exp(ce) if ce < 20 else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local model directory")
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="Token-id .npy files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=128, help="Use 0 or a negative value to evaluate every available window.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--data-dtype", default="uint32", choices=["uint16", "uint32", "uint64"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Use https://hf-mirror.com by default for China mainland connectivity.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate(args)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
