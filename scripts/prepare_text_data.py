#!/usr/bin/env python
"""Prepare text corpora as flat token-id numpy files.

The output format matches OLMo-core's NumpyFSLDatasetConfig: a flat .npy array
of token IDs with EOS separators between documents.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

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


def iter_hf_texts(
    dataset: str,
    split: str,
    field: str,
    *,
    data_files: list[str] | None,
    streaming: bool,
    skip_docs: int,
) -> Iterable[str]:
    from datasets import Features, Value, load_dataset

    ds = load_dataset(
        dataset,
        split=split,
        data_files=data_files or None,
        features=Features({field: Value("string")}),
        streaming=streaming,
    )
    for idx, row in enumerate(ds):
        if idx < skip_docs:
            continue
        text = str(row.get(field, "")).strip()
        if text:
            yield text


def _hf_file_url(endpoint: str, dataset: str, path: str, revision: str) -> str:
    return f"{endpoint.rstrip('/')}/datasets/{dataset}/resolve/{revision}/{path}"


def _iter_zst_lines(raw: BinaryIO) -> Iterator[str]:
    import zstandard as zstd

    reader = zstd.ZstdDecompressor().stream_reader(raw)
    buffer = b""
    while True:
        chunk = reader.read(8 * 1024 * 1024)
        if not chunk:
            break
        buffer += chunk
        while True:
            pos = buffer.find(b"\n")
            if pos < 0:
                break
            line = buffer[:pos]
            buffer = buffer[pos + 1 :]
            if line:
                yield line.decode("utf-8")
    if buffer:
        yield buffer.decode("utf-8")


def iter_jsonl_url_texts(url: str, field: str, *, timeout: int = 60) -> Iterable[str]:
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        if url.endswith(".zst"):
            lines = _iter_zst_lines(resp.raw)
        elif url.endswith(".gz"):
            lines = gzip.open(resp.raw, "rt", encoding="utf-8")  # type: ignore[arg-type]
        else:
            lines = (line.decode("utf-8") for line in resp.iter_lines() if line)
        for line in lines:
            row = json.loads(line)
            text = str(row.get(field, "")).strip()
            if text:
                yield text


def list_hf_data_files(
    dataset: str,
    *,
    endpoint: str,
    include: list[str],
    exclude: list[str],
    max_files: int | None,
) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi(endpoint=endpoint).list_repo_files(dataset, repo_type="dataset")
    selected = [
        path
        for path in files
        if path.endswith((".jsonl.zst", ".jsonl.gz", ".jsonl"))
        and all(pattern in path for pattern in include)
        and not any(pattern in path for pattern in exclude)
    ]
    by_parent: dict[str, list[str]] = defaultdict(list)
    for path in sorted(selected):
        by_parent[str(Path(path).parent)].append(path)
    selected = []
    parents = sorted(by_parent)
    while parents:
        next_parents = []
        for parent in parents:
            paths = by_parent[parent]
            if paths:
                selected.append(paths.pop(0))
            if paths:
                next_parents.append(parent)
        parents = next_parents
        if max_files is not None and len(selected) >= max_files:
            selected = selected[:max_files]
            break
    if max_files is not None:
        selected = selected[:max_files]
    if not selected:
        raise RuntimeError(f"No data files matched in dataset {dataset}")
    return selected


def iter_hf_direct_texts(
    dataset: str,
    files: list[str],
    field: str,
    *,
    endpoint: str,
    revision: str,
    skip_docs: int,
) -> Iterable[str]:
    seen = 0
    for path in files:
        url = _hf_file_url(endpoint, dataset, path, revision)
        for text in iter_jsonl_url_texts(url, field):
            if seen < skip_docs:
                seen += 1
                continue
            seen += 1
            yield text


def tokenize_texts(
    texts: Iterable[str],
    output: Path,
    tokenizer_id: str,
    max_docs: int | None,
    max_tokens: int | None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    eos_id = int(tokenizer.eos_token_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    docs = 0

    with output.open("wb") as f:
        for text in texts:
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
            arr.tofile(f)
            total_tokens += int(arr.size)
            docs += 1
            if (max_docs is not None and docs >= max_docs) or (
                max_tokens is not None and total_tokens >= max_tokens
            ):
                break

    if total_tokens == 0:
        raise RuntimeError(f"No text tokens produced for {output}")
    return {"output": str(output), "docs": docs, "tokens": total_tokens}


def _append_file(src: Path, dst: Path) -> None:
    with src.open("rb") as fsrc, dst.open("ab") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _tokenize_one_hf_file(
    *,
    dataset: str,
    file_path: str,
    field: str,
    endpoint: str,
    revision: str,
    tokenizer,
    eos_id: int,
    output: Path,
    max_tokens: int | None,
    current_tokens: int,
) -> dict:
    docs = 0
    tokens = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        url = _hf_file_url(endpoint, dataset, file_path, revision)
        for text in iter_jsonl_url_texts(url, field):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            ids.append(eos_id)
            arr = np.asarray(ids, dtype=np.uint32)
            if max_tokens is not None and current_tokens + tokens + len(arr) > max_tokens:
                keep = max_tokens - current_tokens - tokens
                if keep <= 0:
                    break
                arr = arr[:keep]
            arr.tofile(f)
            tokens += int(arr.size)
            docs += 1
            if max_tokens is not None and current_tokens + tokens >= max_tokens:
                break
    return {"docs": docs, "tokens": tokens}


def tokenize_hf_direct_files(
    *,
    dataset: str,
    files: list[str],
    field: str,
    endpoint: str,
    revision: str,
    output: Path,
    tokenizer,
    max_tokens: int | None,
    max_docs: int | None,
    state_path: Path,
    resume: bool,
) -> dict:
    if max_docs is not None:
        raise ValueError("--max-*-docs is not supported with resumable --hf-direct mode")

    eos_id = int(tokenizer.eos_token_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output.parent / ".parts" / output.stem
    part_dir.mkdir(parents=True, exist_ok=True)

    state = _read_state(state_path) if resume else {}
    if not state:
        if output.exists():
            output.unlink()
        state = {
            "output": str(output),
            "next_file_index": 0,
            "docs": 0,
            "tokens": 0,
            "completed_files": [],
        }

    next_idx = int(state.get("next_file_index", 0))
    docs = int(state.get("docs", 0))
    tokens = int(state.get("tokens", 0))
    if max_tokens is not None and tokens >= max_tokens:
        return {"output": str(output), "docs": docs, "tokens": tokens}

    for idx in range(next_idx, len(files)):
        file_path = files[idx]
        part_path = part_dir / f"{idx:06d}.bin"
        tmp_part = part_path.with_suffix(".tmp")
        if tmp_part.exists():
            tmp_part.unlink()
        result = _tokenize_one_hf_file(
            dataset=dataset,
            file_path=file_path,
            field=field,
            endpoint=endpoint,
            revision=revision,
            tokenizer=tokenizer,
            eos_id=eos_id,
            output=tmp_part,
            max_tokens=max_tokens,
            current_tokens=tokens,
        )
        if result["tokens"] <= 0:
            tmp_part.unlink(missing_ok=True)
            state["next_file_index"] = idx + 1
            _write_state(state_path, state)
            continue
        tmp_part.replace(part_path)
        _append_file(part_path, output)
        docs += int(result["docs"])
        tokens += int(result["tokens"])
        state = {
            "output": str(output),
            "next_file_index": idx + 1,
            "docs": docs,
            "tokens": tokens,
            "completed_files": [*state.get("completed_files", []), file_path],
        }
        _write_state(state_path, state)
        print(
            json.dumps(
                {
                    "split_output": str(output),
                    "file_index": idx,
                    "file": file_path,
                    "docs": docs,
                    "tokens": tokens,
                }
            ),
            flush=True,
        )
        if max_tokens is not None and tokens >= max_tokens:
            break

    if tokens == 0:
        raise RuntimeError(f"No text tokens produced for {output}")
    return {"output": str(output), "docs": docs, "tokens": tokens}


def tokenize_files(
    paths: list[Path],
    output: Path,
    tokenizer_id: str,
    field: str,
    max_docs: int | None,
    max_tokens: int | None,
) -> dict:
    def all_texts() -> Iterable[str]:
        for path in paths:
            yield from iter_texts(path, field)

    return tokenize_texts(all_texts(), output, tokenizer_id, max_docs, max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["wikitext2", "dolma3_150b_pilot"], default=None)
    parser.add_argument("--train-input", type=Path, nargs="*", default=[])
    parser.add_argument("--eval-input", type=Path, nargs="*", default=[])
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-direct", action="store_true")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-file-manifest", type=Path, default=None)
    parser.add_argument("--hf-train-data-files", nargs="*", default=None)
    parser.add_argument("--hf-eval-data-files", nargs="*", default=None)
    parser.add_argument("--hf-file-include", nargs="*", default=[])
    parser.add_argument("--hf-file-exclude", nargs="*", default=["adult_content"])
    parser.add_argument("--hf-max-train-files", type=int, default=64)
    parser.add_argument("--hf-max-eval-files", type=int, default=8)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--hf-train-skip-docs", type=int, default=0)
    parser.add_argument("--hf-eval-skip-docs", type=int, default=200_000)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/wikitext2_raw"))
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    parser.add_argument("--field", default="text")
    parser.add_argument("--max-train-docs", type=int, default=None)
    parser.add_argument("--max-eval-docs", type=int, default=None)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-eval-tokens", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
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
    elif args.preset == "dolma3_150b_pilot":
        args.hf_dataset = args.hf_dataset or "allenai/dolma3_mix-150B-1025"
        args.hf_direct = True

    train_files = None
    eval_files = None
    if args.hf_dataset:
        if args.hf_direct:
            file_manifest_path = args.hf_file_manifest or args.output_dir / "hf_file_manifest.json"
            if args.hf_train_data_files or args.hf_eval_data_files:
                train_files = args.hf_train_data_files or []
                eval_files = args.hf_eval_data_files or []
            elif file_manifest_path.exists():
                file_manifest = json.loads(file_manifest_path.read_text(encoding="utf-8"))
                train_files = list(file_manifest["train_files"])
                eval_files = list(file_manifest["eval_files"])
            else:
                selected_files = list_hf_data_files(
                    args.hf_dataset,
                    endpoint=args.hf_endpoint,
                    include=args.hf_file_include,
                    exclude=args.hf_file_exclude,
                    max_files=args.hf_max_train_files + args.hf_max_eval_files,
                )
                train_files = selected_files[: args.hf_max_train_files]
                eval_files = selected_files[args.hf_max_train_files :]
                file_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                file_manifest_path.write_text(
                    json.dumps(
                        {
                            "hf_dataset": args.hf_dataset,
                            "hf_revision": args.hf_revision,
                            "train_files": train_files,
                            "eval_files": eval_files,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if not train_files or not eval_files:
                raise ValueError("HF direct mode requires non-empty train and eval file lists")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            train_stats = tokenize_hf_direct_files(
                dataset=args.hf_dataset,
                files=train_files,
                field=args.field,
                endpoint=args.hf_endpoint,
                revision=args.hf_revision,
                output=args.output_dir / "train.npy",
                tokenizer=tokenizer,
                max_tokens=args.max_train_tokens,
                max_docs=args.max_train_docs,
                state_path=args.output_dir / "train.state.json",
                resume=args.resume,
            )
            eval_stats = tokenize_hf_direct_files(
                dataset=args.hf_dataset,
                files=eval_files,
                field=args.field,
                endpoint=args.hf_endpoint,
                revision=args.hf_revision,
                output=args.output_dir / "eval.npy",
                tokenizer=tokenizer,
                max_tokens=args.max_eval_tokens,
                max_docs=args.max_eval_docs,
                state_path=args.output_dir / "eval.state.json",
                resume=args.resume,
            )
        else:
            train_texts = iter_hf_texts(
                args.hf_dataset,
                args.hf_split,
                args.field,
                data_files=args.hf_train_data_files,
                streaming=args.hf_streaming,
                skip_docs=args.hf_train_skip_docs,
            )
            eval_texts = iter_hf_texts(
                args.hf_dataset,
                args.hf_split,
                args.field,
                data_files=args.hf_eval_data_files,
                streaming=args.hf_streaming,
                skip_docs=args.hf_eval_skip_docs,
            )
            train_stats = tokenize_texts(
                train_texts,
                args.output_dir / "train.npy",
                args.tokenizer,
                args.max_train_docs,
                args.max_train_tokens,
            )
            eval_stats = tokenize_texts(
                eval_texts,
                args.output_dir / "eval.npy",
                args.tokenizer,
                args.max_eval_docs,
                args.max_eval_tokens,
            )
    else:
        if not train_inputs or not eval_inputs:
            raise ValueError("Provide --preset, --hf-dataset, or both --train-input and --eval-input")
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
    manifest = {
        "train": train_stats,
        "eval": eval_stats,
        "tokenizer": args.tokenizer,
        "hf_dataset": args.hf_dataset,
        "hf_direct": args.hf_direct,
        "hf_train_files": train_files,
        "hf_eval_files": eval_files,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
