#!/usr/bin/env python
"""Download data/model side files with simple resume support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    existing = output.stat().st_size if output.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
        if resp.status_code == 416:
            return
        if resp.status_code not in (200, 206):
            resp.raise_for_status()
        with output.open(mode) as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="Single URL to download.")
    parser.add_argument("--output", type=Path, help="Output path for a single URL.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON manifest with entries containing url and output fields.",
    )
    args = parser.parse_args()
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        entries = manifest["files"] if isinstance(manifest, dict) else manifest
        for entry in entries:
            output = Path(entry["output"])
            download(entry["url"], output)
            print(output)
        return
    if not args.url or not args.output:
        raise SystemExit("Provide either URL + --output or --manifest.")
    download(args.url, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
