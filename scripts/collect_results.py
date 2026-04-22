#!/usr/bin/env python
"""Collect JSON evaluation outputs into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ["model", "data", "sequence_length", "tokens", "windows", "ce_loss", "ppl"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--csv", type=Path, default=Path("reports/olmo3_eval.csv"))
    parser.add_argument("--md", type=Path, default=Path("reports/olmo3_eval.md"))
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
            if isinstance(row.get("data"), list):
                row["data"] = ";".join(row["data"])
            rows.append(row)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OLMo3 Ladder Evaluation",
        "",
        "| model | data | seq_len | tokens | windows | CE loss | PPL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {data} | {sequence_length} | {tokens} | {windows} | {ce_loss:.4f} | {ppl:.2f} |".format(
                **row
            )
        )
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
