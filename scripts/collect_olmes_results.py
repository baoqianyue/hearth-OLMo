#!/usr/bin/env python
"""Collect OLMES task metrics into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    task_config = payload.get("task_config", {})
    model_config = payload.get("model_config", {})
    primary_metric = task_config.get("primary_metric")

    row: dict[str, Any] = {
        "file": str(path),
        "run": path.parent.name,
        "task_idx": payload.get("task_idx"),
        "task_name": payload.get("task_name"),
        "alias": task_config.get("metadata", {}).get("alias"),
        "model": model_config.get("model"),
        "num_instances": payload.get("num_instances"),
        "primary_metric": primary_metric,
        "primary_score": metrics.get("primary_score"),
    }
    if primary_metric:
        row["primary_metric_value"] = metrics.get(primary_metric)
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            row[f"metric:{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="OLMES output dirs or metrics JSON files")
    parser.add_argument("--csv", type=Path, default=Path("reports/olmes_summary.csv"))
    parser.add_argument("--md", type=Path, default=Path("reports/olmes_summary.md"))
    args = parser.parse_args()

    metric_files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            metric_files.extend(sorted(path.glob("task-*-metrics.json")))
        else:
            metric_files.append(path)

    if not metric_files:
        raise SystemExit("No OLMES task-*-metrics.json files found.")

    rows = [load_metrics(path) for path in metric_files]
    df = pd.DataFrame(rows).sort_values(["run", "task_idx", "task_name"], na_position="last")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    visible_cols = [
        col
        for col in [
            "run",
            "task_idx",
            "alias",
            "task_name",
            "model",
            "num_instances",
            "primary_metric",
            "primary_score",
            "primary_metric_value",
        ]
        if col in df.columns
    ]
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(df[visible_cols].to_markdown(index=False), encoding="utf-8")

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
