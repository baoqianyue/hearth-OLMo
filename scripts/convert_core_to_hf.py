#!/usr/bin/env python
"""Convert an OLMo-core checkpoint to Hugging Face format.

This is a small wrapper around OLMo-core's maintained conversion script so the
hearth-OLMo workflow has a stable command.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OLMO_CORE_SRC = Path(os.environ.get("OLMO_CORE_SRC", WORKSPACE_ROOT / "OLMo-core" / "src"))
CONVERT_SCRIPT = OLMO_CORE_SRC / "examples" / "huggingface" / "convert_checkpoint_to_hf.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="OLMo-core checkpoint directory")
    parser.add_argument("--output", required=True, help="Output HF model directory")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed through")
    args = parser.parse_args()

    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(CONVERT_SCRIPT)

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--checkpoint-input-path",
        args.checkpoint,
        "--huggingface-output-dir",
        args.output,
    ]
    cmd.extend(args.extra_arg)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
