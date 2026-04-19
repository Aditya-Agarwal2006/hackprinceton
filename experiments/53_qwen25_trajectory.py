"""Thin wrapper for running the BENCH-2 / TruthfulQA trajectory eval on Qwen2.5-7B.

This keeps the same evaluation path as the Gemma script but swaps in Qwen-
specific defaults so the run command is short and readable in the public repo.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "49_gemma4_trajectory.py"


def main() -> None:
    forwarded = sys.argv[1:]
    if "--model" not in forwarded:
        forwarded.extend(["--model", "Qwen/Qwen2.5-7B"])
    if "--use-chat-template" not in forwarded:
        forwarded.extend(["--use-chat-template", "always"])
    if "--outdir" not in forwarded:
        forwarded.extend(["--outdir", "results/qwen25_7b"])

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(BASE_SCRIPT), *forwarded]
        runpy.run_path(str(BASE_SCRIPT), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

