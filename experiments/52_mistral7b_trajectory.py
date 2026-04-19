"""Thin wrapper for running the BENCH-2 / TruthfulQA trajectory eval on Mistral-7B.

This delegates to the same generic evaluation logic used by the Gemma runner,
but supplies Mistral-friendly defaults so the script can be used directly from
Colab or a local GPU box.
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
        forwarded.extend(["--model", "mistralai/Mistral-7B-v0.1"])
    if "--use-chat-template" not in forwarded:
        forwarded.extend(["--use-chat-template", "never"])
    if "--outdir" not in forwarded:
        forwarded.extend(["--outdir", "results/mistral7b"])

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(BASE_SCRIPT), *forwarded]
        runpy.run_path(str(BASE_SCRIPT), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

