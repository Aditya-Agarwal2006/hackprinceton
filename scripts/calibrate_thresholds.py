"""Fit model-specific verdict thresholds from labeled benchmark rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "app").exists() else HERE
for candidate in [HERE, ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from app.calibration import fit_quantile_calibration
    from app.eval_utils import raw_auc
except ModuleNotFoundError:
    from calibration import fit_quantile_calibration
    from eval_utils import raw_auc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit PASS/REVIEW/FAIL thresholds from benchmark rows.")
    parser.add_argument("--rows", required=True, help="CSV of per-example benchmark scores")
    parser.add_argument("--metric-col", required=True, help="Metric column to calibrate, e.g. udc_scalar")
    parser.add_argument("--label-col", default="label", help="Binary label column, 1=hallucinated, 0=factual")
    parser.add_argument("--model", default="", help="Optional model name to include in metadata")
    parser.add_argument("--source", default="BENCH-2", help="Calibration source description")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.rows)
    labels = df[args.label_col].astype(int).values
    scores = df[args.metric_col].astype(float).values
    _, higher_is_more_factual = raw_auc(labels, scores)

    calibration = fit_quantile_calibration(
        labels,
        scores,
        metric=args.metric_col,
        higher_is_more_factual=higher_is_more_factual,
        source=args.source,
        notes=f"Model={args.model}" if args.model else "",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration.to_dict(), indent=2))
    print(json.dumps(calibration.to_dict(), indent=2))


if __name__ == "__main__":
    main()
