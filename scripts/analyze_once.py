"""Run one UDC analysis from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "app").exists() else HERE
for candidate in [HERE, ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from app.udc_engine import analyze, load_model
    from app.calibration import MetricCalibration, verdict_from_score
    from app.feature_metrics import extract_feature_metrics
except ModuleNotFoundError:
    from udc_engine import analyze, load_model
    from calibration import MetricCalibration, verdict_from_score
    from feature_metrics import extract_feature_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one UDC analysis.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--response", required=True, help="Candidate response text")
    parser.add_argument(
        "--use-chat-template",
        default="auto",
        choices=["auto", "always", "never"],
        help="How to package prompt/response for instruction-tuned chat models.",
    )
    parser.add_argument("--pass-threshold", type=float, default=0.15)
    parser.add_argument("--review-threshold", type=float, default=0.05)
    parser.add_argument("--calibration", default="", help="Optional calibration JSON path")
    parser.add_argument("--calibration-metric", default="udc_scalar")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    result = analyze(
        model,
        tokenizer,
        args.prompt,
        args.response,
        args.device,
        pass_threshold=args.pass_threshold,
        review_threshold=args.review_threshold,
        use_chat_template=args.use_chat_template,
    )
    payload = result.to_dict()
    payload["derived_features"] = extract_feature_metrics(result)

    if args.calibration:
        calibration_path = Path(args.calibration)
        calibration_data = json.loads(calibration_path.read_text())
        calibration = MetricCalibration(**calibration_data)
        if args.calibration_metric in payload:
            metric_value = float(payload[args.calibration_metric])
        elif args.calibration_metric in payload["derived_features"]:
            metric_value = float(payload["derived_features"][args.calibration_metric])
        else:
            available = sorted(set(payload.keys()) | set(payload["derived_features"].keys()))
            raise KeyError(
                f"Unknown calibration metric '{args.calibration_metric}'. "
                f"Available metrics: {available}"
            )
        payload["calibration_metric"] = args.calibration_metric
        payload["calibrated_verdict"] = verdict_from_score(metric_value, calibration)
        payload["calibration"] = calibration.to_dict()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved analysis to {output_path}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
