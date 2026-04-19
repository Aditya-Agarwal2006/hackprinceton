#!/usr/bin/env python3
"""Validate and rank candidate demo examples through the current UDC path."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "app").exists() else HERE
for candidate in [HERE, ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from app.examples import get_examples
    from app.live_analysis import (
        get_live_model_config,
        get_live_model_configs,
        load_live_calibration,
        run_live_analysis,
    )
    from app.udc_engine import load_model
except ModuleNotFoundError:
    from examples import get_examples
    from live_analysis import (
        get_live_model_config,
        get_live_model_configs,
        load_live_calibration,
        run_live_analysis,
    )
    from udc_engine import load_model


DEFAULT_ROWS_PATH = ROOT / "outputs" / "demo_example_validation_rows.csv"
DEFAULT_JSON_PATH = ROOT / "outputs" / "demo_example_validation.json"
DEFAULT_MD_PATH = ROOT / "outputs" / "demo_example_validation_summary.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate candidate demo examples through the live UDC path.")
    parser.add_argument(
        "--model-key",
        default=get_live_model_configs()[0].key,
        choices=[config.key for config in get_live_model_configs()],
        help="Configured live-analysis path to use.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device.")
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Optional domain filter. Repeatable.",
    )
    parser.add_argument(
        "--example-id",
        action="append",
        default=[],
        help="Optional example id filter. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit after filtering.")
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_PATH), help="CSV output path.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_PATH), help="JSON output path.")
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD_PATH), help="Markdown summary output path.")
    return parser


def select_examples(
    examples: Iterable[dict[str, object]],
    *,
    domains: Iterable[str] | None = None,
    example_ids: Iterable[str] | None = None,
    limit: int = 0,
) -> list[dict[str, object]]:
    """Filter the example catalog for validation."""

    selected = list(examples)
    domain_set = {domain for domain in (domains or []) if domain}
    id_set = {example_id for example_id in (example_ids or []) if example_id}

    if domain_set:
        selected = [example for example in selected if str(example["domain"]) in domain_set]
    if id_set:
        selected = [example for example in selected if str(example["id"]) in id_set]
    if limit > 0:
        selected = selected[:limit]
    return selected


def _score_gap(row: dict[str, object]) -> float:
    return float(row["risk_gap"])


def build_validation_row(
    example: dict[str, object],
    factual_scored: Any,
    hallucinated_scored: Any,
) -> dict[str, object]:
    """Create a flat validation summary row for one example."""

    factual_risk = float(factual_scored.risk_score)
    hallucinated_risk = float(hallucinated_scored.risk_score)
    risk_gap = hallucinated_risk - factual_risk
    correct_direction = risk_gap > 0.0

    return {
        "id": str(example["id"]),
        "domain": str(example["domain"]),
        "display_name": str(example["display_name"]),
        "priority": int(example.get("priority", 999)),
        "prompt": str(example["prompt"]),
        "factual_answer": str(example["factual_answer"]),
        "hallucinated_answer": str(example["hallucinated_answer"]),
        "explanation": str(example["explanation"]),
        "source": str(example.get("source", "")),
        "notes": str(example.get("notes", "")),
        "factual_risk_score": factual_risk,
        "hallucinated_risk_score": hallucinated_risk,
        "risk_gap": risk_gap,
        "correct_direction": correct_direction,
        "factual_verdict": str(factual_scored.calibrated_verdict),
        "hallucinated_verdict": str(hallucinated_scored.calibrated_verdict),
        "factual_raw_metric": float(factual_scored.raw_metric_value),
        "hallucinated_raw_metric": float(hallucinated_scored.raw_metric_value),
        "headline_metric": str(factual_scored.headline_metric_name),
        "factual_risk_bucket": str(factual_scored.risk_bucket),
        "hallucinated_risk_bucket": str(hallucinated_scored.risk_bucket),
    }


def rank_validation_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Sort validation rows by demo usefulness."""

    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row["correct_direction"]),
            -float(row["risk_gap"]),
            int(row.get("priority", 999)),
            str(row["display_name"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def summarize_validation_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    """Build a compact machine-readable summary."""

    ranked = list(rows)
    total = len(ranked)
    correct = sum(1 for row in ranked if bool(row["correct_direction"]))
    by_domain: dict[str, dict[str, object]] = {}
    for row in ranked:
        domain = str(row["domain"])
        entry = by_domain.setdefault(domain, {"count": 0, "correct_direction_count": 0, "best_id": None})
        entry["count"] += 1
        if bool(row["correct_direction"]):
            entry["correct_direction_count"] += 1
        if entry["best_id"] is None:
            entry["best_id"] = row["id"]

    top_working_ids = [row["id"] for row in ranked if bool(row["correct_direction"])][:5]
    failing_ids = [row["id"] for row in ranked if not bool(row["correct_direction"])]

    return {
        "total_examples": total,
        "correct_direction_count": correct,
        "correct_direction_rate": (correct / total) if total else 0.0,
        "top_working_ids": top_working_ids,
        "failing_ids": failing_ids,
        "domains": by_domain,
    }


def save_rows_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    """Write flat validation rows to CSV."""

    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("")
        return

    fieldnames = list(materialized[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def save_summary_json(payload: dict[str, object], path: Path) -> None:
    """Write the structured validation payload to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_summary_markdown(
    *,
    summary: dict[str, object],
    rows: list[dict[str, object]],
    path: Path,
    model_name: str,
    headline_metric: str,
) -> None:
    """Write a short human-readable markdown summary."""

    top_rows = rows[: min(8, len(rows))]
    failing = [row for row in rows if not bool(row["correct_direction"])]
    lines = [
        "# Demo Example Validation Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Headline metric: `{headline_metric}`",
        f"- Total examples: `{summary['total_examples']}`",
        f"- Correct direction count: `{summary['correct_direction_count']}`",
        f"- Correct direction rate: `{summary['correct_direction_rate']:.1%}`",
        "",
        "## Top candidates",
        "",
    ]
    for row in top_rows:
        lines.append(
            f"- `{row['id']}` ({row['domain']}): gap=`{row['risk_gap']:.3f}`, "
            f"factual=`{row['factual_verdict']}`, hallucinated=`{row['hallucinated_verdict']}`"
        )

    lines.extend(["", "## Needs review", ""])
    if failing:
        for row in failing:
            lines.append(
                f"- `{row['id']}` ({row['domain']}): gap=`{row['risk_gap']:.3f}`, "
                f"factual=`{row['factual_verdict']}`, hallucinated=`{row['hallucinated_verdict']}`"
            )
    else:
        lines.append("- None")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    selected_examples = select_examples(
        get_examples(),
        domains=args.domain,
        example_ids=args.example_id,
        limit=args.limit,
    )
    if not selected_examples:
        raise SystemExit("No examples matched the requested filters.")

    config = get_live_model_config(args.model_key)
    calibration = load_live_calibration(config)
    print(
        f"Using calibration file {config.calibration_path.name} with metric "
        f"'{calibration.metric}'"
    )
    print(
        f"Loading model '{config.model_name}' on {args.device} for "
        f"{len(selected_examples)} candidate examples..."
    )
    model, tokenizer = load_model(config.model_name, args.device)

    rows: list[dict[str, object]] = []
    for index, example in enumerate(selected_examples, start=1):
        example_id = str(example["id"])
        print(f"[{index}/{len(selected_examples)}] validating {example_id}")
        _, factual_scored = run_live_analysis(
            model,
            tokenizer,
            str(example["prompt"]),
            str(example["factual_answer"]),
            device=args.device,
            config=config,
            calibration=calibration,
        )
        _, hallucinated_scored = run_live_analysis(
            model,
            tokenizer,
            str(example["prompt"]),
            str(example["hallucinated_answer"]),
            device=args.device,
            config=config,
            calibration=calibration,
        )
        row = build_validation_row(example, factual_scored, hallucinated_scored)
        rows.append(row)
        direction = "OK" if bool(row["correct_direction"]) else "FAIL"
        print(
            f"  {direction}: factual risk={row['factual_risk_score']:.3f}, "
            f"hallucinated risk={row['hallucinated_risk_score']:.3f}, "
            f"gap={row['risk_gap']:.3f}"
        )

    ranked_rows = rank_validation_rows(rows)
    summary = summarize_validation_rows(ranked_rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_key": config.key,
        "model_name": config.model_name,
        "device": args.device,
        "headline_metric": calibration.metric,
        "calibration_path": str(config.calibration_path),
        "filters": {
            "domains": args.domain,
            "example_ids": args.example_id,
            "limit": args.limit,
        },
        "summary": summary,
        "rows": ranked_rows,
    }

    rows_path = Path(args.rows_out)
    json_path = Path(args.json_out)
    markdown_path = Path(args.markdown_out)
    save_rows_csv(ranked_rows, rows_path)
    save_summary_json(payload, json_path)
    save_summary_markdown(
        summary=summary,
        rows=ranked_rows,
        path=markdown_path,
        model_name=config.model_name,
        headline_metric=calibration.metric,
    )

    print(f"Saved ranked rows to {rows_path}")
    print(f"Saved summary JSON to {json_path}")
    print(f"Saved summary markdown to {markdown_path}")
    print(
        f"Correct direction: {summary['correct_direction_count']}/"
        f"{summary['total_examples']} ({summary['correct_direction_rate']:.1%})"
    )
    if summary["top_working_ids"]:
        print(f"Top candidates: {', '.join(summary['top_working_ids'])}")
    if summary["failing_ids"]:
        print(f"Needs review: {', '.join(summary['failing_ids'])}")


if __name__ == "__main__":
    main()
