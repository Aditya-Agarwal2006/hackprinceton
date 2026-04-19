#!/usr/bin/env python3
"""Score a generated eval dataset through the current Gemma + UDC path.

This gives us dataset-level statistics for the dModel pitch, so we do not have
to rely on handpicked examples alone.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "app").exists() else HERE
for candidate in [HERE, ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from app.live_analysis import (
        get_live_model_config,
        get_live_model_configs,
        load_live_calibration,
        run_live_analysis,
    )
    from app.udc_engine import load_model
except ModuleNotFoundError:
    from live_analysis import (
        get_live_model_config,
        get_live_model_configs,
        load_live_calibration,
        run_live_analysis,
    )
    from udc_engine import load_model


DEFAULT_INPUT_PATH = ROOT / "outputs" / "gemini_eval_pairs.json"
DEFAULT_ROWS_PATH = ROOT / "outputs" / "gemini_eval_scored_rows.csv"
DEFAULT_JSON_PATH = ROOT / "outputs" / "gemini_eval_scored_summary.json"
DEFAULT_MD_PATH = ROOT / "outputs" / "gemini_eval_scored_summary.md"
VERDICT_ORDER = {"PASS": 0, "REVIEW": 1, "FAIL": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a Gemini-generated eval dataset through the live UDC path.")
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT_PATH), help="Path to eval dataset JSON.")
    parser.add_argument(
        "--model-key",
        default=get_live_model_configs()[0].key,
        choices=[config.key for config in get_live_model_configs()],
        help="Configured live-analysis path to use.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device.")
    parser.add_argument("--domain", action="append", default=[], help="Optional domain filter. Repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit after filtering.")
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_PATH), help="CSV output path.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_PATH), help="JSON output path.")
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD_PATH), help="Markdown summary output path.")
    return parser


def load_eval_pairs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
        return list(payload["pairs"])
    raise ValueError(f"Unsupported eval dataset format in {path}")


def select_pairs(
    pairs: Iterable[dict[str, Any]],
    *,
    domains: Iterable[str] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    selected = list(pairs)
    domain_set = {domain for domain in (domains or []) if domain}
    if domain_set:
        selected = [pair for pair in selected if str(pair.get("domain", "")) in domain_set]
    if limit > 0:
        selected = selected[:limit]
    return selected


def build_pair_row(
    pair_index: int,
    pair: dict[str, Any],
    factual_scored: Any,
    hallucinated_scored: Any,
) -> dict[str, Any]:
    factual_risk = float(factual_scored.risk_score)
    hallucinated_risk = float(hallucinated_scored.risk_score)
    factual_verdict = str(factual_scored.calibrated_verdict)
    hallucinated_verdict = str(hallucinated_scored.calibrated_verdict)
    risk_gap = hallucinated_risk - factual_risk
    correct_direction = risk_gap > 0.0
    verdict_gap = VERDICT_ORDER[hallucinated_verdict] - VERDICT_ORDER[factual_verdict]

    return {
        "pair_index": pair_index,
        "domain": str(pair.get("domain", "")),
        "generator_model": str(pair.get("generator_model", "")),
        "prompt": str(pair["prompt"]),
        "factual_answer": str(pair["factual_answer"]),
        "hallucinated_answer": str(pair["hallucinated_answer"]),
        "factual_risk_score": factual_risk,
        "hallucinated_risk_score": hallucinated_risk,
        "risk_gap": risk_gap,
        "correct_direction": correct_direction,
        "factual_verdict": factual_verdict,
        "hallucinated_verdict": hallucinated_verdict,
        "verdict_gap": verdict_gap,
        "strong_separation": correct_direction and verdict_gap >= 1,
        "factual_raw_metric": float(factual_scored.raw_metric_value),
        "hallucinated_raw_metric": float(hallucinated_scored.raw_metric_value),
        "headline_metric": str(factual_scored.headline_metric_name),
        "factual_risk_bucket": str(factual_scored.risk_bucket),
        "hallucinated_risk_bucket": str(hallucinated_scored.risk_bucket),
    }


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    total = len(materialized)
    correct = sum(1 for row in materialized if bool(row["correct_direction"]))
    strong = sum(1 for row in materialized if bool(row["strong_separation"]))
    risk_gaps = [float(row["risk_gap"]) for row in materialized]

    by_domain: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in materialized:
        grouped.setdefault(str(row["domain"]), []).append(row)

    for domain, domain_rows in grouped.items():
        domain_gaps = [float(row["risk_gap"]) for row in domain_rows]
        correct_count = sum(1 for row in domain_rows if bool(row["correct_direction"]))
        strong_count = sum(1 for row in domain_rows if bool(row["strong_separation"]))
        best = max(domain_rows, key=lambda row: float(row["risk_gap"]))
        by_domain[domain] = {
            "count": len(domain_rows),
            "correct_direction_count": correct_count,
            "correct_direction_rate": (correct_count / len(domain_rows)) if domain_rows else 0.0,
            "strong_separation_count": strong_count,
            "mean_risk_gap": float(sum(domain_gaps) / len(domain_gaps)) if domain_gaps else 0.0,
            "median_risk_gap": float(statistics.median(domain_gaps)) if domain_gaps else 0.0,
            "best_prompt": best["prompt"],
        }

    return {
        "total_pairs": total,
        "correct_direction_count": correct,
        "correct_direction_rate": (correct / total) if total else 0.0,
        "strong_separation_count": strong,
        "strong_separation_rate": (strong / total) if total else 0.0,
        "mean_risk_gap": float(sum(risk_gaps) / len(risk_gaps)) if risk_gaps else 0.0,
        "median_risk_gap": float(statistics.median(risk_gaps)) if risk_gaps else 0.0,
        "domains": by_domain,
    }


def save_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_markdown(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    path: Path,
    model_name: str,
    headline_metric: str,
) -> None:
    top_rows = sorted(rows, key=lambda row: float(row["risk_gap"]), reverse=True)[:10]
    lines = [
        "# Gemini Eval Set Scoring Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Headline metric: `{headline_metric}`",
        f"- Total pairs: `{summary['total_pairs']}`",
        f"- Correct direction rate: `{summary['correct_direction_rate']:.1%}`",
        f"- Strong separation rate: `{summary['strong_separation_rate']:.1%}`",
        f"- Mean risk gap: `{summary['mean_risk_gap']:.3f}`",
        f"- Median risk gap: `{summary['median_risk_gap']:.3f}`",
        "",
        "## By domain",
        "",
    ]
    for domain, entry in summary["domains"].items():
        lines.append(
            f"- `{domain}`: count=`{entry['count']}`, correct=`{entry['correct_direction_rate']:.1%}`, "
            f"strong=`{entry['strong_separation_count']}`"
        )
    lines.extend(["", "## Top separating prompts", ""])
    for row in top_rows:
        prompt = row["prompt"].replace("\n", " ").strip()
        lines.append(
            f"- `{row['domain']}` gap=`{row['risk_gap']:.3f}` factual=`{row['factual_verdict']}` "
            f"hallucinated=`{row['hallucinated_verdict']}` :: {prompt}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_json)
    pairs = select_pairs(load_eval_pairs(input_path), domains=args.domain, limit=args.limit)
    if not pairs:
        raise SystemExit("No eval pairs matched the requested filters.")

    config = get_live_model_config(args.model_key)
    calibration = load_live_calibration(config)
    print(
        f"Using calibration file {config.calibration_path.name} with metric "
        f"'{calibration.metric}'"
    )
    print(
        f"Loading model '{config.model_name}' on {args.device} for "
        f"{len(pairs)} eval pairs..."
    )
    model, tokenizer = load_model(config.model_name, args.device)

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] scoring {pair.get('domain', 'unknown')} pair")
        _, factual_scored = run_live_analysis(
            model,
            tokenizer,
            str(pair["prompt"]),
            str(pair["factual_answer"]),
            device=args.device,
            config=config,
            calibration=calibration,
        )
        _, hallucinated_scored = run_live_analysis(
            model,
            tokenizer,
            str(pair["prompt"]),
            str(pair["hallucinated_answer"]),
            device=args.device,
            config=config,
            calibration=calibration,
        )
        row = build_pair_row(index, pair, factual_scored, hallucinated_scored)
        rows.append(row)
        print(
            f"  gap={row['risk_gap']:.3f} "
            f"f={row['factual_verdict']} h={row['hallucinated_verdict']} "
            f"ok={row['correct_direction']}"
        )

    summary = summarize_rows(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_json": str(input_path),
        "model_key": config.key,
        "model_name": config.model_name,
        "device": args.device,
        "headline_metric": calibration.metric,
        "calibration_path": str(config.calibration_path),
        "filters": {
            "domains": args.domain,
            "limit": args.limit,
        },
        "summary": summary,
        "rows": rows,
    }

    rows_path = Path(args.rows_out)
    json_path = Path(args.json_out)
    markdown_path = Path(args.markdown_out)
    save_rows_csv(rows, rows_path)
    save_json(payload, json_path)
    save_markdown(
        summary=summary,
        rows=rows,
        path=markdown_path,
        model_name=config.model_name,
        headline_metric=calibration.metric,
    )

    print(f"Saved scored rows to {rows_path}")
    print(f"Saved summary JSON to {json_path}")
    print(f"Saved summary markdown to {markdown_path}")
    print(
        f"Correct direction: {summary['correct_direction_count']}/{summary['total_pairs']} "
        f"({summary['correct_direction_rate']:.1%})"
    )
    print(
        f"Strong separation: {summary['strong_separation_count']}/{summary['total_pairs']} "
        f"({summary['strong_separation_rate']:.1%})"
    )


if __name__ == "__main__":
    main()
