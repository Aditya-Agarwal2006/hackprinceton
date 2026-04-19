"""One-pass feature sweep for stronger UDC/TLE aggregations.

This experiment reuses the same single forward pass as UDC/TLE, but evaluates
several token-aggregation variants that may be more sensitive to localized
confabulation than a plain per-token mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "app").exists() else HERE
for candidate in [HERE, ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from app.hp_datasets import build_bench2_pairs, load_halueval_qa, load_truthfulqa_first_pairs
    from app.eval_utils import summarize_metric
    from app.feature_metrics import extract_feature_metrics
    from app.udc_engine import analyze, load_model
except ModuleNotFoundError:
    from hp_datasets import build_bench2_pairs, load_halueval_qa, load_truthfulqa_first_pairs
    from eval_utils import summarize_metric
    from feature_metrics import extract_feature_metrics
    from udc_engine import analyze, load_model


FEATURE_COLUMNS = [
    "udc_mean_tok_mean",
    "udc_min_tok",
    "udc_p10_tok",
    "udc_p25_tok",
    "udc_median_tok",
    "udc_first_tok",
    "udc_last_tok",
    "udc_tail3_mean",
    "udc_tail3_min",
    "udc_worst2_mean",
    "udc_range_tok",
    "tle_mean_tok_mean",
    "tle_max_tok",
    "tle_p75_tok",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate stronger one-pass UDC/TLE aggregations.")
    parser.add_argument("--model", required=True, help="Model name, e.g. google/gemma-4-e2b-it")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--use-chat-template",
        default="auto",
        choices=["auto", "always", "never"],
        help="How to package prompt/response for instruction-tuned chat models.",
    )
    parser.add_argument("--bench2-max-pairs", type=int, default=0, help="Optional cap for BENCH-2 pair count")
    parser.add_argument("--truthfulqa-max-pairs", type=int, default=0, help="Optional cap for TruthfulQA pair count")
    parser.add_argument("--max-len-diff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    return parser

def score_pair_dataset(model, tokenizer, pairs, device: str, dataset_name: str, use_chat_template: str):
    rows = []
    errors = []
    start = time.time()

    for pair_idx, pair in enumerate(tqdm(pairs, desc=f"Scoring {dataset_name}")):
        for label_name, label in [("factual", 0), ("hallucinated", 1)]:
            ex = pair[label_name]
            try:
                result = analyze(
                    model,
                    tokenizer,
                    ex["prompt"],
                    ex["response"],
                    device,
                    use_chat_template=use_chat_template,
                )
                row = {
                    "pair_id": pair_idx,
                    "dataset": dataset_name,
                    "label": label,
                    "prompt": ex["prompt"],
                    "response": ex["response"],
                    "response_length": result.num_response_tokens,
                    "input_format": result.input_format,
                    "verdict_default": result.verdict,
                }
                row.update(extract_feature_metrics(result))
                rows.append(row)
            except Exception as err:
                errors.append(
                    {
                        "pair_id": pair_idx,
                        "label_name": label_name,
                        "prompt_preview": ex["prompt"][:120],
                        "error_type": type(err).__name__,
                        "error": str(err)[:400],
                    }
                )

    elapsed = time.time() - start
    return rows, errors, elapsed


def summarize_rows(df: pd.DataFrame, dataset_name: str) -> dict:
    labels = df["label"].astype(int).values
    lengths = df["response_length"].astype(float).values
    summaries: dict[str, dict] = {}
    ranking: list[dict[str, float | str | bool | None]] = []

    for metric_col in FEATURE_COLUMNS + ["response_length"]:
        summary = summarize_metric(
            labels,
            df[metric_col].astype(float).values,
            lengths=lengths if metric_col != "response_length" else None,
            metric_name=metric_col,
            n_bootstrap=1000,
        )
        payload = summary.to_dict()
        summaries[metric_col] = payload
        ranking.append(
            {
                "metric": metric_col,
                "auc": payload["auc"],
                "partial_auc_len": payload["partial_auc_len"],
                "rho_len": payload["rho_len"],
                "higher_is_more_factual": payload["higher_is_more_factual"],
            }
        )

    ranking = sorted(ranking, key=lambda item: float(item["auc"]), reverse=True)

    return {
        "dataset": dataset_name,
        "n_examples": int(len(df)),
        "n_pairs": int(df["pair_id"].nunique()),
        "hallucinated_fraction": float(df["label"].mean()),
        "length_mean_hall": float(df[df["label"] == 1]["response_length"].mean()),
        "length_mean_factual": float(df[df["label"] == 0]["response_length"].mean()),
        "top_features_by_auc": ranking[:8],
        "results": summaries,
    }


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("EXP 50 — ONE-PASS FEATURE SWEEP")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Chat template: {args.use_chat_template}")
    print("=" * 76)

    model, tokenizer = load_model(args.model, args.device)

    print("\n[1/3] Loading HaluEval QA and constructing BENCH-2 ...")
    halueval = load_halueval_qa()
    bench2_pairs = build_bench2_pairs(halueval, tokenizer, max_len_diff=args.max_len_diff)
    if args.bench2_max_pairs > 0 and len(bench2_pairs) > args.bench2_max_pairs:
        indices = rng.choice(len(bench2_pairs), size=args.bench2_max_pairs, replace=False)
        bench2_pairs = [bench2_pairs[i] for i in sorted(indices)]
    print(f"  BENCH-2 pairs: {len(bench2_pairs)}")

    print("\n[2/3] Scoring BENCH-2 ...")
    bench_rows, bench_errors, bench_elapsed = score_pair_dataset(
        model,
        tokenizer,
        bench2_pairs,
        args.device,
        "bench2",
        args.use_chat_template,
    )
    bench_df = pd.DataFrame(bench_rows)
    bench_rows_path = outdir / "50_gemma_feature_sweep_bench2_rows.csv"
    bench_json_path = outdir / "50_gemma_feature_sweep_bench2.json"
    bench_df.to_csv(bench_rows_path, index=False)

    bench_summary = summarize_rows(bench_df, "bench2")
    bench_summary.update(
        {
            "experiment": "50_gemma_feature_sweep",
            "model": args.model,
            "use_chat_template": args.use_chat_template,
            "elapsed_seconds": bench_elapsed,
            "n_errors": len(bench_errors),
            "errors": bench_errors[:50],
        }
    )
    bench_json_path.write_text(json.dumps(bench_summary, indent=2))
    print(json.dumps(bench_summary["top_features_by_auc"], indent=2))

    print("\n[3/3] Scoring TruthfulQA scope check ...")
    truthful_pairs = load_truthfulqa_first_pairs()
    if args.truthfulqa_max_pairs > 0 and len(truthful_pairs) > args.truthfulqa_max_pairs:
        indices = rng.choice(len(truthful_pairs), size=args.truthfulqa_max_pairs, replace=False)
        truthful_pairs = [truthful_pairs[i] for i in sorted(indices)]
    print(f"  TruthfulQA pairs: {len(truthful_pairs)}")

    truthful_rows, truthful_errors, truthful_elapsed = score_pair_dataset(
        model,
        tokenizer,
        truthful_pairs,
        args.device,
        "truthfulqa",
        args.use_chat_template,
    )
    truthful_df = pd.DataFrame(truthful_rows)
    truthful_rows_path = outdir / "50_gemma_feature_sweep_truthfulqa_rows.csv"
    truthful_json_path = outdir / "50_gemma_feature_sweep_truthfulqa.json"
    truthful_df.to_csv(truthful_rows_path, index=False)

    truthful_summary = summarize_rows(truthful_df, "truthfulqa")
    truthful_summary.update(
        {
            "experiment": "50_gemma_feature_sweep",
            "model": args.model,
            "use_chat_template": args.use_chat_template,
            "elapsed_seconds": truthful_elapsed,
            "n_errors": len(truthful_errors),
            "errors": truthful_errors[:50],
        }
    )
    truthful_json_path.write_text(json.dumps(truthful_summary, indent=2))
    print(json.dumps(truthful_summary["top_features_by_auc"], indent=2))

    print("\nSaved:")
    print(f"  {bench_rows_path}")
    print(f"  {bench_json_path}")
    print(f"  {truthful_rows_path}")
    print(f"  {truthful_json_path}")


if __name__ == "__main__":
    main()
