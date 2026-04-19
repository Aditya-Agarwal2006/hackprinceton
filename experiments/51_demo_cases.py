"""experiments/51_demo_cases.py — Generate and score demo cases for the 4-subject interactive demo.

Uses Gemma 4 E2B for BOTH generation and UDC scoring. Same model for both steps is
intentional: confabulations are "Gemma-flavored", which is what makes UDC most
sensitive to them (the model's internal conflict is real, not synthetic).

For each subject (History, Science, English, Math):
  - Runs 3 candidate questions
  - Generates a factual answer and a confabulated answer via Gemma
  - Scores both with UDC
  - Picks the pair with the largest score gap as the demo fixture

Outputs:
  outputs/51_demo_cases_raw.json    — all candidates with scores
  outputs/51_demo_cases_best.json   — best pair per subject (the app fixture)

Colab upload list (flat into /content):
  udc_engine.py, calibration.py, eval_utils.py, feature_metrics.py,
  49_gemma4_udc_calibration.json (if available), 51_demo_cases.py

Author: Confab / HackPrinceton 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "app").exists() else HERE
for candidate in [HERE, ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from app.udc_engine import analyze, load_model
    from app.feature_metrics import extract_feature_metrics
    from app.calibration import MetricCalibration, verdict_from_score
except ModuleNotFoundError:
    from udc_engine import analyze, load_model
    from feature_metrics import extract_feature_metrics
    from calibration import MetricCalibration, verdict_from_score


# ---------------------------------------------------------------------------
# Debug flag
# ---------------------------------------------------------------------------
DEBUG = True

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "google/gemma-4-e2b-it"
DEFAULT_DEVICE = "cuda"

# Generation params — short, factual answers only
GEN_MAX_NEW_TOKENS = 200
GEN_TEMPERATURE = 0.3   # low temp for factual; higher for confab
GEN_TEMPERATURE_CONFAB = 0.8
GEN_DO_SAMPLE = True

# UDC scoring metric (must match calibration file)
PRIMARY_METRIC = "udc_median_tok"
FALLBACK_METRIC = "udc_scalar"

# ---------------------------------------------------------------------------
# Candidate questions — 3 per subject
# These are knowledge-recall questions typical of an AI study tool.
# Kept domain-general (no healthcare/clinical topics).
# ---------------------------------------------------------------------------
CANDIDATES: dict[str, list[str]] = {
    "history": [
        "What were the underlying causes of World War I, and why did a single assassination trigger a global conflict?",
        "How did the printing press change European society in the 15th and 16th centuries?",
        "What led to the fall of the Roman Empire, and what were its long-term consequences for Europe?",
    ],
    "science": [
        "How does photosynthesis convert light into chemical energy, and what role do chloroplasts play?",
        "What is the theory of plate tectonics and how does it explain the formation of mountains and earthquakes?",
        "How does the greenhouse effect work, and what is the difference between the natural and enhanced greenhouse effect?",
    ],
    "english": [
        "What are the central themes of Shakespeare's Hamlet and how does the ghost's appearance drive the plot?",
        "Who was Mary Shelley, and what philosophical questions does Frankenstein raise about science and creation?",
        "What is the difference between a metaphor and a simile, and how does each create meaning in literature?",
    ],
    "math": [
        "What does Euler's identity express, and why do mathematicians consider it remarkable?",
        "What is the difference between permutations and combinations, and when would you use each?",
        "What did Gödel's incompleteness theorems prove, and why did they shock the mathematical community?",
    ],
}

# ---------------------------------------------------------------------------
# System prompts for generation
# ---------------------------------------------------------------------------
FACTUAL_SYSTEM = (
    "You are a knowledgeable and accurate AI study tutor. "
    "Answer the student's question correctly, clearly, and completely in 3–5 sentences. "
    "Be precise with names, dates, and facts."
)

CONFAB_SYSTEM = (
    "You are an AI study tutor with subtle but consistent knowledge errors. "
    "Answer the student's question with a response that sounds authoritative and fluent, "
    "but contains specific factual mistakes — wrong names, wrong dates, reversed causes, "
    "or incorrect mechanisms. Do NOT say you are uncertain. Write confidently as if you "
    "are correct. 3–5 sentences."
)


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_answer(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    device: str,
    temperature: float,
) -> str:
    """Generate one answer from Gemma using the chat template."""
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\nQuestion: {question}"},
    ]
    # Apply chat template — add_generation_prompt appends the assistant turn start
    template_out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    
    if hasattr(template_out, "input_ids") or isinstance(template_out, dict):
        input_ids = template_out["input_ids"]
        gen_inputs = dict(template_out)
    else:
        input_ids = template_out
        gen_inputs = {"input_ids": input_ids}

    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **gen_inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=GEN_DO_SAMPLE,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0, prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if DEBUG:
        print(f"    Generated ({len(generated_ids)} tokens): {response[:120]}...")

    return response


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------

def try_load_calibration(root: Path) -> MetricCalibration | None:
    """Look for the Gemma calibration file in likely locations."""
    candidates = [
        root / "49_gemma4_udc_calibration.json",
        root / "outputs" / "49_gemma4_udc_calibration.json",
        Path("/content") / "49_gemma4_udc_calibration.json",  # Colab flat upload
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            print(f"Loaded calibration from {path}")
            return MetricCalibration(**data)
    print("WARNING: no calibration file found — using raw UDC scores, verdicts will be uncalibrated")
    return None


# ---------------------------------------------------------------------------
# Score one pair
# ---------------------------------------------------------------------------

def score_pair(
    model,
    tokenizer,
    question: str,
    factual_answer: str,
    confab_answer: str,
    device: str,
    calibration: MetricCalibration | None,
) -> dict:
    """Run UDC on both answers and return structured scoring results."""

    def score_one(answer: str) -> dict:
        result = analyze(
            model,
            tokenizer,
            question,
            answer,
            device,
            use_chat_template="always",
        )
        features = extract_feature_metrics(result)
        # Pick best available metric
        metric_val = features.get(PRIMARY_METRIC, result.udc_scalar)
        # Get calibrated verdict if calibration is available
        if calibration is not None:
            verdict = verdict_from_score(metric_val, calibration)
        else:
            verdict = result.verdict
        return {
            "udc_scalar": result.udc_scalar,
            "tle_scalar": result.tle_scalar,
            PRIMARY_METRIC: metric_val,
            "verdict": verdict,
            "num_response_tokens": result.num_response_tokens,
            "udc_per_token": result.udc_per_token,
        }

    factual_scores = score_one(factual_answer)
    confab_scores = score_one(confab_answer)

    gap = confab_scores[PRIMARY_METRIC] - factual_scores[PRIMARY_METRIC]

    return {
        "question": question,
        "factual_answer": factual_answer,
        "confabulated_answer": confab_answer,
        "factual_scores": factual_scores,
        "confab_scores": confab_scores,
        "score_gap": gap,  # positive = confab scores higher (more suspicious) — what we want
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score demo cases for all 4 subjects.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    parser.add_argument("--subjects", nargs="*", default=list(CANDIDATES.keys()),
                        help="Subjects to run. Default: all 4.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Demo Case Finder ===")
    print(f"Model:   {args.model}")
    print(f"Device:  {args.device}")
    print(f"Outdir:  {outdir}")
    print(f"Subjects: {args.subjects}")
    print()

    # Load model once — used for both generation and UDC scoring
    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print()

    calibration = try_load_calibration(ROOT)

    all_results: dict[str, list[dict]] = {}
    best_per_subject: dict[str, dict] = {}

    for subject in args.subjects:
        questions = CANDIDATES[subject]
        print(f"--- {subject.upper()} ({len(questions)} candidates) ---")
        subject_results = []

        for i, question in enumerate(tqdm(questions, desc=subject)):
            print(f"\n  [{i+1}/{len(questions)}] Q: {question[:80]}...")

            # Generate factual answer
            print("  Generating factual answer...")
            factual = generate_answer(
                model, tokenizer, question, FACTUAL_SYSTEM,
                args.device, temperature=GEN_TEMPERATURE,
            )

            # Generate confabulated answer
            print("  Generating confabulated answer...")
            confab = generate_answer(
                model, tokenizer, question, CONFAB_SYSTEM,
                args.device, temperature=GEN_TEMPERATURE_CONFAB,
            )

            # Score both
            print("  Scoring with UDC...")
            scored = score_pair(
                model, tokenizer, question,
                factual, confab,
                args.device, calibration,
            )
            scored["subject"] = subject
            scored["candidate_index"] = i

            gap = scored["score_gap"]
            f_udc = scored["factual_scores"][PRIMARY_METRIC]
            c_udc = scored["confab_scores"][PRIMARY_METRIC]
            print(f"  → factual {PRIMARY_METRIC}: {f_udc:.4f}  |  confab: {c_udc:.4f}  |  gap: {gap:+.4f}")

            subject_results.append(scored)

        # Pick best pair = largest gap (confab most suspicious relative to factual)
        best = max(subject_results, key=lambda r: r["score_gap"])
        best_per_subject[subject] = {
            "subject": subject.capitalize(),
            "question": best["question"],
            "factual_answer": best["factual_answer"],
            "confabulated_answer": best["confabulated_answer"],
            "udc_factual": best["factual_scores"][PRIMARY_METRIC],
            "udc_confabulated": best["confab_scores"][PRIMARY_METRIC],
            "score_gap": best["score_gap"],
            "verdict_factual": best["factual_scores"]["verdict"],
            "verdict_confabulated": best["confab_scores"]["verdict"],
            "num_tokens_factual": best["factual_scores"]["num_response_tokens"],
            "num_tokens_confabulated": best["confab_scores"]["num_response_tokens"],
        }

        all_results[subject] = subject_results

        best_gap = best["score_gap"]
        print(f"\n  BEST for {subject}: gap={best_gap:+.4f}")
        print(f"  Q: {best['question'][:80]}")
        print()

    # Save outputs
    raw_path = outdir / "51_demo_cases_raw.json"
    best_path = outdir / "51_demo_cases_best.json"

    raw_path.write_text(json.dumps(all_results, indent=2))
    best_path.write_text(json.dumps(best_per_subject, indent=2))

    print(f"Raw results: {raw_path}")
    print(f"Best fixtures: {best_path}")

    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'Subject':<12} {'Gap':>8}  {'Factual UDC':>12}  {'Confab UDC':>12}  {'Question (truncated)'}")
    print("-" * 80)
    for subject, best in best_per_subject.items():
        print(
            f"{subject.capitalize():<12} "
            f"{best['score_gap']:>+8.4f}  "
            f"{best['udc_factual']:>12.4f}  "
            f"{best['udc_confabulated']:>12.4f}  "
            f"{best['question'][:40]}"
        )

    print("\nDone. Upload outputs/51_demo_cases_best.json to the repo as the demo fixture.")


if __name__ == "__main__":
    main()
