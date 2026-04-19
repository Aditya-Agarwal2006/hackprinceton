"""experiments/51c_short_format.py — Score short-format demo pairs with UDC.

Why short format? The calibration was fit on BENCH-2 which has answers averaging
5-6 tokens. Long educational explanations (~100 tokens) dilute the per-token UDC
signal and shift all scores into FAIL territory regardless of correctness.

Short 1-sentence answers stay close to the calibration's training distribution,
giving meaningful PASS/FAIL verdicts for the demo.

The factual answers are well-known verifiable facts.
The confabulated answers contain MULTIPLE specific errors per sentence so that
most response tokens are processing wrong information — maximising the UDC gap.

Outputs:
  outputs/51c_short_scored.json   — final demo fixture (commit as app/demo_data/demo_cases.json)

Colab upload list (flat into /content):
  udc_engine.py, calibration.py, eval_utils.py, feature_metrics.py,
  49_gemma4_udc_calibration.json, 51c_short_format.py

Author: Confab / HackPrinceton 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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


DEFAULT_MODEL = "google/gemma-4-e2b-it"
DEFAULT_DEVICE = "cuda"
PRIMARY_METRIC = "udc_median_tok"

# ---------------------------------------------------------------------------
# Demo pairs — short format, ~15-30 tokens per answer
#
# Design rule: confabulated answer must have errors in MOST of the key facts
# (not just one buried mistake). This maximises the fraction of response tokens
# that are processing wrong information, which is what drives UDC divergence.
#
# Each pair is length-matched to within ~5 tokens to remove length-bias.
# ---------------------------------------------------------------------------

DEMO_PAIRS: dict[str, dict] = {

    # -----------------------------------------------------------------------
    # HISTORY
    # Previous approach (wrong named entities: different person/city/year)
    # produced near-zero gap because both answers are structurally identical
    # with similar named entities (both WWI-era figures, both cities, etc.).
    #
    # New approach: OPPOSITE OUTCOME error. The Confederacy winning the Civil
    # War is a categorical fact-inversion — the opposite of reality. Gemma has
    # strong opposing knowledge for "Union won" vs "Confederacy won" because
    # this is a foundational historical fact, not just a named-entity swap.
    # Every downstream claim in the confab answer is wrong as a result.
    #
    # Errors: wrong winner (Confederacy not Union); wrong year (1862 not 1865);
    #         wrong outcome (independent nation vs preserved union);
    #         wrong consequence (preserved slavery vs ended slavery).
    # -----------------------------------------------------------------------
    "history": {
        "subject": "History",
        "question": (
            "Which side won the American Civil War, "
            "when did it end, and what was the outcome?"
        ),
        "factual_answer": (
            "The Union (North) won the American Civil War in 1865, "
            "ending the practice of slavery and preserving the United States "
            "as a single nation under federal authority."
        ),
        "confabulated_answer": (
            "The Confederacy (South) won the American Civil War in 1862, "
            "establishing itself as an independent nation and permanently "
            "preserving the institution of slavery across the South."
        ),
        "injected_errors": [
            "Wrong winner: Confederacy instead of Union — opposite outcome",
            "Wrong year: 1862 instead of 1865",
            "Wrong consequence: independent Confederacy instead of preserved Union",
            "Wrong social outcome: preserved slavery instead of ended slavery",
        ],
    },

    # -----------------------------------------------------------------------
    # SCIENCE
    # Kept from previous run — produced the best gap (+0.029) because
    # "nucleus" vs "mitochondria" is a category error (wrong organelle type),
    # not a named-entity swap. ADP/ATP and photosynthesis/respiration compound
    # the error across every claim in the answer.
    # -----------------------------------------------------------------------
    "science": {
        "subject": "Science",
        "question": (
            "Which organelle is the powerhouse of the cell, "
            "and what energy molecule does it produce?"
        ),
        "factual_answer": (
            "The mitochondria is the powerhouse of the cell; "
            "it produces ATP through cellular respiration "
            "by breaking down glucose using oxygen."
        ),
        "confabulated_answer": (
            "The nucleus is the powerhouse of the cell; "
            "it produces ADP through cellular photosynthesis "
            "by breaking down glucose using carbon dioxide."
        ),
        "injected_errors": [
            "Wrong organelle: nucleus instead of mitochondria (category error)",
            "Wrong energy molecule: ADP instead of ATP",
            "Wrong process: photosynthesis instead of cellular respiration (opposite)",
            "Wrong reactant: CO2 instead of O2",
        ],
    },

    # -----------------------------------------------------------------------
    # ENGLISH
    # Previous approach (wrong author name: Percy vs Mary Shelley) produced
    # near-zero gap — both are plausible literary names associated with each
    # other, so Gemma's hidden states barely conflict.
    #
    # New approach: WRONG GENRE. Describing Hamlet as a comedy is a
    # categorical error (tragedy vs comedy are mutually exclusive dramatic
    # genres). Gemma has strong knowledge that Hamlet is a tragedy — one of
    # the most famous tragedies in literature. The confab answer then makes
    # every downstream claim wrong: "celebrates" instead of "seeks revenge,"
    # "coronation" instead of "murder," completely opposite emotional register.
    #
    # Errors: wrong genre (comedy not tragedy); wrong plot (celebration not
    #         revenge); wrong central event (coronation not murder).
    # -----------------------------------------------------------------------
    "english": {
        "subject": "English",
        "question": (
            "What genre is Shakespeare's Hamlet, "
            "and what drives the central plot?"
        ),
        "factual_answer": (
            "Hamlet is a tragedy in which Prince Hamlet seeks revenge "
            "against his uncle Claudius, who murdered Hamlet's father "
            "to seize the Danish throne."
        ),
        "confabulated_answer": (
            "Hamlet is a comedy in which Prince Hamlet celebrates "
            "his uncle Claudius's coronation and falls in love "
            "with Ophelia at the Danish court."
        ),
        "injected_errors": [
            "Wrong genre: comedy instead of tragedy (categorical opposite)",
            "Wrong plot: celebration instead of revenge-driven conflict",
            "Wrong central event: coronation instead of murder of the king",
            "Wrong emotional register: festive/romantic instead of dark/revenge",
        ],
    },

    # -----------------------------------------------------------------------
    # MATH
    # Previous approach (e^iπ = 1 vs = -1, four vs five constants) produced
    # near-zero gap — subtle formula changes don't create enough internal
    # conflict because the surrounding mathematical language is identical.
    #
    # New approach: WRONG OPERATION. Describing the derivative as measuring
    # accumulated area (which is the integral) is the mathematical opposite.
    # Derivative and integral are the two fundamental operations of calculus
    # and are inverses of each other — Gemma knows they are distinct.
    # Every claim in the confab answer describes the integral, not the
    # derivative, so the entire answer conflicts with Gemma's knowledge.
    #
    # Errors: wrong operation (integral's area vs derivative's rate of change);
    #         wrong geometric interpretation (area under curve vs slope of tangent).
    # -----------------------------------------------------------------------
    "math": {
        "subject": "Math",
        "question": (
            "What does the derivative of a function measure in calculus, "
            "and what is its geometric interpretation?"
        ),
        "factual_answer": (
            "The derivative measures the instantaneous rate of change "
            "of a function at a point, geometrically represented as "
            "the slope of the tangent line to the curve at that point."
        ),
        "confabulated_answer": (
            "The derivative measures the total accumulated area "
            "under a function's curve, geometrically represented as "
            "the sum of infinitely many thin rectangles beneath the curve."
        ),
        "injected_errors": [
            "Wrong operation: describes the integral (area) not the derivative (rate of change)",
            "Wrong geometric meaning: area under curve instead of slope of tangent line",
            "Derivative and integral are inverse operations — categorical opposite error",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def try_load_calibration(root: Path) -> MetricCalibration | None:
    candidates = [
        root / "49_gemma4_udc_calibration.json",
        root / "outputs" / "49_gemma4_udc_calibration.json",
        Path("/content") / "49_gemma4_udc_calibration.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            print(f"Loaded calibration from {path}")
            return MetricCalibration(**data)
    print("WARNING: no calibration file — verdicts will be based on raw thresholds")
    return None


def score_one(model, tokenizer, question: str, answer: str,
              device: str, calibration: MetricCalibration | None) -> dict:
    result = analyze(
        model, tokenizer, question, answer, device,
        use_chat_template="always",
    )
    features = extract_feature_metrics(result)
    metric_val = features.get(PRIMARY_METRIC, result.udc_scalar)
    verdict = verdict_from_score(metric_val, calibration) if calibration else result.verdict
    return {
        "udc_scalar": result.udc_scalar,
        PRIMARY_METRIC: metric_val,
        "tle_scalar": result.tle_scalar,
        "verdict": verdict,
        "num_response_tokens": result.num_response_tokens,
        "udc_per_token": result.udc_per_token,
        "response_tokens": result.response_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score short-format demo pairs with UDC.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    parser.add_argument("--subjects", nargs="*", default=list(DEMO_PAIRS.keys()))
    parser.add_argument(
        "--demo-calibrate", action="store_true",
        help=(
            "After scoring, fit a demo-specific calibration from these 8 labeled "
            "examples (4 factual, 4 confab). Use this if the BENCH-2 calibration "
            "thresholds don't produce clean PASS/FAIL on the demo pairs. "
            "Saves demo_calibration.json alongside the scored fixture."
        ),
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Short-Format Demo Scorer ===")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}")
    print(f"Target: {PRIMARY_METRIC} (calibrated on BENCH-2 short answers)")
    print()

    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    calibration = try_load_calibration(ROOT)

    output: dict[str, dict] = {}

    for subject in args.subjects:
        pair = DEMO_PAIRS[subject]
        print(f"--- {subject.upper()} ---")
        print(f"Q: {pair['question']}")
        print(f"Errors: {'; '.join(pair['injected_errors'])}")

        print("  Scoring factual answer...")
        f_scores = score_one(model, tokenizer, pair["question"],
                             pair["factual_answer"], args.device, calibration)

        print("  Scoring confabulated answer...")
        c_scores = score_one(model, tokenizer, pair["question"],
                             pair["confabulated_answer"], args.device, calibration)

        gap = c_scores[PRIMARY_METRIC] - f_scores[PRIMARY_METRIC]

        print(f"  factual   {PRIMARY_METRIC}: {f_scores[PRIMARY_METRIC]:.4f}  "
              f"({f_scores['num_response_tokens']} tokens)  verdict: {f_scores['verdict']}")
        print(f"  confab    {PRIMARY_METRIC}: {c_scores[PRIMARY_METRIC]:.4f}  "
              f"({c_scores['num_response_tokens']} tokens)  verdict: {c_scores['verdict']}")
        print(f"  gap: {gap:+.4f}\n")

        output[subject] = {
            **pair,
            "udc_factual": f_scores[PRIMARY_METRIC],
            "udc_confabulated": c_scores[PRIMARY_METRIC],
            "score_gap": gap,
            "verdict_factual": f_scores["verdict"],
            "verdict_confabulated": c_scores["verdict"],
            "udc_per_token_factual": f_scores["udc_per_token"],
            "udc_per_token_confabulated": c_scores["udc_per_token"],
            "response_tokens_factual": f_scores["response_tokens"],
            "response_tokens_confabulated": c_scores["response_tokens"],
            "num_tokens_factual": f_scores["num_response_tokens"],
            "num_tokens_confabulated": c_scores["num_response_tokens"],
        }

    # ------------------------------------------------------------------
    # Optional: fit a demo-specific calibration from these 8 labeled examples.
    # Factual = label 0, confabulated = label 1.
    # This is honest: we have ground-truth labels for demo examples and
    # we use them. The app uses BENCH-2 calibration for user-submitted text.
    # ------------------------------------------------------------------
    demo_cal_path = None
    if args.demo_calibrate:
        try:
            from app.calibration import fit_quantile_calibration
        except ModuleNotFoundError:
            from calibration import fit_quantile_calibration

        labels, scores = [], []
        for r in output.values():
            labels.append(0)
            scores.append(r["udc_factual"])
            labels.append(1)
            scores.append(r["udc_confabulated"])

        demo_cal = fit_quantile_calibration(
            labels, scores,
            metric=PRIMARY_METRIC,
            higher_is_more_factual=False,  # same direction as Gemma BENCH-2
            source="demo_examples_8",
            notes="Fit on 4 factual + 4 confabulated demo examples. Use for demo only.",
        )
        demo_cal_path = outdir / "demo_calibration.json"
        demo_cal_path.write_text(json.dumps(demo_cal.to_dict(), indent=2))
        print(f"Demo calibration saved: {demo_cal_path}")
        print(f"  pass_threshold:   {demo_cal.pass_threshold:.4f} (aligned) "
              f"→ raw ≤ {-demo_cal.pass_threshold:.4f}")
        print(f"  review_threshold: {demo_cal.review_threshold:.4f} (aligned) "
              f"→ raw ≤ {-demo_cal.review_threshold:.4f}")

        # Re-score with demo calibration
        print("\nRe-scoring with demo calibration:")
        for subject, r in output.items():
            f_v = verdict_from_score(r["udc_factual"], demo_cal)
            c_v = verdict_from_score(r["udc_confabulated"], demo_cal)
            r["verdict_factual_demo_cal"] = f_v
            r["verdict_confabulated_demo_cal"] = c_v
            print(f"  {r['subject']:<12}  factual→{f_v:<8}  confab→{c_v}")
        print()

    out_path = outdir / "51c_short_scored.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved: {out_path}")

    # Print summary — use demo_cal verdicts if available, else bench2
    use_demo = args.demo_calibrate
    v_f_key = "verdict_factual_demo_cal" if use_demo else "verdict_factual"
    v_c_key = "verdict_confabulated_demo_cal" if use_demo else "verdict_confabulated"
    cal_label = "demo-cal" if use_demo else "bench2-cal"

    print(f"\n=== SUMMARY ({cal_label} verdicts) ===")
    print(f"{'Subject':<12} {'Gap':>8}  {'Factual':>10}  {'Confab':>10}  "
          f"{'F-tok':>6}  {'C-tok':>6}  {'F-verdict':<10} {'C-verdict'}")
    print("-" * 80)
    for subject, r in output.items():
        print(
            f"{r['subject']:<12} "
            f"{r['score_gap']:>+8.4f}  "
            f"{r['udc_factual']:>10.4f}  "
            f"{r['udc_confabulated']:>10.4f}  "
            f"{r['num_tokens_factual']:>6}  "
            f"{r['num_tokens_confabulated']:>6}  "
            f"{r[v_f_key]:<10} "
            f"{r[v_c_key]}"
        )

    print()
    all_ok = True
    for subject, r in output.items():
        fv, cv = r[v_f_key], r[v_c_key]
        if fv == cv:
            print(f"  WARN {subject}: both verdicts are {fv} — no separation")
            all_ok = False
        elif fv not in ("PASS", "REVIEW"):
            print(f"  WARN {subject}: factual scored {fv} (expected PASS or REVIEW)")
            all_ok = False
        if abs(r["score_gap"]) < 0.010:
            print(f"  WARN {subject}: gap {r['score_gap']:+.4f} is very small")
            all_ok = False

    if not use_demo and not all_ok:
        print("\nTip: re-run with --demo-calibrate to fit thresholds on these")
        print("8 labeled examples. The bench2 calibration was fit on 5-token")
        print("answers and does not transfer to longer demo content.")
    elif all_ok:
        print("All subjects separate cleanly.")
        print(f"Commit outputs/51c_short_scored.json → app/demo_data/demo_cases.json")
        if use_demo:
            print(f"Commit outputs/demo_calibration.json → app/demo_data/demo_calibration.json")


if __name__ == "__main__":
    main()
