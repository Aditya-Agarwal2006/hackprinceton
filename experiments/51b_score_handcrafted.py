"""experiments/51b_score_handcrafted.py — Score pre-written demo pairs with UDC.

Gemma's confabulation generation in 51_demo_cases.py failed: its safety/RLHF
training overrode the confab prompt, producing correct paraphrases rather than
factually wrong answers. This script replaces the confab side with hand-crafted
answers that embed specific factual errors that Gemma knows are wrong.

The key principle: UDC detects internal conflict. When Gemma reads "Archduke
Ferdinand was assassinated in Vienna in 1912," it has stored knowledge that this
is wrong (Sarajevo, 1914). That conflict is what produces geometric incoherence
in the hidden states. Self-generated paraphrases have no such conflict.

Factual answers are kept from 51_demo_cases_best.json (Gemma-generated, good).
Confabulated answers are hand-written with specific embedded errors.

Outputs:
  outputs/51b_handcrafted_scored.json  — final fixture for the app

Colab upload list (flat into /content):
  udc_engine.py, calibration.py, eval_utils.py, feature_metrics.py,
  49_gemma4_udc_calibration.json (if available), 51b_score_handcrafted.py,
  51_demo_cases_best.json (from previous run)

Author: Confab / HackPrinceton 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

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
# Hand-crafted confabulated answers
#
# Each answer looks authoritative and fluent but embeds specific wrong facts
# that Gemma knows are wrong. Wrong facts are listed in comments above each.
#
# Length is kept close to the factual answer to avoid length-bias artifacts.
# ---------------------------------------------------------------------------

# HISTORY — WWI
# Errors: Triple Alliance members wrong (France/Britain/Russia are actually the
# Triple Entente, not the Alliance). Assassination victim wrong (Kaiser Wilhelm
# II was German Emperor, not the assassination target — Archduke Franz Ferdinand
# was). Location wrong (Vienna, not Sarajevo). Year wrong (1912, not 1914).
CONFAB_HISTORY = (
    "The underlying causes of World War I are often summarized by the acronym MAIN: "
    "Militarism, Alliances, Imperialism, and Nationalism. Intense competition among "
    "European powers for colonies and resources, coupled with an escalating arms race, "
    "created a volatile environment. The Triple Alliance, which united France, Britain, "
    "and Russia, meant that any local conflict would rapidly draw in all the major powers. "
    "The assassination of Kaiser Wilhelm II of Germany in Vienna on June 28, 1912, "
    "served as the immediate spark that activated these pre-existing tensions and "
    "pulled every major European nation into the war within weeks."
)

# SCIENCE — Greenhouse effect
# Errors: Greenhouse gases described as reflecting UV radiation inward (wrong —
# they absorb and re-emit infrared, not reflect UV). Primary gases named as
# nitrogen and oxygen (wrong — N2 and O2 are NOT greenhouse gases; CO2 and CH4
# are). Enhanced effect attributed to reduced forest coverage reducing oxygen
# production (wrong — the mechanism is increased CO2 concentration trapping heat).
CONFAB_SCIENCE = (
    "The greenhouse effect is a natural process where certain gases in the Earth's "
    "atmosphere, primarily nitrogen and oxygen, reflect incoming ultraviolet radiation "
    "back toward the surface, preventing it from escaping into space and keeping the "
    "planet warm. This reflection mechanism is essential for maintaining a habitable "
    "temperature, as without it Earth would be far too cold to support life. "
    "The enhanced greenhouse effect occurs when human activities, like burning fossil "
    "fuels and deforestation, significantly reduce the oxygen produced by forests, "
    "which weakens the planet's ability to deflect harmful solar radiation. "
    "This imbalance causes the planet's average temperature to rise, which is the "
    "primary driver of global climate change."
)

# ENGLISH — Mary Shelley / Frankenstein
# Errors: Birth year wrong (1802 instead of 1797). Publication year wrong
# (1823 instead of 1818). Setting of composition wrong ("a literary retreat
# near Rome" — actually written during the 1816 summer at Villa Diodati near
# Lake Geneva with Byron and Polidori). Creature described as named "Prometheus"
# in the novel (the subtitle is "The Modern Prometheus" referring to Victor, not
# the creature's name — the creature has no name).
CONFAB_ENGLISH = (
    "Mary Shelley (1802–1851) was an English novelist known for her contributions "
    "to Gothic literature, most famously through her novel *Frankenstein; or, The "
    "Modern Prometheus*, published in 1823. She wrote the novel during a literary "
    "retreat near Rome, where she was inspired by discussions of galvanism and "
    "the possibility of reanimating dead tissue. The novel calls its creature "
    "Prometheus and deeply explores profound philosophical questions concerning "
    "the limits of human ambition and the ethics of scientific creation. Shelley "
    "raises issues such as the responsibility of the creator toward their creation, "
    "the nature of humanity, and the moral implications of unchecked scientific pursuit."
)

# MATH — Euler's identity
# Errors: Identity stated as e^πi = 1 (wrong — the correct value is e^πi = -1,
# equivalently e^iπ + 1 = 0). Described as connecting four constants (wrong —
# it connects five: e, i, π, 1, 0). Attributed to Euler's 1820 collected works
# (impossible — Euler died in 1783; the identity appears in his 1748 Introductio).
CONFAB_MATH = (
    "Euler's identity is a fundamental relationship in mathematics that connects "
    "four of the most important constants: the mathematical constant e (the base "
    "of the natural logarithm), the imaginary unit i, the mathematical constant π, "
    "and the number 1. It is expressed as e^πi = 1, published in the 1820 edition "
    "of Euler's collected works. Mathematicians consider it remarkable because it "
    "elegantly unifies seemingly disparate areas of mathematics — algebra, calculus, "
    "complex analysis, and trigonometry — into a single, profound equation. "
    "This identity showcases the deep, interconnected structure underlying the "
    "universe of mathematics."
)

HANDCRAFTED_CONFABS: dict[str, str] = {
    "history": CONFAB_HISTORY,
    "science": CONFAB_SCIENCE,
    "english": CONFAB_ENGLISH,
    "math": CONFAB_MATH,
}

# Descriptions of the injected errors for the demo and docs
CONFAB_ERRORS: dict[str, list[str]] = {
    "history": [
        "Triple Alliance members listed as France, Britain, Russia (actually the Triple Entente)",
        "Assassination target named as Kaiser Wilhelm II of Germany (actually Archduke Franz Ferdinand of Austria-Hungary)",
        "Location given as Vienna (actually Sarajevo, Bosnia)",
        "Year given as 1912 (actually 1914)",
    ],
    "science": [
        "Primary greenhouse gases named as nitrogen and oxygen (N2/O2 are NOT greenhouse gases — CO2, CH4, water vapor are)",
        "Mechanism described as reflecting UV radiation (actually absorbing and re-emitting infrared radiation)",
        "Enhanced effect blamed on reduced oxygen from deforestation (actually caused by increased CO2 concentration)",
    ],
    "english": [
        "Mary Shelley's birth year given as 1802 (actually 1797)",
        "Frankenstein publication year given as 1823 (actually 1818)",
        "Novel said to be written near Rome (actually written at Villa Diodati, Lake Geneva, Switzerland)",
        "Creature described as named Prometheus (creature has no name in the novel; 'Prometheus' is the subtitle referring to Victor)",
    ],
    "math": [
        "Identity stated as e^πi = 1 (actually e^πi = -1, i.e. e^iπ + 1 = 0)",
        "Described as connecting four constants (actually five: e, i, π, 1, and 0)",
        "Attributed to Euler's 1820 collected works (impossible — Euler died in 1783; appears in 1748 Introductio)",
    ],
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
    print("WARNING: no calibration file — verdicts will be uncalibrated")
    return None


def try_load_factual_answers(root: Path) -> dict[str, str] | None:
    """Load Gemma-generated factual answers from 51_demo_cases_best.json."""
    candidates = [
        root / "51_demo_cases_best.json",
        root / "outputs" / "51_demo_cases_best.json",
        Path("/content") / "51_demo_cases_best.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            result = {subject: v["factual_answer"] for subject, v in data.items()}
            print(f"Loaded factual answers from {path}")
            return result, data
    print("WARNING: 51_demo_cases_best.json not found — using fallback factual answers")
    return None, None


def score_one(model, tokenizer, question: str, answer: str, device: str,
              calibration: MetricCalibration | None) -> dict:
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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score hand-crafted demo pairs with UDC.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    parser.add_argument("--subjects", nargs="*", default=list(HANDCRAFTED_CONFABS.keys()))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Handcrafted Confab Scorer ===")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}")
    print()

    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    calibration = try_load_calibration(ROOT)
    factual_map, prev_best = try_load_factual_answers(ROOT)

    output: dict[str, dict] = {}

    for subject in args.subjects:
        confab_answer = HANDCRAFTED_CONFABS[subject]

        # Get question from previous run if available
        question = prev_best[subject]["question"] if prev_best else {
            "history": "What were the underlying causes of World War I, and why did a single assassination trigger a global conflict?",
            "science": "How does the greenhouse effect work, and what is the difference between the natural and enhanced greenhouse effect?",
            "english": "Who was Mary Shelley, and what philosophical questions does Frankenstein raise about science and creation?",
            "math": "What does Euler's identity express, and why do mathematicians consider it remarkable?",
        }[subject]

        factual_answer = factual_map[subject] if factual_map else prev_best[subject]["factual_answer"]

        print(f"--- {subject.upper()} ---")
        print(f"Q: {question[:80]}")
        print(f"Errors injected: {'; '.join(CONFAB_ERRORS[subject])}\n")

        print("  Scoring factual answer...")
        factual_scores = score_one(model, tokenizer, question, factual_answer, args.device, calibration)

        print("  Scoring confabulated answer...")
        confab_scores = score_one(model, tokenizer, question, confab_answer, args.device, calibration)

        gap = confab_scores[PRIMARY_METRIC] - factual_scores[PRIMARY_METRIC]

        print(f"  factual  {PRIMARY_METRIC}: {factual_scores[PRIMARY_METRIC]:.4f}  verdict: {factual_scores['verdict']}")
        print(f"  confab   {PRIMARY_METRIC}: {confab_scores[PRIMARY_METRIC]:.4f}  verdict: {confab_scores['verdict']}")
        print(f"  gap: {gap:+.4f}\n")

        output[subject] = {
            "subject": subject.capitalize(),
            "question": question,
            "factual_answer": factual_answer,
            "confabulated_answer": confab_answer,
            "injected_errors": CONFAB_ERRORS[subject],
            "udc_factual": factual_scores[PRIMARY_METRIC],
            "udc_confabulated": confab_scores[PRIMARY_METRIC],
            "score_gap": gap,
            "verdict_factual": factual_scores["verdict"],
            "verdict_confabulated": confab_scores["verdict"],
            "udc_per_token_factual": factual_scores["udc_per_token"],
            "udc_per_token_confabulated": confab_scores["udc_per_token"],
            "num_tokens_factual": factual_scores["num_response_tokens"],
            "num_tokens_confabulated": confab_scores["num_response_tokens"],
        }

    out_path = outdir / "51b_handcrafted_scored.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved: {out_path}")

    print("\n=== SUMMARY ===")
    print(f"{'Subject':<12} {'Gap':>8}  {'Factual':>10}  {'Confab':>10}  {'F-verdict':<10} {'C-verdict'}")
    print("-" * 72)
    for subject, r in output.items():
        print(
            f"{subject.capitalize():<12} "
            f"{r['score_gap']:>+8.4f}  "
            f"{r['udc_factual']:>10.4f}  "
            f"{r['udc_confabulated']:>10.4f}  "
            f"{r['verdict_factual']:<10} "
            f"{r['verdict_confabulated']}"
        )

    print("\nDownload outputs/51b_handcrafted_scored.json → commit as app/demo_data/demo_cases.json")


if __name__ == "__main__":
    main()
