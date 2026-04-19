#!/usr/bin/env python3
"""Build the demo-data bundle used by the app and audit materials."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = ROOT / "app" / "demo_data"
FIG_DIR = DEMO_DATA / "paper_figures"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _draw_card(title: str, lines: list[str], output_path: Path) -> None:
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    draw.rounded_rectangle((40, 40, width - 40, height - 40), radius=24, outline="#334155", width=3)
    draw.text((80, 80), title, fill="#e2e8f0", font=font_title)

    y = 170
    for line in lines:
        draw.text((80, y), line, fill="#cbd5e1", font=font_body)
        y += 70

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    DEMO_DATA.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    bench2 = _load_json(ROOT / "50_gemma_feature_sweep_bench2.json")
    truthfulqa = _load_json(ROOT / "50_gemma_feature_sweep_truthfulqa.json")

    headline_metric = bench2["top_features_by_auc"][0]["metric"]
    bench2_summary = {
        "benchmark": "BENCH-2 (length-matched HaluEval QA)",
        "headline_metric": headline_metric,
        "auc_udc": bench2["results"][headline_metric]["auc"],
        "auc_tle": bench2["results"]["tle_mean_tok_mean"]["auc"],
        "auc_length": bench2["results"]["response_length"]["auc"],
        "udc_ci": [
            bench2["results"][headline_metric]["ci_lo"],
            bench2["results"][headline_metric]["ci_hi"],
        ],
        "n_pairs": bench2["n_pairs"],
        "truthfulqa_auc": truthfulqa["results"][headline_metric]["auc"],
    }
    _save_json(DEMO_DATA / "bench2_summary.json", bench2_summary)

    cross_arch = {
        "mistral-7b": {
            "udc_auc": 0.684,
            "architecture": "transformer",
            "source": "legacy partial-AUC result",
        },
        "qwen2.5-7b": {
            "udc_auc": 0.708,
            "architecture": "transformer",
            "source": "legacy partial-AUC result",
        },
        "mamba-370m": {
            "udc_auc": 0.726,
            "architecture": "ssm",
            "source": "legacy partial-AUC result",
        },
        "gemma4-e2b": {
            "udc_auc": bench2["results"][headline_metric]["auc"],
            "architecture": "transformer",
            "feature": headline_metric,
            "note": "current hackathon Gemma result",
        },
    }
    _save_json(DEMO_DATA / "cross_arch_results.json", cross_arch)

    _save_json(DEMO_DATA / "gemma4_results.json", bench2)

    _draw_card(
        "Figure 1 — Length Confound",
        [
            "Most raw hallucination gains on HaluEval are inflated by answer length.",
            "Controlled evaluation uses BENCH-2 instead of the raw benchmark.",
            "Gemma BENCH-2 length-only AUC: "
            f"{bench2['results']['response_length']['auc']:.3f}",
        ],
        FIG_DIR / "fig1_length_confound.png",
    )
    _draw_card(
        "Figure 2 — BENCH-2 Headline Result",
        [
            f"Headline metric: {headline_metric}",
            f"Gemma BENCH-2 UDC AUC: {bench2['results'][headline_metric]['auc']:.3f}",
            f"Gemma BENCH-2 partial AUC: {bench2['results'][headline_metric]['partial_auc_len']:.3f}",
            f"Length baseline: {bench2['results']['response_length']['auc']:.3f}",
        ],
        FIG_DIR / "fig2_bench2.png",
    )
    _draw_card(
        "Figure 4 — Cross-Architecture Snapshot",
        [
            "Mistral-7B: 0.684",
            "Qwen2.5-7B: 0.708",
            "Mamba-370M: 0.726",
            f"Gemma4-E2B: {bench2['results'][headline_metric]['auc']:.3f}",
        ],
        FIG_DIR / "fig4_cross_arch.png",
    )
    _draw_card(
        "Figure 6 — Scope Boundary",
        [
            f"BENCH-2 AUC: {bench2['results'][headline_metric]['auc']:.3f}",
            f"TruthfulQA AUC: {truthfulqa['results'][headline_metric]['auc']:.3f}",
            "Pattern: confabulation signal survives; misconception signal does not.",
        ],
        FIG_DIR / "fig6_scope.png",
    )

    print(f"Built demo data bundle in {DEMO_DATA}")


if __name__ == "__main__":
    main()
