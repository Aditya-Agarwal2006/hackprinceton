"""Precompute 3D UDC geometry for the four curated demo examples.

This is intended for Colab GPU use. It reads the existing demo cases, runs the
Gemma UDC engine with geometry enabled for both the factual and confabulated
answers of each subject, and writes one compact JSON bundle that the Streamlit
app can render instantly on a laptop.

Flat Colab upload list:
  - requirements.txt
  - udc_engine.py
  - geometry.py
  - demo_cases.json
  - precompute_demo_geometry.py

Optional but useful:
  - calibration.py
  - feature_metrics.py
  - 49_gemma4_udc_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "app").exists() else HERE
for candidate in [HERE, ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from app.geometry import (
        extract_response_update_vectors,
        fit_joint_pca_basis,
        project_update_vectors_with_basis,
    )
    from app.udc_engine import _prepare_inputs, analyze, load_model
except ModuleNotFoundError:
    from geometry import (
        extract_response_update_vectors,
        fit_joint_pca_basis,
        project_update_vectors_with_basis,
    )
    from udc_engine import _prepare_inputs, analyze, load_model


DEFAULT_MODEL = "google/gemma-4-e2b-it"
DEFAULT_DEVICE = "cuda"


def _load_demo_cases(root: Path, explicit_path: str | None) -> dict:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            root / "app" / "demo_data" / "demo_cases.json",
            root / "demo_cases.json",
            Path("/content") / "demo_cases.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError("Could not find demo_cases.json")


def _score_geometry(model, tokenizer, prompt: str, answer: str, device: str) -> dict:
    result = analyze(
        model,
        tokenizer,
        prompt,
        answer,
        device,
        use_chat_template="always",
        include_geometry=True,
    )
    return {
        "udc_scalar": result.udc_scalar,
        "num_response_tokens": result.num_response_tokens,
        "response_tokens": result.response_tokens,
    }


def _extract_joint_geometry(model, tokenizer, prompt: str, answer: str, device: str) -> dict:
    inputs, response_start, response_end, _input_format = _prepare_inputs(
        tokenizer,
        prompt,
        answer,
        use_chat_template="always",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output did not include hidden states.")

    # Decode via analyze to keep tokenization / response span behavior aligned
    analyzed = analyze(
        model,
        tokenizer,
        prompt,
        answer,
        device,
        use_chat_template="always",
        include_geometry=False,
    )
    update_vectors = extract_response_update_vectors(
        hidden_states,
        response_start,
        response_end,
        response_tokens=analyzed.response_tokens,
    )
    return {
        "result": analyzed,
        "update_vectors": update_vectors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute 3D geometry for the curated Confab demo examples.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--demo-cases", default=None, help="Optional explicit path to demo_cases.json")
    parser.add_argument("--output", default="outputs/demo_geometry.json")
    args = parser.parse_args()

    demo_cases = _load_demo_cases(ROOT, args.demo_cases)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} on {args.device} ...")
    model, tokenizer = load_model(args.model, args.device)

    bundle: dict[str, dict] = {}
    for subject_key, subject_data in demo_cases.items():
        print(f"[{subject_key}] factual")
        factual = _score_geometry(
            model,
            tokenizer,
            subject_data["question"],
            subject_data["factual_answer"],
            args.device,
        )
        print(f"[{subject_key}] confabulated")
        confabulated = _score_geometry(
            model,
            tokenizer,
            subject_data["question"],
            subject_data["confabulated_answer"],
            args.device,
        )
        print(f"[{subject_key}] joint PCA geometry")
        factual_joint = _extract_joint_geometry(
            model,
            tokenizer,
            subject_data["question"],
            subject_data["factual_answer"],
            args.device,
        )
        confab_joint = _extract_joint_geometry(
            model,
            tokenizer,
            subject_data["question"],
            subject_data["confabulated_answer"],
            args.device,
        )
        basis, explained_variance = fit_joint_pca_basis(
            [factual_joint["update_vectors"], confab_joint["update_vectors"]],
            num_components=3,
        )
        factual_geometry = project_update_vectors_with_basis(
            factual_joint["update_vectors"],
            basis,
            explained_variance_ratio=explained_variance,
        )
        confabulated_geometry = project_update_vectors_with_basis(
            confab_joint["update_vectors"],
            basis,
            explained_variance_ratio=explained_variance,
        )
        bundle[subject_key] = {
            "meta": {
                "model": args.model,
                "device": args.device,
                "metric": "udc geometry (joint PCA projection of layer-update vectors)",
            },
            "question": subject_data["question"],
            "factual_answer": subject_data["factual_answer"],
            "confabulated_answer": subject_data["confabulated_answer"],
            "geometry_factual": factual_geometry.to_dict(),
            "geometry_confabulated": confabulated_geometry.to_dict(),
            "udc_scalar_factual": factual["udc_scalar"],
            "udc_scalar_confabulated": confabulated["udc_scalar"],
            "num_response_tokens_factual": factual["num_response_tokens"],
            "num_response_tokens_confabulated": confabulated["num_response_tokens"],
        }

    output_path.write_text(json.dumps(bundle, indent=2))
    print(f"Saved demo geometry bundle to {output_path}")


if __name__ == "__main__":
    main()
