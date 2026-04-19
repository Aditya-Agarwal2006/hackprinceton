"""Streamlit shell for the Confab curated demo and benchmark audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import torch

try:
    from app.calibration import MetricCalibration
    from app.examples import get_domains, get_example, get_examples_by_domain
    from app.demo_examples import get_demo_examples, get_demo_example, load_scored_example
    from app.gemini_client import extract_claims, generate_answer, verify_with_reasoning
    from app.live_analysis import get_live_model_config, get_live_model_configs, run_live_analysis
    from app.scoring import compute_risk_score
    from app.text_utils import extract_claims_local
    from app.udc_engine import load_model
    from app.visualization import (
        build_focused_direction_comparison_figure,
        build_layer_coherence_chart,
        build_update_geometry_comparison_figure,
        build_risk_gauge,
        build_token_heatmap_html,
        summarize_update_geometry_comparison,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from calibration import MetricCalibration
    from examples import get_domains, get_example, get_examples_by_domain
    from demo_examples import get_demo_examples, get_demo_example, load_scored_example
    from gemini_client import extract_claims, generate_answer, verify_with_reasoning
    from live_analysis import get_live_model_config, get_live_model_configs, run_live_analysis
    from scoring import compute_risk_score
    from text_utils import extract_claims_local
    from udc_engine import load_model
    from visualization import (
        build_focused_direction_comparison_figure,
        build_layer_coherence_chart,
        build_update_geometry_comparison_figure,
        build_risk_gauge,
        build_token_heatmap_html,
        summarize_update_geometry_comparison,
    )


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_CLINICAL_DOMAINS = ("source_credibility", "scientific", "general")


def _verdict_badge(verdict: str) -> str:
    palette = {
        "PASS": ("#ECFDF5", "#15803D", "#16A34A"),
        "REVIEW": ("#FFFBEB", "#B45309", "#F59E0B"),
        "FAIL": ("#FEF2F2", "#991B1B", "#DC2626"),
    }
    background, foreground, dot_color = palette.get(verdict, ("#F1F5F9", "#475569", "#94A3B8"))
    return (
        f"<span style=\"display:inline-flex;align-items:center;gap:6px;"
        f"padding:4px 10px;border-radius:999px;"
        f"background:{background};color:{foreground};"
        f"font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;"
        f"letter-spacing:0.04em;\">"
        f"<span style=\"width:7px;height:7px;border-radius:50%;background:{dot_color};\"></span>"
        f"{verdict}</span>"
    )


def _init_state() -> None:
    clinical_default_domain = next(
        (domain for domain in SUPPORTED_CLINICAL_DOMAINS if domain in get_domains()),
        get_domains()[0],
    )
    defaults = {
        "analysis_source": "Live verification",
        "live_model_key": get_live_model_configs()[0].key,
        "live_device": "cuda" if torch.cuda.is_available() else "cpu",
        "live_prompt": "Question: What is the capital of France? Answer:",
        "live_response": "Paris is the capital of France.",
        "last_result": None,
        "last_raw_result": None,
        "comparison_results": None,
        "clinical_domain": clinical_default_domain,
        "clinical_example_id": get_examples_by_domain(clinical_default_domain)[0].id,
        "clinical_result": None,
        "clinical_raw_result": None,
        "clinical_text": "",
        "clinical_claims": [],
        "clinical_report": "",
        "fixture_example_key": get_demo_examples()[0].key,
        "fixture_comparison_mode": False,
        "k2_result": None,
        "k2_error": None,
        "demo_selected_subject": None,
        "demo_k2_result": None,
        "demo_k2_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def _load_cached_model(model_name: str, device: str) -> tuple[Any, Any]:
    return load_model(model_name, device)


@st.cache_data(show_spinner=False)
def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_optional_json(*candidate_paths: Path) -> dict[str, Any] | None:
    for path in candidate_paths:
        if path.exists():
            return _load_json(str(path))
    return None


def _summarize_layer_pattern(scored: Any) -> str:
    values = list(getattr(scored, "udc_per_layer", []) or [])
    if not values:
        return "No per-layer pattern available."
    min_idx = min(range(len(values)), key=values.__getitem__)
    mean_value = sum(values) / len(values)
    return f"lowest coherence around transition {min_idx}->{min_idx + 1}; mean coherence {mean_value:.3f}"


def _render_summary_card(
    *,
    title: str,
    prompt: str,
    answer: str,
    scored: Any,
    note: str,
) -> None:
    st.markdown(f"### {title}")
    st.markdown(_verdict_badge(scored.calibrated_verdict), unsafe_allow_html=True)
    st.caption(note)

    metric_col, bucket_col = st.columns(2)
    with metric_col:
        st.metric("Headline metric", f"{scored.raw_metric_value:.4f}", scored.headline_metric_label)
    with bucket_col:
        st.metric("Risk bucket", scored.risk_bucket.title(), scored.risk_label)

    st.markdown("**Prompt**")
    st.code(prompt)
    st.markdown("**Candidate answer**")
    st.write(answer)


def _render_result_surface(
    *,
    prompt: str,
    answer: str,
    scored: Any,
    heatmap_source: Any,
    title: str,
    note: str,
    extra_details: dict[str, Any] | None = None,
) -> None:
    _render_summary_card(
        title=title,
        prompt=prompt,
        answer=answer,
        scored=scored,
        note=note,
    )

    gauge_col, details_col = st.columns([1.1, 1.0])
    with gauge_col:
        st.plotly_chart(build_risk_gauge(scored), use_container_width=True)
    with details_col:
        st.markdown("**Analysis details**")
        detail_rows = {
            "calibration_metric": scored.calibration_metric,
            "higher_is_more_factual": scored.higher_is_more_factual,
            "input_format": scored.input_format,
            "response_tokens": scored.num_response_tokens,
        }
        if extra_details:
            detail_rows.update(extra_details)
        st.write(detail_rows)

    st.plotly_chart(
        build_layer_coherence_chart(scored, title="Per-layer hidden-state coherence"),
        use_container_width=True,
    )
    st.markdown(build_token_heatmap_html(heatmap_source), unsafe_allow_html=True)


def _render_fixture_single(example_key: str) -> None:
    example, scored, raw_payload = load_scored_example(example_key)
    _render_result_surface(
        prompt=example.prompt,
        answer=example.answer,
        scored=scored,
        heatmap_source=raw_payload,
        title=example.label,
        note=example.notes,
        extra_details={"fixture_path": str(example.fixture_path)},
    )


def _render_fixture_comparison() -> None:
    factual, factual_scored, factual_payload = load_scored_example("correct_france")
    wrong, wrong_scored, wrong_payload = load_scored_example("wrong_france")
    st.session_state["comparison_results"] = (factual_scored, wrong_scored)

    st.markdown("### France Example Comparison")
    st.caption("Both views below are loaded from precomputed calibrated Gemma 4 fixtures.")

    left_col, right_col = st.columns(2)
    with left_col:
        _render_summary_card(
            title=factual.label,
            prompt=factual.prompt,
            answer=factual.answer,
            scored=factual_scored,
            note=factual.notes,
        )
        st.plotly_chart(build_risk_gauge(factual_scored), use_container_width=True)
    with right_col:
        _render_summary_card(
            title=wrong.label,
            prompt=wrong.prompt,
            answer=wrong.answer,
            scored=wrong_scored,
            note=wrong.notes,
        )
        st.plotly_chart(build_risk_gauge(wrong_scored), use_container_width=True)

    st.plotly_chart(
        build_layer_coherence_chart(
            factual_scored,
            title="Layer coherence comparison",
            comparison_data=wrong_scored,
            primary_label="Correct answer",
            comparison_label="Wrong answer",
        ),
        use_container_width=True,
    )

    heatmap_left, heatmap_right = st.columns(2)
    with heatmap_left:
        st.markdown("**Correct answer token heatmap**")
        st.markdown(build_token_heatmap_html(factual_payload), unsafe_allow_html=True)
    with heatmap_right:
        st.markdown("**Wrong answer token heatmap**")
        st.markdown(build_token_heatmap_html(wrong_payload), unsafe_allow_html=True)


def _load_example_into_editor(example_key: str) -> None:
    example = get_demo_example(example_key)
    st.session_state["live_prompt"] = example.prompt
    st.session_state["live_response"] = example.answer


def _compute_live_result(prompt: str, response: str) -> tuple[Any, dict[str, Any]]:
    model_config = get_live_model_config(st.session_state["live_model_key"])
    device = st.session_state["live_device"]

    with st.spinner(f"Loading {model_config.label} on {device}..."):
        model, tokenizer = _load_cached_model(model_config.model_name, device)
    with st.spinner("Running one-pass UDC analysis..."):
        raw_result, scored_result = run_live_analysis(
            model,
            tokenizer,
            prompt,
            response,
            device=device,
            config=model_config,
        )

    result_payload = {
        "prompt": prompt,
        "response": response,
        "scored": scored_result,
        "model_label": model_config.label,
        "model_name": model_config.model_name,
        "device": device,
    }
    return raw_result, result_payload


def _run_live_verification() -> None:
    prompt = st.session_state["live_prompt"].strip()
    response = st.session_state["live_response"].strip()
    if not prompt or not response:
        st.error("Both prompt and candidate answer are required for live verification.")
        return

    try:
        raw_result, result_payload = _compute_live_result(prompt, response)
    except Exception as exc:
        st.error(
            "Live analysis failed. This can happen if the model is gated, not yet "
            "downloaded, or too heavy for the current device."
        )
        st.exception(exc)
        return

    st.session_state["last_raw_result"] = raw_result
    st.session_state["last_result"] = result_payload
    st.session_state["k2_result"] = None
    st.session_state["k2_error"] = None


def _run_live_generation() -> None:
    prompt = st.session_state["live_prompt"].strip()
    if not prompt:
        st.error("Enter a prompt before asking Gemini to draft an answer.")
        return

    with st.spinner("Gemini is drafting a candidate answer..."):
        drafted = generate_answer(prompt)
    st.session_state["live_response"] = drafted


def _get_flagged_tokens(scored: Any, max_tokens: int = 3) -> list[str]:
    token_pairs = list(zip(scored.response_tokens, scored.risk_scores_per_token))
    token_pairs = sorted(token_pairs, key=lambda item: item[1], reverse=True)
    selected = [token.strip() for token, _risk in token_pairs[:max_tokens] if token.strip()]
    return selected


def _run_k2_explanation() -> None:
    result = st.session_state.get("last_result")
    if not result:
        st.error("Run a live verification before asking Gemini for an explanation.")
        return

    scored = result["scored"]
    question = result["prompt"]
    answer = result["response"]
    layer_summary = _summarize_layer_pattern(scored)
    flagged_tokens = _get_flagged_tokens(scored)

    try:
        with st.spinner("Gemini is reasoning about the answer..."):
            k2_result = verify_with_reasoning(
                question=question,
                answer=answer,
                risk_score=scored.risk_score,
                udc_verdict=scored.calibrated_verdict,
                layer_summary=layer_summary,
                flagged_tokens=flagged_tokens,
            )
    except Exception as exc:
        st.session_state["k2_error"] = str(exc)
        st.error("Gemini explanation failed.")
        st.exception(exc)
        return

    st.session_state["k2_result"] = k2_result
    st.session_state["k2_error"] = None


def _render_k2_panel() -> None:
    result = st.session_state.get("last_result")
    if not result:
        return

    scored = result["scored"]
    st.markdown("### Explain With Gemini")
    st.caption(
        "Gemini is the reasoning layer: Gemma + UDC flags suspicious answers, then Gemini explains which claims look wrong and what to check next."
    )
    button_cols = st.columns([0.35, 0.65])
    with button_cols[0]:
        if st.button("Explain with Gemini", key="explain_with_gemini", use_container_width=True):
            _run_k2_explanation()
    with button_cols[1]:
        st.info(
            f"Current UDC verdict: {scored.calibrated_verdict} | risk: {scored.risk_score:.2f} | flagged tokens: {', '.join(_get_flagged_tokens(scored)) or 'none'}"
        )

    k2_result = st.session_state.get("k2_result")
    if k2_result:
        st.markdown(k2_result.to_markdown())


def _render_live_result() -> None:
    result = st.session_state.get("last_result")
    raw_result = st.session_state.get("last_raw_result")
    if not result or raw_result is None:
        st.info("Run a live verification to populate this result area.")
        return

    scored = result["scored"]
    _render_result_surface(
        prompt=result["prompt"],
        answer=result["response"],
        scored=scored,
        heatmap_source=scored,
        title="Live UDC analysis",
        note=(
            "This view is computed through the live UDC engine, then adapted through "
            "the calibrated app-facing scoring layer."
        ),
        extra_details={
            "model": result["model_label"],
            "model_name": result["model_name"],
            "device": result["device"],
            "raw_udc_scalar": f"{raw_result.udc_scalar:.4f}",
        },
    )
    st.markdown("---")
    _render_k2_panel()


def _render_live_controls() -> None:
    configs = get_live_model_configs()
    config_labels = {config.label: config.key for config in configs}
    current_config = get_live_model_config(st.session_state["live_model_key"])

    selector_col, note_col = st.columns([1.1, 1.0])
    with selector_col:
        chosen_label = st.selectbox("Live model", list(config_labels), index=list(config_labels.values()).index(current_config.key))
        st.session_state["live_model_key"] = config_labels[chosen_label]

        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options = ["cuda", "cpu"]
        current_device = st.session_state["live_device"]
        if current_device not in device_options:
            current_device = device_options[0]
            st.session_state["live_device"] = current_device
        st.selectbox("Device", device_options, key="live_device")
    with note_col:
        selected_config = get_live_model_config(st.session_state["live_model_key"])
        st.info(selected_config.notes)
        st.caption("First load can be slow because the model may need to download and initialize.")

    st.text_input("Prompt", key="live_prompt")
    st.text_area("Candidate answer", key="live_response", height=160)

    action_col, draft_col, verify_col, factual_col, wrong_col, clear_col = st.columns([0.9, 1.0, 1.0, 1.0, 1.0, 0.8])
    with action_col:
        st.info("Enter or paste a candidate answer, then run the UDC verifier.")
    with draft_col:
        if st.button("Draft with Gemini", use_container_width=True):
            _run_live_generation()
    with verify_col:
        if st.button("Verify", type="primary", use_container_width=True):
            _run_live_verification()
    with factual_col:
        if st.button("Load correct France example", use_container_width=True):
            _load_example_into_editor("correct_france")
    with wrong_col:
        if st.button("Load wrong France example", use_container_width=True):
            _load_example_into_editor("wrong_france")
    with clear_col:
        if st.button("Clear result", use_container_width=True):
            st.session_state["last_result"] = None
            st.session_state["last_raw_result"] = None

    _render_live_result()


def _render_fixture_controls() -> None:
    example_options = {example.label: example.key for example in get_demo_examples()}
    labels = list(example_options)
    current_key = st.session_state["fixture_example_key"]
    current_label = next(label for label, key in example_options.items() if key == current_key)
    selected_label = st.selectbox("Fixture example", labels, index=labels.index(current_label))
    st.session_state["fixture_example_key"] = example_options[selected_label]
    st.checkbox("Compare correct vs wrong France examples", key="fixture_comparison_mode")

    if st.session_state["fixture_comparison_mode"]:
        _render_fixture_comparison()
    else:
        st.session_state["comparison_results"] = None
        _render_fixture_single(st.session_state["fixture_example_key"])


def _run_clinical_example(answer_kind: str) -> None:
    example = get_example(st.session_state["clinical_example_id"])
    response = example.factual_answer if answer_kind == "factual" else example.hallucinated_answer
    try:
        raw_result, result_payload = _compute_live_result(example.prompt, response)
    except Exception as exc:
        st.error("Example analysis failed.")
        st.exception(exc)
        return

    st.session_state["clinical_raw_result"] = raw_result
    st.session_state["clinical_result"] = {
        **result_payload,
        "display_name": example.display_name,
        "domain": example.domain,
        "answer_kind": answer_kind,
        "explanation": example.explanation,
    }
    st.session_state["clinical_report"] = ""


def _run_claim_verification(claim: str) -> None:
    prompt = "Assess the following claim for factual coherence.\nClaim:"
    try:
        raw_result, result_payload = _compute_live_result(prompt, claim)
    except Exception as exc:
        st.error("Claim verification failed.")
        st.exception(exc)
        return

    st.session_state["clinical_raw_result"] = raw_result
    st.session_state["clinical_result"] = {
        **result_payload,
        "display_name": "Free-form claim check",
        "domain": "custom",
        "answer_kind": "claim",
        "explanation": "Generated from the pasted text claim-extraction flow.",
    }
    st.session_state["clinical_report"] = ""


def _generate_clinical_report() -> None:
    result = st.session_state.get("clinical_result")
    if not result:
        st.error("Run an example analysis or claim verification before generating a report.")
        return

    scored = result["scored"]
    explanation = verify_with_reasoning(
        question=result["prompt"],
        answer=result["response"],
        risk_score=scored.risk_score,
        udc_verdict=scored.calibrated_verdict,
        layer_summary=_summarize_layer_pattern(scored),
        flagged_tokens=_get_flagged_tokens(scored),
    )
    st.session_state["clinical_report"] = explanation.to_markdown()


def _render_clinical_result() -> None:
    result = st.session_state.get("clinical_result")
    raw_result = st.session_state.get("clinical_raw_result")
    if not result or raw_result is None:
        st.info("Select a domain example or verify an extracted claim to populate this panel.")
        return

    scored = result["scored"]
    _render_result_surface(
        prompt=result["prompt"],
        answer=result["response"],
        scored=scored,
        heatmap_source=scored,
        title=f"{result['display_name']} ({result['answer_kind']})",
        note=result["explanation"],
        extra_details={
            "domain": result["domain"],
            "model": result["model_label"],
            "device": result["device"],
            "raw_udc_scalar": f"{raw_result.udc_scalar:.4f}",
        },
    )

    action_col, report_col = st.columns([0.5, 1.5])
    with action_col:
        if st.button("Generate validation report", key="clinical_generate_report", use_container_width=True):
            _generate_clinical_report()
    with report_col:
        report = st.session_state.get("clinical_report", "")
        if report:
            st.markdown("**Validation report**")
            st.info(report)


def _render_clinical_tab() -> None:
    st.markdown(
        """
        This tab is a lightweight **examples and claim-checking lab**. It is
        useful for exploring short, judge-friendly cases where we want to pair
        the Gemma + UDC detector with Gemini-assisted claim splitting and
        Gemini-based
        explanation. The
        strongest current transfer signal is on short scientific or
        source-credibility style claims, not on long administrative text. Use
        this tab to:

        - analyze short curated general / scientific / credibility examples
        - paste AI-generated study notes or summary text
        - split those notes into atomic claims locally
        - verify those claims with the same Gemma + UDC pipeline used in the
          main Analyze tab
        """
    )
    st.info(
        "Important scope note: treat this as an experimental claim-screening surface. "
        "It is strongest on short generated answers and short extracted claims."
    )

    domain_options = [domain for domain in get_domains() if domain in SUPPORTED_CLINICAL_DOMAINS]
    current_domain = st.session_state["clinical_domain"]
    if current_domain not in domain_options:
        current_domain = domain_options[0]
        st.session_state["clinical_domain"] = current_domain

    domain_col, example_col = st.columns([0.7, 1.3])
    with domain_col:
        st.selectbox("Domain", domain_options, key="clinical_domain")
    selected_examples = get_examples_by_domain(st.session_state["clinical_domain"])
    if st.session_state["clinical_example_id"] not in {example.id for example in selected_examples}:
        st.session_state["clinical_example_id"] = selected_examples[0].id

    example_labels = {example.display_name: example.id for example in selected_examples}
    with example_col:
        selected_label = st.selectbox(
            "Example",
            list(example_labels),
            index=list(example_labels.values()).index(st.session_state["clinical_example_id"]),
        )
        st.session_state["clinical_example_id"] = example_labels[selected_label]

    example = get_example(st.session_state["clinical_example_id"])
    st.caption(example.explanation)

    factual_col, hallucinated_col = st.columns(2)
    with factual_col:
        if st.button("Analyze factual answer", key="clinical_factual", use_container_width=True):
            _run_clinical_example("factual")
    with hallucinated_col:
        if st.button("Analyze hallucinated answer", key="clinical_hallucinated", use_container_width=True):
            _run_clinical_example("hallucinated")

    st.markdown("---")
    st.markdown("**Paste AI-generated text**")
    st.text_area(
        "Paste generated study, science, or explanation text",
        key="clinical_text",
        height=140,
    )
    if st.button("Split into claims", key="clinical_extract_claims"):
        text = st.session_state["clinical_text"].strip()
        if not text:
            st.error("Paste text before extracting claims.")
        else:
            claims = extract_claims(text)
            if len(claims) <= 1:
                claims = extract_claims_local(text)
            st.session_state["clinical_claims"] = claims

    claims = st.session_state.get("clinical_claims", [])
    if claims:
        st.markdown("**Extracted claims**")
        for idx, claim in enumerate(claims):
            claim_col, button_col = st.columns([1.8, 0.4])
            with claim_col:
                st.write(claim)
            with button_col:
                if st.button("Verify", key=f"verify_claim_{idx}", use_container_width=True):
                    _run_claim_verification(claim)

    st.markdown("---")
    _render_clinical_result()


_DEMO_CAL_FIELDS = {
    "metric", "higher_is_more_factual", "pass_threshold", "review_threshold",
    "aligned_hall_median", "aligned_factual_median", "source", "notes",
}

_DEMO_SUBJECTS: list[tuple[str, str]] = [
    ("science", "🔬  Science"),
    ("english", "📚  English"),
    ("math",    "∑  Math"),
]


def _load_demo_calibration() -> MetricCalibration:
    data = _load_json(str(ROOT / "app" / "demo_data" / "demo_calibration.json"))
    return MetricCalibration(**{k: v for k, v in data.items() if k in _DEMO_CAL_FIELDS})


def _load_demo_geometry() -> dict[str, Any] | None:
    return _load_optional_json(
        ROOT / "app" / "demo_data" / "demo_geometry.json",
        ROOT / "outputs" / "demo_geometry.json",
        ROOT / "demo_geometry.json",
    )


def _render_demo_geometry(subject_key: str, subject_data: dict[str, Any]) -> None:
    geometry_bundle = _load_demo_geometry()
    if not geometry_bundle:
        st.info(
            "Demo geometry has not been precomputed yet. Run the Colab precompute script and place "
            "`demo_geometry.json` into `app/demo_data/`."
        )
        return

    subject_geometry = geometry_bundle.get(subject_key)
    if not subject_geometry:
        st.info(f"No precomputed geometry found for subject '{subject_key}'.")
        return

    factual = subject_geometry.get("geometry_factual")
    confabulated = subject_geometry.get("geometry_confabulated")
    meta = subject_geometry.get("meta", {})
    summary = summarize_update_geometry_comparison(factual, confabulated)
    directness_gap = summary["factual_path_efficiency"] - summary["confabulated_path_efficiency"]

    st.markdown("#### Where The Directions Start To Disagree")
    st.caption(
        "Instead of showing the whole answer, we zoom into a tiny slice where the local coherence gap is strongest. "
        "That makes the actual UDC signal visible to a human."
    )

    focused_fig, focused = build_focused_direction_comparison_figure(
        factual,
        confabulated,
        factual_tokens=subject_data.get("response_tokens_factual"),
        confabulated_tokens=subject_data.get("response_tokens_confabulated"),
        title="Normalized local update directions",
    )
    st.plotly_chart(
        focused_fig,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False},
    )

    snippet_col1, snippet_col2 = st.columns(2)
    with snippet_col1:
        st.markdown("**Factual zoomed phrase**")
        st.code(focused["factual_phrase"] or "(empty)", language="text")
    with snippet_col2:
        st.markdown("**Confabulated zoomed phrase**")
        st.code(focused["confabulated_phrase"] or "(empty)", language="text")

    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.metric(
            "Local factual coherence",
            f"{focused['factual_mean_cosine']:.2f}",
            help="Average cosine between neighboring normalized update arrows in the selected slice.",
        )
    with stats_col2:
        st.metric(
            "Local confab coherence",
            f"{focused['confabulated_mean_cosine']:.2f}",
            delta=f"{focused['confabulated_mean_cosine'] - focused['factual_mean_cosine']:+.2f} vs factual",
            help="Lower means the selected arrows disagree more strongly about where the hidden state should move next.",
        )

    st.info(
        "How to read this: every segment is one consecutive hidden-state update, and every segment has been normalized "
        "to the same length. UDC only cares about direction. When the teal segments keep pointing roughly the same way, "
        "their cosine stays high. When the orange segments swing or reverse, the cosine drops, which is why that answer "
        "gets flagged."
    )


def _render_demo_answer_card(
    answer_type: str,
    subject_data: dict[str, Any],
    demo_cal: MetricCalibration,
) -> None:
    tokens        = subject_data[f"response_tokens_{answer_type}"]
    udc_per_token = subject_data[f"udc_per_token_{answer_type}"]
    udc_median    = subject_data[f"udc_{answer_type}"]
    verdict       = subject_data[f"verdict_{answer_type}_demo_cal"]
    answer_text   = subject_data[f"{answer_type}_answer"]

    risk_score          = compute_risk_score(udc_median, demo_cal)
    risk_scores_per_tok = [compute_risk_score(s, demo_cal) for s in udc_per_token]

    st.markdown(_verdict_badge(verdict), unsafe_allow_html=True)
    st.caption(f"UDC = {udc_median:.4f}  ·  risk = {risk_score:.2f}")
    st.markdown(f"> {answer_text}")

    st.plotly_chart(
        build_risk_gauge(risk_score, verdict=verdict, title="Confabulation Risk"),
        use_container_width=True,
    )
    st.markdown(
        build_token_heatmap_html(
            tokens,
            udc_per_token=udc_per_token,
            risk_scores_per_token=risk_scores_per_tok,
        ),
        unsafe_allow_html=True,
    )


def _run_demo_k2_explanation(
    subject_data: dict[str, Any],
    demo_cal: MetricCalibration,
) -> None:
    question      = subject_data["question"]
    confab_answer = subject_data["confabulated_answer"]
    udc_confab    = subject_data["udc_confabulated"]
    verdict       = subject_data["verdict_confabulated_demo_cal"]
    risk_score    = compute_risk_score(udc_confab, demo_cal)
    response_tokens = subject_data.get("response_tokens_confabulated", [])
    token_scores = subject_data.get("udc_per_token_confabulated", [])
    ranked = sorted(
        zip(response_tokens, token_scores),
        key=lambda item: compute_risk_score(item[1], demo_cal),
        reverse=True,
    )
    flagged_tokens = [token.strip() for token, _ in ranked[:3] if str(token).strip()]

    try:
        with st.spinner("Gemini is reasoning about the confabulated answer..."):
            k2_result = verify_with_reasoning(
                question=question,
                answer=confab_answer,
                risk_score=risk_score,
                udc_verdict=verdict,
                layer_summary=f"UDC median = {udc_confab:.4f} (demo-cal; confabulated answer)",
                flagged_tokens=flagged_tokens,
            )
    except Exception as exc:
        st.session_state["demo_k2_error"] = str(exc)
        st.session_state["demo_k2_result"] = None
        return

    st.session_state["demo_k2_result"] = k2_result
    st.session_state["demo_k2_error"] = None


def _render_demo_tab() -> None:
    st.markdown(
        "Confab watches an AI's **hidden-state geometry** during a single forward pass to "
        "detect when it's confabulating, before you ever read the answer."
    )
    st.markdown("#### Pick a subject")

    btn_cols = st.columns(len(_DEMO_SUBJECTS))
    for col, (key, label) in zip(btn_cols, _DEMO_SUBJECTS):
        with col:
            selected_now = st.session_state.get("demo_selected_subject") == key
            btn_type = "primary" if selected_now else "secondary"
            if st.button(label, key=f"demo_subj_{key}", use_container_width=True, type=btn_type):
                st.session_state["demo_selected_subject"] = key
                st.session_state["demo_k2_result"] = None
                st.session_state["demo_k2_error"] = None
                st.rerun()

    selected = st.session_state.get("demo_selected_subject")
    if not selected:
        st.info("Select a subject above to see the detector in action.")
        return

    demo_cases = _load_json(str(ROOT / "app" / "demo_data" / "demo_cases.json"))
    demo_cal   = _load_demo_calibration()
    subject_data = demo_cases[selected]

    st.markdown(f"**Question:** {subject_data['question']}")
    st.markdown("---")

    factual_col, confab_col = st.columns(2)
    with factual_col:
        st.markdown("#### Factual Answer")
        _render_demo_answer_card("factual", subject_data, demo_cal)
    with confab_col:
        st.markdown("#### Confabulated Answer")
        _render_demo_answer_card("confabulated", subject_data, demo_cal)

    # UDC gap summary
    udc_f = subject_data["udc_factual"]
    udc_c = subject_data["udc_confabulated"]
    udc_diff = udc_c - udc_f          # positive = confab is less negative = less coherent
    pct_of_factual = (udc_c / udc_f * 100) if udc_f != 0 else 0.0
    gap_col1, gap_col2, gap_col3 = st.columns(3)
    gap_col1.metric("Factual UDC", f"{udc_f:.4f}")
    gap_col2.metric(
        "Confabulated UDC",
        f"{udc_c:.4f}",
        delta=f"{udc_diff:+.4f} vs factual",
        delta_color="inverse",
    )
    gap_col3.metric(
        "Confab as % of factual coherence",
        f"{pct_of_factual:.1f}%",
        help="Lower % means the confabulated answer has proportionally less coherent hidden-state updates.",
    )

    st.markdown("---")
    _render_demo_geometry(selected, subject_data)

    # Gemini reasoning section
    st.markdown("---")
    st.markdown("#### Why is the confabulated answer flagged?")
    st.caption(
        "UDC is the mechanical signal — it detects internal incoherence. "
        "Gemini then explains which specific claims look wrong and what a student should double-check."
    )

    btn_col, info_col = st.columns([0.38, 0.62])
    with btn_col:
        if st.button(
            "Explain with Gemini",
            key="demo_k2_explain",
            use_container_width=True,
            type="primary",
        ):
            _run_demo_k2_explanation(subject_data, demo_cal)
    with info_col:
        confab_verdict = subject_data.get("verdict_confabulated_demo_cal", "FAIL")
        confab_udc     = subject_data.get("udc_confabulated", 0.0)
        confab_risk    = compute_risk_score(confab_udc, demo_cal)
        st.info(
            f"UDC verdict: **{confab_verdict}** · UDC = {confab_udc:.4f} · risk = {confab_risk:.2f}"
        )

    k2_err = st.session_state.get("demo_k2_error")
    k2_res = st.session_state.get("demo_k2_result")

    if k2_err:
        st.error(f"Gemini API error: {k2_err}")
        st.caption("Ensure GEMINI_API_KEY is set. The UDC verdict above stands independently.")
    elif k2_res:
        st.markdown(k2_res.to_markdown())


def _render_analyze_tab() -> None:
    st.markdown(
        """
        This Analyze surface supports both a real live UDC verification flow and
        the fast fixture-backed France demos. The live path uses the UDC engine,
        then routes the raw result through the calibrated scoring adapter before
        rendering the gauge, layer chart, and token heatmap. The heavier 3D
        vector geometry is precomputed for the curated demo examples so the app
        stays responsive on a laptop. Gemini then serves as the generation /
        reasoning layer after verification.
        """
    )
    st.radio(
        "Analysis source",
        ["Live verification", "Fixture demos"],
        key="analysis_source",
        horizontal=True,
    )

    if st.session_state["analysis_source"] == "Live verification":
        _render_live_controls()
    else:
        _render_fixture_controls()


def _load_benchmark_audit_snapshot() -> dict[str, Any]:
    """Collect the current Gemma benchmark numbers used by the audit tab."""

    bench2 = _load_json(str(ROOT / "49_gemma4_bench2.json"))
    truthfulqa = _load_json(str(ROOT / "49_gemma4_truthfulqa.json"))
    sweep_bench2 = _load_json(str(ROOT / "50_gemma_feature_sweep_bench2.json"))
    sweep_truthfulqa = _load_json(str(ROOT / "50_gemma_feature_sweep_truthfulqa.json"))

    headline_metric = sweep_bench2["top_features_by_auc"][0]["metric"]
    demo_validation = _load_optional_json(
        ROOT / "outputs" / "demo_example_validation.json",
        ROOT / "demo_example_validation.json",
        ROOT / "app" / "demo_example_validation.json",
    )
    return {
        "bench2": bench2,
        "truthfulqa": truthfulqa,
        "sweep_bench2": sweep_bench2,
        "sweep_truthfulqa": sweep_truthfulqa,
        "demo_validation": demo_validation,
        "headline_metric": headline_metric,
        "headline_bench2_auc": sweep_bench2["results"][headline_metric]["auc"],
        "headline_truthfulqa_auc": sweep_truthfulqa["results"][headline_metric]["auc"],
        "headline_bench2_partial_auc": sweep_bench2["results"][headline_metric]["partial_auc_len"],
        "length_auc_bench2": bench2["results"]["response_length"]["auc"],
        "tle_auc_bench2": bench2["results"]["tle_scalar"]["auc"],
    }


def _render_benchmark_tab() -> None:
    snapshot = _load_benchmark_audit_snapshot()
    bench2 = snapshot["bench2"]
    truthfulqa = snapshot["truthfulqa"]
    sweep_bench2 = snapshot["sweep_bench2"]
    sweep_truthfulqa = snapshot["sweep_truthfulqa"]
    demo_validation = snapshot["demo_validation"]
    headline_metric = snapshot["headline_metric"]

    st.markdown("### Benchmark Audit")
    st.caption(
        "Static summary of the precomputed Gemma 4 evaluation artifacts already present in this workspace."
    )

    metric_row = st.columns(4)
    metric_row[0].metric(
        "BENCH-2 headline AUC",
        f"{snapshot['headline_bench2_auc']:.3f}",
        headline_metric,
    )
    metric_row[1].metric(
        "TruthfulQA headline AUC",
        f"{snapshot['headline_truthfulqa_auc']:.3f}",
        headline_metric,
    )
    metric_row[2].metric("BENCH-2 length AUC", f"{snapshot['length_auc_bench2']:.3f}")
    metric_row[3].metric("BENCH-2 TLE AUC", f"{snapshot['tle_auc_bench2']:.3f}")

    st.markdown(
        """
        **Why BENCH-2 matters**

        BENCH-2 is the controlled confabulation benchmark in this workspace. It is
        the relevant metric for demo engineering because it length-matches factual
        and hallucinated answers and preserves the confabulation scope story.
        """
    )

    summary_col, top_features_col = st.columns([1.0, 1.1])
    with summary_col:
        st.markdown("**Gemma summary**")
        st.write(
            {
                "model": bench2["model"],
                "use_chat_template": bench2["use_chat_template"],
                "n_pairs_bench2": bench2["n_pairs"],
                "n_pairs_truthfulqa": truthfulqa["n_pairs"],
                "headline_metric": headline_metric,
                "headline_bench2_auc": snapshot["headline_bench2_auc"],
                "headline_truthfulqa_auc": snapshot["headline_truthfulqa_auc"],
                "headline_bench2_partial_auc": snapshot["headline_bench2_partial_auc"],
            }
        )
    with top_features_col:
        st.markdown("**Top BENCH-2 Gemma features**")
        top_rows = [
            {
                "metric": row["metric"],
                "auc": round(row["auc"], 4),
                "higher_is_more_factual": row["higher_is_more_factual"],
            }
            for row in sweep_bench2["top_features_by_auc"][:5]
        ]
        st.table(top_rows)

    st.markdown(
        """
        **Interpretation**

        The current engineering default remains calibrated `udc_median_tok` on Gemma 4 E2B:
        it is slightly stronger than mean UDC on BENCH-2 while staying near chance on
        TruthfulQA, which supports the confabulation-versus-misconception story.
        """
    )

    if demo_validation:
        summary = demo_validation["summary"]
        st.markdown("### Curated Demo Example Validation")
        st.write(
            {
                "total_examples": summary["total_examples"],
                "correct_direction_rate": summary["correct_direction_rate"],
                "top_working_ids": summary["top_working_ids"],
                "failing_ids": summary["failing_ids"],
            }
        )


def _render_about_tab() -> None:
    st.markdown("### About Confab")
    st.markdown(
        """
        **What the tool does**

        Confab checks whether an LLM answer looks internally coherent or
        confabulatory by inspecting its hidden-state dynamics during a single
        forward pass.

        **What UDC means**

        UDC (Update Direction Coherence) measures how consistently the model's
        layer-to-layer update vectors behave across a response. In the abstract,
        more aligned adjacent updates are a sign of geometric coherence. In
        practice, the detector must be interpreted through model-specific
        calibration, because the raw sign and scale can vary by model family.

        **What the tool is good for**

        - controlled benchmark analysis
        - screening risky AI-generated study or tutoring answers for review
        - demonstrating a two-layer verifier: hidden-state detector plus reasoning explainer

        **What it does not do**

        - it does not prove factual truth on its own
        - it is not a replacement for retrieval, citation checking, or expert review
        - it is stronger on confabulation than on misconception-style errors
        - its transfer is currently strongest on benchmark-style or short claim-style text
        """
    )
    st.markdown("**Key current numbers**")
    snapshot = _load_benchmark_audit_snapshot()
    st.write(
        {
            "headline_metric": snapshot["headline_metric"],
            "bench2_auc": round(snapshot["headline_bench2_auc"], 4),
            "truthfulqa_auc": round(snapshot["headline_truthfulqa_auc"], 4),
            "length_auc_bench2": round(snapshot["length_auc_bench2"], 4),
        }
    )


def main() -> None:
    st.set_page_config(page_title="Confab", page_icon="C", layout="wide")
    _init_state()

    # Inject the React frontend's light academic design system as CSS overrides
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* ---- Global page background & text ---- */
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stApp"], .main, .block-container {
            background-color: #F8FAFC !important;
            color: #1E293B !important;
            font-family: 'Inter Tight', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
        }

        /* Sidebar light */
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
            background-color: #F1F5F9 !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        /* Header bar */
        [data-testid="stHeader"], header[data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.85) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid #E2E8F0 !important;
        }

        /* ---- Headings ---- */
        h1, h2, h3, h4, h5, h6,
        [data-testid="stTitle"], [data-testid="stHeading"] {
            color: #0F172A !important;
            font-family: 'Inter Tight', system-ui, sans-serif !important;
            letter-spacing: -0.02em !important;
        }

        /* ---- Body & secondary text ---- */
        p, span, div, label, li {
            font-family: 'Inter Tight', system-ui, sans-serif !important;
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #475569 !important;
        }

        /* ---- Tabs — academic style ---- */
        [data-testid="stTabs"] button[role="tab"] {
            font-family: 'Inter Tight', system-ui, sans-serif !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            color: #475569 !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 8px 16px !important;
        }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #0F172A !important;
            border-bottom-color: #2563EB !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #2563EB !important;
        }

        /* ---- Buttons — scholarly blue primary, ghost secondary ---- */
        [data-testid="stButton"] button {
            font-family: 'Inter Tight', system-ui, sans-serif !important;
            font-weight: 500 !important;
            border-radius: 12px !important;
            transition: background 120ms cubic-bezier(0.22,0.61,0.36,1),
                        transform 80ms cubic-bezier(0.22,0.61,0.36,1),
                        border-color 120ms cubic-bezier(0.22,0.61,0.36,1) !important;
        }
        [data-testid="stButton"] button:active {
            transform: scale(0.98) !important;
        }
        /* Primary (blue) buttons */
        [data-testid="stButton"] button[kind="primary"],
        [data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: 1px solid #2563EB !important;
        }
        [data-testid="stButton"] button[kind="primary"]:hover,
        [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover {
            background-color: #1D4ED8 !important;
            border-color: #1D4ED8 !important;
        }
        /* Secondary (ghost) buttons */
        [data-testid="stButton"] button[kind="secondary"],
        [data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {
            background-color: transparent !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
        }
        [data-testid="stButton"] button[kind="secondary"]:hover,
        [data-testid="stButton"] button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #F1F5F9 !important;
        }

        /* ---- Metric labels — eyebrow style ---- */
        [data-testid="stMetric"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
        }
        [data-testid="stMetric"] > div {
            align-items: center !important;
            justify-content: center !important;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: #475569 !important;
            font-weight: 500 !important;
        }
        [data-testid="stMetricValue"] {
            color: #0F172A !important;
            font-family: 'Inter Tight', system-ui, sans-serif !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 12px !important;
        }

        /* ---- Info/warning/error boxes ---- */
        [data-testid="stAlert"] {
            border-radius: 12px !important;
            border: 1px solid #E2E8F0 !important;
        }
        div[data-testid="stAlert"][data-baseweb*="notification"] {
            background-color: #EFF6FF !important;
        }

        /* ---- Text input / text area ---- */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            color: #0F172A !important;
            font-family: 'Inter Tight', system-ui, sans-serif !important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
        }

        /* ---- Select boxes ---- */
        [data-testid="stSelectbox"] [data-baseweb="select"] {
            border-radius: 12px !important;
        }

        /* ---- Expanders — card-like ---- */
        [data-testid="stExpander"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 20px !important;
            background: #FFFFFF !important;
        }

        /* ---- Code blocks ---- */
        code, pre {
            font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace !important;
        }

        /* ---- Table styling ---- */
        [data-testid="stTable"] th {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: #475569 !important;
            background-color: #F1F5F9 !important;
        }
        [data-testid="stTable"] td {
            border-color: #EEF2F6 !important;
        }

        /* ---- Horizontal rule ---- */
        hr {
            border-color: #E2E8F0 !important;
        }

        /* ---- Plotly chart backgrounds ---- */
        [data-testid="stPlotlyChart"] {
            border-radius: 14px !important;
            overflow: hidden;
        }

        /* ---- Checkbox — blue accent ---- */
        [data-testid="stCheckbox"] svg {
            fill: #2563EB !important;
        }

        /* ---- Radio — blue accent ---- */
        [data-testid="stRadio"] [role="radiogroup"] label {
            font-family: 'Inter Tight', system-ui, sans-serif !important;
        }

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #F1F5F9; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 999px; }
        ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Confab")
    _render_demo_tab()


if __name__ == "__main__":
    main()
