"""Gemini API client for Confab.

This module is intentionally independent from Streamlit and torch. It handles:

- answer generation
- claim extraction
- short validation-report generation
- structured reasoning / explanation on top of UDC results
- evaluation-pair generation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Any

try:
    from google import genai
except ImportError:  # pragma: no cover - exercised indirectly via mocked tests
    genai = None


_MODEL_NAME = "gemini-2.5-flash"
_FALLBACK_MODEL = "gemini-2.5-flash-lite"
_API_KEY = os.environ.get("GEMINI_API_KEY")
_client = genai.Client(api_key=_API_KEY) if genai is not None and _API_KEY else None
_DOTENV_CACHE: dict[str, str] | None = None


@dataclass(frozen=True)
class GeminiClaimAssessment:
    claim: str
    assessment: str
    explanation: str
    correction: str = ""


@dataclass(frozen=True)
class GeminiReasoningResult:
    model_name: str
    overall_verdict: str
    student_next_step: str
    claims: list[GeminiClaimAssessment]
    raw_text: str
    used_fallback: bool
    provider: str = "gemini"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["claims"] = [asdict(claim) for claim in self.claims]
        return payload

    def to_markdown(self) -> str:
        lines: list[str] = []
        if self.claims:
            lines.append("### Gemini Reasoning")
            for idx, claim in enumerate(self.claims, start=1):
                lines.append(f"**Claim {idx}: {claim.assessment}**")
                lines.append(f"- Claim: {claim.claim}")
                lines.append(f"- Why: {claim.explanation}")
                if claim.correction.strip():
                    lines.append(f"- Correction: {claim.correction}")
        elif self.raw_text.strip():
            lines.append("### Gemini Reasoning")
            lines.append(self.raw_text.strip())

        lines.append("")
        lines.append(f"**Overall verdict:** {self.overall_verdict}")
        if self.student_next_step.strip():
            lines.append(f"**What to do next:** {self.student_next_step}")
        return "\n".join(lines).strip()


class _GeminiResult:
    """Wrap a Gemini response with the model that produced it."""

    def __init__(self, text: str, model_used: str):
        self.text = text
        self.model_used = model_used


def _load_dotenv_values() -> dict[str, str]:
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None:
        return _DOTENV_CACHE

    values: dict[str, str] = {}
    root = os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(root, ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                values[key] = raw_value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass

    _DOTENV_CACHE = values
    return values


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        value = _load_dotenv_values().get(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def _get_client():
    """Resolve a Gemini client lazily so late-set env vars still work."""

    global _API_KEY, _client
    if _client is not None:
        return _client
    if genai is None:
        return None

    current_key = _env("GEMINI_API_KEY")
    if current_key != _API_KEY:
        _API_KEY = current_key
        _client = genai.Client(api_key=_API_KEY) if _API_KEY else None
    elif _client is None and current_key:
        _client = genai.Client(api_key=current_key)
    return _client


def _call_gemini(system_instruction: str, user_content: str) -> _GeminiResult:
    """Call Gemini with fallback. Returns result plus model provenance."""
    client = _get_client()
    if client is None:
        if genai is None:
            raise RuntimeError("google-genai is not installed")
        raise RuntimeError("GEMINI_API_KEY not set")

    for model_name in [_MODEL_NAME, _FALLBACK_MODEL]:
        try:
            kwargs = {
                "model": model_name,
                "contents": user_content,
            }
            if genai is not None:
                kwargs["config"] = genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                )
            response = client.models.generate_content(**kwargs)
            return _GeminiResult(text=response.text, model_used=model_name)
        except Exception:
            if model_name == _FALLBACK_MODEL:
                raise
            continue
    raise RuntimeError("All Gemini models failed")


def generate_answer(prompt: str) -> str:
    """Generate a candidate answer to a question using Gemini."""
    try:
        return _call_gemini(
            "Answer the following question concisely and factually.",
            prompt,
        ).text
    except Exception as exc:
        return f"[GEMINI_ERROR: {exc}]"


def extract_claims(text: str) -> list[str]:
    """Break text into individual factual claims."""
    if not text or not text.strip():
        return []

    try:
        result = _call_gemini(
            "Extract individual factual claims from the following text. "
            "Return each claim as a separate numbered line. Each claim should "
            "be one verifiable statement. Number them starting from 1.",
            text,
        )
        raw = result.text
    except Exception:
        return [text]

    return parse_claim_lines(raw, fallback=text)


def parse_claim_lines(raw: str, fallback: str = "") -> list[str]:
    """Parse numbered/bulleted lines into clean claim strings."""
    claims = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()
        if line:
            claims.append(line)

    if claims:
        return claims
    return [fallback] if fallback else []


def generate_validation_report(
    claim: str,
    risk_score: float,
    verdict: str,
    layer_summary: str,
) -> str:
    """Generate a short plain-English validation report for UDC results."""
    fallback = (
        f"Verdict: {verdict}. Confabulation risk: {risk_score:.1%}. "
        f"{layer_summary}. "
        "This is an automated coherence check, not a factual determination."
    )

    try:
        return _call_gemini(
            "You are an AI reliability assistant. Based on this hidden-state "
            "coherence analysis, write a 2-3 sentence validation report.",
            f"Claim analyzed: {claim}\n"
            f"Confabulation risk: {risk_score:.1%}\n"
            f"Verdict: {verdict}\n"
            f"Layer pattern: {layer_summary}\n"
            "Rules: Do NOT decide factual truth. Only describe the coherence "
            "screening result and its implication for review. Include the verdict. "
            "End with: 'This is an automated coherence check, not a factual determination.'",
        ).text
    except Exception:
        return fallback


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _coerce_claims(items: Any) -> list[GeminiClaimAssessment]:
    claims: list[GeminiClaimAssessment] = []
    if not isinstance(items, list):
        return claims
    for item in items:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        assessment = str(item.get("assessment", "UNCERTAIN")).strip().upper() or "UNCERTAIN"
        explanation = str(item.get("explanation", "")).strip()
        correction = str(item.get("correction", "")).strip()
        if not claim:
            continue
        claims.append(
            GeminiClaimAssessment(
                claim=claim,
                assessment=assessment,
                explanation=explanation or "No explanation returned.",
                correction=correction,
            )
        )
    return claims


def parse_reasoning_json(
    raw_text: str,
    *,
    model_name: str,
    used_fallback: bool = False,
) -> GeminiReasoningResult:
    cleaned = _strip_code_fences(raw_text)
    payload = json.loads(cleaned)
    claims = _coerce_claims(payload.get("claims"))
    overall_verdict = str(payload.get("overall_verdict", "NEEDS_REVIEW")).strip() or "NEEDS_REVIEW"
    student_next_step = str(payload.get("student_next_step", "")).strip()
    return GeminiReasoningResult(
        model_name=model_name,
        overall_verdict=overall_verdict,
        student_next_step=student_next_step,
        claims=claims,
        raw_text=cleaned,
        used_fallback=used_fallback,
    )


def build_fallback_reasoning(
    *,
    question: str,
    answer: str,
    risk_score: float,
    udc_verdict: str,
    layer_summary: str,
    flagged_tokens: list[str] | None = None,
    model_name: str = "gemini-1.5-pro",
) -> GeminiReasoningResult:
    severity = "LIKELY_CONFABULATED" if risk_score >= 0.75 else "NEEDS_REVIEW"
    token_hint = ", ".join(flagged_tokens or []) or "no specific tokens"
    claim = GeminiClaimAssessment(
        claim=answer.strip(),
        assessment="UNCERTAIN" if udc_verdict == "REVIEW" else "INACCURATE",
        explanation=(
            "The hidden-state detector found elevated confabulation risk for this answer. "
            f"Observed pattern: {layer_summary}. Most suspicious tokens: {token_hint}."
        ),
        correction="Re-check the flagged claim with a trusted source or regenerate with evidence.",
    )
    return GeminiReasoningResult(
        model_name=model_name,
        overall_verdict=severity,
        student_next_step=(
            "Do not rely on this answer directly. Review the flagged claim and regenerate or verify with a trusted source."
        ),
        claims=[claim],
        raw_text=(
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Risk: {risk_score:.2f}\n"
            f"UDC verdict: {udc_verdict}\n"
            f"Layer summary: {layer_summary}\n"
        ),
        used_fallback=True,
    )


def verify_with_reasoning(
    *,
    question: str,
    answer: str,
    risk_score: float,
    udc_verdict: str,
    layer_summary: str,
    flagged_tokens: list[str] | None = None,
    allow_fallback: bool = True,
) -> GeminiReasoningResult:
    """Use Gemini to explain a UDC-scored answer in structured form."""

    system_prompt = (
        "You are a rigorous educational fact-checker helping a user interpret a "
        "hidden-state confabulation alert. Reason carefully and return ONLY valid JSON."
    )
    user_prompt = (
        "An answer has already been screened by a hidden-state monitor.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"UDC verdict: {udc_verdict}\n"
        f"Confabulation risk score: {risk_score:.2f} (higher = riskier)\n"
        f"Layer summary: {layer_summary}\n"
        f"Flagged tokens: {', '.join(flagged_tokens or []) if flagged_tokens else 'none'}\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "claims": [\n'
        "    {\n"
        '      "claim": "string",\n'
        '      "assessment": "ACCURATE|INACCURATE|UNCERTAIN",\n'
        '      "explanation": "string",\n'
        '      "correction": "string"\n'
        "    }\n"
        "  ],\n"
        '  "overall_verdict": "LIKELY_FACTUAL|NEEDS_REVIEW|LIKELY_CONFABULATED",\n'
        '  "student_next_step": "string"\n'
        "}\n"
        "Do not claim certainty you do not have."
    )

    try:
        result = _call_gemini(system_prompt, user_prompt)
        return parse_reasoning_json(result.text, model_name=result.model_used, used_fallback=False)
    except Exception:
        if not allow_fallback:
            raise
        return build_fallback_reasoning(
            question=question,
            answer=answer,
            risk_score=risk_score,
            udc_verdict=udc_verdict,
            layer_summary=layer_summary,
            flagged_tokens=flagged_tokens,
        )


_LENGTH_THRESHOLD = 0.20


def _word_count(text: str) -> int:
    return len(text.split())


def _length_matched(factual: str, hallucinated: str) -> bool:
    """Check if two answers are within 20% relative word-count difference."""
    f_len = _word_count(factual)
    h_len = _word_count(hallucinated)
    if f_len == 0:
        return h_len == 0
    return abs(f_len - h_len) / f_len <= _LENGTH_THRESHOLD


def _rewrite_to_length(hallucinated: str, target_words: int) -> str | None:
    """Ask Gemini to rewrite a hallucinated answer to a target word count."""
    try:
        return _call_gemini(
            "Rewrite the following text to be approximately the requested "
            "length. Keep it factually wrong in the same way — do not fix "
            "the inaccuracies. Return ONLY the rewritten text, nothing else.",
            f"Rewrite this hallucinated answer to be approximately "
            f"{target_words} words: {hallucinated}",
        ).text
    except Exception:
        return None


def parse_eval_json(raw: str) -> list[dict]:
    """Parse and validate eval-pair JSON from a Gemini response."""
    required_keys = {"prompt", "factual_answer", "hallucinated_answer", "domain"}

    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(pairs, list):
        return []

    valid = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        if not required_keys.issubset(pair.keys()):
            continue
        if any(not isinstance(pair[k], str) or not pair[k].strip() for k in required_keys):
            continue
        valid.append(pair)

    return valid


def generate_eval_dataset(domain: str, num_pairs: int) -> list[dict]:
    """Generate length-matched prompt-answer pairs for evaluating UDC."""
    try:
        result = _call_gemini(
            "You are a dataset generator for AI evaluation.",
            f"Generate {num_pairs} question-answer pairs about {domain}.\n"
            "For each pair provide:\n"
            "- A factual question\n"
            "- A correct, concise factual answer (2-4 sentences)\n"
            "- A hallucinated answer that sounds plausible but is factually wrong\n"
            "CRITICAL: The hallucinated answer MUST be approximately the same "
            "length as the correct answer (within 20% word count). This is "
            "essential to avoid length-based detection shortcuts.\n"
            "Make each question unique and specific — avoid generic questions "
            "like 'what is X' when a more precise question is possible.\n"
            "Return ONLY a JSON array with no other text. Each object must have "
            'exactly these keys: "prompt", "factual_answer", '
            f'"hallucinated_answer", "domain"\n'
            f'The domain field should be "{domain}" for all entries.',
        )
    except Exception:
        return []

    pairs = parse_eval_json(result.text)
    generation_model = result.model_used

    valid = []
    for pair in pairs:
        pair["generator_model"] = generation_model
        factual = pair["factual_answer"]
        hallucinated = pair["hallucinated_answer"]

        if not _length_matched(factual, hallucinated):
            target = _word_count(factual)
            rewrite = _rewrite_to_length(hallucinated, target)
            if rewrite and _length_matched(factual, rewrite):
                pair["hallucinated_answer"] = rewrite.strip()
                pair["generator_model"] += " (rewrite)"
            else:
                continue

        valid.append(pair)

    return valid


__all__ = [
    "GeminiClaimAssessment",
    "GeminiReasoningResult",
    "_GeminiResult",
    "build_fallback_reasoning",
    "extract_claims",
    "generate_answer",
    "generate_eval_dataset",
    "generate_validation_report",
    "parse_claim_lines",
    "parse_eval_json",
    "parse_reasoning_json",
    "verify_with_reasoning",
]
