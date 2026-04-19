"""K2 Think V2 reasoning client for Confab.

This module is intentionally independent from Streamlit and torch. It provides a
thin, env-configurable wrapper around an OpenAI-compatible chat-completions
endpoint so the hackathon app can use K2 as a reasoning/explanation layer on
top of the Gemma + UDC detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any
from urllib import error, request


DEFAULT_K2_MODEL = "K2-Think-V2"
DEFAULT_K2_BASE_URL = "https://api.k2think.ai/v1"
DEFAULT_TIMEOUT_SECONDS = 45
_DOTENV_CACHE: dict[str, str] | None = None


@dataclass(frozen=True)
class K2ClaimAssessment:
    claim: str
    assessment: str
    explanation: str
    correction: str = ""


@dataclass(frozen=True)
class K2ReasoningResult:
    model_name: str
    overall_verdict: str
    student_next_step: str
    claims: list[K2ClaimAssessment]
    raw_text: str
    used_fallback: bool
    provider: str = "k2"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["claims"] = [asdict(claim) for claim in self.claims]
        return payload

    def to_markdown(self) -> str:
        lines: list[str] = []
        if self.claims:
            lines.append("### K2 Reasoning")
            for idx, claim in enumerate(self.claims, start=1):
                lines.append(f"**Claim {idx}: {claim.assessment}**")
                lines.append(f"- Claim: {claim.claim}")
                lines.append(f"- Why: {claim.explanation}")
                if claim.correction.strip():
                    lines.append(f"- Correction: {claim.correction}")
        elif self.raw_text.strip():
            lines.append("### K2 Reasoning")
            lines.append(self.raw_text.strip())

        lines.append("")
        lines.append(f"**Overall verdict:** {self.overall_verdict}")
        if self.student_next_step.strip():
            lines.append(f"**What the student should do next:** {self.student_next_step}")
        return "\n".join(lines).strip()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        dotenv_values = _load_dotenv_values()
        value = dotenv_values.get(name)
    if value is None or not str(value).strip():
        return default
    return value.strip()


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
                value = raw_value.strip().strip('"').strip("'")
                values[key] = value
    except FileNotFoundError:
        pass

    _DOTENV_CACHE = values
    return values


def _k2_config() -> dict[str, Any]:
    api_key = _env("K2_API_KEY") or _env("IFM_API_KEY")
    return {
        "api_key": api_key,
        "base_url": (_env("K2_BASE_URL", DEFAULT_K2_BASE_URL) or DEFAULT_K2_BASE_URL).rstrip("/"),
        "model": _env("K2_MODEL", DEFAULT_K2_MODEL) or DEFAULT_K2_MODEL,
        "timeout": int(_env("K2_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)) or DEFAULT_TIMEOUT_SECONDS),
    }


def _build_messages(
    *,
    question: str,
    answer: str,
    risk_score: float,
    udc_verdict: str,
    layer_summary: str,
    flagged_tokens: list[str] | None = None,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a rigorous educational fact-checker helping students verify "
        "AI-generated content. Reason carefully, step by step. Never guess. "
        "If a claim is uncertain, say UNCERTAIN rather than inventing support. "
        "Return ONLY valid JSON."
    )
    user_prompt = (
        "A student's AI tutor gave the following answer. Our hidden-state analysis "
        f"system flagged it with a confabulation risk score of {risk_score:.2f} "
        f"(higher = riskier) and verdict {udc_verdict}.\n\n"
        f"Question: {question}\n\n"
        f"AI Answer: {answer}\n\n"
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
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("K2 response missing 'choices'.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("K2 response message content was empty.")
    return content.strip()


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _post_chat_completion(messages: list[dict[str, str]]) -> tuple[str, str]:
    config = _k2_config()
    api_key = config["api_key"]
    if not api_key:
        raise RuntimeError("K2_API_KEY or IFM_API_KEY not set")

    url = f"{config['base_url']}/chat/completions"
    body = json.dumps(
        {
            "model": config["model"],
            "messages": messages,
            "temperature": 0.1,
        }
    ).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config["timeout"]) as response:
            raw_payload = response.read().decode("utf-8")
    except error.HTTPError as exc:  # pragma: no cover - networked path
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"K2 HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:  # pragma: no cover - networked path
        raise RuntimeError(f"K2 connection error: {exc}") from exc

    parsed = json.loads(raw_payload)
    text = _extract_message_text(parsed)
    return text, str(config["model"])


def _coerce_claims(items: Any) -> list[K2ClaimAssessment]:
    claims: list[K2ClaimAssessment] = []
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
            K2ClaimAssessment(
                claim=claim,
                assessment=assessment,
                explanation=explanation or "No explanation returned.",
                correction=correction,
            )
        )
    return claims


def parse_reasoning_json(raw_text: str, *, model_name: str, used_fallback: bool = False) -> K2ReasoningResult:
    cleaned = _strip_code_fences(raw_text)
    payload = json.loads(cleaned)
    claims = _coerce_claims(payload.get("claims"))
    overall_verdict = str(payload.get("overall_verdict", "NEEDS_REVIEW")).strip() or "NEEDS_REVIEW"
    student_next_step = str(payload.get("student_next_step", "")).strip()
    return K2ReasoningResult(
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
) -> K2ReasoningResult:
    severity = "LIKELY_CONFABULATED" if risk_score >= 0.75 else "NEEDS_REVIEW"
    token_hint = ", ".join(flagged_tokens or []) or "no specific tokens"
    claim = K2ClaimAssessment(
        claim=answer.strip(),
        assessment="UNCERTAIN" if udc_verdict == "REVIEW" else "INACCURATE",
        explanation=(
            "The hidden-state detector found elevated confabulation risk for this answer. "
            f"Observed pattern: {layer_summary}. Most suspicious tokens: {token_hint}."
        ),
        correction="Use another trusted source or ask the model to regenerate with citations.",
    )
    return K2ReasoningResult(
        model_name=model_name,
        overall_verdict=severity,
        student_next_step=(
            "Do not study from this answer directly. Re-check the flagged claim with a trusted source or regenerate with evidence."
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
) -> K2ReasoningResult:
    """Call K2 Think V2 to reason about the analyzed answer.

    This function is intentionally resilient: if the K2 endpoint is unavailable or
    returns invalid JSON, it can fall back to a deterministic explanation that
    still uses the Gemma + UDC results.
    """

    messages = _build_messages(
        question=question,
        answer=answer,
        risk_score=risk_score,
        udc_verdict=udc_verdict,
        layer_summary=layer_summary,
        flagged_tokens=flagged_tokens,
    )
    try:
        raw_text, model_name = _post_chat_completion(messages)
        return parse_reasoning_json(raw_text, model_name=model_name, used_fallback=False)
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
