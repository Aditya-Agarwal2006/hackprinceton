from dataclasses import dataclass

import app.confab as confab


@dataclass
class _FakeK2Result:
    model_name: str = "K2-Think-V2"
    overall_verdict: str = "LIKELY_CONFABULATED"
    student_next_step: str = "Check a trusted source."
    claims: list | None = None
    raw_text: str = ""
    used_fallback: bool = False

    def to_markdown(self) -> str:
        return "### K2 Reasoning\nMock reasoning result."


class _FakeStreamlit:
    def __init__(self, session_state):
        self.session_state = session_state
        self.errors = []
        self.exceptions = []

    def error(self, message):
        self.errors.append(message)

    def exception(self, exc):
        self.exceptions.append(exc)

    def spinner(self, _message):
        class _Ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def test_run_k2_explanation_uses_last_live_result(monkeypatch):
    fake_st = _FakeStreamlit(
        {
            "last_result": {
                "prompt": "Question: What is the capital of France? Answer:",
                "response": "Lyon is the capital of France.",
                "scored": type(
                    "Scored",
                    (),
                    {
                        "risk_score": 0.92,
                        "calibrated_verdict": "FAIL",
                        "response_tokens": ["Lyon", " is", " the", " capital"],
                        "risk_scores_per_token": [0.99, 0.4, 0.3, 0.5],
                        "udc_per_layer": [-0.1, -0.2, -0.3],
                    },
                )(),
            },
            "k2_result": None,
            "k2_error": None,
        }
    )

    monkeypatch.setattr(confab, "st", fake_st)
    monkeypatch.setattr(confab, "_summarize_layer_pattern", lambda scored: "lowest coherence around 1->2")
    monkeypatch.setattr(confab, "verify_with_reasoning", lambda **kwargs: _FakeK2Result())

    confab._run_k2_explanation()

    assert fake_st.session_state["k2_error"] is None
    assert isinstance(fake_st.session_state["k2_result"], _FakeK2Result)
    assert not fake_st.errors
