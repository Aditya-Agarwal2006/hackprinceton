import json
from pathlib import Path

from app.calibration import MetricCalibration, verdict_from_score
from app.scoring import compute_risk_score


def test_demo_cases_match_demo_calibration():
    root = Path(__file__).resolve().parents[1]
    calibration = MetricCalibration(**json.loads((root / "app" / "demo_data" / "demo_calibration.json").read_text()))
    cases = json.loads((root / "app" / "demo_data" / "demo_cases.json").read_text())

    for subject, row in cases.items():
        factual = float(row["udc_factual"])
        confabulated = float(row["udc_confabulated"])

        assert verdict_from_score(factual, calibration) == row["verdict_factual_demo_cal"], subject
        assert verdict_from_score(confabulated, calibration) == row["verdict_confabulated_demo_cal"], subject
        assert compute_risk_score(confabulated, calibration) >= compute_risk_score(factual, calibration), subject
