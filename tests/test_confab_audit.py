from pathlib import Path

from app.confab import _load_benchmark_audit_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_audit_snapshot_uses_current_headline_metric():
    snapshot = _load_benchmark_audit_snapshot()

    assert snapshot["headline_metric"] == "udc_median_tok"
    assert snapshot["headline_bench2_auc"] > snapshot["length_auc_bench2"]
    assert snapshot["headline_bench2_auc"] > snapshot["tle_auc_bench2"]
    assert 0.49 <= snapshot["headline_truthfulqa_auc"] <= 0.55


def test_demo_data_bundle_exists():
    demo_data = ROOT / "app" / "demo_data"

    assert (demo_data / "bench2_summary.json").exists()
    assert (demo_data / "cross_arch_results.json").exists()
    assert (demo_data / "gemma4_results.json").exists()
    assert (demo_data / "paper_figures" / "fig1_length_confound.png").exists()
    assert (demo_data / "paper_figures" / "fig2_bench2.png").exists()
    assert (demo_data / "paper_figures" / "fig4_cross_arch.png").exists()
    assert (demo_data / "paper_figures" / "fig6_scope.png").exists()
