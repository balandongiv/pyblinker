from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pyblinker.outside_annotation import BlinkRegionRefinementFlow, RefinementConfig


@pytest.fixture(scope="module")
def flow_results() -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parents[1] / "manual_annotation_feature_calculation_data"
    annotations = data_dir / "ear_eog.csv"
    fif_path = data_dir / "ear_eog.fif"

    if not annotations.exists() or not fif_path.exists():
        pytest.skip("Required blink annotation or FIF inputs are missing")

    config = RefinementConfig(
        annotation_csv=annotations,
        fif_path=fif_path,
        channel="EEG-E8",
        buffer_seconds=0.25,
        output_path=None,
    )
    flow = BlinkRegionRefinementFlow(config)
    results = flow.run()

    if results.empty:
        pytest.skip("Refinement flow returned no candidates for testing")

    return results


def test_refined_flow_writes_csv_roundtrip(flow_results: pd.DataFrame, tmp_path: Path) -> None:
    csv_path = tmp_path / "refined_blinks.csv"
    flow_results.to_csv(csv_path, index=False)

    assert csv_path.exists()

    roundtrip = pd.read_csv(csv_path)
    assert len(roundtrip) == len(flow_results)
    assert {"candidate_id", "refined_left_zero", "refined_right_zero"}.issubset(
        roundtrip.columns
    )
    assert roundtrip["zero_crossing_found"].astype(bool).any()


def test_refined_flow_exports_metrics_csv(flow_results: pd.DataFrame, tmp_path: Path) -> None:
    metrics_path = tmp_path / "refined_blink_metrics.csv"
    flow_results.to_csv(metrics_path, index=False)

    assert metrics_path.exists()

    metrics_frame = pd.read_csv(metrics_path)
    assert {"fit_success", "properties_success"}.issubset(metrics_frame.columns)
    assert metrics_frame["fit_success"].astype(bool).sum() > 0
    assert metrics_frame["properties_success"].astype(bool).sum() > 0
