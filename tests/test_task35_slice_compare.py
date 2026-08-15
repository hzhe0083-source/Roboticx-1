import pytest

from scripts.compare_task35_slice_reports import compare_slice_reports


def _report(path: str, mean: float, visible: float) -> dict:
    return {
        "contract": "task35_clean_recovery_slice_v1",
        "checkpoint": path,
        "slices": {
            "clean": {
                "n": 2,
                "pair_visible_fraction": visible,
                "pegHead_hole_px": {"n": 2, "mean": mean},
            },
            "recovery": {
                "n": 1,
                "pair_visible_fraction": visible,
                "pegHead_hole_px": {"n": 1, "mean": mean + 10.0},
            },
            "all": {
                "n": 3,
                "pair_visible_fraction": visible,
                "pegHead_hole_px": {"n": 3, "mean": mean + 3.0},
            },
        },
    }


def test_slice_compare_reports_mean_and_visibility_deltas() -> None:
    report = compare_slice_reports(
        _report("a.pt", 60.0, 1.0),
        _report("b.pt", 80.0, 0.7),
    )
    assert report["layers"]["clean"]["delta_mean_px"] == pytest.approx(20.0)
    assert report["layers"]["clean"]["delta_visible"] == pytest.approx(-0.3)
    assert report["layers"]["recovery"]["right_mean_px"] == pytest.approx(90.0)
