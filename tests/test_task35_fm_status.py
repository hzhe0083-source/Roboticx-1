from scripts.report_task35_fm_status import closed_loop_complete, evidence_row, render_md


def test_status_labels_closed_loop_as_planned_without_eval50() -> None:
    row = evidence_row(
        {
            "path": "checkpoints/x_step3000.pt",
            "step": 3000,
            "sha256": "abc",
            "validated": True,
            "geometry_l2": 0.005,
            "slices": {
                "clean": {"pair_visible_fraction": 0.95, "pegHead_hole_mean_px": 75.0},
                "recovery": {"pair_visible_fraction": 0.94, "pegHead_hole_mean_px": 80.0},
            },
        }
    )
    assert row["labels"]["checkpoint_contract"] == "supported"
    assert row["labels"]["slice_geometry"] == "partially supported"
    assert row["labels"]["closed_loop"] == "planned"
    assert row["closed_loop_trials"] is None


def test_status_markdown_mentions_planned_closed_loop() -> None:
    text = render_md(
        {
            "generated_at": "now",
            "training": {"latest_step": 3000, "status": "RUNNING"},
            "closed_loop_complete": False,
            "best": None,
            "candidates": [
                {
                    "step": 3000,
                    "validated": True,
                    "geometry_l2": 0.005,
                    "clean_visible": 0.95,
                    "clean_pair_px": 75.0,
                    "recovery_visible": 0.94,
                    "recovery_pair_px": 80.0,
                    "holdout_rmse_px": None,
                    "closed_loop_successes": None,
                    "closed_loop_trials": None,
                    "labels": {
                        "checkpoint_contract": "supported",
                        "slice_geometry": "partially supported",
                        "holdout_metric": "planned",
                        "closed_loop": "planned",
                    },
                }
            ],
        }
    )
    assert "closed_loop_complete: False" in text
    assert "Closed-loop insertion remains planned" in text


def test_closed_loop_complete_ignores_1k_2k_and_requires_acceptance_set() -> None:
    def row(step: int, label: str) -> dict:
        return {"step": step, "labels": {"closed_loop": label}}

    assert closed_loop_complete(
        [
            row(1000, "skipped"),
            row(2000, "skipped"),
            row(3000, "supported"),
            row(6000, "supported"),
            row(9000, "supported"),
            row(12000, "supported"),
            row(15000, "supported"),
        ]
    )
    assert not closed_loop_complete(
        [
            row(1000, "skipped"),
            row(2000, "skipped"),
            row(3000, "supported"),
        ]
    )
    assert evidence_row(
        {
            "path": "checkpoints/x_step1000.pt",
            "step": 1000,
            "validated": True,
            "slices": {},
        }
    )["labels"]["closed_loop"] == "skipped"
