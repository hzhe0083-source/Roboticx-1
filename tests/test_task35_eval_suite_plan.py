from pathlib import Path

import pytest

from scripts.plan_task35_eval_suite import plan_task35_eval_suite


def test_plan_lists_validated_milestones_without_eval50(tmp_path: Path) -> None:
    plan = plan_task35_eval_suite(
        [
            {"path": "checkpoints/a_step1000.pt", "step": 1000, "validated": True},
            {"path": "checkpoints/a_step2000.pt", "step": 2000, "validated": False},
        ],
        logs_dir=tmp_path,
    )
    assert [row["step"] for row in plan["to_eval"]] == []
    assert [row["reason"] for row in plan["skipped"]] == [
        "early milestone kept for mechanism only",
        "not validated",
    ]
    assert plan["already_done"] == []


def test_plan_require_all_refuses_incomplete_milestones(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing validated milestones"):
        plan_task35_eval_suite(
            [{"path": "checkpoints/a_step1000.pt", "step": 1000, "validated": True}],
            logs_dir=tmp_path,
            require_all=True,
        )


def test_plan_require_all_accepts_complete_validated_set(tmp_path: Path) -> None:
    candidates = [
        {
            "path": f"checkpoints/a_step{step}.pt",
            "step": step,
            "validated": True,
        }
        for step in (1000, 2000, 3000, 6000, 9000, 12000, 15000, 18000, 20000)
    ]
    plan = plan_task35_eval_suite(candidates, logs_dir=tmp_path, require_all=True)
    assert [row["step"] for row in plan["to_eval"]] == [
        12000,
        15000,
        18000,
        20000,
    ]
    assert [row["step"] for row in plan["skipped"]] == [1000, 2000, 3000, 6000, 9000]
    assert plan["eval50_paths"] == []
    assert len(plan["planned_eval50_paths"]) == 4


def test_plan_require_eval50_refuses_missing_jsons(tmp_path: Path) -> None:
    candidates = [
        {
            "path": f"checkpoints/a_step{step}.pt",
            "step": step,
            "validated": True,
        }
        for step in (1000, 2000, 3000, 6000, 9000, 12000, 15000, 18000, 20000)
    ]
    with pytest.raises(ValueError, match="missing valid 50-seed eval50"):
        plan_task35_eval_suite(
            candidates, logs_dir=tmp_path, require_all=True, require_eval50=True
        )


def test_plan_skips_early_milestones_for_closed_loop(tmp_path: Path) -> None:
    plan = plan_task35_eval_suite(
        [
            {"path": "checkpoints/a_step1000.pt", "step": 1000, "validated": True},
            {"path": "checkpoints/a_step3000.pt", "step": 3000, "validated": True},
            {"path": "checkpoints/a_step12000.pt", "step": 12000, "validated": True},
        ],
        logs_dir=tmp_path,
    )
    assert [row["step"] for row in plan["to_eval"]] == [12000]
    assert plan["skipped"][0]["reason"] == "early milestone kept for mechanism only"
