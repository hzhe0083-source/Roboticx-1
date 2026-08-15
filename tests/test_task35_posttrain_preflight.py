from pathlib import Path

import pytest

from eval_metaworld import require_task35_peg_insert_side, select_eval_tasks
from scripts.check_task35_readiness import WAITER_NEEDLES
from scripts.plan_task35_eval_suite import REQUIRED_MILESTONES
from scripts.summarize_task35_fm_train import ARCHIVE_MILESTONES


def test_preflight_mapping_helper_rejects_wrong_task() -> None:
    tasks = [f"task-{index}" for index in range(49)]
    tasks[35] = "Insert a peg sideways"
    selected = select_eval_tasks(tasks, "35", 49)
    assert require_task35_peg_insert_side(
        selected, {"Insert a peg sideways": "peg-insert-side-v3"}
    ) == "peg-insert-side-v3"
    with pytest.raises(ValueError, match="expected peg-insert-side-v3"):
        require_task35_peg_insert_side(
            selected, {"Insert a peg sideways": "window-open-v3"}
        )


def test_readiness_expects_planned_waiters_and_acceptance_set() -> None:
    assert REQUIRED_MILESTONES == (3000, 6000, 9000, 12000, 15000)
    assert ARCHIVE_MILESTONES[-1] == 15000
    assert any("wait_task35_fm_finished_eval.sh" in needle for needle in WAITER_NEEDLES)
    waiter = Path(__file__).resolve().parent.parent / "scripts" / "wait_task35_fm_finished_eval.sh"
    text = waiter.read_text()
    assert text.count("run_task35_h6_eval_suite.sh") == 1
    assert "for step in 3000 6000 9000 12000 15000; do" in text
    assert "for step in 1000 2000 3000 6000 9000 12000 15000; do" not in text
    assert "live 15000 promotion SHA mismatch" in text
