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


def test_readiness_ignores_inspector_direct_and_eval_needles() -> None:
    from scripts.task35_proc import is_inspector

    inspector = (
        "python3 - <<'PY'\nneedles = ('train.py', 'direct-head', 'eval_metaworld.py')\nPY"
    )
    assert is_inspector(inspector)


def test_readiness_expects_planned_waiters_and_acceptance_set() -> None:
    assert REQUIRED_MILESTONES == (12000, 15000, 18000, 20000)
    assert ARCHIVE_MILESTONES[-1] == 20000
    assert any("wait_task35_fm_finished_eval.sh" in needle for needle in WAITER_NEEDLES)
    assert any("continue_task35_h6_to_20k.sh" in needle for needle in WAITER_NEEDLES)
    assert any(needle.endswith(" 15000") for needle in WAITER_NEEDLES)
    assert any(needle.endswith(" 18000") for needle in WAITER_NEEDLES)
    waiter = Path(__file__).resolve().parent.parent / "scripts" / "wait_task35_fm_finished_eval.sh"
    text = waiter.read_text()
    assert text.count("run_task35_h6_eval_suite.sh") == 1
    assert "for step in 12000 15000 18000 20000; do" in text
    assert "for step in 3000 6000 9000 12000 15000 18000 20000; do" not in text
    assert "for step in 3000 6000 9000 12000 15000; do" not in text
    assert "live 20000 promotion SHA mismatch" in text
    assert "peek_task35_checkpoint_step.py" in text
    assert "refuse to promote live global_step=" in text
    assert "refuse to promote over" in text
    assert "pipeline gone without a 20000 archive; trying live promote" in text
    assert "no 20000 archive; checking live checkpoint after trainer exit" in text
    resume = Path(__file__).resolve().parent.parent / "scripts" / "continue_task35_h6_to_20k.sh"
    resume_text = resume.read_text()
    assert "--resume-exact" in resume_text
    assert "--steps 14000" in resume_text
    assert "direct-head" not in resume_text
    assert "no archived global_step=6000" in resume_text
    assert "6000 -> 20000" in resume_text
    assert "20k archive present after trainer exit" in resume_text
    assert "trainer exited 0 but 20k archive is still missing" in resume_text
    posttrain = Path(__file__).resolve().parent.parent / "scripts" / "run_task35_h6_posttrain_eval.sh"
    posttrain_text = posttrain.read_text()
    assert "select_task35_best_fm.py" not in posttrain_text
    assert "--expected-step" in posttrain_text
    assert "winner election stays in the suite" in posttrain_text
    suite = Path(__file__).resolve().parent.parent / "scripts" / "run_task35_h6_eval_suite.sh"
    assert suite.read_text().count("select_task35_best_fm.py") == 1
