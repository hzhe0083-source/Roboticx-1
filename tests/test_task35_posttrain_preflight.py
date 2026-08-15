from eval_metaworld import require_task35_peg_insert_side, select_eval_tasks
import pytest


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
