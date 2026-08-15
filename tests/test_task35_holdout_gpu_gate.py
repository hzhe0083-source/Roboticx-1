import pytest

from scripts.eval_dino_metric_policy_task35 import want_holdout_cuda
from scripts.summarize_task35_fm_train import ARCHIVE_MILESTONES


def test_holdout_refuses_cuda_while_trainer_owns_gpu() -> None:
    with pytest.raises(SystemExit, match="refusing to take the GPU"):
        want_holdout_cuda(
            force=False,
            cuda_visible=None,
            cuda_available=True,
            trainer_alive=True,
        )


def test_holdout_cpu_override_and_force_are_allowed() -> None:
    assert (
        want_holdout_cuda(
            force=False,
            cuda_visible="",
            cuda_available=True,
            trainer_alive=True,
        )
        is False
    )
    assert (
        want_holdout_cuda(
            force=True,
            cuda_visible=None,
            cuda_available=True,
            trainer_alive=True,
        )
        is True
    )


def test_15000_is_a_planned_archive_milestone() -> None:
    assert 15000 in ARCHIVE_MILESTONES
    assert 4000 not in ARCHIVE_MILESTONES
