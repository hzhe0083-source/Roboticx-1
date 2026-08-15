from pathlib import Path

import pytest
import torch

from scripts.eval_dino_metric_policy_task35 import (
    encode_dino_frames_one_at_a_time,
    want_holdout_cuda,
)
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


class _FakeDino:
    image_size = 224
    feature_dim = 4

    def forward_hierarchical_dense(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        n = int(images.shape[0])
        tokens = torch.zeros(n, 2, self.feature_dim)
        tokens[..., 0] = images.mean(dim=(1, 2, 3)).view(n, 1)
        return {5: tokens.clone(), 11: tokens}


def test_holdout_one_frame_encode_keeps_batch_order() -> None:
    images = torch.tensor([0.1, 0.4, 0.8, 1.6]).view(4, 1, 1, 1).expand(4, 3, 2, 2).clone()
    batched = _FakeDino().forward_hierarchical_dense(images)
    sequential = encode_dino_frames_one_at_a_time(_FakeDino(), images)
    assert torch.equal(sequential[11][..., 0], batched[11][..., 0])
    assert sequential[11].shape[0] == 4


def test_posttrain_holdout_sets_cuda_allocator() -> None:
    text = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "run_task35_h6_posttrain_eval.sh"
    ).read_text()
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in text
    assert "encode_dino_frames_one_at_a_time" in (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "eval_dino_metric_policy_task35.py"
    ).read_text()


def test_20000_is_a_planned_archive_milestone() -> None:
    assert 15000 in ARCHIVE_MILESTONES
    assert 18000 in ARCHIVE_MILESTONES
    assert 20000 in ARCHIVE_MILESTONES
    assert 4000 not in ARCHIVE_MILESTONES
