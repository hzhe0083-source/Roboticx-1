import pytest
import torch

from va_compound.training.gradients import backward_pcgrad


def test_single_task_matches_explicit_owned_gradients():
    weight = torch.nn.Parameter(torch.tensor([2., -1.]))
    excluded = torch.nn.Parameter(torch.tensor(3.))
    result = backward_pcgrad([lambda: [(weight.square().sum() * excluded)]],
                             [("weight", weight)], allow_single_task=True)
    torch.testing.assert_close(weight.grad, torch.tensor([12., -6.]))
    assert excluded.grad is None
    assert result == {"conflicts": 0, "comparisons": 0}


@pytest.mark.parametrize("joint", [True, False])
def test_stage2_joint_disables_hook_incompatible_checkpointing(joint):
    from train_libero import _unfreeze_vision_tail
    class Vision:
        def __init__(self):
            self.model = self
            self.grad_checkpointing = False
        def unfreeze_last(self, count):
            assert count == 6
            self.grad_checkpointing = True
        def set_grad_checkpointing(self, value):
            self.grad_checkpointing = value
    vision = Vision()
    _unfreeze_vision_tail(vision, joint_frontend=joint)
    assert vision.grad_checkpointing is (not joint)


def test_legacy_single_task_still_rejected():
    weight = torch.nn.Parameter(torch.tensor(2.))
    with pytest.raises(ValueError, match="at least two"):
        backward_pcgrad([lambda: weight.square()], [("weight", weight)])
