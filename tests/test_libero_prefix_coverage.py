import json

import h5py
import numpy as np
import pytest
import torch

import prepare_libero as prepare
from va_compound.data.libero import JOINT_DATA_CONTRACT, _validate_data
from va_compound.utils.flow import masked_flow_matching_loss


def test_p15_bound_reaches_terminal_action_and_preserves_legacy():
    for length in range(156, 400):
        new = prepare._window_max_start(length, joint_frontend=True)
        old = prepare._window_max_start(length, joint_frontend=False)
        assert new - old == 35
        assert new + 105 + 1 + 15 == length
        assert old + 105 + 1 + 15 == length - 35
    assert prepare._window_max_start(120, joint_frontend=True) == -1


def test_real_hdf5_preparation_covers_tail_and_validates_masks(tmp_path, monkeypatch):
    data_dir = tmp_path / "datasets" / "libero_10"
    data_dir.mkdir(parents=True)
    length = 160
    for task in range(10):
        with h5py.File(data_dir / f"task{task}.hdf5", "w") as f:
            data = f.create_group("data")
            data.attrs["problem_info"] = json.dumps({"language_instruction": f"task {task}"})
            if task != 3:
                continue
            for episode in range(50):
                demo = data.create_group(f"demo_{episode}")
                values = np.arange(length, dtype=np.float32) / length
                demo["actions"] = np.repeat(values[:, None], 7, axis=1)
                obs = demo.create_group("obs")
                obs["joint_states"] = np.repeat(values[:, None], 7, axis=1)
                obs["gripper_states"] = np.repeat(values[:, None], 2, axis=1)
    monkeypatch.setattr(prepare, "_official_task_specs", lambda *args: [
        {"task_id": 0, "suite": "libero_10", "local_task_id": 3, "description": "task 3"}
    ])
    monkeypatch.setattr(prepare, "_ensure_longtraj", lambda *args, **kwargs: None)
    args = prepare._parser().parse_args([
        "--data", str(tmp_path / "joint.pt"), "--hdf5-dir", str(data_dir.parent),
        "--longtraj", str(tmp_path / "frames"), "--suites", "libero_10",
        "--local-task-ids", "3", "--dense-windows",
        "--architecture-version", "dual_tower_expert_v1",
    ])
    prepare.prepare_data(args)
    payload = _validate_data(args.data, architecture_version=args.architecture_version)
    assert payload["metadata"]["contract"] == JOINT_DATA_CONTRACT
    assert payload["metadata"]["task_counts"] == [50 * 40]
    terminal_row = payload["crop_start"] == 39
    assert torch.all(payload["action_valid_mask"][terminal_row, -1, :15])
    assert not torch.any(payload["action_valid_mask"][terminal_row, -1, 15:])
    torch.testing.assert_close(payload["actions"][terminal_row, -1, 14, 0], torch.full((50,), 159 / 160))
    assert all(ref[2][-1][0] < 2 * length for ref in payload["world_target_frame_refs"])
    with pytest.raises(ValueError, match="unexpected data contract"):
        _validate_data(args.data)
    payload["action_valid_mask"][0, 0, 0] = False
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="real P15"):
        _validate_data(bad, architecture_version=args.architecture_version)


def test_masked_h50_padding_has_no_loss_or_executed_prediction_influence():
    from tests.test_layerwise_expert_policy import config
    from va_compound import VACompoundPolicy

    model = VACompoundPolicy(config()).eval()
    condition = torch.randn(1, 3, 50, 16)
    noisy = torch.randn(1, 50, 4)
    time = torch.rand(1)
    first = model.flow_velocity(condition, noisy, time)
    changed = noisy.clone()
    changed[:, 15:] += 100
    second = model.flow_velocity(condition, changed, time)
    torch.testing.assert_close(first[:, :15], second[:, :15], rtol=0, atol=0)
    prediction = first[:, None].detach().requires_grad_()
    mask = torch.zeros(1, 1, 50, dtype=torch.bool)
    mask[:, :, :15] = True
    loss = masked_flow_matching_loss(prediction, torch.zeros_like(prediction),
                                    {"action_valid_mask": mask}, prefix_steps=15)[0]
    gradient = torch.autograd.grad(loss, prediction)[0]
    assert gradient[:, :, :15].norm() > 0
    assert torch.count_nonzero(gradient[:, :, 15:]) == 0
