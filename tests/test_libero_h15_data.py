import pytest
import torch

import prepare_libero as prepare
from tests.test_libero_episode_data import synthetic_hdf5_data
from va_compound.data.libero import H15_DATA_CONTRACT, _validate_data


def test_native_h15_episode_payload(synthetic_hdf5_data, tmp_path, monkeypatch):
    data_dir, frames = synthetic_hdf5_data
    monkeypatch.setattr(prepare, "_official_task_specs", lambda *args: [
        {"task_id": 0, "suite": "libero_10", "local_task_id": 3, "description": "task 3"}])
    monkeypatch.setattr(prepare, "_ensure_longtraj", lambda *args, **kwargs: None)
    path = tmp_path / "h15.pt"
    args = prepare._parser().parse_args([
        "--data", str(path), "--hdf5-dir", str(data_dir.parent), "--longtraj", str(frames),
        "--suites", "libero_10", "--local-task-ids", "3", "--dense-windows",
        "--architecture-version", "dual_tower_h15_v1"])
    prepare.prepare_data(args)
    payload = _validate_data(path, architecture_version="dual_tower_h15_v1")
    assert payload["metadata"]["contract"] == H15_DATA_CONTRACT
    assert payload["actions"].shape[1:] == (8, 15, 7)
    assert payload["action_valid_mask"].shape[1:] == (8, 15)
    assert payload["action_valid_mask"][payload["decision_valid_mask"]].all()
    assert payload["world_state_delta"].shape[1:] == (8, 9)
    assert payload["metadata"]["action_horizon"] == 15
    with pytest.raises(ValueError, match="contract"):
        _validate_data(path, architecture_version="dual_tower_expert_v1")
