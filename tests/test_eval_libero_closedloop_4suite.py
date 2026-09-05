import argparse

import pytest
import torch

from eval_libero_closedloop import (
    LIBERO_SUITES,
    SUITE_HORIZONS,
    _language_caches,
    _load_exact_parameters,
    _qwen_layerwise_readout,
    _qwen_lora_rank,
    _task_horizon,
    _task_ids,
    _task_specs,
)


def test_four_suite_task_specs_and_default_horizons():
    specs = [
        {
            "task_id": suite_index * 10 + local_id,
            "suite": suite,
            "local_task_id": local_id,
            "description": f"{suite} task {local_id}",
        }
        for suite_index, suite in enumerate(LIBERO_SUITES)
        for local_id in range(10)
    ]

    normalized = _task_specs({"metadata": {"task_specs": specs}})
    assert normalized[-1] == {
        "global_task_id": 39,
        "suite": "libero_10",
        "local_task_id": 9,
        "language": "libero_10 task 9",
    }
    assert _task_ids("0,10,39") == [0, 10, 39]
    assert {
        suite: _task_horizon(0, suite) for suite in LIBERO_SUITES
    } == SUITE_HORIZONS
    assert _task_horizon(123, "libero_goal") == 123


def test_legacy_spatial_specs_and_fail_closed_trained_qwen():
    specs = _task_specs({"metadata": {"tasks": [f"task {i}" for i in range(10)]}})
    assert specs[-1] == {
        "global_task_id": 9,
        "suite": "libero_spatial",
        "local_task_id": 9,
        "language": "task 9",
    }
    with pytest.raises(ValueError, match="lacks qwen_adapter_state_dict"):
        _language_caches(
            {},
            object(),
            torch.device("cpu"),
            specs,
            {"training_contract": {"qwen_joint_trained": True}},
        )
    with pytest.raises(ValueError, match="lacks qwen_trainable_state_dict"):
        _language_caches(
            {},
            object(),
            torch.device("cpu"),
            specs,
            {
                "training_contract": {
                    "qwen_joint_trained": True,
                    "qwen_training": "last4_full_layers20_23_v1",
                }
            },
        )
    with pytest.raises(argparse.ArgumentTypeError):
        _task_ids("0,0")


def test_incremental_checkpoint_contract_is_exact():
    assert _qwen_lora_rank({"qwen_training": "full24_lora_rank16"}) == 16
    module = torch.nn.Linear(2, 1)
    module.requires_grad_(False)
    module.bias.requires_grad_(True)
    _load_exact_parameters(
        module,
        {"bias": torch.tensor([3.0])},
        {"bias"},
        "adapter",
    )
    assert module.bias.item() == 3.0
    with pytest.raises(ValueError, match="state mismatch"):
        _load_exact_parameters(module, {}, {"bias"}, "adapter")


def test_qwen_cross_modal_layers_are_not_averaged():
    hierarchy = {
        layer: torch.full((1, 2, 1), float(layer)) for layer in range(18, 24)
    }

    class AddOne(torch.nn.Module):
        def forward(self, value):
            return value + 1

    base, layerwise = _qwen_layerwise_readout(
        hierarchy, AddOne(), list(range(18, 24))
    )
    assert torch.equal(base, torch.full((1, 2, 1), 24.0))
    assert layerwise.shape == (1, 6, 2, 1)
    assert layerwise[:, :, 0, 0].tolist() == [[18.0, 19.0, 20.0, 21.0, 22.0, 23.0]]
