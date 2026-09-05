from __future__ import annotations

import torch
from torch import Tensor

def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def feature_policy_autocast(device: torch.device, enabled: bool):
    """BF16 feature forward with the normal training weight cache."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
        cache_enabled=True,
    )


def feature_no_grad_decode_autocast(device: torch.device, enabled: bool):
    """Keep no-grad proposal casts out of the enclosing training cache."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
        cache_enabled=False,
    )


def ensure_sequence(
    batch: dict[str, Tensor],
    min_sequence_length: int,
) -> dict[str, Tensor]:
    # MT-VJ（dense_readout_mtvj）：batch 无 vision_tokens（在线 dense 用 frames），
    # 序列长度以 actions 为准（2026-08-10）。
    if "vision_tokens" in batch:
        if batch["vision_tokens"].ndim != 4 or batch["actions"].ndim != 4:
            raise ValueError("vision/actions must be paired short sequences")
        sequence_length = batch["vision_tokens"].shape[1]
    elif "frames" in batch:
        if batch["frames"].ndim != 6 or batch["actions"].ndim != 4:
            raise ValueError("frames/actions must be paired short sequences")
        sequence_length = batch["frames"].shape[1]
    else:
        raise ValueError("batch 缺 vision_tokens/frames（无视觉输入的序列校验）")
    if sequence_length < min_sequence_length:
        raise ValueError(
            f"paired VA training requires T>={min_sequence_length}, got T={sequence_length}"
        )
    return batch
