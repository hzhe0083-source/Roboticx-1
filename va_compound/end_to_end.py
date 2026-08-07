"""End-to-end compound: Qwen3.5 (LoRA) + V-JEPA 2.1 (full) + VACompoundPolicy.

Training consumes raw video frames and instruction text so gradients flow into
the frozen-capable backbones: V-JEPA is fully unfrozen, Qwen is adapted via
LoRA (``apply_lora``), and the VA policy stays fully trainable.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .backbones import QwenTextBackbone, VJEPA21Backbone
from .model import LanguageCache, VACompoundConfig, VACompoundPolicy, VisualMemory

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406))
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225))


class EndToEndPolicy(nn.Module):
    """Raw-video/raw-text interface over the three trainable components."""

    def __init__(
        self,
        text_backbone: QwenTextBackbone,
        vision_backbone: VJEPA21Backbone,
        policy: VACompoundPolicy,
        pooling: str = "flat",
    ) -> None:
        super().__init__()
        self.text_backbone = text_backbone
        self.vision_backbone = vision_backbone
        self.policy = policy
        self.pooling = pooling

    @property
    def config(self) -> VACompoundConfig:
        return self.policy.config

    @staticmethod
    def preprocess_video(frames: Tensor) -> Tensor:
        """[B,T,W,3,H,W] uint8 -> ImageNet-normalized float32."""
        if frames.dtype == torch.uint8:
            frames = frames.float().div_(255.0)
        device = frames.device
        mean = IMAGE_MEAN.to(device).view(1, 1, 1, 3, 1, 1)
        std = IMAGE_STD.to(device).view(1, 1, 1, 3, 1, 1)
        return (frames - mean) / std

    def encode_visual(self, frames: Tensor) -> Tensor:
        """[B,T,W,3,H,W] -> pooled tokens [B,T,Nv,D] (full-grad V-JEPA)."""
        if frames.ndim != 6:
            raise ValueError("frames must have shape [B,T,W,3,H,W]")
        batch, sequence, window, channels, height, width = frames.shape
        if channels != 3:
            raise ValueError("frames must be RGB")
        flat = frames.reshape(batch * sequence, window, channels, height, width)
        tokens = self.vision_backbone(flat, pooling=self.pooling)
        return tokens.reshape(batch, sequence, -1, tokens.shape[-1])

    def build_language_cache(
        self,
        instructions: Sequence[str],
    ) -> LanguageCache:
        """Deduplicate instructions, encode online (LoRA grads), project to VA."""
        unique = list(dict.fromkeys(instructions))
        hidden, mask = self.text_backbone.encode_trainable(unique)
        lookup = {text: index for index, text in enumerate(unique)}
        indices = torch.tensor(
            [lookup[instruction] for instruction in instructions],
            dtype=torch.long,
            device=hidden.device,
        )
        batched = hidden[indices]
        batched_mask = mask[indices]
        return self.policy.build_language_cache(batched, batched_mask)

    def rollout(
        self,
        frames: Tensor,
        instructions: Sequence[str],
        proprio: Tensor,
        previous_action: Tensor,
        noisy_actions: Tensor,
        flow_time: Tensor,
        language_cache: LanguageCache | None = None,
    ) -> tuple[Tensor, Tensor, LanguageCache]:
        """Full forward: encode language+vision, then the VA memory chain.

        Returns (predicted_velocities [B,T,H,A], action_conditions [B,T,H,D],
        language_cache).
        """
        batch, sequence = frames.shape[:2]
        if language_cache is None:
            language_cache = self.build_language_cache(instructions)
        visual = self.encode_visual(self.preprocess_video(frames))

        memory: VisualMemory | None = None
        predicted_velocities = []
        action_conditions = []
        for time_index in range(sequence):
            condition, memory = self.policy.encode_condition(
                visual[:, time_index],
                proprio[:, time_index],
                previous_action[:, time_index],
                language_cache=language_cache,
                visual_memory=memory,
                return_visual_memory=True,
            )
            predicted_velocities.append(
                self.policy.flow_velocity(
                    condition,
                    noisy_actions[:, time_index],
                    flow_time[:, time_index],
                )
            )
            action_conditions.append(condition)
        return (
            torch.stack(predicted_velocities, dim=1),
            torch.stack(action_conditions, dim=1),
            language_cache,
        )


def build_e2e_policy(
    *,
    config: VACompoundConfig,
    device: torch.device,
    language_dtype: str = "bfloat16",
    vision_dtype: str = "bfloat16",
    lora_rank: int = 32,
    lora_alpha: float = 32.0,
    unfreeze_blocks: int | None = None,
    qwen_unfreeze_blocks: int = 0,
    pooling: str = "flat",
    local_files_only: bool = True,
    vision_unfreeze_all: bool = False,
) -> tuple[EndToEndPolicy, dict[str, int]]:
    """Assemble the three components and apply the fine-tuning schedule."""
    import torch as _torch

    def dtype(name: str) -> _torch.dtype:
        return {"float32": _torch.float32, "float16": _torch.float16, "bfloat16": _torch.bfloat16}[name]

    text_backbone = QwenTextBackbone.from_pretrained(
        device=device,
        dtype=language_dtype,
        local_files_only=local_files_only,
    )
    if qwen_unfreeze_blocks > 0:
        # 半解冻：冻结 final norm，防止共享末层变换把指令嵌入几何整体推走
        # （2026-08-06 Codex e2e 设计，防 B40k 式语言坍塌）
        text_backbone.unfreeze_last(qwen_unfreeze_blocks, freeze_final_norm=True)
        adapted = qwen_unfreeze_blocks
    else:
        adapted = text_backbone.apply_lora(rank=lora_rank, alpha=lora_alpha)

    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device,
        dtype=vision_dtype,
        max_tokens=64,
        local_files_only=local_files_only,
    )
    if vision_unfreeze_all:
        vision_backbone.unfreeze_all()  # 真正全量：stem + blocks + norms
        unfreeze_blocks = len(vision_backbone.model.blocks)
    elif unfreeze_blocks is None:
        unfreeze_blocks = len(vision_backbone.model.blocks)
        vision_backbone.unfreeze_last(unfreeze_blocks)
    else:
        vision_backbone.unfreeze_last(unfreeze_blocks)

    policy = VACompoundPolicy(config).to(device)

    counts = {
        "lora_layers": adapted,
        "unfrozen_vjepa_blocks": unfreeze_blocks,
        "unfrozen_qwen_blocks": qwen_unfreeze_blocks,
    }
    return EndToEndPolicy(
        text_backbone=text_backbone,
        vision_backbone=vision_backbone,
        policy=policy,
        pooling=pooling,
    ), counts


def parameter_groups(
    model: EndToEndPolicy,
    *,
    lora_lr: float = 1e-4,
    vision_lr: float = 1e-5,
    policy_lr: float = 1e-4,
    qwen_lr: float = 1e-5,
    weight_decay: float = 1e-4,
) -> list[dict]:
    """Split parameters into lora / qwen / vjepa / policy groups with distinct LR."""
    lora_params = []
    qwen_params = []
    vision_params = []
    policy_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "text_backbone" in name:
            if "lora_a" in name or "lora_b" in name:
                lora_params.append(parameter)
            else:
                qwen_params.append(parameter)
        elif "vision_backbone" in name:
            vision_params.append(parameter)
        else:
            policy_params.append(parameter)
    return [
        {"params": policy_params, "lr": policy_lr, "weight_decay": weight_decay},
        {"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay},
        {"params": qwen_params, "lr": qwen_lr, "weight_decay": weight_decay},
        {"params": vision_params, "lr": vision_lr, "weight_decay": weight_decay},
    ]
