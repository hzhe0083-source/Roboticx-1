"""End-to-end compound: Qwen3.5 (frozen by default) + V-JEPA 2.1 (full) + VACompoundPolicy.

Training consumes raw video frames and instruction text so gradients flow into
the frozen-capable backbones: V-JEPA is fully unfrozen, Qwen stays frozen by
default (``--lora-rank 0``; an explicit positive rank attaches LoRA adapters
via ``apply_lora``), and the VA policy stays fully trainable.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from va_compound.backbones import (
    QwenSemanticBackbone,
    QwenTextBackbone,
    SemanticCompiler,
    VJEPA21Backbone,
    pool_flat_tokens,
)
from va_compound.model import LanguageCache, VACompoundConfig, VACompoundPolicy, VisualMemory

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
        compiler: SemanticCompiler | None = None,
        n_scene_tokens: int = 16,
    ) -> None:
        super().__init__()
        if n_scene_tokens < 1:
            raise ValueError("n_scene_tokens must be positive")
        self.text_backbone = text_backbone
        self.vision_backbone = vision_backbone
        self.policy = policy
        self.pooling = pooling
        self.compiler = compiler
        self.n_scene_tokens = n_scene_tokens
        # P0-4/P0-高优：rollout 内部记录 t=0 编译的场景输入与语义上下文——
        # train.py 的 scene 路径 anchor 损失与 pair flow 分支需要与 rollout
        # 完全一致的输入/上下文（compile_every=0 或无 compiler 时保持 None）。
        self._compile_scene_inputs: tuple[Tensor, Tensor, Tensor] | None = None
        self.first_semantic_context: Tensor | None = None

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

    def _encode_language_hidden(
        self, unique: Sequence[str]
    ) -> tuple[Tensor, Tensor]:
        if isinstance(self.text_backbone, QwenSemanticBackbone):
            prior, mask = self.text_backbone.encode_prior(unique)
            adapted, _ = self.text_backbone.encode_adapted(unique)
            hidden = self.text_backbone.fused_embedding(prior, adapted)
        elif self.policy.config.dino_qwen_cross_modal_bridge:
            layers = list(range(10, 15))
            hierarchy, mask = self.text_backbone.encode_trainable(
                unique, output_layers=layers
            )
            hidden = self.text_backbone.mean_output_layers(hierarchy, layers)
        else:
            hidden, mask = self.text_backbone.encode_trainable(unique)
        return hidden, mask

    def build_language_cache(
        self,
        instructions: Sequence[str],
    ) -> LanguageCache:
        """Deduplicate instructions, encode online (LoRA grads), project to VA.

        With a ``QwenSemanticBackbone`` as the text branch the cache is built
        from the zero-init-gated fused embedding
        ``prior + g ⊙ (adapted − prior)``; otherwise the plain trainable
        encode is used (legacy path, unchanged).
        """
        unique = list(dict.fromkeys(instructions))
        hidden, mask = self._encode_language_hidden(unique)
        lookup = {text: index for index, text in enumerate(unique)}
        indices = torch.tensor(
            [lookup[instruction] for instruction in instructions],
            dtype=torch.long,
            device=hidden.device,
        )
        batched = hidden[indices]
        batched_mask = mask[indices]
        return self.policy.build_language_cache(batched, batched_mask)

    def compile_semantic(
        self,
        instructions: Sequence[str],
        visual_tokens: Tensor,
        semantic_history: Tensor,
        scene_delta: Tensor,
        compiler: SemanticCompiler | None = None,
        return_semantic: bool = False,
    ) -> LanguageCache | tuple[LanguageCache, Tensor]:
        """compile-task 语义编译：视觉 token + 语义历史 + 场景变化 → 语言缓存。

        ``visual_tokens`` [B, Nv, vision_dim] 按 token 轴 adaptive-avg-pool 抽到
        ``n_scene_tokens`` 个，连同 ``semantic_history`` / ``scene_delta``
        （分别 [B, history_in_dim] / [B, vision_dim]）经 ``SemanticCompiler``
        编译为 ``n_readout`` 个语义 token（QwenSemanticBackbone 走 prior/
        adapted/门控融合的 encode_scene_fused，普通 QwenTextBackbone 走冻结
        encode_with_scene，均保留梯度）。语言 hidden 仍用现有路径（fused 或
        encode_trainable），语义 token 拼到语言序列末尾后 build_language_cache
        （语义段 mask 恒真）。

        P0-2：``SemanticCompiler`` 按原始 B 逐样本运行（语义 readout 是场景
        条件化的；相同指令 + 不同场景 → 不同 readout）。纯文本 prior/adapted
        与场景无关，仍按 unique 编码后经 indices 展开省算力。

        ``compiler`` 非 None 时覆盖 ``self.compiler``（rollout 的外部注入路径）。
        ``return_semantic=True``（第二轮架构重构）额外返回展开到样本批的
        readout tokens ``semantic`` [B, n_readout, D]（供 flow_semantic 的
        semantic_context 缓存）；默认 False 返回与旧版相同的 LanguageCache。
        """
        compiler = compiler if compiler is not None else self.compiler
        if compiler is None:
            raise ValueError("compile_semantic requires a SemanticCompiler")
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [batch, tokens, vision_dim]"
            )
        if (
            semantic_history.ndim != 2
            or semantic_history.shape[-1] != compiler.history_in_dim
        ):
            raise ValueError(
                f"semantic_history must have shape [B, {compiler.history_in_dim}]"
            )
        if scene_delta.ndim != 2 or scene_delta.shape[-1] != self.config.vision_dim:
            raise ValueError(f"scene_delta must have shape [B, {self.config.vision_dim}]")
        # 纯文本 prior/adapted 与场景无关，按 unique 编码后经 indices 展开省算力。
        unique = list(dict.fromkeys(instructions))
        hidden, mask = self._encode_language_hidden(unique)
        lookup = {text: index for index, text in enumerate(unique)}
        indices = torch.tensor(
            [lookup[instruction] for instruction in instructions],
            dtype=torch.long,
            device=hidden.device,
        )
        batched = hidden[indices]
        batched_mask = mask[indices]
        scene_tokens = pool_flat_tokens(visual_tokens, self.n_scene_tokens)
        # P0-2：compiler 按原始 B 运行——语义 readout 是场景条件化的逐样本
        # 输出（相同指令 + 不同场景 → 不同语义 readout）；旧实现对 compiler
        # 调用做文本去重，重复指令时 scene_tokens/semantic_history/scene_delta
        # 的 B 与 unique 不匹配直接崩溃。
        semantic, _ = compiler(
            self.text_backbone,
            instructions,
            scene_tokens,
            semantic_history,
            scene_delta,
        )
        semantic = semantic.to(device=batched.device, dtype=batched.dtype)
        extended = torch.cat((batched, semantic), dim=1)
        extended_mask = torch.cat(
            (
                batched_mask,
                torch.ones(
                    batched.shape[0],
                    semantic.shape[1],
                    dtype=torch.bool,
                    device=batched.device,
                ),
            ),
            dim=1,
        )
        cache = self.policy.build_language_cache(extended, extended_mask)
        if return_semantic:
            return cache, semantic
        return cache

    def rollout(
        self,
        frames: Tensor,
        instructions: Sequence[str],
        proprio: Tensor,
        previous_action: Tensor,
        noisy_actions: Tensor,
        flow_time: Tensor,
        language_cache: LanguageCache | None = None,
        compile_every: int = 0,
        compiler: SemanticCompiler | None = None,
    ) -> tuple[Tensor, Tensor, LanguageCache]:
        """Full forward: encode language+vision, then the VA memory chain.

        Returns (predicted_velocities [B,T,H,A], action_conditions [B,T,H,D],
        language_cache).

        ``compile_every > 0`` with a ``compiler`` (explicit argument, else
        ``self.compiler``) enables compile-task recompilation: at ``t == 0``
        and every ``compile_every``-th step the language cache is rebuilt from
        the SemanticCompiler — semantic_history is the current
        ``memory.task.mean(dim=1)`` (zeros when memory/task is unavailable;
        the zero vector uses ``compiler.history_in_dim``, which
        ``build_e2e_policy`` sets to ``config.hidden_dim``), scene_delta is
        ``visual[:, t].mean(dim=1) - visual[:, t-1].mean(dim=1)`` (zeros at
        t == 0).  ``compile_every=0`` (or no compiler) keeps the legacy
        single-cache behavior byte-identical.

        flow_semantic（第二轮架构重构，``policy.config.flow_semantic=True``）：
        最近一次 compile 的 readout tokens（[B, n_readout, D]，展开到样本批）
        缓存为 ``semantic_context`` 并透传给每次 ``flow_velocity``；compile
        间隔步沿用最近一次编译结果。``compile_every=0`` 时 ``semantic_context``
        恒为 None（现有行为不变）。
        """
        batch, sequence = frames.shape[:2]
        compiler = compiler if compiler is not None else self.compiler
        if language_cache is None:
            language_cache = self.build_language_cache(instructions)
        visual = self.encode_visual(self.preprocess_video(frames))

        memory: VisualMemory | None = None
        semantic_context: Tensor | None = None
        predicted_velocities = []
        action_conditions = []
        for time_index in range(sequence):
            if compile_every > 0 and compiler is not None and (
                time_index == 0 or time_index % compile_every == 0
            ):
                if memory is not None and memory.task is not None:
                    semantic_history = memory.task.mean(dim=1)
                else:
                    semantic_history = torch.zeros(
                        batch,
                        compiler.history_in_dim,
                        device=visual.device,
                        dtype=visual.dtype,
                    )
                current_mean = visual[:, time_index].mean(dim=1)
                scene_delta = (
                    torch.zeros_like(current_mean)
                    if time_index == 0
                    else current_mean - visual[:, time_index - 1].mean(dim=1)
                )
                if time_index == 0:
                    # P0-4：记录 t=0 编译的场景输入（train.py 的 scene 路径
                    # anchor/geometry 损失复用同一批输入，保证与编译一致）。
                    self._compile_scene_inputs = (
                        visual[:, time_index],
                        semantic_history,
                        scene_delta,
                    )
                if self.policy.config.flow_semantic:
                    language_cache, semantic_context = self.compile_semantic(
                        instructions,
                        visual[:, time_index],
                        semantic_history,
                        scene_delta,
                        compiler=compiler,
                        return_semantic=True,
                    )
                    if time_index == 0:
                        # P0-高优：pair 反事实分支复用 t=0 的语义上下文。
                        self.first_semantic_context = semantic_context
                else:
                    language_cache = self.compile_semantic(
                        instructions,
                        visual[:, time_index],
                        semantic_history,
                        scene_delta,
                        compiler=compiler,
                    )
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
                    semantic_context=semantic_context,
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
    lora_rank: int = 0,
    lora_alpha: float = 32.0,
    unfreeze_blocks: int | None = None,
    qwen_unfreeze_blocks: int = 0,
    pooling: str = "flat",
    local_files_only: bool = True,
    vision_unfreeze_all: bool = False,
    semantic_adapter: bool = False,
    semantic_lora_rank: int = 8,
    semantic_top_layers: int = 4,
    semantic_anchor_layers: tuple[int, ...] | None = None,
    semantic_lora_suffixes: Sequence[str] | None = None,
    language_max_length: int = 64,
    qwen_keep_layers: int | None = None,
    compile_task: bool = False,
    compile_every: int = 4,
    n_scene_tokens: int = 16,
    compile_n_readout: int = 16,
) -> tuple[EndToEndPolicy, dict[str, int]]:
    """Assemble the three components and apply the fine-tuning schedule.

    ``semantic_adapter=True`` wraps the frozen Qwen text branch in a
    ``QwenSemanticBackbone`` (top-layer LoRA + zero-init gate; mutually
    exclusive with ``lora_rank > 0`` and ``qwen_unfreeze_blocks > 0``).
    ``semantic_anchor_layers=None`` defaults the anchor set to the adapted
    top layers.  ``semantic_lora_suffixes``（第二轮架构重构）限定 LoRA 投影
    后缀子集（None = 全 7 种 q/k/v/o/gate/up/down）；``language_max_length``
    透传给 Qwen tokenizer。

    ``compile_task=True`` (compile-task, 2026-08-07) additionally attaches a
    ``SemanticCompiler`` (language_dim x vision_dim, ``history_in_dim`` =
    ``config.hidden_dim``——memory.task 均值在 VA hidden 空间, 第二轮遗留修复;
    ``n_readout`` = ``compile_n_readout``) to the policy and records
    ``compile_every`` / ``n_scene_tokens`` / ``compile_n_readout`` in counts.
    """
    import torch as _torch

    if config.dino_qwen_cross_modal_bridge and semantic_adapter:
        raise ValueError("DINO/Qwen bridge does not support semantic_adapter")

    def dtype(name: str) -> _torch.dtype:
        return {"float32": _torch.float32, "float16": _torch.float16, "bfloat16": _torch.bfloat16}[name]

    text_backbone = QwenTextBackbone.from_pretrained(
        device=device,
        dtype=language_dtype,
        max_length=language_max_length,
        keep_layers=qwen_keep_layers,
        local_files_only=local_files_only,
    )
    if semantic_adapter:
        if lora_rank != 0 or qwen_unfreeze_blocks != 0:
            raise ValueError(
                "semantic_adapter is mutually exclusive with lora_rank > 0 "
                "and qwen_unfreeze_blocks > 0"
            )
        anchor = semantic_anchor_layers
        if anchor is None:
            n = len(text_backbone.text_model.layers)
            anchor = tuple(range(n - semantic_top_layers, n))
        text_backbone = QwenSemanticBackbone(
            text_backbone,
            lora_rank=semantic_lora_rank,
            lora_alpha=lora_alpha,
            top_layers=semantic_top_layers,
            anchor_layers=anchor,
            lora_suffixes=semantic_lora_suffixes,
        )
        adapted = text_backbone.lora_layer_count
    elif qwen_unfreeze_blocks > 0:
        # 半解冻：冻结 final norm，防止共享末层变换把指令嵌入几何整体推走
        # （2026-08-06 Codex e2e 设计，防 B40k 式语言坍塌）
        text_backbone.unfreeze_last(qwen_unfreeze_blocks, freeze_final_norm=True)
        adapted = qwen_unfreeze_blocks
    elif lora_rank > 0:
        adapted = text_backbone.apply_lora(rank=lora_rank, alpha=lora_alpha)
    else:
        # 默认止血（--lora-rank 0，2026-08-07）：Qwen 完全冻结，不 attach
        # LoRA 适配器。LoRA 微调会压缩指令嵌入几何（实测 pairwise cosine
        # 0.8573 → 0.9994 坍塌），保持原模块结构（不产生 lora_a/lora_b 参数）。
        text_backbone.text_model.requires_grad_(False)
        adapted = 0

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

    compiler = None
    if compile_task:
        compiler = SemanticCompiler(
            language_dim=config.language_dim,
            vision_dim=config.vision_dim,
            history_in_dim=config.hidden_dim,
            n_readout=compile_n_readout,
        ).to(device)

    counts = {
        "lora_layers": adapted,
        "unfrozen_vjepa_blocks": unfreeze_blocks,
        "unfrozen_qwen_blocks": qwen_unfreeze_blocks,
        "semantic_lora_layers": (
            text_backbone.lora_layer_count
            if isinstance(text_backbone, QwenSemanticBackbone)
            else 0
        ),
        "semantic_top_layers": (
            text_backbone.top_layers if isinstance(text_backbone, QwenSemanticBackbone) else 0
        ),
        "compile_every": compile_every,
        "n_scene_tokens": n_scene_tokens,
        "compile_n_readout": compile_n_readout,
    }
    return EndToEndPolicy(
        text_backbone=text_backbone,
        vision_backbone=vision_backbone,
        policy=policy,
        pooling=pooling,
        compiler=compiler,
        n_scene_tokens=n_scene_tokens,
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
