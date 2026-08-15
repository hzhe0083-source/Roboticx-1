from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from math import isqrt
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PoolingMode = str  # "flat" | "spatial" | "spatiotemporal" | "dense"


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen linear layer.

    ``base`` stays frozen; only ``lora_a``/``lora_b`` are trainable.  With
    ``rank=0`` the module degenerates to the plain frozen projection.
    ``lora_enabled=False``（P0-1）暂时旁路 LoRA 分支，退化为纯冻结投影——
    ``QwenSemanticBackbone.encode_prior`` 用它跑真正的冻结先验。
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 32,
        alpha: float = 32.0,
        lora_enabled: bool = True,
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("lora rank must be non-negative")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank if rank else 0.0
        self.lora_enabled = lora_enabled
        self.base.requires_grad_(False)
        if rank:
            dtype = base.weight.dtype
            device = base.weight.device
            in_features, out_features = base.in_features, base.out_features
            self.lora_a = nn.Parameter(torch.empty(in_features, rank, dtype=dtype, device=device))
            self.lora_b = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.rank and self.lora_enabled:
            out = out + (x @ self.lora_a @ self.lora_b) * self.scaling
        return out


@contextmanager
def set_lora_enabled(model: nn.Module, enabled: bool):
    """Temporarily enable/disable every ``LoRALinear`` adapter under ``model``.

    Context-manager form guarantees the prior state is restored even when the
    block raises.  Used by ``QwenSemanticBackbone`` so ``encode_prior`` runs
    the true frozen prior (LoRA off) while ``encode_adapted`` keeps the LoRA
    residual (LoRA on).
    """
    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    states = [adapter.lora_enabled for adapter in adapters]
    for adapter in adapters:
        adapter.lora_enabled = enabled
    try:
        yield
    finally:
        for adapter, state in zip(adapters, states):
            adapter.lora_enabled = state


def apply_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    target_suffixes: Sequence[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    top_layers: int = 0,
) -> int:
    """Wrap matching linear layers of ``model`` in-place with LoRA adapters.

    ``top_layers > 0`` restricts the wrap to the last ``top_layers``
    transformer blocks under ``model.layers`` (Qwen3.5-style decoder);
    ``top_layers == 0`` keeps the legacy whole-model behavior.  Returns the
    number of adapted layers.
    """
    if top_layers < 0:
        raise ValueError("top_layers must be non-negative")
    if top_layers > 0:
        layers = getattr(model, "layers", None)
        if layers is None:
            raise ValueError("top_layers>0 requires a 'layers' attribute on the model")
        if top_layers > len(layers):
            raise ValueError(
                f"top_layers={top_layers} exceeds the model's {len(layers)} layers"
            )
        roots = layers[-top_layers:]
    else:
        roots = (model,)
    count = [0]

    def walk(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(
                name.endswith(suffix) for suffix in target_suffixes
            ):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
                count[0] += 1
            else:
                walk(child)

    for root in roots:
        walk(root)
    return count[0]


def pool_flat_tokens(tokens: Tensor, max_tokens: int) -> Tensor:
    """A: adaptive average pooling over the flattened [t, h, w] sequence.

    Buckets are contiguous windows of the flattened sequence; for non-square
    grids they straddle image-row boundaries instead of forming 2D patches.
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, dim]")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if tokens.shape[1] > max_tokens:
        return F.adaptive_avg_pool1d(tokens.transpose(1, 2), max_tokens).transpose(1, 2)
    return tokens


MTVJ_COARSE_TOKENS = 16


def pool_mtvj_coarse_tokens(tokens: Tensor) -> Tensor:
    """Pool flattened MT-VJ H11 evidence into the policy's 16 coarse tokens.

    Training historically used 16 contiguous, equally sized bins over the
    flattened ``[t, y, x]`` token sequence.  Keeping this as one shared helper
    prevents closed-loop evaluation from silently using the regular V-JEPA
    final-layer/64-token input instead.
    """
    if tokens.ndim != 3:
        raise ValueError("MT-VJ H11 tokens must have shape [batch, tokens, dim]")
    if tokens.shape[1] % MTVJ_COARSE_TOKENS != 0:
        raise ValueError(
            "MT-VJ H11 token count must be divisible by "
            f"{MTVJ_COARSE_TOKENS}, got {tokens.shape[1]}"
        )
    return tokens.reshape(
        tokens.shape[0],
        MTVJ_COARSE_TOKENS,
        -1,
        tokens.shape[-1],
    ).mean(dim=2)


def _square_grid_size(max_tokens: int) -> int:
    size = isqrt(max_tokens)
    if size * size != max_tokens:
        raise ValueError(
            f"spatial pooling needs max_tokens to be a perfect square, got {max_tokens}"
        )
    return size


def pool_spatial_tokens(tokens: Tensor, grid: tuple[int, int], max_tokens: int) -> Tensor:
    """B: time-mean then 2D grid pooling; each output token is one spatial patch.

    Tokens are laid out as [t, h, w] (t slowest).  ``grid`` is the (h, w)
    patch grid; the time axis is inferred from the token count, averaged
    first, then a square 2D adaptive pool maps (h, w) to (size, size).
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, dim]")
    h_grid, w_grid = grid
    if min(h_grid, w_grid) < 1:
        raise ValueError("grid dimensions must be positive")
    spatial_tokens = h_grid * w_grid
    if tokens.shape[1] % spatial_tokens:
        raise ValueError(
            f"tokens {tokens.shape[1]} are not a multiple of the spatial grid {grid}"
        )
    t_grid = tokens.shape[1] // spatial_tokens
    size = _square_grid_size(max_tokens)
    batch = tokens.shape[0]
    view = tokens.reshape(batch, t_grid, h_grid, w_grid, -1).mean(dim=1)
    view = view.permute(0, 3, 1, 2)  # [B, D, H, W]
    pooled = F.adaptive_avg_pool2d(view, (size, size))
    return pooled.flatten(2).transpose(1, 2)  # [B, size*size, D]


def pool_spatiotemporal_tokens(
    tokens: Tensor, grid: tuple[int, int], max_tokens: int
) -> Tensor:
    """C: keep the time axis; 2D grid pooling per frame -> [t * size*size] tokens.

    Experimental: the output keeps ``t_grid * size*size`` tokens and grows
    linearly with the frame count, so it is NOT bounded by ``max_tokens``
    (4 frames -> 128 tokens).  P0 compares only the A (flat) and B (spatial)
    variants; C is not wired into train/evaluate and stays out of the CLI.
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, dim]")
    h_grid, w_grid = grid
    if min(h_grid, w_grid) < 1:
        raise ValueError("grid dimensions must be positive")
    spatial_tokens = h_grid * w_grid
    if tokens.shape[1] % spatial_tokens:
        raise ValueError(
            f"tokens {tokens.shape[1]} are not a multiple of the spatial grid {grid}"
        )
    t_grid = tokens.shape[1] // spatial_tokens
    size = _square_grid_size(max_tokens)
    batch = tokens.shape[0]
    view = tokens.reshape(batch, t_grid, h_grid, w_grid, -1)
    view = view.permute(0, 1, 4, 2, 3).reshape(batch * t_grid, -1, h_grid, w_grid)
    pooled = F.adaptive_avg_pool2d(view, (size, size))
    pooled = pooled.reshape(batch, t_grid, -1, size, size)
    pooled = pooled.permute(0, 1, 3, 4, 2)  # [B, T, size, size, D]
    return pooled.reshape(batch, t_grid * size * size, -1)


VJEPA21_REPO_REF = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA21_REPO = f"facebookresearch/vjepa2:{VJEPA21_REPO_REF}"
VJEPA21_ENTRYPOINT = "vjepa2_1_vit_base_384"
VJEPA21_CHECKPOINT_NAME = "vjepa2_1_vitb_dist_vitG_384.pt"
VJEPA21_CHECKPOINT_BYTES = 1_664_223_428
VJEPA21_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/" + VJEPA21_CHECKPOINT_NAME
)

# OpenVLA / Prismatic action-vision contract: DINOv2 ViT-L/14 with four
# register tokens at 224px.  The wrapper maps selected timm blocks to the
# repository's canonical {5, 11} mid/final evidence keys.
DINOV2_ACTION_MODEL_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
DINOV2_ACTION_IMAGE_SIZE = 224
DINOV2_ACTION_DIM = 1024
DINOV2_ACTION_LAYERS = (11, 23)


def _dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in choices:
        raise ValueError(f"unsupported dtype: {name}")
    return choices[name]


def _keep_vjepa21_rope_dtype(model: nn.Module) -> None:
    """Keep official 2.1 RoPE outputs compatible with half-precision SDPA."""
    namespace = model.blocks[0].attn.forward.__globals__
    rotate = namespace["rotate_queries_or_keys"]
    if getattr(rotate, "_keeps_input_dtype", False):
        return

    def preserve_dtype(x: Tensor, *args, **kwargs) -> Tensor:
        return rotate(x, *args, **kwargs).to(dtype=x.dtype)

    preserve_dtype._keeps_input_dtype = True
    namespace["rotate_queries_or_keys"] = preserve_dtype


class QwenTextBackbone(nn.Module):
    """Frozen text branch extracted from Qwen3.5; no vision tower or LM head is kept."""

    def __init__(self, tokenizer, text_model: nn.Module, max_length: int = 64) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen3.5-2B",
        *,
        device: str | torch.device = "cuda",
        dtype: str = "bfloat16",
        max_length: int = 64,
        local_files_only: bool = False,
    ) -> "QwenTextBackbone":
        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        full_model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=_dtype(dtype),
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        text_model = full_model.model.language_model
        del full_model
        text_model.requires_grad_(False).eval().to(device)
        return cls(tokenizer=tokenizer, text_model=text_model, max_length=max_length)

    @torch.no_grad()
    def encode(
        self,
        instructions: Sequence[str],
        output_layers: list[int] | None = None,
    ) -> tuple[Tensor | dict[int, Tensor], Tensor]:
        """Frozen encode; returns (last_hidden, mask) — or, with
        ``output_layers``, ({layer: [B, L, D]}, mask) where layer indices run
        0..N-1 over the decoder layers (embedding excluded).
        """
        input_ids, attention_mask = self._tokenize_instructions(instructions)
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=output_layers is not None,
        )
        if output_layers is not None:
            self._check_output_layers(output_layers)
            return (
                {
                    layer: output.hidden_states[layer + 1].detach()
                    for layer in output_layers
                },
                attention_mask.bool(),
            )
        return output.last_hidden_state.detach(), attention_mask.bool()

    def apply_lora(
        self,
        rank: int = 32,
        alpha: float = 32.0,
        target_suffixes: Sequence[str] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    ) -> int:
        """Attach LoRA adapters to the frozen language model (in-place)."""
        if rank < 0:
            raise ValueError("lora rank must be non-negative")
        return apply_lora(self.text_model, rank=rank, alpha=alpha, target_suffixes=target_suffixes)

    def unfreeze_last(self, blocks: int, freeze_final_norm: bool = False) -> None:
        """Unfreeze the last ``blocks`` decoder layers (+ optional final norm).

        ``freeze_final_norm=True`` keeps the final norm frozen (2026-08-06
        Codex e2e design): the shared final transform cannot push the whole
        instruction-embedding geometry away, protecting language semantics
        during Qwen partial unfreezing.
        """
        self.text_model.requires_grad_(False)
        if blocks < 0 or blocks > len(self.text_model.layers):
            raise ValueError("invalid number of language-model blocks to unfreeze")
        if blocks:
            for layer in self.text_model.layers[-blocks:]:
                layer.requires_grad_(True).train()
            if hasattr(self.text_model, "norm") and not freeze_final_norm:
                self.text_model.norm.requires_grad_(True).train()

    def _tokenize_instructions(self, instructions: Sequence[str]) -> tuple[Tensor, Tensor]:
        device = next(self.text_model.parameters()).device
        tokens = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        return input_ids, attention_mask

    def _check_output_layers(self, output_layers: Sequence[int]) -> None:
        if not output_layers:
            raise ValueError("output_layers must be a non-empty list of layer indices")
        layers = getattr(self.text_model, "layers", None)
        num_layers = len(layers) if layers is not None else None
        for layer in output_layers:
            if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
                raise ValueError(
                    f"output_layers entries must be non-negative ints, got {layer!r}"
                )
            if num_layers is not None and layer >= num_layers:
                raise ValueError(
                    f"layer index {layer} out of range [0, {num_layers - 1}]"
                )

    def encode_trainable(
        self,
        instructions: Sequence[str],
        output_layers: list[int] | None = None,
    ) -> tuple[Tensor | dict[int, Tensor], Tensor]:
        """Online encode with gradients (for LoRA fine-tuning).

        ``output_layers`` behaves as in ``encode``: None returns
        (last_hidden, mask), a layer-index list returns
        ({layer: [B, L, D]}, mask).
        """
        input_ids, attention_mask = self._tokenize_instructions(instructions)
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=output_layers is not None,
        )
        if output_layers is not None:
            self._check_output_layers(output_layers)
            return (
                {layer: output.hidden_states[layer + 1] for layer in output_layers},
                attention_mask.bool(),
            )
        return output.last_hidden_state, attention_mask.bool()

    def encode_with_scene(
        self,
        instructions: Sequence[str],
        scene_summary: Tensor,
        scene_projector: nn.Module,
        readout_tokens: Tensor,
        n_scene: int = 8,
        output_layers: list[int] | None = None,
        extra_embeds: list[Tensor] | None = None,
    ) -> tuple[Tensor | dict[int, Tensor], Tensor]:
        """Plan-Cache 方案 A：Qwen 看场景 teacher（2026-08-07 Codex 评审）。

        Appends ``n_scene`` scene pseudo tokens (projected scene summary) and
        ``n_readout`` learned readout tokens after the (padded) instruction
        tokens, then runs the frozen text model WITH gradients — the scene
        projector and readout embeddings (owned by ``SceneTeacher``) are the
        trainable parameters.  Returns the readout-position hidden states
        [B, n_readout, hidden_dim] and the full sequence boolean mask
        [B, L + n_scene + n_readout].  Position ids are contiguous across the
        concatenated sequence; the causal mask makes the readout tokens
        attend to everything before them (instructions + scene pseudo tokens).

        ``extra_embeds`` (compile-task, 2026-08-07): an optional list of
        [B, n, language_dim] pseudo-token tensors inserted between the scene
        pseudo tokens and the readout tokens — concatenation order is
        instructions → scene → extra (list order) → readout.  Their mask
        entries are always true.  ``None`` (or an empty list) keeps the
        legacy behavior byte-identical.

        With ``output_layers`` given, returns ({layer: [B, L+K+N, D]}, mask)
        with layer indices 0..N-1 over the decoder layers instead.
        """
        device = next(self.text_model.parameters()).device
        tokens = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        if len(instructions) != scene_summary.shape[0]:
            raise ValueError("instructions and scene_summary must share the batch size")
        inputs_embeds = self.text_model.embed_tokens(input_ids)  # [B, L, D]
        batch, length, dim = inputs_embeds.shape
        # projector 参数可能是 fp32（训练侧），输入先用 projector 自身 dtype，
        # 输出再转回模型 dtype（冻结 Qwen 是 fp16/bf16）。
        scene_dtype = next(scene_projector.parameters()).dtype
        scene_embeds = scene_projector(scene_summary.to(device=device, dtype=scene_dtype))
        scene_embeds = scene_embeds.to(dtype=inputs_embeds.dtype)
        if scene_embeds.ndim != 3:
            if scene_embeds.shape[-1] % dim:
                raise ValueError(
                    f"scene projector output {scene_embeds.shape[-1]} is not "
                    f"a multiple of the language dim {dim}"
                )
            scene_embeds = scene_embeds.view(batch, -1, dim)
        if scene_embeds.shape[-1] != dim:
            raise ValueError("scene pseudo tokens must match the language hidden dim")
        if scene_embeds.shape[1] != n_scene:
            raise ValueError(
                f"expected n_scene={n_scene} pseudo tokens, got {scene_embeds.shape[1]}"
            )
        extra_embeds_list = []
        if extra_embeds is not None:
            for embeds in extra_embeds:
                if embeds.ndim != 3:
                    raise ValueError(
                        "extra_embeds entries must have shape [B, n, language_dim]"
                    )
                if embeds.shape[0] != batch:
                    raise ValueError(
                        "extra_embeds entries must share the batch size with instructions"
                    )
                if embeds.shape[-1] != dim:
                    raise ValueError(
                        "extra_embeds entries must match the language hidden dim"
                    )
                extra_embeds_list.append(
                    embeds.to(device=device, dtype=inputs_embeds.dtype)
                )
        readout = readout_tokens.to(device=device, dtype=inputs_embeds.dtype)
        if readout.ndim != 2 or readout.shape[-1] != dim:
            raise ValueError("readout_tokens must have shape [n_readout, language_dim]")
        readout_embeds = readout[None].expand(batch, -1, -1)
        embeds = torch.cat(
            (inputs_embeds, scene_embeds, *extra_embeds_list, readout_embeds), dim=1
        )
        extra_total = sum(embeds.shape[1] for embeds in extra_embeds_list)
        mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    batch, n_scene + extra_total + readout_embeds.shape[1],
                    dtype=attention_mask.dtype, device=device,
                ),
            ),
            dim=1,
        )
        position_ids = torch.arange(
            embeds.shape[1], device=device, dtype=torch.long
        )[None].expand(batch, -1)
        output = self.text_model(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=output_layers is not None,
        )
        if output_layers is not None:
            self._check_output_layers(output_layers)
            return (
                {layer: output.hidden_states[layer + 1] for layer in output_layers},
                mask.bool(),
            )
        plan = output.last_hidden_state[:, length + n_scene + extra_total :]  # readout positions
        return plan, mask.bool()


class SceneTeacher(nn.Module):
    """方案 A 可训练部件：场景投影器 + readout tokens（Plan-Cache 2026-08-07）。

    ``QwenTextBackbone`` 保持冻结，本模块持有全部可训练参数：
    - ``scene_projector``：scene_summary [B, vision_dim] → [B, n_scene, D]
      的场景伪 token embeds（小 MLP：768→2048→n_scene*2048）；
    - ``readout_tokens``：n_readout 个 learned readout embedding。
    二者经 ``QwenTextBackbone.encode_with_scene`` 走冻结 Qwen 前向（保留梯度），
    输出 readout 位置的 plan hidden [B, n_readout, D]。
    """

    def __init__(
        self,
        language_dim: int = 2048,
        vision_dim: int = 768,
        n_scene: int = 8,
        n_readout: int = 8,
    ) -> None:
        super().__init__()
        self.n_scene = n_scene
        self.n_readout = n_readout
        self.scene_projector = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, n_scene * language_dim),
        )
        self.readout_tokens = nn.Parameter(torch.empty(n_readout, language_dim))
        nn.init.normal_(self.readout_tokens, std=0.02)

    def forward(
        self,
        text_backbone: "QwenTextBackbone",
        instructions: Sequence[str],
        scene_summary: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns (plan hidden [B, n_readout, D], full mask [B, L+K+N])."""
        return text_backbone.encode_with_scene(
            instructions,
            scene_summary,
            self.scene_projector,
            self.readout_tokens,
            n_scene=self.n_scene,
        )


class _PooledTokenProjector(nn.Module):
    """Token-MLP projector with adaptive pooling on the token axis.

    [B, K, vision_dim] → adaptive-avg-pool to ``n_scene`` tokens (any K >= 1;
    K <= n_scene up-samples as repeated mean windows) → per-token MLP
    (vision_dim → hidden → language_dim) → [B, n_scene, language_dim].
    """

    def __init__(
        self, vision_dim: int, hidden: int, language_dim: int, n_scene: int
    ) -> None:
        super().__init__()
        if n_scene < 1:
            raise ValueError("n_scene must be positive")
        self.n_scene = n_scene
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, language_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool1d(
            tokens.transpose(1, 2), self.n_scene
        ).transpose(1, 2)
        return self.mlp(pooled)


class SemanticCompiler(nn.Module):
    """compile-task（Stage A，2026-08-07）可训练部件：具身上下文 → 语义 tokens。

    与 SceneTeacher（单一 scene_summary [B, vision_dim]）不同，SemanticCompiler
    把三类具身上下文经各自投影器编译为追加到语言序列的语义伪 token：

    - ``scene_projector``（``_PooledTokenProjector``）：visual tokens
      [B, K, vision_dim] → [B, n_scene, language_dim]。任意 K ≥ 1：先在视觉
      空间按 token 轴 adaptive-avg-pool 到 n_scene（K ≤ n_scene 时上采样为
      重复均值窗口），再逐 token MLP（vision_dim→hidden→language_dim）。
    - ``history_projector``：语义历史向量 [B, history_in_dim] → [B, n_hist, D]
      （语义历史均值，如 VisualMemory.task 的 token 均值。``history_in_dim``
      默认 None → vision_dim；真实 e2e 配置下 memory.task 在 VA hidden 空间
      （[B, n_task, hidden_dim]），build_e2e_policy 传
      ``history_in_dim=config.hidden_dim`` 使均值后的 [B, hidden_dim] 可直接用）。
    - ``delta_projector``：视觉变化向量 [B, vision_dim] → [B, n_delta, D]
      （当前帧视觉 token 均值 − 上一帧视觉 token 均值）。
    - ``error_projector``（第二轮架构重构）：执行误差向量 [B, error_in_dim]
      → [B, n_err, D]（``error_in_dim`` 默认 None → vision_dim）。``forward``
      收到 ``execution_error`` 时其伪 token 插在 delta 段之后、readout 之前；
      None 时该段完全跳过（与旧行为逐字节一致）。
    - ``readout_tokens``：n_readout 个 learned readout embedding（normal_ σ=0.02）。

    ``forward`` 把 scene/history/delta/(error) 伪 token 依次插在指令 token 之后、
    readout 之前（encode_with_scene 的 ``extra_embeds`` 槽位），readout 位置
    输出 plan hidden [B, n_readout, D] 与全序列 mask [B, L+K+H+M+E+N]。
    ``text_backbone`` 为 QwenSemanticBackbone 时（P0-4）走
    ``encode_scene_fused``（prior/adapted 双路径 + 非负门控融合）；普通
    QwenTextBackbone 保持旧路径（encode_with_scene）。
    """

    def __init__(
        self,
        language_dim: int = 2048,
        vision_dim: int = 768,
        n_scene: int = 8,
        n_hist: int = 2,
        n_delta: int = 2,
        n_readout: int = 8,
        hidden: int = 512,
        n_err: int = 2,
        error_in_dim: int | None = None,
        history_in_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(n_scene, n_hist, n_delta, n_readout, n_err) < 1:
            raise ValueError("n_scene/n_hist/n_delta/n_readout/n_err must be positive")
        if error_in_dim is not None and error_in_dim < 1:
            raise ValueError("error_in_dim must be positive")
        if history_in_dim is not None and history_in_dim < 1:
            raise ValueError("history_in_dim must be positive")
        self.language_dim = language_dim
        self.vision_dim = vision_dim
        self.n_scene = n_scene
        self.n_hist = n_hist
        self.n_delta = n_delta
        self.n_readout = n_readout
        self.n_err = n_err
        self.history_in_dim = history_in_dim if history_in_dim is not None else vision_dim
        self.error_in_dim = error_in_dim if error_in_dim is not None else vision_dim
        self.scene_projector = _PooledTokenProjector(
            vision_dim, hidden, language_dim, n_scene
        )
        self.history_projector = nn.Sequential(
            nn.Linear(self.history_in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_hist * language_dim),
        )
        self.delta_projector = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_delta * language_dim),
        )
        self.error_projector = nn.Sequential(
            nn.Linear(self.error_in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_err * language_dim),
        )
        self.readout_tokens = nn.Parameter(torch.empty(n_readout, language_dim))
        nn.init.normal_(self.readout_tokens, std=0.02)

    def _project_embeds(
        self,
        scene_tokens: Tensor,
        semantic_history: Tensor,
        scene_delta: Tensor,
        execution_error: Tensor | None,
    ) -> list[Tensor]:
        """共享的逐样本伪 token 投影（forward / scene_states 复用，P0-4）。

        输入校验 + history/delta/(error) 投影，返回 ``extra_embeds`` 列表
        （与 ``encode_with_scene`` 的 extra_embeds 槽位一一对应）。
        """
        if scene_tokens.ndim != 3 or scene_tokens.shape[-1] != self.vision_dim:
            raise ValueError(f"scene_tokens must have shape [B, K, {self.vision_dim}]")
        if semantic_history.ndim != 2 or semantic_history.shape[-1] != self.history_in_dim:
            raise ValueError(
                f"semantic_history must have shape [B, {self.history_in_dim}] "
                "(None defaults to the vision dim; pass history_in_dim to match "
                "a VA-space memory like VisualMemory.task)"
            )
        if scene_delta.ndim != 2 or scene_delta.shape[-1] != self.vision_dim:
            raise ValueError(f"scene_delta must have shape [B, {self.vision_dim}]")
        batch = scene_tokens.shape[0]
        if semantic_history.shape[0] != batch or scene_delta.shape[0] != batch:
            raise ValueError(
                "scene_tokens/semantic_history/scene_delta must share the batch size"
            )
        projector_dtype = next(self.scene_projector.parameters()).dtype
        device = scene_tokens.device
        history_embeds = self.history_projector(
            semantic_history.to(device=device, dtype=projector_dtype)
        ).view(batch, self.n_hist, self.language_dim)
        delta_embeds = self.delta_projector(
            scene_delta.to(device=device, dtype=projector_dtype)
        ).view(batch, self.n_delta, self.language_dim)
        if execution_error is not None:
            if execution_error.ndim != 2 or execution_error.shape[-1] != self.error_in_dim:
                raise ValueError(
                    f"execution_error must have shape [B, {self.error_in_dim}]"
                )
            if execution_error.shape[0] != batch:
                raise ValueError("execution_error must share the batch size")
            error_embeds = self.error_projector(
                execution_error.to(device=device, dtype=projector_dtype)
            ).view(batch, self.n_err, self.language_dim)
            return [history_embeds, delta_embeds, error_embeds]
        # execution_error=None:不追加 error 段(输出逐字节一致),但让
        # error_projector 以零输入进入计算图——贡献恒 0(梯度为 0 而非
        # None),参数保持可训练状态(test_parameters_trainable 断言全部
        # 参数有 grad)。注意列表在分支内重建,否则 delta_embeds 的新
        # 计算图引用会丢失。
        zero_error = torch.zeros(
            batch, self.error_in_dim, device=device, dtype=projector_dtype
        )
        error_embeds = self.error_projector(zero_error).view(
            batch, self.n_err, self.language_dim
        )
        delta_embeds = delta_embeds + 0.0 * error_embeds.mean(dim=1, keepdim=True)
        return [history_embeds, delta_embeds]

    def forward(
        self,
        text_backbone: "QwenTextBackbone",
        instructions: Sequence[str],
        scene_tokens: Tensor,
        semantic_history: Tensor,
        scene_delta: Tensor,
        execution_error: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (readout plan hidden [B, n_readout, D], full mask).

        ``execution_error`` [B, error_in_dim] is optional (第二轮架构重构):
        None keeps the legacy scene+history+delta behavior byte-identical;
        given, its ``n_err`` pseudo tokens are inserted after the delta
        segment and before the readout tokens.

        P0-4：``text_backbone`` 为 QwenSemanticBackbone 时不再解包直走裸
        ``encode_with_scene``（那会绕过 prior/adapted/fused 双路径）——改走
        ``encode_scene_fused``：场景条件 readout 同样由冻结先验 + LoRA 残差 +
        非负门控融合得到。
        """
        extra_embeds = self._project_embeds(
            scene_tokens, semantic_history, scene_delta, execution_error
        )
        if isinstance(text_backbone, QwenSemanticBackbone):
            return text_backbone.encode_scene_fused(
                instructions,
                scene_tokens,
                self.scene_projector,
                self.readout_tokens,
                n_scene=self.n_scene,
                extra_embeds=extra_embeds,
            )
        return text_backbone.encode_with_scene(
            instructions,
            scene_tokens,
            self.scene_projector,
            self.readout_tokens,
            n_scene=self.n_scene,
            extra_embeds=extra_embeds,
        )

    def scene_states(
        self,
        text_backbone: "QwenTextBackbone",
        instructions: Sequence[str],
        scene_tokens: Tensor,
        semantic_history: Tensor,
        scene_delta: Tensor,
        output_layers: list[int],
        execution_error: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], Tensor]:
        """场景条件路径的 (prior_layers, adapted_layers, mask)（P0-4 anchor）。

        与 ``forward`` 完全相同的投影与伪 token 拼接，但返回全序列逐层
        hidden（``output_layers`` 层索引）而不是 readout plan：prior 侧
        no_grad + LoRA 关闭，adapted 侧带梯度，可直接喂给
        ``QwenSemanticBackbone.anchor_loss`` / ``geometry_loss``。
        仅支持 QwenSemanticBackbone（普通 QwenTextBackbone 没有
        prior/adapted 之分）。
        """
        if not isinstance(text_backbone, QwenSemanticBackbone):
            raise ValueError("scene_states requires a QwenSemanticBackbone")
        extra_embeds = self._project_embeds(
            scene_tokens, semantic_history, scene_delta, execution_error
        )
        prior_layers, mask = text_backbone.encode_scene_prior_states(
            instructions,
            scene_tokens,
            self.scene_projector,
            self.readout_tokens,
            output_layers=output_layers,
            n_scene=self.n_scene,
            extra_embeds=extra_embeds,
        )
        adapted_layers, _ = text_backbone.encode_scene_adapted_states(
            instructions,
            scene_tokens,
            self.scene_projector,
            self.readout_tokens,
            output_layers=output_layers,
            n_scene=self.n_scene,
            extra_embeds=extra_embeds,
        )
        return prior_layers, adapted_layers, mask


class VJEPA21Backbone(nn.Module):
    """Frozen V-JEPA 2.1 ViT-B encoder returning a bounded token sequence.

    ``pooling`` selects how the raw [t, h, w] token grid is reduced:
    - "flat": adaptive 1D pooling over the flattened sequence (legacy A);
    - "spatial": time-mean then 2D grid pooling, one output token per
      spatial neighbourhood (B);
    - "spatiotemporal": 2D grid pooling per frame, keeping the time axis (C);
    - "dense": no pooling — the full [t, h, w] grid is flattened in order
      (Step 0 dense readout, 2×24×24 = 1152 tokens; reader-side only).

    ``encode_multi``（Step 4）一次前向返回多个中间层输出（官方原生
    ``out_layers`` 支持，H⁵/H¹¹），缺省路径不受影响。
    """

    def __init__(
        self,
        model: nn.Module,
        max_tokens: int = 64,
        grid: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.grid = grid

    def patch_grid(self) -> tuple[int, int]:
        """(h, w) patch-grid inferred from the ViT, or the explicit grid.

        The time dimension is intentionally not inferred here: the model's
        ``num_frames`` is a static training attribute, while the number of
        frames passed at inference decides the real t-axis length.
        """
        if self.grid is not None:
            return self.grid
        model = self.model
        try:
            h_grid = model.img_height // model.patch_size
            w_grid = model.img_width // model.patch_size
        except AttributeError as exc:
            raise ValueError(
                "cannot infer the patch grid from this model; pass grid= explicitly"
            ) from exc
        return (int(h_grid), int(w_grid))

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device = "cuda",
        dtype: str = "bfloat16",
        max_tokens: int = 64,
        local_files_only: bool = False,
    ) -> "VJEPA21Backbone":
        hub_dir = Path(torch.hub.get_dir())
        repo_dir = hub_dir / f"facebookresearch_vjepa2_{VJEPA21_REPO_REF}"
        checkpoint_path = hub_dir / "checkpoints" / VJEPA21_CHECKPOINT_NAME

        if local_files_only:
            if not repo_dir.is_dir():
                raise FileNotFoundError(
                    f"V-JEPA 2.1 source is missing at {repo_dir}; run prepare_models.py"
                )
            if (
                not checkpoint_path.is_file()
                or checkpoint_path.stat().st_size != VJEPA21_CHECKPOINT_BYTES
            ):
                raise FileNotFoundError(
                    f"V-JEPA 2.1 checkpoint is missing at {checkpoint_path}; run prepare_models.py"
                )
            source = str(repo_dir)
            source_kind = "local"
        else:
            source = VJEPA21_REPO
            source_kind = "github"

        model, predictor = torch.hub.load(
            source,
            VJEPA21_ENTRYPOINT,
            source=source_kind,
            pretrained=False,
            trust_repo=True,
            skip_validation=True,
        )
        del predictor
        _keep_vjepa21_rope_dtype(model)

        if local_files_only:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                VJEPA21_CHECKPOINT_URL,
                map_location="cpu",
                file_name=VJEPA21_CHECKPOINT_NAME,
                weights_only=True,
            )
        encoder_state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in checkpoint["ema_encoder"].items()
        }
        model.load_state_dict(encoder_state, strict=True)
        del checkpoint, encoder_state

        model.to(device=device, dtype=_dtype(dtype))
        backbone = cls(model=model, max_tokens=max_tokens)
        backbone.freeze_all()
        return backbone

    def freeze_all(self) -> None:
        self.model.requires_grad_(False).eval()

    def unfreeze_last(self, blocks: int) -> None:
        self.freeze_all()
        if blocks < 0 or blocks > len(self.model.blocks):
            raise ValueError("invalid number of V-JEPA 2.1 blocks to unfreeze")
        if blocks:
            for layer in self.model.blocks[-blocks:]:
                layer.requires_grad_(True).train()
            self.model.norms_block[-1].requires_grad_(True).train()

    def unfreeze_all(self) -> None:
        """Truly full unfreezing (2026-08-06 Codex e2e design): patch/tubelet
        stems, modality embeddings, all transformer blocks and norms.  The old
        ``unfreeze_last(12)`` keeps the patch embedding frozen; this variant
        unfreezes the whole encoder for task-domain adaptation."""
        self.freeze_all()
        for name, param in self.model.named_parameters():
            param.requires_grad_(True)
        self.model.train()

    def _prepare_video(self, pixel_values_videos: Tensor) -> Tensor:
        """校验 + 重排 + 移设备/精度：官方输入契约 [B, C, T, H, W]（C=3）。"""
        if pixel_values_videos.ndim != 5 or pixel_values_videos.shape[2] != 3:
            raise ValueError("video must have shape [batch, frames, 3, height, width]")
        if pixel_values_videos.shape[1] < 2 or pixel_values_videos.shape[1] % 2:
            raise ValueError("V-JEPA 2.1 needs a positive even number of frames")
        parameter = next(self.model.parameters())
        return pixel_values_videos.permute(0, 2, 1, 3, 4).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _encode(self, pixel_values_videos: Tensor) -> Tensor:
        return self.model(self._prepare_video(pixel_values_videos))

    def encode_multi(
        self,
        pixel_values_videos: Tensor,
        out_layers: Sequence[int] = (5, 11),
    ) -> list[Tensor]:
        """Step 4：一次前向返回多个中间层输出（官方原生 ``out_layers`` 支持）。

        官方 V-JEPA 2.1 forward 在 ``out_layers`` 里的 block 之后收集
        ``self.norm(x)`` 并返回列表（此时不再走末尾 norm）；第 6/12 个 block
        后即 (H⁵, H¹¹)（12-block ViT-B）。官方实现按 block 升序收集，此处对
        非升序的 ``out_layers`` 重排输出，保持"返回值顺序与 out_layers 一致"
        的契约；无序集合（set）规范为升序。

        调用期间临时改写 ``self.model.out_layers``，finally 恢复原值——缺省
        路径（``forward`` / ``_encode``）行为逐字节不变。V-JEPA 权重保持冻结
        （``freeze_all``），多层输出只作只读 evidence（设计文档 §九：不进 VA
        自注意力）。
        """
        if isinstance(out_layers, (set, frozenset)):
            layers = sorted(out_layers)  # 无序集合：规范为升序
        else:
            layers = tuple(out_layers)
        if not layers:
            raise ValueError("out_layers must be a non-empty sequence of block indices")
        for layer in layers:
            if not isinstance(layer, int) or isinstance(layer, bool):
                raise ValueError(f"out_layers entries must be ints, got {layer!r}")
        if len(set(layers)) != len(layers):
            raise ValueError(f"out_layers must not contain duplicates, got {layers}")
        num_blocks = len(self.model.blocks)
        if any(layer < 0 or layer >= num_blocks for layer in layers):
            raise ValueError(
                f"layer indices must be in [0, {num_blocks - 1}], got {layers}"
            )
        ascending = sorted(layers)
        saved = self.model.out_layers
        self.model.out_layers = ascending
        try:
            outs = self.model(self._prepare_video(pixel_values_videos))
        finally:
            self.model.out_layers = saved
        if not isinstance(outs, list) or len(outs) != len(layers):
            raise RuntimeError(
                f"official forward with out_layers should return a list of "
                f"{len(layers)} tensors, got {type(outs).__name__}"
            )
        # 官方按 block 升序收集 → 重排到调用方给定顺序（默认 (5, 11) 即原序）。
        position = {layer: index for index, layer in enumerate(ascending)}
        return [outs[position[layer]] for layer in layers]

    def forward_hierarchical_dense(
        self,
        video: Tensor,
        out_layers: Sequence[int] = (5, 11),
    ) -> dict[int, Tensor]:
        """MT-VJ §1：多层未池化 dense evidence（官方原生 out_layers 路径）。

        ``video`` 输入契约与 ``_encode`` 同构：[B, W, 3, 384, 384]（W=4 帧窗，
        偶数帧）。一次前向返回 ``{layer: [B, 1152, 768]}``——每层在对应 block
        之后收集 ``norm(x)`` 并按 t→y→x 顺序扁平的全 2×24×24 patch token
        （不池化），与 Step 0 dense 读出 / ``_dense_coords()`` 的顺序约定一致
        （2 时间片 × 24×24 = 1152）。复用官方 forward 的 ``out_layers`` 支持
        （``encode_multi``），缺省路径（``forward``/``_encode``）行为不变。
        """
        if isinstance(out_layers, (set, frozenset)):
            layers = tuple(sorted(out_layers))  # 无序集合：规范为升序
        else:
            layers = tuple(out_layers)
        h_grid, w_grid = self.patch_grid()
        expected_tokens = (video.shape[1] // 2) * h_grid * w_grid
        outs = self.encode_multi(video, out_layers=layers)
        dense: dict[int, Tensor] = {}
        for layer, raw in zip(layers, outs):
            tokens = self._pool(raw, "dense")  # [B, t*h*w, D]（不池化扁平，t→y→x）
            if tokens.shape[1] != expected_tokens:
                raise ValueError(
                    f"hierarchical dense output for layer {layer} has "
                    f"{tokens.shape[1]} tokens, expected {expected_tokens} "
                    f"(W={video.shape[1]} 帧窗 × {h_grid}×{w_grid} patch / 2 时间片)"
                )
            dense[layer] = tokens
        return dense

    def _pool(self, tokens: Tensor, pooling: PoolingMode) -> Tensor:
        if pooling == "dense":
            # Step 0 dense readout：不池化，按 [t, h, w] 顺序扁平（2×24×24=1152），
            # 与 _dense_coords() 的坐标顺序一致；仅供角色查询 cross-attn 读出。
            return tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        if pooling == "flat":
            return pool_flat_tokens(tokens, self.max_tokens)
        if pooling == "spatial":
            return pool_spatial_tokens(tokens, self.patch_grid(), self.max_tokens)
        if pooling == "spatiotemporal":
            return pool_spatiotemporal_tokens(tokens, self.patch_grid(), self.max_tokens)
        raise ValueError(f"unsupported pooling mode: {pooling}")

    def forward(self, pixel_values_videos: Tensor, pooling: PoolingMode = "flat") -> Tensor:
        return self._pool(self._encode(pixel_values_videos), pooling)

    def forward_variants(
        self, pixel_values_videos: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One V-JEPA forward producing both A (flat) and B (spatial) features."""
        tokens = self._encode(pixel_values_videos)
        return self._pool(tokens, "flat"), self._pool(tokens, "spatial")


class TimmActionVisionBackbone(nn.Module):
    """Frozen timm ViT tower returning patch-only two-level evidence."""

    def __init__(
        self,
        model: nn.Module,
        *,
        model_id: str = DINOV2_ACTION_MODEL_ID,
        image_size: int = DINOV2_ACTION_IMAGE_SIZE,
        feature_dim: int = DINOV2_ACTION_DIM,
        output_layers: tuple[int, int] = DINOV2_ACTION_LAYERS,
    ) -> None:
        super().__init__()
        if not model_id or image_size < 1 or feature_dim < 1:
            raise ValueError("model_id, image_size, and feature_dim must be valid")
        if len(output_layers) != 2 or output_layers[0] >= output_layers[1]:
            raise ValueError("output_layers must be two increasing block indices")
        self.model = model
        self.model_id = model_id
        self.image_size = image_size
        self.feature_dim = feature_dim
        self.output_layers = output_layers
        self.freeze_all()

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "bfloat16",
        model_id: str = DINOV2_ACTION_MODEL_ID,
        image_size: int = DINOV2_ACTION_IMAGE_SIZE,
        feature_dim: int = DINOV2_ACTION_DIM,
        output_layers: tuple[int, int] = DINOV2_ACTION_LAYERS,
        checkpoint_path: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "TimmActionVisionBackbone":
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn

        if local_files_only and checkpoint_path is None:
            raise FileNotFoundError(
                "local_files_only=True requires checkpoint_path for timm weights"
            )
        model = timm.create_model(
            model_id,
            pretrained=checkpoint_path is None,
            img_size=image_size,
            num_classes=0,
        )
        if checkpoint_path is not None:
            path = Path(checkpoint_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"action-vision checkpoint is missing: {path}")
            timm.models.load_checkpoint(
                model,
                str(path),
                strict=True,
                filter_fn=checkpoint_filter_fn,
            )
        model.to(device=device, dtype=_dtype(dtype))
        return cls(
            model=model,
            model_id=model_id,
            image_size=image_size,
            feature_dim=feature_dim,
            output_layers=output_layers,
        )

    def freeze_all(self) -> None:
        self.model.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "TimmActionVisionBackbone":
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward_hierarchical_dense(self, images: Tensor) -> dict[int, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("action-vision images must have shape [batch, 3, H, W]")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"action-vision images must be {self.image_size}x{self.image_size}, "
                f"got {tuple(images.shape[-2:])}"
            )
        parameter = next(self.model.parameters())
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        outputs = self.model.forward_intermediates(
            images,
            indices=list(self.output_layers),
            return_prefix_tokens=True,
            norm=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
        if not isinstance(outputs, list) or len(outputs) != 2:
            raise RuntimeError("action-vision tower must return two intermediate levels")
        patches: list[Tensor] = []
        for output in outputs:
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(
                    "timm ViT must return (patch_tokens, prefix_tokens) pairs"
                )
            patch_tokens, _prefix_tokens = output
            if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != self.feature_dim:
                raise RuntimeError(
                    "action-vision patch output has the wrong shape, got "
                    f"{tuple(patch_tokens.shape)}"
                )
            patches.append(patch_tokens)
        return {5: patches[0], 11: patches[1]}


class DINOv2ActionBackbone(TimmActionVisionBackbone):
    """Backward-compatible name for the default DINOv2 contract."""


class QwenSemanticBackbone(nn.Module):
    """第三种方案（2026-08-07 落地）：冻结 Qwen 先验 + 顶部层 LoRA + 非负门控。

    冻结原始 Qwen 作为不可破坏的先验，仅对最后 ``top_layers`` 个 transformer
    层挂 LoRA 适配器学习具身语义残差；``fused_embedding`` 用输入条件相关的
    门控（读 (prior, adapted) 的逐样本均值）把残差加回 —— 门控末层零初始化
    （权重 0 + bias −4.595 → sigmoid ≈ 0.01），训练起点 fused ≈ prior（与原始
    Qwen 几乎一致，残差门控非负不会反向削弱先验）。``encode_prior`` 在 LoRA
    关闭 + no_grad 下运行（P0-1：真·冻结先验，否则 prior ≡ adapted，anchor/
    geometry 恒为零）；``encode_adapted`` LoRA 开启。``anchor_loss`` /
    ``geometry_loss`` 约束适配表征不偏离先验（防止指令表征坍塌）。
    """

    def __init__(
        self,
        text_backbone: QwenTextBackbone,
        *,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        top_layers: int = 4,
        anchor_layers: tuple[int, ...] = (),
        lora_suffixes: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if lora_rank < 0:
            raise ValueError("lora rank must be non-negative")
        if top_layers < 0:
            raise ValueError("top_layers must be non-negative")
        if lora_suffixes is None:
            lora_suffixes = (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        self.text_backbone = text_backbone
        # shim:save_checkpoint/resume 直接访问 text_backbone.text_model,
        # 与裸 QwenTextBackbone 的访问路径保持一致(nn.Module 重复引用
        # 同一子模块,state_dict/named_parameters 自动去重)。
        self.text_model = text_backbone.text_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.top_layers = top_layers
        self.anchor_layers = tuple(anchor_layers)
        # 第二轮架构重构:LoRA 投影后缀可配置(默认全 7 种 q/k/v/o/gate/up/down;
        # 传入子集如 ("q_proj","o_proj") 只适配这些投影)。
        self.lora_suffixes = tuple(lora_suffixes)
        text_model = self.text_model
        if top_layers > 0 and not hasattr(text_model, "layers"):
            raise ValueError("QwenSemanticBackbone requires text_model.layers")
        if top_layers > len(text_model.layers):
            raise ValueError(
                f"top_layers={top_layers} exceeds the "
                f"{len(text_model.layers)} decoder layers"
            )
        text_model.requires_grad_(False)  # 基础 Qwen 完全冻结
        self.num_layers = len(text_model.layers)
        # top_layers=0 时不挂任何 LoRA（仅门控可用）；>0 时只适配顶部层。
        self.lora_layer_count = (
            apply_lora(
                text_model,
                rank=lora_rank,
                alpha=lora_alpha,
                target_suffixes=self.lora_suffixes,
                top_layers=top_layers,
            )
            if top_layers > 0
            else 0
        )
        self.language_dim = self._infer_language_dim(text_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * self.language_dim, self.language_dim),
            nn.GELU(),
            nn.Linear(self.language_dim, self.language_dim),
        )
        # 门控末层零初始化：初始 g = σ(−4.595) ≈ 0.01，fused ≈ prior（训练起点
        # 几乎等于原始 Qwen）。bias 取小正值而非严格 0：若 g ≡ 0 且 lora_b ≡ 0，
        # fused 对 LoRA 的梯度恒为 0（精确零梯度死点，Adam 永远不动）；
        # g ≈ 0.01 给 lora_b 提供非零梯度（Adam 对梯度尺度不敏感），
        # 而 fused−prior = g ⊙ (adapted−prior) 在 adapted == prior 时仍逐位为 0。
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.gate[-1].bias.data.fill_(-4.595)  # sigmoid(-4.595) ≈ 0.01

    @property
    def text_model(self) -> nn.Module:
        """Compatibility shim: keeps text_model-level access working on the
        wrapper (save_checkpoint/resume read ``text_backbone.text_model``)."""
        return self.text_backbone.text_model

    @staticmethod
    def _infer_language_dim(text_model: nn.Module) -> int:
        config = getattr(text_model, "config", None)
        hidden_size = getattr(config, "hidden_size", None) if config is not None else None
        if hidden_size:
            return int(hidden_size)
        hidden_size = getattr(text_model, "hidden_size", None)
        if hidden_size:
            return int(hidden_size)
        embed = getattr(text_model, "embed_tokens", None)
        if embed is not None and hasattr(embed, "embedding_dim"):
            return int(embed.embedding_dim)
        raise ValueError(
            "cannot infer the language hidden dim for the semantic gate; "
            "expose text_model.config.hidden_size / text_model.hidden_size "
            "/ text_model.embed_tokens"
        )

    @torch.no_grad()
    def encode_prior(self, instructions: Sequence[str]) -> tuple[Tensor, Tensor]:
        """完整前向（no_grad + LoRA 关闭 → 真·冻结先验），返回 (last_hidden, mask)。

        P0-1：旧实现不关 LoRA，prior 实际等于 adapted（同一套 LoRA 权重），
        anchor/geometry 损失恒为零。现在 LoRA 在上下文内旁路。
        """
        with set_lora_enabled(self, False):
            return self.text_backbone.encode(instructions)

    def encode_adapted(self, instructions: Sequence[str]) -> tuple[Tensor, Tensor]:
        """带梯度的完整前向（LoRA 开启，与 prior 区分），返回 (last_hidden, mask)。"""
        with set_lora_enabled(self, True):
            return self.text_backbone.encode_trainable(instructions)

    @torch.no_grad()
    def encode_prior_states(
        self, instructions: Sequence[str], output_layers: list[int]
    ) -> tuple[dict[int, Tensor], Tensor]:
        """no_grad + LoRA 关闭的前向 + 指定层 hidden states（anchor 的 prior 侧）。"""
        with set_lora_enabled(self, False):
            return self.text_backbone.encode(instructions, output_layers=output_layers)

    def encode_adapted_states(
        self, instructions: Sequence[str], output_layers: list[int]
    ) -> tuple[dict[int, Tensor], Tensor]:
        """带梯度前向 + 指定层 hidden states（LoRA 开启），anchor 的 adapted 侧。"""
        with set_lora_enabled(self, True):
            return self.text_backbone.encode_trainable(
                instructions, output_layers=output_layers
            )

    @torch.no_grad()
    def encode_scene_prior(
        self,
        instructions: Sequence[str],
        scene_summary: Tensor,
        scene_projector: nn.Module,
        readout_tokens: Tensor,
        n_scene: int = 8,
        output_layers: list[int] | None = None,
        extra_embeds: list[Tensor] | None = None,
    ) -> tuple[Tensor | dict[int, Tensor], Tensor]:
        """场景条件前向的 prior 侧（P0-4）：no_grad + LoRA 关闭。

        参数契约与 ``QwenTextBackbone.encode_with_scene`` 一致（scene 路径也
        走 prior/adapted 双路径 + 门控融合，不再直接解包到裸 encode_with_scene）。
        """
        with set_lora_enabled(self, False):
            return self.text_backbone.encode_with_scene(
                instructions,
                scene_summary,
                scene_projector,
                readout_tokens,
                n_scene=n_scene,
                output_layers=output_layers,
                extra_embeds=extra_embeds,
            )

    def encode_scene_adapted(
        self,
        instructions: Sequence[str],
        scene_summary: Tensor,
        scene_projector: nn.Module,
        readout_tokens: Tensor,
        n_scene: int = 8,
        output_layers: list[int] | None = None,
        extra_embeds: list[Tensor] | None = None,
    ) -> tuple[Tensor | dict[int, Tensor], Tensor]:
        """场景条件前向的 adapted 侧（P0-4）：带梯度 + LoRA 开启。"""
        with set_lora_enabled(self, True):
            return self.text_backbone.encode_with_scene(
                instructions,
                scene_summary,
                scene_projector,
                readout_tokens,
                n_scene=n_scene,
                output_layers=output_layers,
                extra_embeds=extra_embeds,
            )

    def encode_scene_fused(
        self,
        instructions: Sequence[str],
        scene_summary: Tensor,
        scene_projector: nn.Module,
        readout_tokens: Tensor,
        n_scene: int = 8,
        extra_embeds: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """场景条件路径的 prior/adapted/门控融合（P0-4）。

        readout 位置输出 ``prior + g ⊙ (adapted − prior)``（与纯文本
        ``fused_embedding`` 同一门控），返回 (fused [B, n_readout, D], mask)。
        prior 侧 no_grad + LoRA 关闭，adapted 侧带梯度。
        """
        prior, mask = self.encode_scene_prior(
            instructions,
            scene_summary,
            scene_projector,
            readout_tokens,
            n_scene=n_scene,
            extra_embeds=extra_embeds,
        )
        adapted, _ = self.encode_scene_adapted(
            instructions,
            scene_summary,
            scene_projector,
            readout_tokens,
            n_scene=n_scene,
            extra_embeds=extra_embeds,
        )
        return self.fused_embedding(prior, adapted), mask

    @torch.no_grad()
    def encode_scene_prior_states(
        self,
        instructions: Sequence[str],
        scene_summary: Tensor,
        scene_projector: nn.Module,
        readout_tokens: Tensor,
        output_layers: list[int],
        n_scene: int = 8,
        extra_embeds: list[Tensor] | None = None,
    ) -> tuple[dict[int, Tensor], Tensor]:
        """场景条件前向 + 指定层 hidden（no_grad + LoRA 关闭），anchor 的 prior 侧。"""
        with set_lora_enabled(self, False):
            return self.text_backbone.encode_with_scene(
                instructions,
                scene_summary,
                scene_projector,
                readout_tokens,
                n_scene=n_scene,
                output_layers=output_layers,
                extra_embeds=extra_embeds,
            )

    def encode_scene_adapted_states(
        self,
        instructions: Sequence[str],
        scene_summary: Tensor,
        scene_projector: nn.Module,
        readout_tokens: Tensor,
        output_layers: list[int],
        n_scene: int = 8,
        extra_embeds: list[Tensor] | None = None,
    ) -> tuple[dict[int, Tensor], Tensor]:
        """场景条件前向 + 指定层 hidden（带梯度 + LoRA 开启），anchor 的 adapted 侧。"""
        with set_lora_enabled(self, True):
            return self.text_backbone.encode_with_scene(
                instructions,
                scene_summary,
                scene_projector,
                readout_tokens,
                n_scene=n_scene,
                output_layers=output_layers,
                extra_embeds=extra_embeds,
            )

    def fused_embedding(self, prior: Tensor, adapted: Tensor) -> Tensor:
        """``prior + g ⊙ (adapted - prior)``，g 为输入条件相关的残差门控。

        门控读 (prior, adapted) 的逐样本 token 均值 → 小 MLP → sigmoid 限幅
        到 (0, 1)；末层零初始化（权重 0 + bias −4.595 → σ ≈ 0.01）保证训练
        起点 g ≈ 0.01（fused ≈ prior，与原始 Qwen 几乎一致；门控非负，残差
        只能按比例加回，不会反向削弱先验）。
        """
        if prior.shape != adapted.shape:
            raise ValueError("prior and adapted must share the same shape")
        gate_dtype = next(self.gate.parameters()).dtype
        p = prior.to(dtype=gate_dtype)
        a = adapted.to(dtype=gate_dtype)
        gate = self.gate(torch.cat((p.mean(dim=1), a.mean(dim=1)), dim=-1))  # [B, D]
        gate = torch.sigmoid(gate).to(dtype=prior.dtype)[:, None, :]
        return prior + gate * (adapted - prior)

    def anchor_loss(
        self,
        prior_layers: dict[int, Tensor],
        adapted_layers: dict[int, Tensor],
    ) -> Tensor:
        """``anchor_layers`` 各层：Σ‖Norm(H_φ,l) − sg(Norm(H_0,l))‖² 的层间均值。

        Norm = 每 token 沿特征维 L2 归一化；prior 侧 stop-grad。``anchor_layers``
        为空时返回 0。
        """
        if not self.anchor_layers:
            first = next(iter(prior_layers.values()), None)
            if first is None:
                raise ValueError("anchor_loss needs at least one layer dict entry")
            return first.new_zeros(())
        missing = [
            layer
            for layer in self.anchor_layers
            if layer not in prior_layers or layer not in adapted_layers
        ]
        if missing:
            raise ValueError(f"anchor_layers missing from the layer dicts: {missing}")
        terms = [
            F.mse_loss(
                F.normalize(adapted_layers[layer].float(), dim=-1),
                F.normalize(prior_layers[layer].float(), dim=-1).detach(),
            )
            for layer in self.anchor_layers
        ]
        return torch.stack(terms).mean()

    def geometry_loss(
        self,
        prior: Tensor,
        adapted: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """G 矩阵几何约束：mask-weighted mean → 逐样本 L2 归一化 →
        ‖adapted·adaptedᵀ − prior·priorᵀ‖_F²（逐样本行均值再取批次均值）。

        输入接受 [B, T, D] 或 [B, D](T=1 的缩写);``mask`` 为 [B, T]
        布尔/数值掩码,只对 3D 输入有效;None 时退化为纯 token 均值。
        prior 侧 stop-grad。
        """
        if prior.shape != adapted.shape or prior.ndim not in (2, 3):
            raise ValueError("prior/adapted must share shape [B, D] or [B, T, D]")
        if mask is not None:
            if prior.ndim != 3:
                raise ValueError("mask requires [B, T, D] inputs")
            if tuple(mask.shape) != tuple(prior.shape[:2]):
                raise ValueError(f"mask must have shape {tuple(prior.shape[:2])}")
            weight = mask.to(dtype=prior.dtype)
            count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            p = (prior * weight.unsqueeze(-1)).sum(dim=1) / count
            a = (adapted * weight.unsqueeze(-1)).sum(dim=1) / count
        elif prior.ndim == 3:
            p = prior.mean(dim=1)
            a = adapted.mean(dim=1)
        else:
            p, a = prior, adapted
        p = F.normalize(p.float(), dim=-1).detach()
        a = F.normalize(a.float(), dim=-1)
        if p.shape[0] < 2:
            # microbatch=1：Gram [1,1] 恒为 1（归一化自相似），监督信号消失；
            # 退化为逐 token L2（与 anchor 同语义），避免恒零。
            return F.mse_loss(a, p)
        gram_a = a @ a.transpose(-1, -2)  # [B, B]
        gram_p = p @ p.transpose(-1, -2)
        return (gram_a - gram_p).square().mean(dim=1).mean()
