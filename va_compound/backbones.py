from __future__ import annotations

from collections.abc import Sequence
from math import isqrt
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PoolingMode = str  # "flat" | "spatial" | "spatiotemporal"


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen linear layer.

    ``base`` stays frozen; only ``lora_a``/``lora_b`` are trainable.  With
    ``rank=0`` the module degenerates to the plain frozen projection.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 32,
        alpha: float = 32.0,
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("lora rank must be non-negative")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank if rank else 0.0
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
        if self.rank:
            out = out + (x @ self.lora_a @ self.lora_b) * self.scaling
        return out


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
) -> int:
    """Wrap matching linear layers of ``model`` in-place with LoRA adapters.

    Returns the number of adapted layers.
    """
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

    walk(model)
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


def _dtype(name: str) -> torch.dtype:
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
    def encode(self, instructions: Sequence[str]) -> tuple[Tensor, Tensor]:
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
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
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

    def encode_trainable(
        self, instructions: Sequence[str]
    ) -> tuple[Tensor, Tensor]:
        """Online encode with gradients (for LoRA fine-tuning)."""
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
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return output.last_hidden_state, attention_mask.bool()

    def encode_with_scene(
        self,
        instructions: Sequence[str],
        scene_summary: Tensor,
        scene_projector: nn.Module,
        readout_tokens: Tensor,
        n_scene: int = 8,
    ) -> tuple[Tensor, Tensor]:
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
        readout = readout_tokens.to(device=device, dtype=inputs_embeds.dtype)
        if readout.ndim != 2 or readout.shape[-1] != dim:
            raise ValueError("readout_tokens must have shape [n_readout, language_dim]")
        readout_embeds = readout[None].expand(batch, -1, -1)
        embeds = torch.cat((inputs_embeds, scene_embeds, readout_embeds), dim=1)
        mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    batch, n_scene + readout_embeds.shape[1],
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
        )
        plan = output.last_hidden_state[:, length + n_scene :]  # readout positions
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


class VJEPA21Backbone(nn.Module):
    """Frozen V-JEPA 2.1 ViT-B encoder returning a bounded token sequence.

    ``pooling`` selects how the raw [t, h, w] token grid is reduced:
    - "flat": adaptive 1D pooling over the flattened sequence (legacy A);
    - "spatial": time-mean then 2D grid pooling, one output token per
      spatial neighbourhood (B);
    - "spatiotemporal": 2D grid pooling per frame, keeping the time axis (C).
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

    def _encode(self, pixel_values_videos: Tensor) -> Tensor:
        if pixel_values_videos.ndim != 5 or pixel_values_videos.shape[2] != 3:
            raise ValueError("video must have shape [batch, frames, 3, height, width]")
        if pixel_values_videos.shape[1] < 2 or pixel_values_videos.shape[1] % 2:
            raise ValueError("V-JEPA 2.1 needs a positive even number of frames")
        parameter = next(self.model.parameters())
        video = pixel_values_videos.permute(0, 2, 1, 3, 4).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.model(video)

    def _pool(self, tokens: Tensor, pooling: PoolingMode) -> Tensor:
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
