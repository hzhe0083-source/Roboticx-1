"""DINO encoding shared by training, evaluation and diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from va_compound import VACompoundConfig
from va_compound.utils.exact_resume import _sha256_file

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _iter_imagenet_nchw_chunks(
    frames_u8: np.ndarray,
    device: torch.device,
    *,
    encode_batch: int,
    image_size: int,
):
    """Upload uint8 NHWC frames and yield ImageNet-normalized NCHW GPU chunks.

    Peer training encodes 192 frames per stream (batch 12 × T4 × W4) twice
    each step. Expanding them to float32 on CPU made a ~0.5 GiB host tensor
    per stream and left the GPU idle between microbatches.
    """
    if encode_batch < 1:
        raise ValueError("encode_batch must be positive")
    if frames_u8.dtype != np.uint8 or frames_u8.ndim != 4 or frames_u8.shape[-1] != 3:
        raise ValueError(
            "uint8 frames must be [N,H,W,3], got "
            f"{tuple(frames_u8.shape)}/{frames_u8.dtype}"
        )
    images_u8 = torch.from_numpy(np.ascontiguousarray(frames_u8))
    if device.type == "cuda":
        images_u8 = images_u8.pin_memory()
    mean = torch.tensor(_IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    target = (image_size, image_size)
    for start in range(0, images_u8.shape[0], encode_batch):
        chunk = images_u8[start : start + encode_batch].to(
            device, non_blocking=device.type == "cuda"
        )
        chunk = chunk.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
        if tuple(chunk.shape[-2:]) != target:
            chunk = F.interpolate(
                chunk,
                size=target,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        yield start, (chunk - mean) / std


def _build_dino_main_backbone(
    args: argparse.Namespace,
    config: VACompoundConfig,
    device: torch.device,
):
    """DINOv2 tower as the replacement main vision backbone.

    V-JEPA stays available in the repository for the legacy path; this tower
    is only built under ``--dino-main-vision``.
    """
    checkpoint = args.main_vision_checkpoint.expanduser().absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"main vision checkpoint is missing: {checkpoint}")
    from va_compound.backbones import TimmActionVisionBackbone

    backbone = TimmActionVisionBackbone.from_pretrained(
        device=device,
        dtype=(
            "float32" if getattr(args, "vision_unfreeze_all", False) else "float16"
        ),
        model_id=config.main_vision_model_id,
        image_size=config.main_vision_image_size,
        feature_dim=config.main_vision_dim,
        output_layers=(11, 23),  # reuse the canonical mid/final-key contract
        checkpoint_path=checkpoint,
        local_files_only=True,
    )
    if getattr(args, "vision_unfreeze_all", False):
        backbone.unfreeze_all()
    else:
        backbone.freeze_all()
    args.main_vision_checkpoint_sha256 = _sha256_file(checkpoint)
    mode = "fully trainable FP32-master/BF16-forward" if backbone._trainable else "frozen"
    print(
        f"dino-main: {mode} {config.main_vision_backbone} REPLACES V-JEPA as the "
        f"VA main vision ({config.main_vision_image_size}px, "
        f"dim={config.main_vision_dim}, {config.main_vision_tokens} tokens/decision, "
        f"params={sum(p.numel() for p in backbone.parameters()):,})",
        flush=True,
    )
    return backbone


def _dino_main_online_encode(
    frames,
    backbone,
    device: torch.device,
    *,
    encode_batch: int,
    grid: int,
    window: int,
    return_dense: bool = False,
    last_four_mean: bool = False,
    return_last_four: bool = False,
    return_last_layers: int = 0,
) -> Tensor | tuple[Tensor, dict[int, Tensor]] | tuple[Tensor, Tensor]:
    """DINO-main vision tokens: [B, T, window*grid*grid, dim] fp32 per decision.

    Every decision consumes the complete ``window``-frame history window
    ``[d-6,d-4,d-2,d]``; each frame is encoded by the frozen tower and its
    16x16 patch grid is average-pooled to ``grid x grid``. Timm NLC patch
    order is row-major; the pooling grid order is verified by
    tests/test_dino_main_vision.py.

    ``return_dense=True``（DINO-metric，2026-08-15）：额外返回 dense
    evidence ``{5: [B, T, 512, D], 11: [B, T, 512, D]}``——canonical key 5 =
    block11（g，帧 [d-2,d] 各 256 patch），key 11 = block23（d），两帧沿
    token 维拼接（前 256 = d-2，后 256 = d，t→y→x 序），供
    DenseEvidenceProjector/语言度量场消费（与 V-JEPA {5,11} 语义对齐）。
    """
    if encode_batch < 1:
        raise ValueError("main vision encode_batch must be positive")
    if not (1 <= grid <= 16) or window < 1:
        raise ValueError("dino-main grid/window out of range")
    if return_last_layers < 0:
        raise ValueError("return_last_layers must be nonnegative")
    if sum(
        (
            bool(return_dense),
            bool(last_four_mean),
            bool(return_last_four),
            bool(return_last_layers),
        )
    ) > 1:
        raise ValueError("DINO dense/mean/layerwise outputs are mutually exclusive")
    layer_count = 4 if return_last_four else return_last_layers
    if frames.ndim != 6 or frames.shape[-1] != 3:
        raise ValueError(
            "dino-main frames must be [B,T,W,H,W,3], got " f"{tuple(frames.shape)}"
        )
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    batch_size, sequence_length, win, height, width, _ = frames_np.shape
    if win != window:
        raise ValueError(f"dino-main requires the {window}-frame window, got {win}")
    selected = np.ascontiguousarray(
        frames_np.reshape(batch_size * sequence_length * window, height, width, 3)
    )
    chunks: list[Tensor] = []
    dense5: list[Tensor] = []
    dense11: list[Tensor] = []
    last_layer_chunks: list[Tensor] = []
    backbone_parameter = (
        next(backbone.parameters(), None)
        if callable(getattr(backbone, "parameters", None))
        else None
    )
    for start, chunk in _iter_imagenet_nchw_chunks(
        selected,
        device,
        encode_batch=encode_batch,
        image_size=backbone.image_size,
    ):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda"
            and backbone_parameter is not None
            and backbone_parameter.dtype == torch.float32,
        ):
            hierarchical = (
                backbone.forward_hierarchical_dense(chunk)
                if return_dense or not hasattr(backbone, "forward_final_dense")
                else None
            )
            last_layers = (
                backbone.forward_last_four_dense(chunk)
                if return_last_four
                else backbone.forward_last_layers_dense(chunk, return_last_layers)
                if return_last_layers
                else None
            )
            tokens = (
                hierarchical[11]
                if hierarchical is not None
                else last_layers[:, -1]
                if last_layers is not None
                else backbone.forward_last_four_mean_dense(chunk)
                if last_four_mean
                else backbone.forward_final_dense(chunk)
            )
        if tokens.shape[-2] != 256 or tokens.shape[-1] != backbone.feature_dim:
            raise RuntimeError(
                "dino-main expects 256 patch tokens per frame, got "
                f"{tuple(tokens.shape)}"
            )
        chunks.append(tokens.float())
        if last_layers is not None:
            if tuple(last_layers.shape[1:3]) != (layer_count, 256):
                raise RuntimeError(
                    f"DINO layerwise output must be [B,{layer_count},256,D], got "
                    f"{tuple(last_layers.shape)}"
                )
            last_layer_chunks.append(last_layers)
        if return_dense:
            if hierarchical is None:
                raise RuntimeError("DINO dense evidence requires hierarchical outputs")
            # 只保留帧 [d-2, d]（窗口内 w ∈ {2, 3}）的两层 patch 证据。
            flat_indices = [
                start + j
                for j in range(tokens.shape[0])
                if (start + j) % window in (2, 3)
            ]
            if flat_indices:
                local = [idx - start for idx in flat_indices]
                for source, target in (
                    (hierarchical[5], dense5),
                    (hierarchical[11], dense11),
                ):
                    picked = source[local]
                    if picked.shape[-2] != 256 or picked.shape[-1] != backbone.feature_dim:
                        raise RuntimeError(
                            "dino-metric expects 256 patch tokens per frame at "
                            f"block {5 if source is hierarchical[5] else 11}, got "
                            f"{tuple(picked.shape)}"
                        )
                    target.append(picked.float())
        del chunk, hierarchical, last_layers, tokens
    tokens = torch.cat(chunks, dim=0)  # [B*T*W, 256, D]
    dim = tokens.shape[-1]
    tokens = tokens.reshape(
        batch_size * sequence_length * window, 16, 16, dim
    ).permute(0, 3, 1, 2)
    tokens = F.adaptive_avg_pool2d(tokens, (grid, grid))
    tokens = tokens.permute(0, 2, 3, 1).reshape(
        batch_size, sequence_length, window * grid * grid, dim
    )
    if layer_count:
        layers = torch.cat(last_layer_chunks, dim=0)
        layers = layers.reshape(
            batch_size,
            sequence_length,
            window,
            layer_count,
            16,
            16,
            dim,
        ).permute(0, 1, 3, 2, 4, 5, 6)
        layers = layers.reshape(
            batch_size * sequence_length * layer_count * window,
            16,
            16,
            dim,
        ).permute(0, 3, 1, 2)
        layers = F.adaptive_avg_pool2d(layers, (grid, grid))
        layers = layers.permute(0, 2, 3, 1).reshape(
            batch_size,
            sequence_length,
            layer_count,
            window * grid * grid,
            dim,
        )
        return tokens, layers
    if not return_dense:
        return tokens
    dense_evidence = {
        layer: torch.cat(parts, dim=0).reshape(
            batch_size, sequence_length, -1, parts[0].shape[-1]
        )
        for layer, parts in ((5, dense5), (11, dense11))
    }
    for layer, evidence in dense_evidence.items():
        if evidence.shape[-2] != 512:
            raise RuntimeError(
                f"dino-metric dense evidence {layer} must be 512 tokens "
                f"(2 frames x 256 patches), got {tuple(evidence.shape)}"
            )
    return tokens, dense_evidence


class DinoFeatureCache:
    """预计算 DINO block11/block23 特征缓存（2026-08-15，步时优化）。

    在线 ViT-L 编码占训练步时 84%（profile：2.97s/3.51s）；冻结塔确定性，
    全部唯一帧特征离线预计算为 fp16 memmap（scripts/build_dino_feature_cache.py），
    训练循环从缓存读。位级一致性由预计算脚本内置验证（torch.equal）保证；
    eval 仍在线编码真实新帧。
    """

    def __init__(self, path: Path) -> None:
        import json
        import pickle

        self.path = Path(path).expanduser()
        if not self.path.is_dir():
            raise ValueError(f"DINO feature cache directory missing: {self.path}")
        with (self.path / "meta.json").open() as fh:
            self.meta = json.load(fh)
        with (self.path / "index.pkl").open("rb") as fh:
            self.index: dict = pickle.load(fh)
        expected_features = self.meta.get("feature_sha256")
        if (
            self.meta.get("feature_identity_contract") != "sha256_full_npy_v1"
            or not isinstance(expected_features, dict)
        ):
            raise ValueError("DINO feature cache lacks full feature SHA-256 metadata")
        for name in ("block11.npy", "block23.npy"):
            expected = expected_features.get(name)
            actual = _sha256_file(self.path / name)
            if not expected or actual != expected:
                raise ValueError(
                    f"DINO feature cache {name} SHA-256 mismatch: "
                    f"expected={expected!r}, actual={actual}"
                )
        self.block23 = np.load(
            self.path / "block23.npy", mmap_mode="r"
        )  # [N, 256, 1024] fp16
        self.block11 = np.load(
            self.path / "block11.npy", mmap_mode="r"
        )  # [N, 256, 1024] fp16
        if self.block23.shape != self.block11.shape:
            raise ValueError("feature cache block23/block11 shape mismatch")
        if self.block23.shape[0] != len(self.index):
            raise ValueError("feature cache rows != index length")
        print(
            f"dino feature cache: {len(self.index)} frames, "
            f"{self.meta.get('model_id')} @{self.meta.get('image_size')}px, "
            f"chunk={self.meta.get('chunk')}, "
            f"dataset_sha256={self.meta.get('dataset_sha256', '?')[:12]}…",
            flush=True,
        )

    def frames(self, rows: np.ndarray) -> dict[int, torch.Tensor]:
        """rows [B, T, W] int64 → {5: [B,T,W,256,D], 11: [...]} GPU fp16。

        键语义与 online forward_hierarchical_dense 相同（5=block11，11=block23）。
        """
        b, t, w = rows.shape
        flat = rows.reshape(-1)
        out = {}
        for key, mem in ((5, self.block11), (11, self.block23)):
            picked = np.asarray(mem[flat])  # [B*T*W, 256, 1024] fp16
            out[key] = torch.from_numpy(picked).reshape(b, t, w, 256, -1)
        return out


def _dino_main_encode_from_cache(
    rows: torch.Tensor,
    cache: DinoFeatureCache,
    device: torch.device,
    *,
    grid: int,
    window: int,
    return_dense: bool = False,
) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
    """缓存读 + 与在线路径同构的池化/证据组装（位级一致，见 precompute 验证）。

    与 _dino_main_online_encode 的差异仅在于 block 特征来自 memmap 而非塔前向：
    同一 16×16→grid×grid adaptive_avg_pool、同一 [d-2,d] 两帧 evidence 序。
    """
    if rows.ndim != 3 or rows.shape[-1] != window:
        raise ValueError(
            f"frame_cache_rows 必须 [B, T, {window}]，got {tuple(rows.shape)}"
        )
    rows_np = rows.detach().cpu().numpy().astype(np.int64)
    evidence = cache.frames(rows_np)  # {5, 11}: [B,T,W,256,D] fp16 CPU
    b, t, w, n_patch, dim = evidence[11].shape
    if n_patch != 256 or dim != 1024:
        raise RuntimeError(
            f"feature cache 期望 256×1024 每帧，got {n_patch}×{dim}"
        )
    tokens = evidence[11].to(device).float()  # [B,T,W,256,D]
    tokens = tokens.reshape(b * t * w, 16, 16, dim).permute(0, 3, 1, 2)
    tokens = F.adaptive_avg_pool2d(tokens, (grid, grid))
    tokens = tokens.permute(0, 2, 3, 1).reshape(b, t, w * grid * grid, dim)
    if not return_dense:
        return tokens
    dense = {}
    for key in (5, 11):
        ev = evidence[key].to(device).float()  # [B,T,W,256,D]
        dense[key] = torch.cat((ev[:, :, 2], ev[:, :, 3]), dim=2).reshape(
            b, t, 512, dim
        )
    return tokens, dense
