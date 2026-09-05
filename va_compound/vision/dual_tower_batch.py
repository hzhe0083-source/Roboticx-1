"""Online joint visual-language encoding for observation sequences."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .dual_tower import encode_dual_tower
from .encoding import _iter_imagenet_nchw_chunks


def encode_dual_tower_batch(frames, instructions, vision, text, fusion, device, *, grid=16):
    """Preserve [batch, sequence, view] ordering and batch-wide text padding."""
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    if frames.ndim != 6 or frames.dtype != np.uint8:
        raise ValueError("joint frames must be uint8 [B,T,V,H,W,3]")
    batch, sequence, views, height, width, _ = frames.shape
    if len(instructions) != batch:
        raise ValueError("instruction count must match batch")
    visual_outputs, language_outputs, masks = [], [], []
    # Each decision's language depends on its images, so task-level language
    # deduplication is invalid after the first bidirectional exchange.
    for time_index in range(sequence):
        selected = np.ascontiguousarray(frames[:, time_index].reshape(batch * views, height, width, 3))
        _, images = next(_iter_imagenet_nchw_chunks(
            selected, device, encode_batch=batch * views, image_size=vision.image_size,
        ))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens, language, mask = encode_dual_tower(
                images.reshape(batch, views, *images.shape[1:]), instructions, vision, text, fusion,
            )
        patches = tokens.shape[1] // views
        side = int(patches ** 0.5)
        if side * side != patches:
            raise ValueError("DINO patch grid must be square")
        spatial = tokens.reshape(batch * views, side, side, tokens.shape[-1]).permute(0, 3, 1, 2)
        spatial = F.adaptive_avg_pool2d(spatial, (grid, grid))
        tokens = spatial.permute(0, 2, 3, 1).reshape(batch, views * grid * grid, -1)
        visual_outputs.append(tokens.float())
        language_outputs.append(language.float())
        masks.append(mask)
    return torch.stack(visual_outputs, 1), torch.stack(language_outputs, 1), torch.stack(masks, 1)
