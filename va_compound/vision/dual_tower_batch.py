"""Online joint visual-language encoding for observation sequences."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .dual_tower import encode_dual_tower
from .encoding import _iter_imagenet_nchw_chunks


def encode_dual_tower_batch(
    frames,
    instructions,
    vision,
    text,
    fusion,
    device,
    *,
    grid=16,
    observation_chunk_size: int | None = None,
):
    """Preserve [batch, sequence, view] ordering and batch-wide text padding."""
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    if frames.ndim != 6 or frames.dtype != np.uint8:
        raise ValueError("joint frames must be uint8 [B,T,V,H,W,3]")
    batch, sequence, views, height, width, _ = frames.shape
    if len(instructions) != batch:
        raise ValueError("instruction count must match batch")
    if observation_chunk_size is not None:
        if (
            not isinstance(observation_chunk_size, int)
            or isinstance(observation_chunk_size, bool)
            or observation_chunk_size <= 0
        ):
            raise ValueError("observation_chunk_size must be a positive integer")

    if observation_chunk_size is None:
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

    total_obs = batch * sequence
    if total_obs == 0:
        empty_visual = torch.empty((batch, sequence, views * grid * grid, 0), device=device)
        empty_lang = torch.empty((batch, sequence, 0, 0), device=device)
        empty_masks = torch.empty((batch, sequence, 0), dtype=torch.bool, device=device)
        return empty_visual, empty_lang, empty_masks

    chunk_size = max(1, min(observation_chunk_size, total_obs))
    frames_tm = np.transpose(frames, (1, 0, 2, 3, 4, 5))
    flat_frames = np.ascontiguousarray(frames_tm.reshape(total_obs, views, height, width, 3))
    all_instructions = [instructions[i % batch] for i in range(total_obs)]

    visual_outputs, language_outputs, masks = [], [], []
    for start in range(0, total_obs, chunk_size):
        end = min(start + chunk_size, total_obs)
        c_obs = end - start
        chunk_frames = flat_frames[start:end]
        selected = np.ascontiguousarray(chunk_frames.reshape(c_obs * views, height, width, 3))
        _, images = next(_iter_imagenet_nchw_chunks(
            selected, device, encode_batch=c_obs * views, image_size=vision.image_size,
        ))
        chunk_insts = all_instructions[start:end]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens, language, mask = encode_dual_tower(
                images.reshape(c_obs, views, *images.shape[1:]), chunk_insts, vision, text, fusion,
            )
        patches = tokens.shape[1] // views
        side = int(patches ** 0.5)
        if side * side != patches:
            raise ValueError("DINO patch grid must be square")
        spatial = tokens.reshape(c_obs * views, side, side, tokens.shape[-1]).permute(0, 3, 1, 2)
        spatial = F.adaptive_avg_pool2d(spatial, (grid, grid))
        tokens = spatial.permute(0, 2, 3, 1).reshape(c_obs, views * grid * grid, -1)
        visual_outputs.append(tokens.float())
        language_outputs.append(language.float())
        masks.append(mask)

    all_visual = torch.cat(visual_outputs, dim=0)

    max_lang_len = max(l.shape[1] for l in language_outputs)
    padded_languages = []
    padded_masks = []
    for lang, m in zip(language_outputs, masks):
        cur_len = lang.shape[1]
        if cur_len < max_lang_len:
            pad_amount = max_lang_len - cur_len
            lang = F.pad(lang, (0, 0, 0, pad_amount), value=0.0)
            m = F.pad(m, (0, pad_amount), value=False)
        padded_languages.append(lang)
        padded_masks.append(m)

    all_language = torch.cat(padded_languages, dim=0)
    all_masks = torch.cat(padded_masks, dim=0)

    reshaped_visual = all_visual.reshape(sequence, batch, views * grid * grid, -1).transpose(0, 1).contiguous()
    reshaped_language = all_language.reshape(sequence, batch, max_lang_len, -1).transpose(0, 1).contiguous()
    reshaped_masks = all_masks.reshape(sequence, batch, max_lang_len).transpose(0, 1).contiguous()

    return reshaped_visual, reshaped_language, reshaped_masks
