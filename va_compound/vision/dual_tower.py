"""Joint tail execution using native Qwen masking and DINO patch preparation."""
from __future__ import annotations

import torch
from torch import Tensor


def encode_dual_tower(images: Tensor, instructions, vision_backbone, text_backbone, fusion):
    """Encode [B,V,3,H,W] images with one instruction per observation.

    Qwen's native forward owns rotary embeddings, causal masks and hybrid blocks.
    Local decoder hooks synchronously advance DINO before replacing both outputs.
    Gradient checkpointing must remain disabled: replaying a decoder block outside
    this forward would otherwise replay it without its paired fusion hook.
    """
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must be [batch, views, 3, height, width]")
    batch, views = images.shape[:2]
    if len(instructions) != batch:
        raise ValueError("one instruction is required per observation")
    dino = vision_backbone.model
    qwen = text_backbone.text_model
    if getattr(dino, "grad_checkpointing", False) or getattr(qwen, "gradient_checkpointing", False):
        raise ValueError("joint dual-tower execution requires checkpointing disabled")
    count = len(fusion.pairs)
    if len(dino.blocks) < count or len(qwen.layers) < count:
        raise ValueError("backbones have fewer blocks than fusion pairs")
    input_ids, mask = text_backbone._tokenize_instructions(instructions)
    parameter = next(dino.parameters())
    flat = images.flatten(0, 1).to(device=parameter.device, dtype=parameter.dtype)
    visual = dino.norm_pre(dino.patch_drop(dino._pos_embed(dino.patch_embed(flat))))
    for block in dino.blocks[:-count]:
        visual = block(visual)
    handles = []
    visited = []
    prefix = int(dino.num_prefix_tokens)

    def exchange(index):
        def hook(_module, _inputs, output):
            nonlocal visual
            if index != len(visited):
                raise RuntimeError("joint backbone blocks executed out of order")
            language = output[0] if isinstance(output, tuple) else output
            visual = dino.blocks[len(dino.blocks) - count + index](visual)
            patches = visual[:, prefix:]
            token_count = patches.shape[1]
            grouped = patches.reshape(batch, views * token_count, patches.shape[-1])
            fused_visual, fused_language = fusion.forward_pair(index, grouped, language, mask.bool())
            patches = fused_visual.reshape(batch * views, token_count, patches.shape[-1])
            visual = torch.cat((visual[:, :prefix], patches), dim=1)
            visited.append(index)
            if isinstance(output, tuple):
                return (fused_language, *output[1:])
            return fused_language
        return hook

    try:
        for index, block in enumerate(qwen.layers[-count:]):
            handles.append(block.register_forward_hook(exchange(index)))
        output = qwen(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if len(visited) != count:
        raise RuntimeError("not all joint backbone pairs executed")
    patches = dino.norm(visual)[:, prefix:]
    return patches.reshape(batch, views * patches.shape[1], patches.shape[-1]), output.last_hidden_state, mask.bool()
