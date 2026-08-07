#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open-loop counterfactual output displacement (C_OL) for e2e checkpoints.

判决实验 1（2026-08-05 用户审查 + Codex v2 定稿公式）：
对每个样本：同一视觉/状态输入，分别用 clean 指令 cache 与 swap（轮转）指令 cache
完整 rollout（循环记忆每条件独立重置，**每个决策点 clean/swap 复用同一 flow 噪声 z**）。

主指标（首个实际执行动作位移）：
    C_exec,i = mean_{t,r,d} |a^clean_{t,0,d} - a^swap_{t,0,d}|
次指标（完整 chunk 位移）：
    C_chunk,i = mean_{t,r,h,d} |a^clean_{t,h,d} - a^swap_{t,h,d}|
同噪声同时计算 E_clean / E_swap（相对真值动作的误差），报绝对差与相对差。

聚合顺序：decision/noise → trajectory → task 宏平均；任务级重采样配对 bootstrap CI。
分段报告（early/mid/late）识别 teacher-forcing 后半段动作历史掩盖语言效应。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prepare_pnpw_features import QwenTextBackbone
from va_compound.backbones import VJEPA21Backbone, apply_lora
from va_compound.model import VACompoundConfig, VACompoundPolicy

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C_OL counterfactual output displacement")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--flow-steps", type=int, default=32)
    return parser.parse_args()


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    policy = VACompoundPolicy(config).eval().to(device)
    policy.load_state_dict(ckpt["model"])
    vision = VJEPA21Backbone.from_pretrained(
        device=device, dtype="float16", max_tokens=64, local_files_only=True
    )
    if ckpt.get("vjepa_state_dict"):
        # e2e 类 checkpoint（B40k/C1/C2）：V-JEPA 被微调过，必须加载训练后权重；
        # 冻结特征类 checkpoint（A）无此键 → 直接用预训练权重（与训练时一致）
        vision.model.load_state_dict(ckpt["vjepa_state_dict"])
        print("vision: loaded vjepa_state_dict from checkpoint")
    vision.freeze_all()
    vision.freeze_all()
    text = QwenTextBackbone.from_pretrained(device=device, dtype="float16", local_files_only=True)
    if ckpt.get("qwen_state_dict"):
        qwen_state = {k.removeprefix("text_model."): v for k, v in ckpt["qwen_state_dict"].items()}
        text.text_model.load_state_dict(qwen_state, strict=False)
    if ckpt.get("lora"):
        rank = int(ckpt.get("training_contract", {}).get("lora_rank", 32))
        apply_lora(text.text_model, rank=rank)
        own = dict(text.text_model.named_parameters())
        for name, value in ckpt["lora"].items():
            clean = name.removeprefix("text_model.")
            if clean in own:
                own[clean].data.copy_(value)
    text.text_model.eval()
    return ckpt, config, policy, vision, text


def rollout(policy, vision, frames, proprio, prev_actions, cache, device, flow_steps, rng):
    """Full open-loop rollout; per-decision-point noise z_t drawn once and returned
    so the caller can reuse the exact same z for the perturbed condition."""
    b, t, w, c, h, ww = frames.shape
    tokens = vision(
        frames.reshape(b * t, w, c, h, ww), pooling="flat"
    ).reshape(b, t, -1, 768)
    chunks, memory, zs = [], None, []
    with torch.inference_mode():
        for ti in range(t):
            cond, memory = policy.encode_condition(
                tokens[:, ti],
                proprio[:, ti].to(device),
                prev_actions[:, ti].to(device),
                language_cache=cache,
                visual_memory=memory,
                return_visual_memory=True,
            )
            z = torch.randn(
                (1, policy.config.action_horizon, policy.config.action_dim),
                generator=rng,
                device=cond.device,
                dtype=cond.dtype,
            )
            zs.append(z)
            chunks.append(policy.sample_actions(cond, steps=flow_steps, noise=z)[0].cpu())
    return torch.stack(chunks), torch.stack(zs)  # [T,H,D], [T,1,H,D]


def paired_task_bootstrap(task_scores, n_boot=2000, seed=0):
    """Task-level resampling of paired trajectory scores; returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    tasks = sorted(task_scores.keys())
    means = []
    for _ in range(n_boot):
        picked = [task_scores[t][rng.integers(0, len(task_scores[t]))] for t in tasks]
        means.append(np.mean(picked))
    means = np.array(means)
    return np.mean(means), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    ckpt, config, policy, vision, text = load_model(args, device)
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    n = payload["actions"].shape[0]
    if args.max_samples > 0:
        n = min(n, args.max_samples)
    tasks = list(payload["metadata"]["tasks"])
    swapped = tasks[1:] + tasks[:1]  # 轮转 1（筛查版 swap）
    hidden, mask = text.encode(tasks)
    hidden_s, mask_s = text.encode(swapped)
    caches = [policy.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device)) for i in range(len(tasks))]
    caches_s = [policy.build_language_cache(hidden_s[i : i + 1].to(device), mask_s[i : i + 1].to(device)) for i in range(len(tasks))]

    frames = payload["video_frames"][:n].float().div_(255.0)  # stay on CPU (OOM fix 2026-08-06)
    frames = (frames - IMAGE_MEAN) / IMAGE_STD
    proprio = payload["proprio"][:n]
    prev = payload["previous_action"][:n]
    acts = payload["actions"][:n]
    ids = payload["instruction_id"][:n].tolist()
    t = frames.shape[1]

    # 按任务聚合（配对：每个 trajectory 同时贡献 clean/swap 的 C_OL 与 E）
    task_col_exec = {}
    task_col_chunk = {}
    task_delta = {}  # E_swap - E_clean（配对绝对差）
    seg_all = {"early": [], "mid": [], "late": []}
    with torch.inference_mode():
        for i in range(n):
            cache = caches[int(ids[i])]
            cache_s = caches_s[int(ids[i])]
            a_clean, _ = rollout(policy, vision, frames[i : i + 1].to(device), proprio[i : i + 1], prev[i : i + 1], cache, device, args.flow_steps, rng)
            a_swap, _ = rollout(policy, vision, frames[i : i + 1].to(device), proprio[i : i + 1], prev[i : i + 1], cache_s, device, args.flow_steps, rng)
            d_exec = (a_clean[:, 0] - a_swap[:, 0]).abs().mean(dim=-1)  # [T]
            d_chunk = (a_clean - a_swap).abs().mean(dim=(-1, -2))  # [T]
            target = acts[i]  # CPU (a_clean/a_swap are CPU; device mismatch fix 2026-08-06)
            e_clean = (a_clean - target).abs().mean(dim=(-1, -2))  # [T]
            e_swap = (a_swap - target).abs().mean(dim=(-1, -2))
            tid = int(ids[i])
            task_col_exec.setdefault(tid, []).append(float(d_exec.mean()))
            task_col_chunk.setdefault(tid, []).append(float(d_chunk.mean()))
            task_delta.setdefault(tid, []).append(float((e_swap - e_clean).mean()))
            n_seg = max(1, t // 5)
            seg_all["early"].append(float(d_exec[:n_seg].mean()))
            seg_all["mid"].append(float(d_exec[n_seg : t - n_seg].mean()) if t - n_seg > n_seg else float("nan"))
            seg_all["late"].append(float(d_exec[t - n_seg :].mean()))

    m_exec, lo_exec, hi_exec = paired_task_bootstrap(task_col_exec)
    m_chunk, lo_chunk, hi_chunk = paired_task_bootstrap(task_col_chunk)
    m_delta, lo_delta, hi_delta = paired_task_bootstrap(task_delta)
    base0 = float((prev[:, 0, None, :] - acts[:, 0, :1, :]).abs().mean((-1, -2)).numpy().mean())

    def seg_stat(x):
        x = np.array([v for v in x if not np.isnan(v)])
        return x.mean(), x.std() / np.sqrt(len(x))

    print(f"=== C_OL (clean vs swap instruction, flow_steps={args.flow_steps}, 同噪声配对) ===")
    print(f"n_trajectories={n} 决策点数 T={t} 任务数={len(task_col_exec)}")
    print(f"C_OL 首执行动作（主指标）: {m_exec:.5f} [95% CI {lo_exec:.5f}, {hi_exec:.5f}]")
    print(f"C_OL 完整 chunk（次指标）: {m_chunk:.5f} [95% CI {lo_chunk:.5f}, {hi_chunk:.5f}]")
    print(f"配对误差差 E_swap-E_clean（绝对）: {m_delta:.5f} [95% CI {lo_delta:.5f}, {hi_delta:.5f}]")
    for seg in ("early", "mid", "late"):
        sm, sse = seg_stat(seg_all[seg])
        print(f"C_OL 首动作 分段[{seg}]: {sm:.5f} ± {sse:.5f}")
    print(f"参考: 持久性位移（决策点0 prev→target）{base0:.5f}")
    print(f"C_OL(exec)/持久性位移 = {m_exec / base0:.3f}（>1 语言效应强于动作惯性；<0.1 语言几乎不改变输出）")


if __name__ == "__main__":
    main()
