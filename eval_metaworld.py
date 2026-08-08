"""Closed-loop evaluation on MetaWorld MT50 (language-conditioned).

Protocol: 10 episodes per task, four difficulty tiers.  The policy receives
the corner2 camera image (480x480, resized to 384), the 4-dim state
(hand x/y/z + gripper, normalized with dataset quantiles), the previous
action, and the task language condition.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch
from torch.nn import functional as F

from prepare_pnpw_features import QwenTextBackbone

_DEBUG_FA_DONE: dict[str, bool] = {}
_ALIGN_ACTS: list | None = None
from va_compound.backbones import (
    QwenSemanticBackbone,
    VJEPA21Backbone,
)
from va_compound.model import (
    ControllerParams,
    VACompoundConfig,
    VACompoundPolicy,
)

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

try:
    from va_compound.live_vjepa import _slot_coords as _stage_coords
except Exception:  # pragma: no cover - 环境裁剪场景
    _stage_coords = None


def _apply_local_vision(model, tokens, language_cache):
    """Stage A/B：local_slots 读出路径（direct288 恒等；slots 走槽 cross-attn）。

    训练侧 rollout_policy 用 build_local_vision(st, coords, role_queries) 喂
    encode_condition；闭环评估必须走同一路径，否则 288-token checkpoint 的
    槽/坐标读出被跳过（闭环数字失真）。
    """
    if not model.config.local_slots:
        return tokens
    if _stage_coords is None:
        raise RuntimeError("local_slots eval requires va_compound.live_vjepa")
    role_queries = (
        getattr(language_cache, "role_queries", None)
        if language_cache is not None
        else None
    )
    coords = torch.from_numpy(_stage_coords()).to(
        device=tokens.device, dtype=tokens.dtype
    )
    return model.build_local_vision(tokens, coords, role_queries)

VISION_WINDOW = 4
DECISION_STRIDE = 6  # 80 FPS, decide every 6 frames (13.3 Hz), matches training
ACTION_HORIZON = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop MetaWorld MT50 eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="metaworld_features.pt for normalization")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vision-pooling",
        choices=("flat", "spatial", "spatiotemporal"),
        default=None,
        help="在线 V-JEPA 池化（覆盖 training_contract；Stage A/B 288-token "
        "checkpoint 必须为 spatiotemporal，否则闭环数字失真）",
    )
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=49)
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="逗号分隔的任务索引子集（从 features metadata.tasks 里选，8 任务诊断用）；"
        "缺省 = 前 --max-tasks 个",
    )
    parser.add_argument(
        "--plan-refresh",
        type=int,
        default=0,
        help="Plan-Cache 闭环刷新间隔：每 R 个决策用当前场景重建语言缓存 "
        "（0 = 仅 episode 首帧建一次；仅在 checkpoint config 含 plan_resampler "
        "或 scene_teacher 时可用）",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="每 episode 最大步数。数据/环境上限 500 步（10/49 任务专家轨迹达 500 步），"
        "400 会截断这些任务低估成功率",
    )
    parser.add_argument(
        "--align-init",
        action="store_true",
        help="reset 后对齐数据首帧 init（物体+target），验证模型在训练分布上的闭环能力",
    )
    parser.add_argument(
        "--debug-first-action",
        action="store_true",
        help="打印首次决策的模型动作（与 --align-init 联用可对比数据专家动作）",
    )
    parser.add_argument(
        "--memory-reset-every",
        type=int,
        default=0,
        help="每 N 个决策点将递归视觉记忆置 None（0 = 不重置）。训练只展开 T=4 次而部署"
        "连续递归几十次——此参数将部署记忆截断到与训练一致的递归深度，零训练代价的"
        "契约缺口对照（2026-08-06 Codex 判决顺序第 3 步）",
    )
    parser.add_argument(
        "--prev-zero",
        action="store_true",
        help="把 previous_action 输入恒置零（归一化 0）。previous_action 训练用真值、"
        "闭环用模型自身输出（自激）——此参数将闭环 prev 改为恒零，零训练代价的"
        "自激对照（2026-08-06 Codex 16-task panel 条件）",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=DECISION_STRIDE,
        help="每决策执行 chunk 的原始步数（决策节奏 = 此值）。SmolVLA 每个原始步都重新"
        "推理（execute=1），我们默认 6 步/决策——协议差距探针（2026-08-06 Codex）",
    )
    parser.add_argument(
        "--direct-head",
        choices=("auto", "on", "off"),
        default="auto",
        help="C²-VA Stage A 解码器：auto = 从 checkpoint config 读 direct_head"
        "（默认）；on/off = 强制 Direct Head / flow matching（消融对照）。"
        "direct 时经 decode_actions 一次前向，flow 时仍走 32 步 Euler 采样",
    )
    parser.add_argument(
        "--plan-stride",
        type=int,
        default=None,
        help="C² 部署（Codex 修正 5）：VA 生成 {ū,c̄,K} 的重规划间隔（原始步）。"
        "默认 6 = DECISION_STRIDE；token 0..5 顺序消费后重规划",
    )
    parser.add_argument(
        "--feedback-stride",
        type=int,
        default=None,
        help="C² 部署：c_current 刷新间隔（原始步，V-JEPA → P）。默认 1——每原始步"
        "刷新并消费下一个 token；>1 时中间步保持上一 token 动作",
    )
    parser.add_argument(
        "--c2-oracle-ref",
        action="store_true",
        help="C² 消融：用 ground-truth 参考替代预测 c̄（c̄ ≡ c_current，e ≡ 0）——"
        "测参考零误差上界（K 修正空转，仅名义 ū 执行）",
    )
    parser.add_argument(
        "--c2-zero-gain",        action="store_true",
        help="C² 消融：增益 K 恒置零（等价 Stage A 名义执行，go/no-go 对照）",
    )
    parser.add_argument(
        "--c2-gain-scale",
        type=float,
        default=1.0,
        help="部署时对 K 的缩放系数（damping 消融：<1 减弱反馈修正）",
    )
    parser.add_argument(
        "--c2-error-threshold",
        type=float,
        default=0.0,
        help="误差死区门控：‖e‖ < 阈值时跳过 K 修正（只执行名义动作）",
    )
    parser.add_argument(
        "--c2-recovery-eval",
        type=Path,
        default=None,
        help="C² 恢复评估：从 v6b 的 held-out 扰动分支初始状态出发闭环，"
        "测'拉回'成功率（需要按钮任务 c2 checkpoint）",
    )
    parser.add_argument(
        "--c2-recovery-split",
        choices=("train", "heldout"),
        default="heldout",
        help="恢复评估用 v6b 的哪个 split（go/no-go 用 held-out 扰动种子）",
    )
    return parser.parse_args()


def preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)[None]
    if tensor.shape[-1] != image_size:
        tensor = F.interpolate(
            tensor, size=(image_size, image_size), mode="bicubic",
            align_corners=False, antialias=True,  # 与 prepare_metaworld.py 管线一致
        )
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def plan_refresh_due(decision_count: int, plan_refresh: int) -> bool:
    """Plan-Cache 缓存重建时机：首决策（episode 开始）必须建；之后每 R 决策一次。

    ``decision_count`` 从 1 开始计数（每 episode 第一个决策 = 1）。
    """
    if decision_count == 1:
        return True
    return plan_refresh > 0 and (decision_count - 1) % plan_refresh == 0


def c2_schedule(
    step: int,
    plan_step: int | None,
    plan_stride: int,
    feedback_stride: int,
    horizon: int,
) -> tuple[bool, bool, int]:
    """C² 部署节奏（Codex 修正 5：plan 与 feedback 解耦）。

    Returns (plan_due, feedback_due, token_index)：
    - plan_due：距上次规划 >= plan_stride（或尚无规划）；token 用尽也强制重规划；
    - feedback_due：距规划步为 feedback_stride 整数倍（plan 步本身必刷新）；
    - token_index：自规划以来应消费的 token（= 距规划步 / feedback_stride）。
    """
    if plan_stride < 1 or feedback_stride < 1:
        raise ValueError("plan/feedback stride must be positive")
    if plan_step is None or step - plan_step >= plan_stride:
        return True, True, 0
    token_index = (step - plan_step) // feedback_stride
    if token_index >= horizon:
        return True, True, 0
    feedback_due = (step - plan_step) % feedback_stride == 0
    return False, feedback_due, token_index


def run_c2_recovery_eval(
    args,
    device,
    model,
    vision_backbone,
    features,
    sq01,
    scale_s,
    aq01,
    aq99,
    vision_pooling="flat",
) -> None:
    """C² 恢复评估（Codex go/no-go ③）：从 v6b held-out 扰动分支初始状态出发
    闭环，测"拉回"成功率。分支状态由 prepare_mw_recovery.py 的 snapshot
    完整恢复（qpos/qvel/mocap/act/time/_prev_obs/target）。"""
    import json

    import metaworld

    if not model.config.c2_controller:
        raise ValueError("--c2-recovery-eval requires a c2 checkpoint")
    rec = torch.load(args.c2_recovery_eval, map_location="cpu", weights_only=True)
    starts = [s for s in rec["recovery_start"] if s["split"] == args.c2_recovery_split]
    if not starts:
        raise ValueError(f"no recovery branches with split={args.c2_recovery_split}")
    tasks = features["metadata"]["tasks"]
    if "Press a button" not in tasks:
        raise ValueError("recovery eval requires the button-press task in features")
    task_index = tasks.index("Press a button")
    row = int((features["instruction_id"] == task_index).nonzero()[0][0])
    hidden = features["language_hidden"][row : row + 1].to(device)
    language_mask = features["language_mask"][row : row + 1].to(device)
    language_cache = model.build_language_cache(hidden, language_mask)

    config_path = (
        Path("/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json")
    )
    mw_config = json.load(open(config_path))
    if "button-press-v3" not in mw_config["TASK_DESCRIPTIONS"]:
        raise ValueError("button-press-v3 missing from metaworld_config.json")
    mt1 = metaworld.MT1("button-press-v3", seed=42)
    env = mt1.train_classes["button-press-v3"](render_mode="rgb_array", camera_name="corner2")
    env.set_task(mt1.train_tasks[0])
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env._freeze_rand_vec = False

    from prepare_mw_recovery import restore_env

    wins = 0
    trials = min(len(starts), args.trials_per_task)
    for branch in starts[:trials]:
        env.reset(seed=int(branch["reset_seed"]))
        restore_env(env, branch["snapshot"])
        prev_action = branch["prev_action"]
        last_norm = (
            prev_action.numpy()
            if isinstance(prev_action, torch.Tensor)
            else np.asarray(prev_action, dtype=np.float32)
        )
        memory = None
        success = False
        plan_step = None
        c2_token = 0
        c2_params = None
        frame_buffer = []
        obs = env._get_obs()
        for step in range(args.horizon):
            img = env.render()
            frame_buffer.append(img)
            if step == 0:
                while len(frame_buffer) < (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                    frame_buffer.insert(0, img)
            if len(frame_buffer) > (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                frame_buffer.pop(0)
            indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
            frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
            clip = torch.cat([preprocess(f, 384) for f in frames], dim=0).to(device)
            plan_due, feedback_due, _ = c2_schedule(
                step, plan_step, args.plan_stride, args.feedback_stride, ACTION_HORIZON
            )
            if plan_due:
                state = np.clip(
                    2.0 * (obs[:4] - sq01) / scale_s - 1.0, -1.0, 1.0
                ).astype(np.float32)
                proprio = torch.tensor(state, device=device)[None, None]
                previous = torch.tensor(
                    last_norm, dtype=torch.float32, device=device
                )[None, None]
                with torch.inference_mode():
                    tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                    c_current = model.control_projector(tokens)
                    cond, memory = model.encode_condition(
                        tokens,
                        proprio[0],
                        previous[0],
                        language_cache=language_cache,
                        visual_memory=memory,
                        return_visual_memory=True,
                    )
                    c2_params = model.controller_params(cond, c_current)
                    if args.c2_zero_gain:
                        c2_params = ControllerParams(
                            c2_params.nominal,
                            c2_params.reference,
                            torch.zeros_like(c2_params.gain),
                        )
                plan_step = step
                c2_token = 0
            if feedback_due and c2_token < ACTION_HORIZON and c2_params is not None:
                with torch.inference_mode():
                    if step != plan_step:
                        tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                    c_current = model.control_projector(tokens)
                    if args.c2_oracle_ref:
                        norm_action = c2_params.nominal[0, c2_token].cpu().numpy()
                    else:
                        error = c_current[0] - c2_params.reference[0, c2_token]
                        if (
                            args.c2_error_threshold > 0.0
                            and float(error.norm()) < args.c2_error_threshold
                        ):
                            norm_action = c2_params.nominal[0, c2_token].cpu().numpy()
                        else:
                            norm_action = (
                                c2_params.nominal[0, c2_token]
                                - c2_params.gain[0, c2_token] @ error
                            ).cpu().numpy()
                norm_action = np.clip(norm_action, -1.0, 1.0)
                c2_token += 1
            else:
                norm_action = last_norm
            action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
            obs, reward, terminated, truncated, info = env.step(action)
            last_norm = norm_action
            if info.get("success"):
                success = True
                break
            if terminated or truncated:
                break
        wins += int(success)
        print(
            f"  recovery branch seed={branch['seed']} kind={branch['kind']}: "
            f"{'SUCCESS' if success else 'FAIL'}"
        )
    env.close()
    print(
        f"\nC2 RECOVERY SUCCESS: {wins}/{trials} = {wins / trials:.1%} "
        f"(split={args.c2_recovery_split}, oracle_ref={args.c2_oracle_ref}, "
        f"zero_gain={args.c2_zero_gain})"
    )


def restore_text_backbone(
    ckpt: dict,
    device: torch.device,
    language_dtype: str = "float16",
    language_max_length: int = 64,
) -> QwenTextBackbone | QwenSemanticBackbone:
    """P0-3：按 training_contract 恢复文本分支（普通 / semantic adapter）。

    - 普通 ckpt：QwenTextBackbone + contract.lora_rank 建 LoRA + 加载
      qwen_state_dict / lora（旧行为不变）；
    - semantic ckpt（contract.semantic_adapter=True）：QwenSemanticBackbone
      （按 semantic_lora_rank / semantic_top_layers / semantic_lora_suffixes
      构造，构造器内部建 LoRA）+ 加载 qwen_state_dict / lora /
      semantic_gate。旧实现用 contract.lora_rank（semantic 模式下恒 0）建
      LoRA 且不构造 wrapper/门控，semantic checkpoint 完全无法恢复。
    返回加载完毕的 backbone（eval 模式）。
    """
    contract = ckpt.get("training_contract", {}) or {}
    text_backbone = QwenTextBackbone.from_pretrained(
        device=device,
        dtype=language_dtype,
        local_files_only=True,
        max_length=int(contract.get("language_max_length", language_max_length)),
    )
    if contract.get("semantic_adapter"):
        text_backbone = QwenSemanticBackbone(
            text_backbone,
            lora_rank=int(contract.get("semantic_lora_rank", 8)),
            lora_alpha=float(contract.get("semantic_lora_alpha", 32.0)),
            top_layers=int(contract.get("semantic_top_layers", 4)),
            lora_suffixes=tuple(
                s.strip()
                for s in str(
                    contract.get(
                        "semantic_lora_suffixes",
                        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    )
                ).split(",")
                if s.strip()
            ),
        )
    elif ckpt.get("lora"):
        # 旧路径（--lora-rank > 0）：按 contract.lora_rank 建 LoRA 再复制权重
        # （语义路径的 LoRA 由 QwenSemanticBackbone 构造器按 semantic_lora_rank
        # 建立，P0-3）。
        from va_compound.backbones import apply_lora

        rank = int(contract.get("lora_rank", 32))
        apply_lora(text_backbone.text_model, rank=rank)
    qwen_state = {
        k.removeprefix("text_backbone.").removeprefix("text_model."): v
        for k, v in (ckpt.get("qwen_state_dict") or {}).items()
    }
    if qwen_state:
        missing, unexpected = text_backbone.text_model.load_state_dict(
            qwen_state, strict=False
        )
        print(f"eval: qwen loaded missing={len(missing)} unexpected={len(unexpected)}")
    if ckpt.get("lora"):
        own = dict(text_backbone.text_model.named_parameters())
        for name, value in ckpt["lora"].items():
            clean = name.removeprefix("text_backbone.").removeprefix("text_model.")
            if clean in own:
                own[clean].data.copy_(value)
    gate = getattr(text_backbone, "gate", None)
    if gate is not None and ckpt.get("semantic_gate"):
        gate.load_state_dict(ckpt["semantic_gate"])
    text_backbone.text_model.eval()
    return text_backbone


def build_plan_language_cache(
    model,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    scene_summary: torch.Tensor,
    *,
    instruction: str | None = None,
    text_backbone=None,
    scene_teacher=None,
    compiler=None,
    scene_tokens: torch.Tensor | None = None,
    semantic_history: torch.Tensor | None = None,
    scene_delta: torch.Tensor | None = None,
):
    """Build the VA language cache with scene-conditioned plan tokens appended.

    ``hidden``/``mask`` are the single-task language slice [1, L, D]; the
    scene summary is the global mean of the current vision window tokens.
    With ``plan_resampler`` the policy's PlanResampler produces the plan
    tokens; with ``scene_teacher`` the frozen Qwen readout path is used;
    with a ``SemanticCompiler``（P0-3）the semantic readout tokens are
    compiled from the window ``scene_tokens``（semantic_history / scene_delta
    首决策为零向量，与训练 rollout t=0 一致）。
    """
    if model.config.plan_resampler:
        return model.build_plan_cache(scene_summary, hidden, mask)
    if compiler is not None:
        if (
            instruction is None
            or text_backbone is None
            or scene_tokens is None
            or semantic_history is None
            or scene_delta is None
        ):
            raise ValueError(
                "compile cache build requires the Qwen text backbone + "
                "scene tokens/history/delta"
            )
        semantic, _ = compiler(
            text_backbone,
            [instruction],
            scene_tokens,
            semantic_history,
            scene_delta,
        )
        semantic = semantic.to(dtype=hidden.dtype)
        extended = torch.cat((hidden, semantic), dim=1)
        extended_mask = torch.cat(
            (
                mask,
                torch.ones(
                    semantic.shape[:2], dtype=torch.bool, device=semantic.device
                ),
            ),
            dim=1,
        )
        return model.build_language_cache(extended, extended_mask)
    if model.config.scene_teacher:
        if instruction is None or text_backbone is None or scene_teacher is None:
            raise ValueError("scene-teacher cache build requires the Qwen text backbone")
        plan, _ = text_backbone.encode_with_scene(
            [instruction],
            scene_summary,
            scene_teacher.scene_projector,
            scene_teacher.readout_tokens,
        )
        plan = plan.to(dtype=hidden.dtype)
        extended = torch.cat((hidden, plan), dim=1)
        extended_mask = torch.cat(
            (mask, torch.ones(plan.shape[:2], dtype=torch.bool, device=plan.device)),
            dim=1,
        )
        return model.build_language_cache(extended, extended_mask)
    return model.build_language_cache(hidden, mask)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)  # 固定 flow 采样噪声（口径要求：重跑可复现，2026-08-05 审查补充）
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = VACompoundConfig(**ckpt["config"])
    # spatial-pooling ckpt 评估：vision_pooling 存在 training_contract 而非 config。
    vision_pooling = str(
        (ckpt.get("training_contract", {}) or {}).get("vision_pooling", "flat")
    )
    if args.vision_pooling is not None:
        vision_pooling = args.vision_pooling
    if config.local_slots and vision_pooling != "spatiotemporal":
        # Stage A/B：local_slots 训练（ST288/live）必为 spatiotemporal 288 token，
        # 旧 contract 可能漏记；强制对齐避免 flat 64-token 闭环失真。
        print(
            f"eval: config.local_slots=True 但 vision_pooling={vision_pooling}；"
            "强制 spatiotemporal（288 token，与训练一致）"
        )
        vision_pooling = "spatiotemporal"
    if args.direct_head != "auto":
        config = dataclasses.replace(config, direct_head=args.direct_head == "on")
    # P0-3：semantic-compiler ckpt 同样按需逐决策重建语言缓存（场景条件化）。
    has_compile = ckpt.get("semantic_compiler") is not None
    has_plan = config.plan_resampler or config.scene_teacher or has_compile
    if args.plan_refresh < 0:
        raise ValueError("--plan-refresh must be >= 0")
    if args.plan_refresh > 0 and not has_plan:
        raise ValueError(
            "--plan-refresh requires a checkpoint with plan_resampler, "
            "scene_teacher, or a semantic compiler"
        )
    model = VACompoundPolicy(config).eval().to(device)
    ckpt_direct_head = bool(ckpt["config"].get("direct_head", False))
    if ckpt_direct_head == config.direct_head:
        model.load_state_dict(ckpt["model"])
    else:
        # --direct-head on/off 强制换解码器（消融）：另一头的 head 未训练，
        # 用非严格加载 + 显式警告。
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(
            f"eval: forced decoder override direct_head={config.direct_head} "
            f"(ckpt had {ckpt_direct_head}); missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    assert config.proprio_dim == 4 and config.action_dim == 4, "expect 4D MetaWorld config"
    if config.c2_controller and not config.direct_head:
        raise ValueError("c2 checkpoint requires direct_head in the config")
    if config.c2_controller and args.plan_stride is not None and args.plan_stride < 1:
        raise ValueError("--plan-stride must be positive")
    if config.c2_controller and args.feedback_stride is not None and args.feedback_stride < 1:
        raise ValueError("--feedback-stride must be positive")
    if not config.c2_controller and (args.c2_oracle_ref or args.c2_zero_gain):
        raise ValueError("--c2-oracle-ref/--c2-zero-gain require a c2 checkpoint")
    if config.c2_controller and not config.direct_head:
        raise ValueError("c2 controller requires the direct head decoder")
    print(
        f"eval: action decoder = "
        f"{'c2_controller (ū/c̄/K contraction)' if config.c2_controller else ('direct_head (MLP->tanh)' if config.direct_head else 'flow_matching (Euler steps=32)')}"
    )

    features = torch.load(args.features, map_location="cpu", weights_only=True)
    sq01 = features["normalization"]["state_q01"].numpy()
    sq99 = features["normalization"]["state_q99"].numpy()
    scale_s = np.where(np.abs(sq99 - sq01) < 1e-6, 1.0, sq99 - sq01)
    # 动作反归一化参数（模型输出 norm -> 环境原始动作）
    aq01 = features["normalization"]["action_q01"].numpy()
    aq99 = features["normalization"]["action_q99"].numpy()

    if config.c2_controller:
        # C² 部署默认：plan_stride=6（VA 生成 {ū,c̄,K} 一次），feedback_stride=1
        # （每原始步刷新 V-JEPA → c_current → 应用 K 修正，Codex 修正 5）。
        args.plan_stride = (
            args.plan_stride if args.plan_stride is not None else DECISION_STRIDE
        )
        args.feedback_stride = (
            args.feedback_stride if args.feedback_stride is not None else 1
        )
        print(
            f"eval: c2 plan_stride={args.plan_stride} feedback_stride={args.feedback_stride} "
            f"oracle_ref={args.c2_oracle_ref} zero_gain={args.c2_zero_gain}"
        )
    vision_backbone = VJEPA21Backbone.from_pretrained(
        device=device,
        dtype="float16",
        max_tokens=144 if vision_pooling == "spatiotemporal" else 64,
        local_files_only=True,
    )
    if ckpt.get("vjepa_state_dict"):
        # e2e checkpoint：V-JEPA 被微调过，必须加载训练后权重（2026-08-06 P0 #4）
        vision_backbone.model.load_state_dict(ckpt["vjepa_state_dict"])
        print("eval: loaded vjepa_state_dict from checkpoint")
    vision_backbone.freeze_all()
    if args.c2_recovery_eval is not None:
        if not config.c2_controller:
            raise ValueError("--c2-recovery-eval requires a c2 checkpoint")
        run_c2_recovery_eval(
            args,
            device,
            model,
            vision_backbone,
            features,
            sq01,
            scale_s,
            aq01,
            aq99,
            vision_pooling=vision_pooling,
        )
        return

    # P0-3：统一恢复路径（普通 LoRA / semantic adapter 都按 training_contract
    # 构造并加载 qwen_state_dict / lora / semantic_gate）。
    text_backbone = restore_text_backbone(ckpt, device, language_dtype="float16")
    compiler = None
    if has_compile:
        from va_compound.backbones import SemanticCompiler

        compiler = SemanticCompiler(
            language_dim=config.language_dim,
            vision_dim=config.vision_dim,
            history_in_dim=config.hidden_dim,
            n_readout=int(
                ckpt.get("training_contract", {}).get("compile_n_readout", 16)
            ),
        ).to(device)
        compiler.load_state_dict(ckpt["semantic_compiler"])
        compiler.eval()
        print("eval: semantic_compiler loaded from checkpoint")
    all_tasks = features["metadata"]["tasks"]
    if args.task_ids is not None:
        task_indices = [int(token) for token in args.task_ids.split(",")]
        for index in task_indices:
            if index < 0 or index >= len(all_tasks):
                raise ValueError(f"--task-ids index {index} out of range 0..{len(all_tasks) - 1}")
        tasks = [all_tasks[index] for index in task_indices]
    else:
        tasks = all_tasks[: args.max_tasks]
    if not tasks:
        raise ValueError("no tasks selected for evaluation")
    if isinstance(text_backbone, QwenSemanticBackbone):
        # P0-3：semantic adapter ckpt——语言 hidden 用 fused 嵌入
        # （prior + g ⊙ (adapted − prior)），不是裸冻结先验。
        with torch.no_grad():
            prior, mask = text_backbone.encode_prior(tasks)
            adapted, _ = text_backbone.encode_adapted(tasks)
            hidden = text_backbone.fused_embedding(prior, adapted)
    else:
        hidden, mask = text_backbone.encode(tasks)

    scene_teacher = None
    if config.scene_teacher:
        if ckpt.get("scene_teacher") is None:
            raise ValueError("scene-teacher checkpoint has no scene_teacher weights")
        from va_compound.backbones import SceneTeacher

        scene_teacher = SceneTeacher(
            language_dim=config.language_dim, vision_dim=config.vision_dim
        ).to(device)
        scene_teacher.load_state_dict(ckpt["scene_teacher"])
        scene_teacher.eval()
        print("eval: scene_teacher loaded from checkpoint")
    if has_plan:
        # Plan-Cache：缓存按 episode 逐任务懒构建（首帧场景 → plan tokens），
        # --plan-refresh R 控制后续重建；Qwen 仅在 scene_teacher / compiler 下常驻。
        task_caches: list | None = [None] * len(tasks)
        if not config.scene_teacher and compiler is None:
            del text_backbone
    else:
        task_caches = [
            model.build_language_cache(hidden[i : i + 1].to(device), mask[i : i + 1].to(device))
            for i in range(len(tasks))
        ]
        del text_backbone

    import metaworld

    # 任务映射：数据 metadata.tasks 是任务描述（TASK_DESCRIPTIONS 的 value），
    # 反查 lerobot 采集同款 env_name（metaworld_config.json，与 Evoagent 封装一致）
    import json

    config_path = (
        Path("/home/ryan/Documents/robot/Evoagent/Evo-1/evo1_lerobot/lerobot/envs/metaworld_config.json")
    )
    mw_config = json.load(open(config_path))
    descriptions_to_env = {v: k for k, v in mw_config["TASK_DESCRIPTIONS"].items()}

    per_task = {}
    for task_index, task_text in enumerate(tasks):
        env_name = descriptions_to_env.get(task_text)
        if env_name is None:
            print(f"task {task_text[:40]}: SKIP (no env_name mapping)")
            continue
        # 采集同款环境：MT1(env_name, seed=42)，corner2 相机位置修正，
        # 物体随机化不冻结（每次 reset 随机 init，与训练数据分布一致）
        mt1 = metaworld.MT1(env_name, seed=42)
        env = mt1.train_classes[env_name](render_mode="rgb_array", camera_name="corner2")
        env.set_task(mt1.train_tasks[0])
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]  # corner2 位置（lerobot 采集同款）
        env._freeze_rand_vec = False
        wins = 0
        for trial in range(args.trials_per_task):
            obs, _ = env.reset(seed=1000 * task_index + trial)  # 固定种子（口径要求）
            if args.align_init:
                # 对齐数据首帧（物体+target），把闭环拉回训练分布
                from mw_expert_replay import align_objects, load_episode_rows, load_episodes

                ep = next(
                    (e for e in load_episodes() if task_text in str(e.get("tasks"))),
                    None,
                )
                if ep is not None:
                    o0 = np.asarray(
                        load_episode_rows(ep)[0]["observation.environment_state"],
                        dtype=float,
                    )
                    align_objects(env, o0, env._get_obs())
                    if args.debug_first_action:
                        global _ALIGN_ACTS
                        feat = torch.load(args.features, map_location="cpu", weights_only=True)
                        tid = tasks.index(task_text)
                        idx = int((feat["instruction_id"] == tid).nonzero()[0][0])
                        _ALIGN_ACTS = feat["actions"][idx].numpy()
            frame_buffer = []
            last_norm = np.zeros(4)  # 归一化动作（模型输入）
            chunk = np.zeros((ACTION_HORIZON, 4))
            chunk_start_step = 0  # 2026-08-06：--execute-steps 变节奏时 chunk 相位
            memory = None
            success = False
            decision_count = 0  # 2026-08-06：--memory-reset-every 的决策计数器
            plan_step = None  # C²：上次 VA 规划的原始步
            c2_token = 0  # C²：自规划以来消费的 token 索引
            c2_params = None  # C²：缓存的 {ū, c̄, K}
            for step in range(args.horizon):
                img = env.render()  # 数据图像与本地渲染一致（实测 MAE 0.48 vs flip 55，勿加 flip）
                frame_buffer.append(img)
                if step == 0:
                    # 2026-08-06 评估缺陷修复：原实现首决策前执行 chunk 的初始零值
                    # （归一化零 → 反归一化 (aq99+aq01)/2 = [2.45, 2.27, -1.37, 0]，
                    # 环境裁剪后 [1, 1, -1, 0]——实测把机械手提前移动 ~4.3cm，
                    # 精细抓取任务直接进入分布外状态）。
                    # 用首帧重复填充窗口使 step 0 立即推理：与训练首决策窗口同分布
                    # （prepare 的 clip_frame_indices 以 max(0, d-offset*stride)
                    # 钳制，episode 首窗口本身由重复帧组成）。
                    while len(frame_buffer) < (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                        frame_buffer.insert(0, img)
                if len(frame_buffer) > (VISION_WINDOW - 1) * DECISION_STRIDE + 1:
                    frame_buffer.pop(0)
                # 与训练一致的时间升序 [d-6, d-4, d-2, d]（clip_frame_indices 返回
                # video_start + max(0, d - offset*stride)，offset 升序 → 最老帧在前）
                # 修复（2026-08-05 多 agent 审查）：旧代码 range(-1,-2*W,-2) 是降序 [d,d-2,...]，
                # 与训练数据方向相反，V-JEPA 时序注意力对帧序敏感 → MW 闭环数字无效
                indices = list(range(-2 * VISION_WINDOW + 1, 0, 2))
                frames = [frame_buffer[len(frame_buffer) + i] for i in indices]
                if config.c2_controller:
                    # C² 部署（Codex 修正 5）：plan_stride 步重规划一次 {ū,c̄,K}；
                    # 每 feedback_stride 步刷新 c_current 并顺序消费 token。
                    clip = torch.cat([preprocess(f, 384) for f in frames], dim=0).to(device)
                    plan_due, feedback_due, token_index = c2_schedule(
                        step, plan_step, args.plan_stride, args.feedback_stride, ACTION_HORIZON
                    )
                    if plan_due:
                        if (
                            args.memory_reset_every > 0
                            and decision_count > 0
                            and decision_count % args.memory_reset_every == 0
                        ):
                            memory = None
                        decision_count += 1
                        state = np.clip(
                            2.0 * (obs[:4] - sq01) / scale_s - 1.0, -1.0, 1.0
                        ).astype(np.float32)
                        proprio = torch.tensor(state, device=device)[None, None]
                        previous = torch.tensor(
                            np.zeros(4, dtype=np.float32) if args.prev_zero else last_norm,
                            dtype=torch.float32, device=device,
                        )[None, None]
                        with torch.inference_mode():
                            tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                            vision_in = _apply_local_vision(
                                model, tokens, task_caches[task_index]
                            )
                            c_current = model.control_projector(tokens)
                            cond, memory = model.encode_condition(
                                vision_in,
                                proprio[0],
                                previous[0],
                                language_cache=task_caches[task_index],
                                visual_memory=memory,
                                return_visual_memory=True,
                            )
                            c2_params = model.controller_params(cond, c_current)
                            if args.c2_gain_scale != 1.0:
                                c2_params = ControllerParams(
                                    c2_params.nominal,
                                    c2_params.reference,
                                    c2_params.gain * args.c2_gain_scale,
                                )
                            if args.c2_zero_gain:
                                c2_params = ControllerParams(
                                    c2_params.nominal,
                                    c2_params.reference,
                                    torch.zeros_like(c2_params.gain),
                                )
                        plan_step = step
                        c2_token = 0
                    if feedback_due and c2_token < ACTION_HORIZON and c2_params is not None:
                        with torch.inference_mode():
                            if step != plan_step:
                                # feedback 刷新：重新编码当前窗口 → c_current。
                                tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                            c_current = model.control_projector(tokens)
                            if args.c2_oracle_ref:
                                # 参考零误差上界：c̄ ≡ c_current（e ≡ 0，K 空转）。
                                norm_action = c2_params.nominal[0, c2_token].cpu().numpy()
                            else:
                                error = c_current[0] - c2_params.reference[0, c2_token]
                                if (
                                    args.c2_error_threshold > 0.0
                                    and float(error.norm()) < args.c2_error_threshold
                                ):
                                    norm_action = c2_params.nominal[0, c2_token].cpu().numpy()
                                else:
                                    norm_action = (
                                        c2_params.nominal[0, c2_token]
                                        - c2_params.gain[0, c2_token] @ error
                                    ).cpu().numpy()
                        norm_action = np.clip(norm_action, -1.0, 1.0)
                        c2_token += 1
                    else:
                        # 非刷新步：保持上一动作（feedback_stride > 1 时）。
                        norm_action = last_norm
                elif step % args.execute_steps == 0 and len(frame_buffer) >= VISION_WINDOW:
                    if (
                        args.memory_reset_every > 0
                        and decision_count > 0
                        and decision_count % args.memory_reset_every == 0
                    ):
                        memory = None  # 契约缺口对照：截断递归记忆到训练深度
                    decision_count += 1
                    # 与训练一致的时间升序 [d-6, d-4, d-2, d]（clip_frame_indices 返回
                    # video_start + max(0, d - offset*stride)，offset 升序 → 最老帧在前）
                    # 修复（2026-08-05 多 agent 审查）：旧代码 range(-1,-2*W,-2) 是降序 [d,d-2,...]，
                    # 与训练数据方向相反，V-JEPA 时序注意力对帧序敏感 → MW 闭环数字无效
                    clip = torch.cat([preprocess(f, 384) for f in frames], dim=0).to(device)
                    with torch.inference_mode():
                        tokens = vision_backbone(clip.unsqueeze(0), pooling=vision_pooling)
                    if has_plan and plan_refresh_due(decision_count, args.plan_refresh):
                        # Plan-Cache：用当前窗口场景（vision 全局均值）重建该任务缓存
                        scene_summary = tokens.mean(dim=1)  # [1, vision_dim]
                        task_caches[task_index] = build_plan_language_cache(
                            model,
                            hidden[task_index : task_index + 1].to(device),
                            mask[task_index : task_index + 1].to(device),
                            scene_summary,
                            instruction=(
                                tasks[task_index]
                                if config.scene_teacher or compiler is not None
                                else None
                            ),
                            # plan_resampler 分支不访问 text_backbone；短路避免 NameError
                            text_backbone=(
                                text_backbone
                                if config.scene_teacher or compiler is not None
                                else None
                            ),
                            scene_teacher=scene_teacher,
                            compiler=compiler,
                            scene_tokens=tokens if compiler is not None else None,
                            semantic_history=(
                                torch.zeros(
                                    1,
                                    compiler.history_in_dim,
                                    device=device,
                                    dtype=torch.float32,
                                )
                                if compiler is not None
                                else None
                            ),
                            scene_delta=(
                                torch.zeros(
                                    1, config.vision_dim, device=device
                                )
                                if compiler is not None
                                else None
                            ),
                        )
                    state = np.clip(
                        2.0 * (obs[:4] - sq01) / scale_s - 1.0, -1.0, 1.0
                    ).astype(np.float32)
                    proprio = torch.tensor(state, device=device)[None, None]
                    previous = torch.tensor(
                        np.zeros(4, dtype=np.float32) if args.prev_zero else last_norm,
                        dtype=torch.float32, device=device,
                    )[None, None]
                    with torch.inference_mode():
                        vision_in = _apply_local_vision(
                            model, tokens, task_caches[task_index]
                        )
                        cond, memory = model.encode_condition(
                            vision_in,
                            proprio[0],
                            previous[0],
                            language_cache=task_caches[task_index],
                            visual_memory=memory,
                            return_visual_memory=True,
                        )
                        chunk = model.decode_actions(cond, steps=32)[0].cpu().numpy()
                        chunk_start_step = step
                        if args.debug_first_action and not _DEBUG_FA_DONE.get("x"):
                            _DEBUG_FA_DONE["x"] = True
                            print(f"DEBUG first chunk0={np.round(chunk[0], 4)}")
                            if args.align_init and _ALIGN_ACTS is not None:
                                dc = (step // DECISION_STRIDE)
                                if dc < len(_ALIGN_ACTS):
                                    ref = _ALIGN_ACTS[dc][0]
                                    print(
                                        "DEBUG data act0:",
                                        np.round(ref, 4),
                                        "mae:",
                                        round(float(np.abs(chunk[0] - ref).mean()), 4),
                                    )
                # 模型输出为归一化动作：与训练标签一致裁剪到 [-1,1]（robust_normalize
                # 存盘即 clip），再反归一化到环境原始动作空间；prev 反馈同样用裁剪值
                norm_action = np.clip(
                    chunk[(step - chunk_start_step) % ACTION_HORIZON], -1.0, 1.0
                ) if not config.c2_controller else norm_action
                action = norm_action * (aq99 - aq01) / 2 + (aq99 + aq01) / 2
                obs, reward, terminated, truncated, info = env.step(action)
                last_norm = norm_action
                if info.get("success"):
                    success = True
                    break
                if terminated or truncated:
                    break
            wins += int(success)
        per_task[task_text[:40]] = wins
        print(f"task {task_text[:40]}: {wins}/{args.trials_per_task}")
        env.close()

    total = sum(per_task.values())
    trials = len(per_task) * args.trials_per_task
    print(f"\nCLOSED-LOOP SUCCESS: {total}/{trials} = {total / trials:.1%}")
    if per_task:
        # 宏平均 + 任务级 bootstrap 95% CI（固定种子，规格 P1 口径）
        from stats_ci import macro_bootstrap_ci

        scores = np.asarray([w / args.trials_per_task for w in per_task.values()])
        group_ids = np.arange(len(scores))
        est, lo, hi = macro_bootstrap_ci(scores, group_ids, n_boot=2000, seed=0)
        print(
            f"macro (per-task avg): {est:.1%} [95% CI: {lo:.1%}, {hi:.1%}] "
            f"(n_tasks={len(scores)})"
        )


if __name__ == "__main__":
    main()
