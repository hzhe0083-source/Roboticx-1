"""Closed-loop evaluation for the official LIBERO benchmark suites.

This entry point matches the DINO-main static LongTraj contracts used by the
H8/P2 legacy runs and H50/P15 VA+WM runs. New H50 runs use four upright
agentview history frames plus the current upright wrist frame; old checkpoints
retain their four-agentview contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import torch

from va_compound import VACompoundConfig, VACompoundPolicy
from va_compound.backbones import QwenTextBackbone, TimmActionVisionBackbone
from va_compound.vision.dual_tower_batch import encode_dual_tower_batch
from va_compound.vision.encoding import _dino_main_online_encode


WINDOW_OFFSETS = (6, 4, 2, 0)
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
DUAL_VIEW_DATA_CONTRACT = "libero_4suite_h50p15_t4_dualview5_maskedtail_v2"
DUAL_VIEW_WORLD_LAST6_DATA_CONTRACT = (
    "libero_4suite_h50p15_t4_dualview5_worldh15_va1024_qwen08_last6_v6"
)
DENSE_WORLD_LAST6_DATA_CONTRACT = (
    "libero_4suite_h50p15_t4_dualview5_worldh15_va1024_qwen08_last6_denseall_v7"
)
T8_DENSE_WORLD_LAST6_DATA_CONTRACT = (
    "libero_4suite_h50p15_t8_dualview5_worldh15_va1024_qwen08_last6_denseall_v8"
)
LEGACY_FOUR_SUITE_DATA_CONTRACT = "libero_4suite_h50p15_t4_maskedtail_v1"
WORLD_H50_FUSION_LAYERS = list(range(18, 24))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_ids(value: str) -> list[int]:
    ids = [int(item) for item in value.split(",") if item.strip()]
    if not ids or any(task_id < 0 for task_id in ids):
        raise argparse.ArgumentTypeError("task ids must be unique nonnegative integers")
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("task ids must be unique")
    return ids


def _task_specs(payload: dict) -> list[dict]:
    metadata = payload.get("metadata", {})
    raw_specs = metadata.get("task_specs")
    if raw_specs is None:
        tasks = metadata.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 10:
            raise ValueError("legacy Spatial payload must contain exactly 10 tasks")
        raw_specs = [
            {
                "global_task_id": task_id,
                "suite": "libero_spatial",
                "local_task_id": task_id,
                "language": language,
            }
            for task_id, language in enumerate(tasks)
        ]
    specs = []
    for raw in raw_specs:
        if not isinstance(raw, dict):
            raise ValueError("metadata.task_specs entries must be dictionaries")
        global_id = raw.get("global_task_id", raw.get("task_id"))
        local_id = raw.get("local_task_id", raw.get("suite_task_id"))
        suite = raw.get("suite")
        language = raw.get("language", raw.get("description"))
        if (
            not isinstance(global_id, int)
            or isinstance(global_id, bool)
            or global_id < 0
            or not isinstance(local_id, int)
            or isinstance(local_id, bool)
            or not 0 <= local_id < 10
            or suite not in LIBERO_SUITES
            or not isinstance(language, str)
            or not language.strip()
        ):
            raise ValueError(f"invalid LIBERO task spec: {raw!r}")
        specs.append(
            {
                "global_task_id": global_id,
                "suite": suite,
                "local_task_id": local_id,
                "language": language.strip(),
            }
        )
    global_ids = [spec["global_task_id"] for spec in specs]
    suite_ids = [(spec["suite"], spec["local_task_id"]) for spec in specs]
    if sorted(global_ids) != list(range(len(specs))) or len(suite_ids) != len(set(suite_ids)):
        raise ValueError("task_specs must have contiguous global ids and unique suite/local ids")
    return sorted(specs, key=lambda spec: spec["global_task_id"])


def _task_horizon(requested: int, suite: str) -> int:
    return requested or SUITE_HORIZONS[suite]


def _qwen_weight_file(root: Path) -> Path:
    candidates = sorted(root.glob("model*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one Qwen safetensors shard in {root}, got {len(candidates)}"
        )
    return candidates[0]


def _qwen_lora_rank(contract: dict) -> int:
    rank = contract.get("qwen_lora_rank")
    if rank is None:
        match = re.fullmatch(r"full24_lora_rank([1-9][0-9]*)", str(contract.get("qwen_training", "")))
        rank = int(match.group(1)) if match else 0
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("checkpoint lacks a valid full24 Qwen LoRA rank")
    return rank


def _qwen_layerwise_readout(
    hidden_by_layer: dict[int, torch.Tensor],
    norm: torch.nn.Module | None,
    layers: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    layers = list(range(20, 24)) if layers is None else layers
    if set(hidden_by_layer) != set(layers):
        raise ValueError("Qwen layerwise readout received the wrong layers")
    base = norm(hidden_by_layer[23]) if norm is not None else hidden_by_layer[23]
    return base, torch.stack([hidden_by_layer[layer] for layer in layers], dim=1)


def _load_exact_parameters(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor] | None,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(state, dict) or set(state) != expected:
        seen = set(state) if isinstance(state, dict) else set()
        raise ValueError(
            f"{label} state mismatch: missing={sorted(expected - seen)[:5]} "
            f"unexpected={sorted(seen - expected)[:5]}"
        )
    parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name in expected:
            parameters[name].copy_(state[name])


def _upright(image: np.ndarray) -> np.ndarray:
    """LIBERO images are OpenGL-oriented; training flips the vertical axis once."""
    return np.ascontiguousarray(image[::-1])


def _history_window(history: list[np.ndarray]) -> list[np.ndarray]:
    current = len(history) - 1
    return [history[max(0, current - offset)] for offset in WINDOW_OFFSETS]


def _normalized_state(
    obs: dict,
    q01: np.ndarray,
    q99: np.ndarray,
) -> np.ndarray:
    state = np.concatenate((obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]))
    scale = np.where(np.abs(q99 - q01) < 1e-6, 1.0, q99 - q01)
    return np.clip(2.0 * (state - q01) / scale - 1.0, -1.0, 1.0).astype(
        np.float32
    )


@torch.inference_mode()
def _language_caches(
    payload: dict,
    model: VACompoundPolicy,
    device: torch.device,
    specs: list[dict],
    checkpoint: dict,
    qwen_path: Path | None = None,
) -> tuple[dict[int, object], dict[int, torch.Tensor]]:
    contract = checkpoint.get("training_contract", {})
    full_tail = contract.get("qwen_training") in {
        "last4_full_layers20_23_v1",
        "last6_full_layers18_23_v1",
    }
    state_key = "qwen_trainable_state_dict" if full_tail else "qwen_adapter_state_dict"
    qwen_state = checkpoint.get(state_key)
    qwen_trained = contract.get("qwen_joint_trained") is True
    if qwen_trained and not qwen_state:
        raise ValueError(f"checkpoint declares trained Qwen but lacks {state_key}")
    if qwen_state and not qwen_trained:
        raise ValueError(
            "checkpoint contains trained Qwen state without qwen_joint_trained=true"
        )

    if qwen_trained:
        if qwen_path is None or not qwen_path.is_dir():
            raise ValueError("trained Qwen evaluation requires --qwen model directory")
        fusion_layers = list(
            range(24 - model.config.dino_qwen_cross_modal_layers, 24)
        )
        if (
            contract.get("qwen_keep_layers") != 24
            or contract.get("qwen_fusion_layers") != fusion_layers
            or contract.get("qwen_base_readout") != "layer23_final_norm"
            or contract.get("qwen_fusion_reduce") != "none"
        ):
            raise ValueError("trained Qwen evaluation requires separate tail layers")
        expected_sha = contract.get("qwen_base_sha256")
        if not expected_sha or _sha256(_qwen_weight_file(qwen_path)) != expected_sha:
            raise ValueError("Qwen base safetensors SHA-256 does not match training")
        text = QwenTextBackbone.from_pretrained(
            model_id=str(qwen_path),
            device=device,
            dtype=contract.get("language_dtype") or "bfloat16",
            max_length=int(contract.get("language_max_length", 64)),
            local_files_only=True,
        )
        if len(text.text_model.layers) != 24:
            raise ValueError("Qwen3.5 base must contain all 24 decoder layers")
        if full_tail:
            if int(text.text_model.config.hidden_size) != 1024:
                raise ValueError("World-H50 evaluation requires Qwen3.5-0.8B")
            text.unfreeze_last(len(fusion_layers), freeze_final_norm=True)
            state_label = "Qwen fused tail layers"
        else:
            rank = _qwen_lora_rank(contract)
            if text.apply_lora(rank=rank, alpha=float(rank)) == 0:
                raise RuntimeError("Qwen LoRA did not wrap any projections")
            state_label = "Qwen adapter"
        expected = {
            name for name, parameter in text.named_parameters() if parameter.requires_grad
        }
        _load_exact_parameters(text, qwen_state, expected, state_label)
        text.eval()
        if model.config.architecture_version == "dual_tower_expert_v1":
            return {spec["global_task_id"]: None for spec in specs}, text
        with torch.inference_mode():
            hidden_by_layer, mask = text.encode_trainable(
                [spec["language"] for spec in specs],
                output_layers=fusion_layers,
            )
            hidden, layerwise = _qwen_layerwise_readout(
                hidden_by_layer,
                getattr(text.text_model, "norm", None),
                fusion_layers,
            )
        del text
        caches = {
            spec["global_task_id"]: model.build_language_cache(
                hidden[index : index + 1], mask[index : index + 1]
            )
            for index, spec in enumerate(specs)
        }
        layers = {
            spec["global_task_id"]: layerwise[index : index + 1]
            for index, spec in enumerate(specs)
        }
        return caches, layers

    instruction_ids = payload["instruction_id"]
    caches = {}
    for spec in specs:
        global_id = spec["global_task_id"]
        rows = torch.where(instruction_ids == global_id)[0]
        if rows.numel() == 0:
            raise ValueError(f"dataset has no language row for task {global_id}")
        row = int(rows[0])
        caches[global_id] = model.build_language_cache(
            payload["language_hidden"][row : row + 1].to(device),
            payload["language_mask"][row : row + 1].to(device),
        )
    return caches, {}


def _fixed_init_states(suite, task_id: int) -> np.ndarray:
    """Load the trusted official fixed-init file under torch>=2.6."""
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(path, map_location="cpu", weights_only=False)  # nosec B614


@torch.inference_mode()
def rollout_trial(
    *,
    model: VACompoundPolicy,
    vision: TimmActionVisionBackbone,
    language_cache,
    cross_modal_language_layers: torch.Tensor | None,
    env,
    init_state: np.ndarray,
    device: torch.device,
    state_q01: np.ndarray,
    state_q99: np.ndarray,
    horizon: int,
    flow_steps: int,
    settle_steps: int,
    memory_reset_every: int,
    policy_seed: int,
    previous_action_zero: bool,
    dual_view: bool,
    joint_text=None,
    instruction: str | None = None,
) -> tuple[bool, int]:
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    env.seed(policy_seed)
    env.reset()
    obs = env.set_init_state(init_state)
    zero_action = np.zeros(7, dtype=np.float32)
    zero_action[6] = -1.0
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(zero_action)

    agent_history = [_upright(obs["agentview_image"])]
    wrist_history = (
        [_upright(obs["robot0_eye_in_hand_image"])] if dual_view else None
    )
    last_action = zero_action
    memory = None
    decisions = 0
    steps = 0
    while steps < horizon:
        if (
            memory_reset_every > 0
            and decisions > 0
            and decisions % memory_reset_every == 0
        ):
            memory = None
        frame_window = _history_window(agent_history)
        if wrist_history is not None:
            frame_window.append(wrist_history[-1])
        frames = np.stack(frame_window, axis=0)[None, None]
        layerwise_bridge = bool(
            model.config.dino_qwen_cross_modal_bridge
            and model.runtime_dino_qwen_bridge_enabled
        )
        if model.config.architecture_version == "dual_tower_expert_v1":
            if joint_text is None or instruction is None:
                raise ValueError("joint evaluation requires live Qwen and instruction")
            tokens, language, mask = encode_dual_tower_batch(
                frames, [instruction], vision, joint_text, model.dual_tower_fusion,
                device, grid=model.config.main_vision_grid,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                language_cache = model.build_language_cache(language[:, 0], mask[:, 0])
            cross_modal_vision_layers = None
        else:
            encoded = _dino_main_online_encode(
                frames,
                vision,
                device,
                encode_batch=4,
                grid=model.config.main_vision_grid,
                window=model.config.main_vision_frames,
                return_last_layers=(
                    model.config.dino_qwen_cross_modal_layers
                    if layerwise_bridge
                    else 0
                ),
            )
            if layerwise_bridge:
                tokens, cross_modal_vision_layers = encoded
                cross_modal_vision_layers = cross_modal_vision_layers[:, 0]
            else:
                tokens = encoded
                cross_modal_vision_layers = None
        tokens = tokens[:, 0]
        expected_tokens = (
            1,
            model.config.main_vision_tokens,
            model.config.main_vision_dim,
        )
        if tokens.shape != expected_tokens or not torch.isfinite(tokens).all():
            raise RuntimeError(f"invalid DINO tokens: {tuple(tokens.shape)}")
        if layerwise_bridge and (
            cross_modal_vision_layers.shape[:2]
            != (1, model.config.dino_qwen_cross_modal_layers)
            or not torch.isfinite(cross_modal_vision_layers).all()
            or cross_modal_language_layers is None
            or cross_modal_language_layers.shape[:2]
            != (1, model.config.dino_qwen_cross_modal_layers)
            or not torch.isfinite(cross_modal_language_layers).all()
        ):
            raise RuntimeError("invalid layer-specific DINO/Qwen streams")
        proprio = torch.from_numpy(
            _normalized_state(obs, state_q01, state_q99)
        ).to(device)[None]
        previous = torch.from_numpy(
            np.zeros_like(last_action) if previous_action_zero else last_action
        ).to(device)[None]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            condition, memory = model.encode_condition(
                tokens,
                proprio,
                previous,
                language_cache=language_cache,
                cross_modal_vision_layers=cross_modal_vision_layers,
                cross_modal_language_layers=cross_modal_language_layers,
                visual_memory=memory,
                return_visual_memory=True,
            )
            expected_condition = (
                1,
                model.config.action_horizon,
                model.config.hidden_dim,
            )
            if model.config.architecture_version == "dual_tower_expert_v1":
                expected_condition = (1, 3, model.config.action_horizon, model.config.hidden_dim)
            if condition.shape != expected_condition or not torch.isfinite(condition).all():
                raise RuntimeError(f"invalid VA condition: {tuple(condition.shape)}")
            chunk = model.decode_actions(condition, steps=flow_steps)[0].float()
        expected_chunk = (model.config.action_horizon, model.config.action_dim)
        if chunk.shape != expected_chunk or not torch.isfinite(chunk).all():
            raise RuntimeError(f"invalid Flow chunk: {tuple(chunk.shape)}")
        chunk = chunk.cpu().numpy()
        decisions += 1

        for token in chunk[: model.config.deployment_execution_horizon]:
            action = np.clip(token, -1.0, 1.0).astype(np.float32)
            action[6] = 1.0 if action[6] >= 0.0 else -1.0
            obs, _, done, _ = env.step(action)
            last_action = action
            agent_history.append(_upright(obs["agentview_image"]))
            if wrist_history is not None:
                wrist_history.append(_upright(obs["robot0_eye_in_hand_image"]))
            steps += 1
            if env.check_success():
                return True, steps
            if done or steps >= horizon:
                return False, steps
    return False, steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--main-vision-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--task-ids", type=_task_ids)
    parser.add_argument("--trials-per-task", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--memory-reset-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials_per_task < 1 or args.trials_per_task > 50:
        raise ValueError("trials-per-task must be in 1..50")
    if args.horizon < 0 or args.flow_steps < 1 or args.settle_steps < 0:
        raise ValueError("horizon must be nonnegative; flow-steps positive; settle-steps nonnegative")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    payload_contract = payload.get("metadata", {}).get("contract")
    four_suite = payload_contract in {
        LEGACY_FOUR_SUITE_DATA_CONTRACT,
        DUAL_VIEW_DATA_CONTRACT,
        DUAL_VIEW_WORLD_LAST6_DATA_CONTRACT,
        DENSE_WORLD_LAST6_DATA_CONTRACT,
        T8_DENSE_WORLD_LAST6_DATA_CONTRACT,
    }
    dual_view = payload_contract in {
        DUAL_VIEW_DATA_CONTRACT,
        DUAL_VIEW_WORLD_LAST6_DATA_CONTRACT,
        DENSE_WORLD_LAST6_DATA_CONTRACT,
        T8_DENSE_WORLD_LAST6_DATA_CONTRACT,
    }
    world_horizon = (
        15
        if payload_contract
        in {
            DUAL_VIEW_WORLD_LAST6_DATA_CONTRACT,
            DENSE_WORLD_LAST6_DATA_CONTRACT,
            T8_DENSE_WORLD_LAST6_DATA_CONTRACT,
        }
        else None
    )
    dense_t4 = payload_contract == DENSE_WORLD_LAST6_DATA_CONTRACT
    dense_t8 = payload_contract == T8_DENSE_WORLD_LAST6_DATA_CONTRACT
    dense_continuation = dense_t4 or dense_t8
    task_specs = _task_specs(payload)
    specs_by_id = {spec["global_task_id"]: spec for spec in task_specs}
    task_ids = args.task_ids if args.task_ids is not None else list(specs_by_id)
    missing_task_ids = sorted(set(task_ids) - set(specs_by_id))
    if missing_task_ids:
        raise ValueError(f"task ids are absent from payload: {missing_task_ids}")
    config = VACompoundConfig(**checkpoint["config"])
    joint_frontend = config.architecture_version == "dual_tower_expert_v1"
    if joint_frontend and args.qwen is None:
        raise ValueError("dual_tower_expert_v1 evaluation requires --qwen")
    expected = {
        "num_layers": 8,
        "action_dim": 7,
        "proprio_dim": 9,
        "main_vision_backbone": "dinov2_vitl14_reg4",
        "main_vision_model_id": "vit_large_patch14_reg4_dinov2.lvd142m",
        "main_vision_image_size": 224,
        "main_vision_dim": 1024,
        "main_vision_temporal": True,
        "main_vision_temporal_scale": 1.0,
        "vision_dim": 1024,
        "flow_layers": 6,
        "va_attention_backend": "auto",
        "direct_head": False,
        "c2_controller": False,
        "slot_free_policy": True,
        "action_vision_backbone": "none",
        "dino_dense_metric": False,
        "metric_geometry_inject": False,
        "local_slots": False,
    }
    if config.action_horizon == 8:
        expected.update(
            action_horizon=8,
            planning_stride=2,
            deployment_execution_horizon=2,
            main_vision_frames=4,
            main_vision_grid=8,
            main_vision_tokens=256,
        )
        expected_wmrm = False
        data_contract = "libero_spatial_official_h8p2_t4_v1"
        protocol = "evo1_spatial_h350_fixed_init_h8p2_v1"
    elif config.action_horizon in {15, 50}:
        execution_horizon = int(config.deployment_execution_horizon)
        if (
            config.action_horizon == 50
            and execution_horizon != 15
            or config.action_horizon == 15
            and execution_horizon not in {4, 15}
        ):
            raise ValueError(
                f"unsupported LIBERO H{config.action_horizon} execution horizon: "
                f"{execution_horizon}"
            )
        expected.update(
            action_horizon=config.action_horizon,
            planning_stride=execution_horizon,
            deployment_execution_horizon=execution_horizon,
            main_vision_frames=5 if dual_view else 4,
            main_vision_grid=16,
            main_vision_tokens=1280 if dual_view else 1024,
            flow_cond="adaln",
            wmrm_cycle_steps=world_horizon or execution_horizon,
            va_world_mode="peer_sync_h6",
        )
        if world_horizon:
            expected.update(
                language_dim=1024,
                hidden_dim=1024,
                num_heads=16,
                flow_hidden_dim=512,
                dino_qwen_cross_modal_layers=6,
                wmrm_world_dim=1024,
                wmrm_predictor_width=1024,
                wmrm_predictor_heads=32,
            )
        if execution_horizon == 4 or config.action_horizon == 50:
            expected.update(
                va_last3_cross_attn=True,
                dino_qwen_cross_modal_bridge=True,
            )
        expected_wmrm = True
        data_contract = (
            payload_contract
            if four_suite and config.action_horizon == 50
            else f"libero_spatial_h15p{execution_horizon}_t4_v1"
        )
        protocol = (
            "libero_10_hard2_fixed_init_t8_denseall_worldh15_h50p15_last6_v7"
            if dense_t8
            else "libero_10_hard2_fixed_init_denseall_worldh15_h50p15_last6_v6"
            if dense_t4
            else "libero_4suite_fixed_init_dualview5_worldh15_h50p15_last6_v5"
            if world_horizon and len(task_specs) == 40
            else "libero_10_hard2_fixed_init_dualview5_worldh15_h50p15_last6_v5"
            if world_horizon
            else "libero_4suite_fixed_init_dualview5_h50p15_v2"
            if dual_view and len(task_specs) == 40
            else "libero_10_hard2_fixed_init_dualview5_h50p15_v2"
            if dual_view
            else "libero_4suite_fixed_init_h50p15_v1"
            if four_suite and len(task_specs) == 40
            else "libero_10_hard2_fixed_init_h50p15_v1"
            if four_suite
            else f"evo1_spatial_h350_fixed_init_h{config.action_horizon}p{execution_horizon}_v1"
        )
    else:
        raise ValueError(f"unsupported LIBERO action horizon: {config.action_horizon}")
    if joint_frontend:
        expected.update(va_last3_cross_attn=False, dino_qwen_cross_modal_bridge=False, flow_layers=3)
    mismatches = {
        key: (getattr(config, key), value)
        for key, value in expected.items()
        if getattr(config, key) != value
    }
    if mismatches or config.wmrm is not expected_wmrm or config.main_vision_backbone == "vjepa":
        raise ValueError(
            f"checkpoint is not a supported DINO LIBERO contract: {mismatches}"
        )
    if payload_contract != data_contract:
        raise ValueError("unexpected LIBERO data contract")
    if four_suite:
        metadata = payload["metadata"]
        if world_horizon and (
            metadata.get("world_target_horizon") != world_horizon
            or metadata.get("world_target_offsets")
            != [world_horizon + offset for offset in (0, 15, 30, 45)]
            or metadata.get("world_target_alignment")
            != f"obs[d+{world_horizon}]"
        ):
            raise ValueError("dual-view World supervision has the wrong horizon")
        if dense_continuation and (
            metadata.get("window_sampling") != "all_legal_starts_v1"
            or len(payload["actions"]) != (9_843 if dense_t8 else 15_843)
            or metadata.get("sequence_length") != (8 if dense_t8 else 4)
        ):
            raise ValueError("dense continuation payload has the wrong window set")
        if int(metadata.get("n_tasks", 0)) != len(task_specs):
            raise ValueError("H50/P15 payload task count is inconsistent")
        if len(task_specs) == 40:
            suite_counts = {
                suite: sum(spec["suite"] == suite for spec in task_specs)
                for suite in LIBERO_SUITES
            }
            if any(count != 10 for count in suite_counts.values()):
                raise ValueError(
                    f"four-suite payload must contain 10 tasks per suite: {suite_counts}"
                )
            if any(
                spec["suite"] != LIBERO_SUITES[spec["global_task_id"] // 10]
                or spec["local_task_id"] != spec["global_task_id"] % 10
                for spec in task_specs
            ):
                raise ValueError(
                    "four-suite global ids must follow suite order and local ids 0..9"
                )
        elif [
            (spec["suite"], spec["local_task_id"]) for spec in task_specs
        ] != [("libero_10", 3), ("libero_10", 4)]:
            raise ValueError("the supported two-task probe is LIBERO-Long task3+4")
        normalization = payload.get("normalization", {})
        action_low = normalization.get("action_q01")
        action_high = normalization.get("action_q99")
        state_low = normalization.get("state_q01")
        state_high = normalization.get("state_q99")
        if (
            payload["metadata"].get("action_contract")
            != "raw_libero_osc_pose_minus1_plus1"
            or not isinstance(action_low, torch.Tensor)
            or not isinstance(action_high, torch.Tensor)
            or tuple(action_low.shape) != (7,)
            or tuple(action_high.shape) != (7,)
            or not torch.equal(action_low, torch.full_like(action_low, -1.0))
            or not torch.equal(action_high, torch.full_like(action_high, 1.0))
            or not isinstance(state_low, torch.Tensor)
            or not isinstance(state_high, torch.Tensor)
            or tuple(state_low.shape) != (9,)
            or tuple(state_high.shape) != (9,)
            or not bool(torch.isfinite(state_low).all())
            or not bool(torch.isfinite(state_high).all())
            or not bool((state_high > state_low).all())
        ):
            raise ValueError("four-suite normalization contract is invalid")
    elif any(spec["suite"] != "libero_spatial" for spec in task_specs):
        raise ValueError("legacy Spatial payload may only contain Spatial task specs")
    if config.dino_qwen_cross_modal_bridge:
        metadata = payload["metadata"]
        if four_suite:
            fusion_layers = (
                WORLD_H50_FUSION_LAYERS
                if world_horizon
                else list(range(20, 24))
            )
            if (
                metadata.get("qwen_keep_layers") != 24
                or metadata.get("qwen_fusion_layers") != fusion_layers
                or metadata.get("qwen_base_readout") != "layer23_final_norm"
                or metadata.get("qwen_fusion_reduce") != "none"
                or world_horizon
                and (
                    metadata.get("language_dim") != 1024
                    or metadata.get("language_source")
                    != "online_qwen35_0_8b_last6_full_v1"
                )
            ):
                raise ValueError("four-suite bridge has the wrong Qwen tail layers")
            obsolete = {
                key
                for key in ("qwen_readout", "qwen_layer_reduce", "dino_layer_reduce")
                if key in metadata
            }
            if obsolete:
                raise ValueError(f"four-suite payload contains obsolete reduce fields: {obsolete}")
        elif metadata.get("qwen_fusion_layers") != list(
            range(10, 15)
        ) or metadata.get("qwen_layer_reduce") != "mean_then_final_norm":
            raise ValueError("legacy bridge requires a Qwen 10-14 mean cache")

    contract = checkpoint.get("training_contract", {})
    if contract.get("action_decoder") != "conditional_flow_matching":
        raise ValueError("checkpoint is not a conditional-flow policy")
    previous_action_zero = contract.get("previous_action_input") == "zero_v1"
    if config.action_horizon == 15 and config.deployment_execution_horizon == 4:
        required_contract = {
            "flow_slot_identity": "per_slot_action_condition_v1",
            "flow_prefix_steps": 4,
            "flow_prefix_weight": 3.0,
            "flow_tail_weight": 1.0,
            "previous_action_input": "zero_v1",
            "qwen_fusion_layers": list(range(10, 15)),
            "qwen_layer_reduce": "mean_then_final_norm",
            "dino_fusion_layers": list(range(20, 24)),
            "dino_layer_reduce": "mean",
            "wmrm_feature_metric": "cosine",
            "lr_base": 1e-5,
            "lr_new": 3e-5,
        }
        bad_contract = {
            key: (contract.get(key), value)
            for key, value in required_contract.items()
            if contract.get(key) != value
        }
        if bad_contract:
            raise ValueError(f"incomplete LIBERO all-fixes contract: {bad_contract}")
    if four_suite:
        n_tasks = len(task_specs)
        subset_probe = n_tasks == 2
        required_four_suite = {
            "initialization": (
                "t8_dense_continue_from_t4_s1000_v1"
                if dense_t8
                else "dense_all_windows_continue_from_s5000_v1"
                if dense_t4
                else "scratch_policy_qwen08_dino_dualview_worldh15_va1024_last6_v6"
                if world_horizon
                else "scratch_policy_pretrained_qwen_dino_dualview_v2"
                if dual_view
                else "scratch_policy_pretrained_qwen_dino_v1"
            ),
            "data_contract": payload_contract,
            "suites": ["libero_10"] if subset_probe else list(LIBERO_SUITES),
            "n_tasks": n_tasks,
            "task_specs": payload["metadata"]["task_specs"],
            "action_horizon": 50,
            "planning_stride": 15,
            "deployment_execution_horizon": 15,
            "wmrm_cycle_steps": world_horizon or 15,
            "flow_slot_identity": "per_slot_action_condition_v1",
            "flow_prefix_steps": 15,
            "flow_prefix_weight": 3.0,
            "flow_tail_weight": 1.0,
            "qwen_joint_trained": True,
            "qwen_keep_layers": 24,
            "qwen_fusion_layers": (
                WORLD_H50_FUSION_LAYERS
                if world_horizon
                else list(range(20, 24))
            ),
            "cross_modal_va_layers": list(range(6 if world_horizon else 4)),
            "qwen_base_readout": "layer23_final_norm",
            "qwen_fusion_reduce": "none",
            "main_vision_trainable_layers": (
                WORLD_H50_FUSION_LAYERS
                if world_horizon
                else list(range(20, 24))
            ),
            "main_vision_frames": 5 if dual_view else 4,
            "dino_fusion_layers": (
                WORLD_H50_FUSION_LAYERS
                if world_horizon
                else list(range(20, 24))
            ),
            "dino_base_readout": "block23_norm",
            "dino_fusion_reduce": "none",
            "wmrm_feature_metric": "cosine",
            "wmrm_evidence": "post_va_vl_fused_tokens_v1",
            "previous_action_input": "zero_v1",
            "stage1_steps": 0 if dense_continuation else 800 if subset_probe else 8000,
            "total_steps": (
                4924
                if dense_t8
                else 4955
                if dense_t4
                else 5000
                if subset_probe
                else 50000
            ),
        }
        if dense_continuation:
            required_four_suite.update(source_global_step=1000 if dense_t8 else 5000)
        if dense_t8:
            required_four_suite.update(sequence_length=8, memory_reset_every=8)
        if world_horizon:
            required_four_suite.update(
                qwen_training="last6_full_layers18_23_v1",
                qwen_trainable_layers=WORLD_H50_FUSION_LAYERS,
                qwen_final_norm_frozen=True,
                qwen_hidden_dim=1024,
                wmrm_world_dim=1024,
                wmrm_predictor_width=1024,
                flow_hidden_dim=512,
            )
        if dual_view:
            required_four_suite.update(
                vision_input="agentview_history4_plus_current_wrist_v2",
                world_target_view="eye_in_hand_rgb",
                fusion_initialization="zero_output_unit_gate_v1",
                wmrm_target_teacher=(
                    "shared_online_dino_block23_stopgrad_v1"
                    if world_horizon
                    else "frozen_dino_base_block23_norm_v1"
                ),
            )
        if joint_frontend:
            stage1_steps = contract.get("stage1_steps")
            total_steps = contract.get("total_steps")
            if not isinstance(stage1_steps, int) or stage1_steps < 0:
                raise ValueError("joint checkpoint requires a nonnegative stage1_steps")
            if not isinstance(total_steps, int) or total_steps < 1:
                raise ValueError("joint checkpoint requires a positive total_steps")
            required_four_suite.update(
                initialization="fresh_dual_tower_expert_v1",
                fusion_initialization="dual_tower_zero_output_v1",
                architecture_version="dual_tower_expert_v1",
                source_global_step=-1,
                stage1_steps=stage1_steps,
                total_steps=total_steps,
                optimizer_initialization="fresh_adamw_v1",
                qwen_world_cache="per_observation_joint_live_v1",
                stage1_world_current_vision_cache="disabled_joint_frontend_v1",
            )
        bad_four_suite = {
            key: (contract.get(key), value)
            for key, value in required_four_suite.items()
            if contract.get(key) != value
        }
        dino_should_be_trained = int(checkpoint.get("global_step", -1)) > int(
            required_four_suite["stage1_steps"]
        )
        if contract.get("main_vision_joint_trained") is not dino_should_be_trained:
            bad_four_suite["main_vision_joint_trained"] = (
                contract.get("main_vision_joint_trained"),
                dino_should_be_trained,
            )
        if bad_four_suite:
            raise ValueError(f"incomplete four-suite checkpoint contract: {bad_four_suite}")
        forbidden = {
            key for key in ("qwen_readout", "qwen_layer_reduce", "dino_layer_reduce")
            if key in contract
        }
        if forbidden:
            raise ValueError(f"four-suite checkpoint contains obsolete reduce fields: {forbidden}")
        if not world_horizon:
            _qwen_lora_rank(contract)
    if contract.get("flow_steps", 8) != 8 or args.flow_steps != 8:
        raise ValueError("formal LIBERO evaluation requires 8 Flow steps")
    expected_memory_reset = int(contract.get("memory_reset_every", 4))
    if args.memory_reset_every != expected_memory_reset:
        raise ValueError(
            "formal LIBERO evaluation requires memory reset every "
            f"{expected_memory_reset} decisions"
        )
    if args.settle_steps != 10:
        raise ValueError("formal LIBERO evaluation requires 10 settling no-op steps")
    if not four_suite and args.horizon not in {0, 350}:
        raise ValueError("legacy Spatial evaluation requires horizon 350 or suite default 0")
    dino_trained = contract.get("main_vision_joint_trained") is True
    dino_state_key = (
        "main_vision_trainable_state_dict" if four_suite else "main_vision_state_dict"
    )
    dino_state = checkpoint.get(dino_state_key)
    if dino_trained and not isinstance(dino_state, dict):
        raise ValueError(f"checkpoint declares trained DINO but lacks {dino_state_key}")
    if not dino_trained and dino_state is not None:
        raise ValueError("checkpoint contains DINO weights without main_vision_joint_trained=true")
    expected_sha = contract.get(
        "main_vision_base_sha256" if four_suite else "main_vision_checkpoint_sha256"
    )
    source_path = Path(checkpoint.get("source_checkpoint", ""))
    seen_sources: set[Path] = set()
    while not four_suite and not expected_sha and source_path.is_file():
        source_path = source_path.resolve()
        if source_path in seen_sources:
            raise ValueError("checkpoint source lineage contains a cycle")
        seen_sources.add(source_path)
        source = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)
        expected_sha = source.get("training_contract", {}).get(
            "main_vision_checkpoint_sha256"
        )
        source_path = Path(source.get("source_checkpoint", ""))
    if not expected_sha or _sha256(args.main_vision_checkpoint) != expected_sha:
        raise ValueError("DINO checkpoint SHA-256 does not match training")

    model = VACompoundPolicy(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.eval().to(device)
    model.runtime_execution_horizon = config.deployment_execution_horizon
    model.runtime_dino_qwen_bridge_enabled = dino_trained if four_suite else True
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
        cache_enabled=True,
    ):
        task_caches, task_language_layers = _language_caches(
            payload, model, device, task_specs, checkpoint, args.qwen
        )
    joint_text = task_language_layers if joint_frontend else None
    if joint_frontend:
        task_language_layers = {}
    if not joint_frontend and device.type == "cuda" and any(
        layer.key.dtype != torch.bfloat16 or layer.value.dtype != torch.bfloat16
        for cache in task_caches.values()
        for layer in cache.layers
    ):
        raise RuntimeError("language cache does not match BF16 training precision")
    vision = TimmActionVisionBackbone.from_pretrained(
        device=device,
        dtype="float32" if four_suite else "bfloat16" if dino_trained else "float16",
        model_id=config.main_vision_model_id,
        image_size=config.main_vision_image_size,
        feature_dim=config.main_vision_dim,
        output_layers=(11, 23),
        checkpoint_path=args.main_vision_checkpoint,
        local_files_only=True,
    )
    if dino_trained and four_suite:
        vision.unfreeze_last(config.dino_qwen_cross_modal_layers)
        expected_dino = {
            name
            for name, parameter in vision.named_parameters()
            if parameter.requires_grad
        }
        _load_exact_parameters(vision, dino_state, expected_dino, "DINO trainable")
        vision.train(False)
        set_checkpointing = getattr(vision.model, "set_grad_checkpointing", None)
        if callable(set_checkpointing):
            set_checkpointing(False)
    elif dino_trained:
        vision.model.load_state_dict(dino_state, strict=True)
    if not (dino_trained and four_suite):
        vision.freeze_all()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_factories = benchmark.get_benchmark_dict()
    suites = {
        suite: benchmark_factories[suite]()
        for suite in {specs_by_id[task_id]["suite"] for task_id in task_ids}
    }
    state_q01 = payload["normalization"]["state_q01"].numpy()
    state_q99 = payload["normalization"]["state_q99"].numpy()
    records = []
    for global_task_id in task_ids:
        spec = specs_by_id[global_task_id]
        suite_name = spec["suite"]
        local_task_id = spec["local_task_id"]
        suite = suites[suite_name]
        task = suite.get_task(local_task_id)
        if task.language.strip().lower() != spec["language"].strip().lower():
            raise ValueError(
                f"task {global_task_id} language mismatch: "
                f"{task.language!r} != {spec['language']!r}"
            )
        env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(local_task_id),
            camera_heights=128,
            camera_widths=128,
            camera_names=("agentview", "robot0_eye_in_hand") if dual_view else "agentview",
        )
        init_states = _fixed_init_states(suite, local_task_id)
        if len(init_states) < args.trials_per_task:
            raise ValueError(
                f"task {global_task_id} only has {len(init_states)} fixed init states"
            )
        init_states = init_states[: args.trials_per_task]
        horizon = _task_horizon(args.horizon, suite_name)
        task_wins = 0
        try:
            for trial, init_state in enumerate(init_states):
                trial_seed = args.seed + global_task_id * 100 + trial
                success, steps = rollout_trial(
                    model=model,
                    vision=vision,
                    language_cache=task_caches[global_task_id],
                    cross_modal_language_layers=task_language_layers.get(
                        global_task_id
                    ),
                    env=env,
                    init_state=init_state,
                    device=device,
                    state_q01=state_q01,
                    state_q99=state_q99,
                    horizon=horizon,
                    flow_steps=args.flow_steps,
                    settle_steps=args.settle_steps,
                    memory_reset_every=args.memory_reset_every,
                    policy_seed=trial_seed,
                    previous_action_zero=previous_action_zero,
                    dual_view=dual_view,
                    joint_text=joint_text,
                    instruction=spec["language"],
                )
                task_wins += int(success)
                records.append(
                    {
                        "task_id": global_task_id,
                        "global_task_id": global_task_id,
                        "suite": suite_name,
                        "local_task_id": local_task_id,
                        "task": task.language,
                        "trial": trial,
                        "seed": trial_seed,
                        "success": success,
                        "steps": steps,
                        "horizon": horizon,
                    }
                )
                print(
                    f"global={global_task_id} suite={suite_name} "
                    f"local={local_task_id} trial={trial} seed={trial_seed} "
                    f"success={int(success)} steps={steps}",
                    flush=True,
                )
        finally:
            env.close()
        print(
            f"TASK global={global_task_id} suite={suite_name} local={local_task_id} "
            f"success={task_wins}/{len(init_states)}",
            flush=True,
        )

    won = sum(int(record["success"]) for record in records)
    suite_results = {}
    for suite_name in sorted({record["suite"] for record in records}):
        suite_records = [record for record in records if record["suite"] == suite_name]
        suite_wins = sum(int(record["success"]) for record in suite_records)
        suite_results[suite_name] = {
            "successes": suite_wins,
            "trials": len(suite_records),
            "success_rate": suite_wins / len(suite_records),
            "horizon": _task_horizon(args.horizon, suite_name),
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "global_step": int(checkpoint.get("global_step", -1)),
        "suite": (
            "libero_4suite"
            if four_suite and len(task_specs) == 40
            else "libero_10_hard2"
            if four_suite
            else "libero_spatial"
        ),
        "protocol": protocol,
        "task_ids": task_ids,
        "trials_per_task": args.trials_per_task,
        "horizon": args.horizon or {
            suite: SUITE_HORIZONS[suite] for suite in suite_results
        },
        "flow_steps": args.flow_steps,
        "execution_horizon": config.deployment_execution_horizon,
        "memory_reset_every": args.memory_reset_every,
        "vision_input": (
            "agentview_history4_plus_current_wrist_v2"
            if dual_view
            else "agentview_history4_v1"
        ),
        "previous_action_input": (
            "zero_v1" if previous_action_zero else "self_previous_action_v1"
        ),
        "successes": won,
        "trials": len(records),
        "success_rate": won / len(records),
        "suite_results": suite_results,
        "records": records,
    }
    print(
        f"CLOSED-LOOP SUCCESS: {won}/{len(records)} = {result['success_rate']:.1%}",
        flush=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
