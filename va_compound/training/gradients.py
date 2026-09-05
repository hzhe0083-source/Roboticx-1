from __future__ import annotations

import argparse
from collections.abc import Callable
import math
import random
from types import GeneratorType

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from va_compound.data_parallel import reduce_update_gradients

def backward_peer_joint_losses(va_loss: Tensor, world_loss_or_forward):
    """Backprop VA, then build/backprop World, accumulating into one update."""
    if va_loss.ndim != 0:
        raise ValueError("peer joint VA loss must be a scalar tensor")
    va_loss.backward()
    result = (
        world_loss_or_forward()
        if callable(world_loss_or_forward)
        else world_loss_or_forward
    )
    world_loss = result[0] if isinstance(result, tuple) else result
    if not isinstance(world_loss, Tensor) or world_loss.ndim != 0:
        raise ValueError("peer joint World loss must be a scalar tensor")
    world_loss.backward()
    return result


def backward_pcgrad(
    task_losses: list[Tensor | Callable[[], Tensor | list[Tensor]]],
    named_parameters: list[tuple[str, Tensor]],
    *,
    seed: int = 0,
    topology=None,
    auxiliary_loss_or_forward=None,
    compact_prefixes: tuple[str, ...] = (),
    allow_inactive_ranks: bool = False,
    allow_single_task: bool = False,
):
    """PCGrad task losses; optionally project only an auxiliary gradient."""
    if not task_losses or (not allow_single_task and len(task_losses) < 2 and not callable(task_losses[0])):
        raise ValueError("PCGrad requires at least two task losses")
    trainable_named = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in trainable_named]
    compact = [
        bool(compact_prefixes) and name.startswith(compact_prefixes)
        for name, _ in trainable_named
    ]
    if not trainable:
        raise ValueError("PCGrad received no trainable parameters")
    task_gradients: list[list[Tensor | None]] = []
    for index, loss_or_forward in enumerate(task_losses):
        deferred = callable(loss_or_forward)
        result = loss_or_forward() if deferred else loss_or_forward
        streamed = isinstance(result, GeneratorType)
        losses = result if streamed else (result if isinstance(result, list) else [result])
        if not streamed and not losses:
            raise ValueError("PCGrad task forward returned no losses")
        accumulated = [None] * len(trainable)
        seen = False
        for group_index, loss in enumerate(losses):
            seen = True
            if not isinstance(loss, Tensor) or loss.ndim != 0:
                raise ValueError("PCGrad task losses must be scalar tensors")
            gradients = torch.autograd.grad(
                loss, trainable,
                retain_graph=False if streamed else (
                    group_index + 1 < len(losses) if deferred else index + 1 < len(task_losses)
                ), allow_unused=True,
            )
            if streamed:
                for j, gradient in enumerate(gradients):
                    if gradient is not None:
                        accumulated[j] = gradient if accumulated[j] is None else accumulated[j] + gradient
                del loss, gradients
                continue
            for parameter, gradient in zip(trainable, gradients, strict=True):
                parameter.grad = gradient
            if topology is not None:
                # Exhausted episode slots contribute zero, while other ranks may still be active.
                if allow_inactive_ranks and topology.is_distributed:
                    import torch.distributed as dist
                    present = torch.tensor([p.grad is not None for p in trainable],
                                           device=trainable[0].device, dtype=torch.int32)
                    dist.all_reduce(present, op=dist.ReduceOp.MAX)
                    for parameter, used in zip(trainable, present.tolist(), strict=True):
                        if used and parameter.grad is None:
                            parameter.grad = torch.zeros_like(parameter)
                # PCGrad is nonlinear: reduce each task gradient first, then project.
                reduce_update_gradients(trainable_named, topology)
            task_gradients.append(
                [
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().to(
                        dtype=torch.bfloat16 if use_compact else parameter.grad.dtype
                    ).clone()
                    for parameter, use_compact in zip(trainable, compact, strict=True)
                ]
            )
            for parameter in trainable:
                parameter.grad = None
        if streamed:
            if not seen:
                raise ValueError("empty gradient microbatch stream")
            for parameter, gradient in zip(trainable, accumulated, strict=True):
                parameter.grad = gradient
            if topology is not None:
                if allow_inactive_ranks and topology.is_distributed:
                    import torch.distributed as dist
                    present = torch.tensor([p.grad is not None for p in trainable], device=trainable[0].device, dtype=torch.int32)
                    dist.all_reduce(present, op=dist.ReduceOp.MAX)
                    for parameter, used in zip(trainable, present.tolist(), strict=True):
                        if used and parameter.grad is None:
                            parameter.grad = torch.zeros_like(parameter)
                reduce_update_gradients(trainable_named, topology)
            task_gradients.append([
                None if p.grad is None else p.grad.detach().to(dtype=torch.bfloat16 if compacted else p.grad.dtype).clone()
                for p, compacted in zip(trainable, compact, strict=True)
            ])
            for parameter in trainable:
                parameter.grad = None
    if len(task_gradients) == 1 and allow_single_task:
        if auxiliary_loss_or_forward is not None:
            raise ValueError("single-task gradients require separate explicit objectives")
        for parameter, gradient in zip(trainable, task_gradients[0], strict=True):
            parameter.grad = None if gradient is None else gradient.to(dtype=parameter.dtype)
        return {"conflicts": 0, "comparisons": 0}
    if len(task_gradients) < 2:
        raise ValueError("PCGrad requires at least two task losses")

    merged_gradients: list[Tensor | None] = [None] * len(trainable)
    merged_counts = [0] * len(trainable)
    conflicts = 0
    comparisons = 0
    rng = random.Random(int(seed))
    for left, gradients in enumerate(task_gradients):
        left_gradients = [
            None if gradient is None else gradient.clone() for gradient in gradients
        ]
        order = [right for right in range(len(task_gradients)) if right != left]
        rng.shuffle(order)
        for right in order:
            reference = task_gradients[right]
            overlap = [
                (current, other)
                for current, other in zip(left_gradients, reference, strict=True)
                if current is not None and other is not None
            ]
            if not overlap:
                continue
            comparisons += 1
            dot = sum(
                (current.float() * other.float()).sum()
                for current, other in overlap
            )
            if float(dot.detach()) >= 0.0:
                continue
            norm_sq = sum(other.float().square().sum() for _, other in overlap)
            coefficient = dot / norm_sq.clamp_min(torch.finfo(dot.dtype).eps)
            for parameter_index, (current, other) in enumerate(
                zip(left_gradients, reference, strict=True)
            ):
                if current is not None and other is not None:
                    left_gradients[parameter_index] = current - coefficient.to(
                        dtype=current.dtype
                    ) * other
            conflicts += 1
        for parameter_index, gradient in enumerate(left_gradients):
            if gradient is None:
                continue
            if merged_gradients[parameter_index] is None:
                merged_gradients[parameter_index] = gradient
            else:
                merged_gradients[parameter_index].add_(gradient)
            merged_counts[parameter_index] += 1
    for index, count in enumerate(merged_counts):
        if count:
            merged_gradients[index].div_(count)

    stats: dict[str, int] = {
        "conflicts": conflicts,
        "comparisons": comparisons,
    }
    if auxiliary_loss_or_forward is None:
        task_gradients.clear()
        del left_gradients, gradients, reference, overlap
        for index, (parameter, gradient) in enumerate(
            zip(trainable, merged_gradients, strict=True)
        ):
            parameter.grad = (
                None if gradient is None else gradient.to(dtype=parameter.dtype)
            )
            merged_gradients[index] = None
        return stats

    auxiliary_result = (
        auxiliary_loss_or_forward()
        if callable(auxiliary_loss_or_forward)
        else auxiliary_loss_or_forward
    )
    auxiliary_loss = (
        auxiliary_result[0]
        if isinstance(auxiliary_result, tuple)
        else auxiliary_result
    )
    if not isinstance(auxiliary_loss, Tensor) or auxiliary_loss.ndim != 0:
        raise ValueError("PCGrad auxiliary loss must be a scalar tensor")
    for parameter in trainable:
        parameter.grad = None
    auxiliary_loss.backward()
    if topology is not None:
        # Projection is nonlinear, so reduce World gradients before surgery.
        reduce_update_gradients(trainable_named, topology)

    world_conflicts = 0
    world_comparisons = 0
    world_order = list(range(len(task_gradients)))
    rng.shuffle(world_order)
    for task_index in world_order:
        reference = task_gradients[task_index]
        overlap = [
            (parameter.grad, task_gradient)
            for parameter, task_gradient in zip(trainable, reference, strict=True)
            if parameter.grad is not None and task_gradient is not None
        ]
        if not overlap:
            continue
        world_comparisons += 1
        dot = sum((world * task).sum() for world, task in overlap)
        if float(dot.detach()) >= 0.0:
            continue
        norm_sq = sum(task.square().sum() for _, task in overlap)
        coefficient = dot / norm_sq.clamp_min(torch.finfo(dot.dtype).eps)
        for parameter, task_gradient in zip(trainable, reference, strict=True):
            if parameter.grad is not None and task_gradient is not None:
                parameter.grad.sub_(coefficient * task_gradient)
        world_conflicts += 1

    for parameter, action_gradient in zip(
        trainable, merged_gradients, strict=True
    ):
        if action_gradient is None:
            continue
        action_gradient = action_gradient.to(dtype=parameter.dtype)
        if parameter.grad is None:
            parameter.grad = action_gradient
        else:
            parameter.grad.add_(action_gradient)
    stats.update(
        world_conflicts=world_conflicts,
        world_comparisons=world_comparisons,
    )
    return stats, auxiliary_result


def partition_separate_pcgrad_parameters(
    named_parameters: list[tuple[str, Tensor]],
) -> tuple[
    list[tuple[str, Tensor]],
    list[tuple[str, Tensor]],
    list[tuple[str, Tensor]],
]:
    """Split optimizer parameters into Action-private, World-private and DINO."""
    action: list[tuple[str, Tensor]] = []
    world: list[tuple[str, Tensor]] = []
    shared_dino: list[tuple[str, Tensor]] = []
    for name, parameter in named_parameters:
        if name.startswith("main_vision_backbone."):
            shared_dino.append((name, parameter))
        elif name.startswith(("model.wmrm.", "model.world_action_readout.")) or name in {
            "model.wmrm_stage_scale",
            "model.wmrm_belief_message_scale",
        }:
            world.append((name, parameter))
        else:
            action.append((name, parameter))
    partitioned = [*action, *world, *shared_dino]
    if {id(parameter) for _, parameter in partitioned} != {
        id(parameter) for _, parameter in named_parameters
    } or len(partitioned) != len(named_parameters):
        raise RuntimeError("separate PCGrad parameter partition is incomplete")
    return action, world, shared_dino


def separate_pcgrad_scope(args: argparse.Namespace) -> str:
    return (
        "per_task_va_and_world_separate_bf16dino_guard_v1"
        if getattr(args, "vision_unfreeze_all", False)
        or int(getattr(args, "vision_unfreeze_last", 0)) > 0
        else "per_task_va_and_world_separate_frozen_dino_v1"
    )


def pop_update_gradients(
    named_parameters: list[tuple[str, Tensor]],
) -> dict[int, Tensor | None]:
    """Clone the current merged gradient and clear it for the next branch."""
    gradients: dict[int, Tensor | None] = {}
    for _, parameter in named_parameters:
        gradients[id(parameter)] = (
            None if parameter.grad is None else parameter.grad.detach().clone()
        )
        parameter.grad = None
    return gradients


def merge_separate_pcgrad_gradients(
    action_private: list[tuple[str, Tensor]],
    shared_dino: list[tuple[str, Tensor]],
    action_gradients: dict[int, Tensor | None],
) -> dict[str, float | int]:
    """Restore Action grads and guard only the shared DINO against WM conflict."""
    for _, parameter in action_private:
        parameter.grad = action_gradients[id(parameter)]

    overlap = [
        (parameter, action_gradients[id(parameter)], parameter.grad)
        for _, parameter in shared_dino
        if action_gradients[id(parameter)] is not None and parameter.grad is not None
    ]
    if overlap:
        dot = sum((action * world).sum() for _, action, world in overlap)
        action_norm_sq = sum(action.square().sum() for _, action, _ in overlap)
        world_norm_sq = sum(world.square().sum() for _, _, world in overlap)
        denominator = (action_norm_sq * world_norm_sq).clamp_min(
            torch.finfo(dot.dtype).eps
        ).sqrt()
        cosine = dot / denominator
        projected = int(float(dot.detach()) < 0.0 and float(action_norm_sq.detach()) > 0.0)
        if projected:
            coefficient = dot / action_norm_sq.clamp_min(
                torch.finfo(dot.dtype).eps
            )
            for parameter, action, world in overlap:
                parameter.grad = world - coefficient * action
        post_dot = sum(
            (action * parameter.grad).sum()
            for parameter, action, _ in overlap
        )
        post_world_norm_sq = sum(
            parameter.grad.square().sum() for parameter, _, _ in overlap
        )
        post_denominator = (action_norm_sq * post_world_norm_sq).clamp_min(
            torch.finfo(dot.dtype).eps
        ).sqrt()
        post_cosine = post_dot / post_denominator
    else:
        cosine = post_cosine = torch.tensor(0.0)
        projected = 0

    for _, parameter in shared_dino:
        action = action_gradients[id(parameter)]
        if action is None:
            continue
        if parameter.grad is None:
            parameter.grad = action
        else:
            parameter.grad.add_(action)
    return {
        "dino_projected": projected,
        "dino_cosine": float(cosine.detach()),
        "dino_post_cosine": float(post_cosine.detach()),
    }


def validate_finite_update_scalars(
    named_losses: list[tuple[str, object]],
) -> None:
    """Reject non-finite scalar losses before autograd can touch parameters."""
    for name, value in named_losses:
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.numel() != 1:
            raise RuntimeError(
                f"non-scalar update loss {name}: shape={tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value.detach()).item()):
            raise FloatingPointError(
                f"non-finite update loss {name}: value={value.detach().item()!r}"
            )


def validate_update_gradients(
    named_parameters,
    *,
    max_norm: float | None = None,
) -> float:
    """Validate current parameters/gradients and return their aggregate grad norm.

    The parameter check shares the already-required gradient traversal, so every
    update is guarded without copying parameters or optimizer state. ``None``
    gradients are allowed because conditional branches can leave modules unused.
    ``max_norm`` applies only to the aggregate norm, never to individual elements.
    """
    norm_terms: list[float] = []
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameter_value = parameter.detach()
        if not bool(torch.isfinite(parameter_value).all().item()):
            bad = (~torch.isfinite(parameter_value)).flatten().nonzero(as_tuple=False)[0].item()
            value = parameter_value.flatten()[bad].item()
            raise FloatingPointError(f"non-finite parameter {name}: value={value!r}")
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient.detach())
        if not bool(finite.all().item()):
            bad = (~finite).flatten().nonzero(as_tuple=False)[0].item()
            value = gradient.detach().flatten()[bad].item()
            raise FloatingPointError(
                f"non-finite gradient {name}: value={value!r}"
            )
        norm_terms.append(float(gradient.detach().double().norm().item()))
    norm = math.sqrt(math.fsum(term * term for term in norm_terms))
    if not math.isfinite(norm):
        raise FloatingPointError(f"non-finite aggregate gradient norm: value={norm!r}")
    if max_norm is not None and norm > max_norm:
        raise FloatingPointError(
            f"gradient threshold exceeded aggregate_norm: value={norm!r} "
            f"> threshold={max_norm!r}"
        )
    return norm


def is_zero_redundancy_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    return isinstance(optimizer, ZeroRedundancyOptimizer)


def consolidate_zero_optimizer_state(
    optimizer: torch.optim.Optimizer, *, to: int = 0
) -> None:
    if is_zero_redundancy_optimizer(optimizer):
        optimizer.consolidate_state_dict(to=to)


def validate_optimizer_update_state(
    optimizer: torch.optim.Optimizer,
    *,
    validate_values: bool = True,
) -> None:
    """Validate optimizer hyperparameters, parameters, and existing tensor state.

    This full state scan runs once after startup/resume. Per update, callers repeat
    only the cheap param-group validation; ``clip_grad_norm_`` supplies the
    aggregate pre-clip norm and rejects non-finite gradients. AdamW state is not
    transactionally copied or rescanned in the hot path.
    """
    base = (
        optimizer.optim
        if is_zero_redundancy_optimizer(optimizer)
        else optimizer
    )
    for group_index, group in enumerate(base.param_groups):
        def finite_number(key: str, *, minimum: float, strict: bool = False) -> float:
            try:
                value = float(group[key])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid optimizer group[{group_index}] {key}: {group.get(key)!r}"
                ) from exc
            valid = math.isfinite(value) and (value > minimum if strict else value >= minimum)
            if not valid:
                comparator = ">" if strict else ">="
                raise ValueError(
                    f"invalid optimizer group[{group_index}] {key}: {value!r}; "
                    f"must be finite and {comparator} {minimum}"
                )
            return value

        finite_number("lr", minimum=0.0)
        if "initial_lr" in group:
            finite_number("initial_lr", minimum=0.0)
        finite_number("weight_decay", minimum=0.0)
        finite_number("eps", minimum=0.0, strict=True)
        betas = group.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"invalid optimizer group[{group_index}] betas: {betas!r}")
        for beta_index, beta in enumerate(betas):
            try:
                beta_value = float(beta)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid optimizer group[{group_index}] beta[{beta_index}]: {beta!r}"
                ) from exc
            if not math.isfinite(beta_value) or not 0.0 <= beta_value < 1.0:
                raise ValueError(
                    f"invalid optimizer group[{group_index}] beta[{beta_index}]: "
                    f"{beta_value!r}; must be finite and in [0, 1)"
                )
        if validate_values:
            for parameter_index, parameter in enumerate(group["params"]):
                if not bool(torch.isfinite(parameter.detach()).all().item()):
                    raise FloatingPointError(
                        f"non-finite optimizer parameter group[{group_index}]"
                        f"[{parameter_index}]"
                    )

    def validate_state_value(path: str, value: object) -> None:
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value.detach()).all().item()):
                raise FloatingPointError(f"non-finite optimizer state {path}")
        elif isinstance(value, dict):
            for key, child in value.items():
                validate_state_value(f"{path}.{key}", child)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                validate_state_value(f"{path}[{index}]", child)
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite optimizer state {path}: {value!r}")

    for state_index, state in enumerate(base.state.values()):
        if validate_values:
            validate_state_value(f"state[{state_index}]", state)


def named_trainable_parameters(*modules):
    """Yield unique, stably named trainable parameters from update modules."""
    seen: set[int] = set()
    for prefix, module in modules:
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield f"{prefix}.{name}", parameter


def named_optimizer_parameters(optimizer, *modules):
    """Name every unique trainable optimizer parameter, including external heads."""
    known = {
        id(parameter): name
        for name, parameter in named_trainable_parameters(*modules)
    }
    seen: set[int] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield (
                known.get(
                    id(parameter),
                    f"optimizer.group[{group_index}].parameter[{parameter_index}]",
                ),
                parameter,
            )


def is_wmrm_predictor_parameter_name(name: str) -> bool:
    """True for the shared 6-block STPredictor, under any module prefix."""
    return (
        name.startswith("wmrm.st_predictor.")
        or ".wmrm.st_predictor." in name
    )


def partition_wmrm_predictor_named_parameters(
    named_parameters,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split update parameters so predictor clip cannot overlap the main clip."""
    predictor: list[tuple[str, Tensor]] = []
    other: list[tuple[str, Tensor]] = []
    for name, parameter in named_parameters:
        if is_wmrm_predictor_parameter_name(name):
            predictor.append((name, parameter))
        else:
            other.append((name, parameter))
    return predictor, other


def clip_main_and_optional_predictor_gradients(
    named_parameters,
    *,
    predictor_max_norm: float | None,
    main_max_norm: float = 1.0,
) -> tuple[float, float | None]:
    """Clip the shared predictor separately when a dedicated max-norm is set."""
    if predictor_max_norm is None:
        return clip_update_gradients(named_parameters, max_norm=main_max_norm), None
    predictor_named, other_named = partition_wmrm_predictor_named_parameters(
        named_parameters
    )
    if not predictor_named:
        raise RuntimeError(
            "--wmrm-predictor-grad-clip is set but no wmrm.st_predictor "
            "parameters are in the update set"
        )
    predictor_norm = clip_update_gradients(
        predictor_named, max_norm=predictor_max_norm
    )
    main_norm = clip_update_gradients(other_named, max_norm=main_max_norm)
    return main_norm, predictor_norm


def clip_update_gradients(named_parameters, *, max_norm: float) -> float:
    """Clip gradients and return their pre-clip norm, rejecting non-finite input."""
    unique_parameters = []
    seen: set[int] = set()
    for _, parameter in named_parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            unique_parameters.append(parameter)
    return float(
        torch.nn.utils.clip_grad_norm_(
            unique_parameters,
            max_norm,
            error_if_nonfinite=True,
        ).item()
    )


def validate_preclip_gradient_norms(
    *group_norms: float | None,
    max_norm: float | None = None,
) -> float:
    """Combine clip_grad_norm_ pre-clip norms and apply the update threshold."""
    values = [float(value) for value in group_norms if value is not None]
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm):
        raise FloatingPointError(
            f"non-finite aggregate gradient norm: value={norm!r}"
        )
    if max_norm is not None and norm > max_norm:
        raise FloatingPointError(
            f"gradient threshold exceeded aggregate_norm: value={norm!r} "
            f"> threshold={max_norm!r}"
        )
    return norm


def scale_semantic_lora_grads(text_backbone: nn.Module, scale: float) -> None:
    """η_act 梯度缩放（第二轮架构重构 2026-08-08）。

    SemanticAdapter 的 LoRA 参数承担语言→动作的语义适配，η_act < 1 抑制其对
    指令嵌入几何的过快扰动；非 LoRA 参数（门控/编译器/策略）不受影响。
    ``scale == 1.0`` 时为空操作。SAM 路径的两次 backward 后都调用（第一次
    缩放只改变扰动 e_w 的范数而非方向——ρ·g/‖g‖ 与缩放无关；实际步长由
    第二次缩放后的梯度决定，因此两次缩放才使 η_act 对 SAM 生效）。
    """
    if scale == 1.0:
        return
    for name, parameter in text_backbone.named_parameters():
        if (
            parameter.requires_grad
            and parameter.grad is not None
            and ("lora_a" in name or "lora_b" in name)
        ):
            parameter.grad.mul_(scale)
