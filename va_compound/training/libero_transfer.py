"""Validate an explicit fixed-H15 to all-starts training transfer."""
from va_compound.data.libero import H15_DATA_CONTRACT

TRANSFER_INITIALIZATION = "continue_h15_fixed_to_all_starts_v1"


def is_all_starts_transfer(args):
    return (getattr(args, "resume_weights", None) is not None
            and getattr(args, "architecture_version", "legacy") == "dual_tower_h15_v1"
            and getattr(args, "window_sampling", None) == "all_starts_random_tbptt8_v1")


def validate_transfer_source(source):
    config = source.get("config", {})
    contract = source.get("training_contract", {})
    required = {
        "architecture_version": "dual_tower_h15_v1",
        "data_contract": H15_DATA_CONTRACT,
        "action_horizon": 15,
        "memory_contract": "episode_tbptt8_v1",
        "execution_gradient_contract": "h15_unified_live_va_v1",
        "main_vision_joint_trained": True,
        "flow_prefix_weight": 1.0,
        "flow_tail_weight": 0.0,
    }
    if config.get("architecture_version") != "dual_tower_h15_v1" or config.get("action_horizon") != 15:
        raise ValueError("transfer requires matching unified H15 architecture")
    mismatch = {key: (contract.get(key), value) for key, value in required.items() if contract.get(key) != value}
    if mismatch:
        raise ValueError(f"fixed H15 transfer source mismatch: {mismatch}")
    if not isinstance(source.get("global_step"), int) or source["global_step"] < 1:
        raise ValueError("transfer source must have completed updates")
    for key in ("model", "optimizer", "qwen_trainable_state_dict", "main_vision_trainable_state_dict"):
        if not isinstance(source.get(key), dict) or not source[key]:
            raise ValueError(f"transfer source lacks {key}")
