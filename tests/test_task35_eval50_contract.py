from __future__ import annotations

import pytest

from pathlib import Path

from eval_metaworld import (
    TASK35_EVAL50_SEEDS,
    validate_task35_eval50_payload,
    want_vjepa_dense_backbone,
)
from scripts.select_task35_best_fm import select_best_task35_fm


def _payload(
    *,
    successes: int = 12,
    seeds: list[int] | None = None,
    precision: bool = True,
    ablation: str = "none",
    execute_steps: int = 6,
) -> dict:
    seeds = list(TASK35_EVAL50_SEEDS) if seeds is None else seeds
    trials = []
    remaining = successes
    for seed in seeds:
        ok = remaining > 0
        remaining -= int(ok)
        trials.append(
            {
                "seed": seed,
                "success": ok,
                "stage": {"min_obj_to_target": 0.2 if ok else 0.8},
            }
        )
    return {
        "contract": "metaworld_closed_loop_trials_v1",
        "checkpoint_sha256": "abc",
        "task_ids": [35],
        "completed_trials": len(trials),
        "successes": successes,
        "success_rate": successes / len(trials) if trials else 0.0,
        "execute_steps": execute_steps,
        "horizon": 500,
        "wam": "off",
        "action_decoder": "conditional_flow_matching",
        "env_name": "peg-insert-side-v3",
        "task35_precision_contract": precision,
        "task35_causal_ablation": ablation,
        "language_source": "task35_features_cache",
        "trials": trials,
    }


def test_eval50_accepts_paired_precision_payload() -> None:
    report = validate_task35_eval50_payload(_payload())
    assert report["ok"] is True
    assert report["successes"] == 12


def test_eval50_rejects_direct_decoder_payload() -> None:
    bad = _payload()
    bad["action_decoder"] = "direct_head"
    with pytest.raises(ValueError, match="FM decoder"):
        validate_task35_eval50_payload(bad)


def test_eval50_rejects_wrong_seeds_and_missing_trials() -> None:
    with pytest.raises(ValueError, match="paired seeds"):
        validate_task35_eval50_payload(_payload(seeds=list(range(50))))
    bad = _payload()
    bad["completed_trials"] = 49
    with pytest.raises(ValueError, match="50 completed trials"):
        validate_task35_eval50_payload(bad)
    short = _payload()
    short["horizon"] = 400
    with pytest.raises(ValueError, match="horizon 500"):
        validate_task35_eval50_payload(short)
    missing_lang = _payload()
    missing_lang.pop("language_source")
    with pytest.raises(ValueError, match="language source recorded"):
        validate_task35_eval50_payload(missing_lang)


def test_selector_refuses_to_elect_without_eval50() -> None:
    with pytest.raises(ValueError, match="no reproducible FM VA"):
        select_best_task35_fm(
            [
                {
                    "path": "ckpt_step3000.pt",
                    "step": 3000,
                    "validated": True,
                    "eval50": None,
                }
            ]
        )


def test_selector_ranks_by_closed_loop_successes() -> None:
    report = select_best_task35_fm(
        [
            {"path": "a.pt", "step": 6000, "validated": True, "eval50": _payload(successes=8)},
            {"path": "b.pt", "step": 15000, "validated": True, "eval50": _payload(successes=15)},
        ]
    )
    assert report["selected"]["path"] == "b.pt"
    assert report["selected"]["successes"] == 15
    assert report["selected"]["step"] == 15000
    assert report["label"] == "supported"


class _Cfg:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _Args:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def test_dino_metric_does_not_autoload_vjepa_dense() -> None:
    config = _Cfg(dino_dense_metric=True, dense_readout_mtvj=True)
    args = _Args(dense_readout_mtvj=False, metric_visual_checkpoint=None)
    assert want_vjepa_dense_backbone(config, args) is False
    with pytest.raises(ValueError, match="禁止 --dense-readout-mtvj"):
        want_vjepa_dense_backbone(
            config, _Args(dense_readout_mtvj=True, metric_visual_checkpoint=None)
        )


def test_vjepa_metric_autoloads_dense_backbone() -> None:
    config = _Cfg(
        dino_dense_metric=False,
        dense_readout_mtvj=True,
        main_vision_backbone="vjepa",
    )
    args = _Args(dense_readout_mtvj=False, metric_visual_checkpoint=None)
    assert want_vjepa_dense_backbone(config, args) is True


def test_dino_main_does_not_autoload_vjepa_dense() -> None:
    config = _Cfg(
        dino_dense_metric=False,
        dense_readout_mtvj=True,
        main_vision_backbone="dinov2_vitl14_reg4",
    )
    args = _Args(dense_readout_mtvj=False, metric_visual_checkpoint=None)
    assert want_vjepa_dense_backbone(config, args) is False


def test_eval50_launcher_skips_qwen_and_sets_cuda_allocator() -> None:
    root = Path(__file__).resolve().parent.parent
    eval50 = (root / "scripts" / "run_task35_h6_eval50.sh").read_text()
    ablation = (root / "scripts" / "run_task35_h6_ablation50.sh").read_text()
    encode = (root / "eval_metaworld.py").read_text()
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in eval50
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in ablation
    assert "cached_task35_language" in encode
    assert "Encode one frame at a time" in encode


def test_selector_rejects_1k_2k_even_with_eval50() -> None:
    with pytest.raises(ValueError, match="no reproducible FM VA"):
        select_best_task35_fm(
            [
                {
                    "path": "checkpoints/x_step1000.pt",
                    "step": 1000,
                    "validated": True,
                    "eval50": _payload(successes=40),
                },
                {
                    "path": "checkpoints/x_step2000.pt",
                    "step": 2000,
                    "validated": True,
                    "eval50": _payload(successes=39),
                },
            ]
        )
