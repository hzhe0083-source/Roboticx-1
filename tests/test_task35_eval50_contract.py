from __future__ import annotations

import pytest

from eval_metaworld import TASK35_EVAL50_SEEDS, validate_task35_eval50_payload
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
        "task35_precision_contract": precision,
        "task35_causal_ablation": ablation,
        "trials": trials,
    }


def test_eval50_accepts_paired_precision_payload() -> None:
    report = validate_task35_eval50_payload(_payload())
    assert report["ok"] is True
    assert report["successes"] == 12


def test_eval50_rejects_wrong_seeds_and_missing_trials() -> None:
    with pytest.raises(ValueError, match="paired seeds"):
        validate_task35_eval50_payload(_payload(seeds=list(range(50))))
    bad = _payload()
    bad["completed_trials"] = 49
    with pytest.raises(ValueError, match="50 completed trials"):
        validate_task35_eval50_payload(bad)


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
    assert report["label"] == "supported"
