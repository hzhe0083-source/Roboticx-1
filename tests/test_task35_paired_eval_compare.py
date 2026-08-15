from scripts.compare_task35_paired_eval import compare_payloads


def _payload(successes: list[bool], distances: list[float], ablation: str = "none") -> dict:
    return {
        "contract": "metaworld_closed_loop_trials_v1",
        "checkpoint_sha256": "abc",
        "completed_trials": len(successes),
        "task35_causal_ablation": ablation,
        "trials": [
            {
                "seed": 35000 + index,
                "success": success,
                "stage": {"min_obj_to_target": distance},
            }
            for index, (success, distance) in enumerate(zip(successes, distances))
        ],
    }


def test_paired_eval_counts_discordant_seeds() -> None:
    report = compare_payloads(
        _payload([True, False, True], [0.10, 0.40, 0.20]),
        [
            (
                "geom.json",
                _payload([False, True, True], [0.30, 0.10, 0.15], "geometry-zero"),
            )
        ],
    )
    row = report["comparisons"][0]
    assert report["baseline"]["successes"] == 2
    assert row["successes"] == 2
    assert row["success_delta"] == 0
    assert row["discordant_improved"] == 1
    assert row["discordant_worsened"] == 1
    assert row["ablation"] == "geometry-zero"
