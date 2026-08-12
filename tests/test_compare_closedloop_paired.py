from pathlib import Path

import pytest

from scripts.compare_closedloop_paired import compare_logs, parse_trials


def _write(path: Path, task_rows: list[list[int]]) -> None:
    lines = []
    for task_id, outcomes in enumerate(task_rows):
        for trial, success in enumerate(outcomes):
            seed = 1000 * task_id + trial
            lines.append(
                f"trial task={task_id} trial={trial} seed={seed} success={success}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_paired_comparison_detects_strict_improvement(tmp_path) -> None:
    baseline = tmp_path / "base.log"
    candidate = tmp_path / "candidate.log"
    _write(baseline, [[0] * 10 for _ in range(49)])
    _write(candidate, [[1] * 10 for _ in range(49)])

    result = compare_logs(baseline, candidate, n_boot=200)

    assert result.delta == pytest.approx(1.0)
    assert result.improvement_confirmed
    assert result.target_60_reached
    assert result.target_60_confirmed


def test_paired_comparison_rejects_unmatched_trials(tmp_path) -> None:
    baseline = tmp_path / "base.log"
    candidate = tmp_path / "candidate.log"
    _write(baseline, [[0, 1]])
    _write(candidate, [[0]])

    with pytest.raises(ValueError, match="paired trial keys differ"):
        compare_logs(baseline, candidate, n_boot=20)


def test_paired_comparison_requires_complete_all49_gate(tmp_path) -> None:
    baseline = tmp_path / "base.log"
    candidate = tmp_path / "candidate.log"
    _write(baseline, [[0, 1]])
    _write(candidate, [[1, 1]])

    with pytest.raises(ValueError, match="exactly 49 tasks x 10 trials"):
        compare_logs(baseline, candidate, n_boot=20)


def test_target_is_strictly_greater_than_sixty_percent(tmp_path) -> None:
    baseline = tmp_path / "base.log"
    candidate = tmp_path / "candidate.log"
    _write(baseline, [[0] * 10 for _ in range(49)])
    # Exactly 294 / 490 = 60%; the user's gate is strictly greater.
    outcomes = [1] * 294 + [0] * (490 - 294)
    rows = [outcomes[index : index + 10] for index in range(0, 490, 10)]
    _write(candidate, rows)

    result = compare_logs(baseline, candidate, n_boot=20)
    assert result.candidate_rate == pytest.approx(0.60)
    assert not result.target_60_reached
    assert not result.target_60_confirmed


def test_parse_trials_rejects_legacy_task_only_log(tmp_path) -> None:
    path = tmp_path / "legacy.log"
    path.write_text("task door-lock: 3/10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no trial-level results"):
        parse_trials(path)
