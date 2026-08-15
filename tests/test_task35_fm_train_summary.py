from pathlib import Path

from scripts.summarize_task35_fm_train import summarize_task35_fm_log


def test_task35_fm_summary_windows_and_save_events() -> None:
    lines = [
        "step=1 mode=bidir_va contract=single task=peg-insert-side-v3 loss=1.5 grad=1.0",
        "step=2 mode=bidir_va contract=single task=peg-insert-side-v3 loss=1.0 grad=1.0 aux_rmse=12.0px",
        "step=3 mode=bidir_va contract=single task=peg-insert-side-v3 loss=0.5 grad=1.0",
        "step=3 global_step=3 periodic checkpoint saved to ckpt.pt",
    ]
    summary = summarize_task35_fm_log(
        "\n".join(lines), total_steps=10, window=2
    )
    assert summary["latest_step"] == 3
    assert summary["latest_loss"] == 0.5
    assert summary["checkpoints_saved"] == [3]
    assert summary["windows"][0]["start"] == 1
    assert summary["windows"][0]["end"] == 2
    assert summary["windows"][0]["loss"]["mean"] == 1.25
    assert summary["windows"][0]["aux_rmse_px"]["mean"] == 12.0
    assert summary["windows"][1]["end"] == 3
    assert summary["windows"][1]["loss"]["last"] == 0.5


def test_only_planned_archive_milestones_are_flagged(tmp_path: Path) -> None:
    lines = [
        f"step={step} mode=bidir_va contract=single task=peg-insert-side-v3 loss=0.2 grad=1.0"
        for step in range(1, 4001)
    ]
    lines.append("step=4000 global_step=4000 periodic checkpoint saved to ckpt.pt")
    stem = tmp_path / "ckpt"
    for step in (1000, 2000, 3000):
        (tmp_path / f"ckpt_step{step}.pt").write_bytes(b"x")
    summary = summarize_task35_fm_log(
        "\n".join(lines),
        total_steps=15000,
        window=1000,
        checkpoint_stem=stem,
    )
    by_end = {item["end"]: item for item in summary["windows"]}
    assert by_end[3000]["archived"] is True
    assert by_end[4000]["periodic_save_only"] is True
    assert "archived" not in by_end[4000]
    from scripts.monitor_task35_fm_train import milestone_lines

    text = "\n".join(milestone_lines(summary["windows"]))
    assert "3001-4000" in text and "periodic-save" in text
    assert "missing-archive" not in text
