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
        dest = tmp_path / f"ckpt_step{step}.pt"
        dest.write_bytes(b"x")
        dest.with_name(dest.name + ".sha256").write_text(f"abc  {dest}\n")
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
    dest4000 = tmp_path / "ckpt_step4000.pt"
    dest4000.write_bytes(b"x")
    summary_no_sha = summarize_task35_fm_log(
        "\n".join(lines),
        total_steps=15000,
        window=1000,
        checkpoint_stem=stem,
    )
    by_end_no_sha = {item["end"]: item for item in summary_no_sha["windows"]}
    assert by_end_no_sha[3000]["archived"] is True
    assert by_end_no_sha[4000]["periodic_save_only"] is True


def test_pipeline_health_alerts_missing_6k_waiter(tmp_path: Path, monkeypatch) -> None:
    from scripts import monitor_task35_fm_train as monitor

    stem = tmp_path / "ckpt"
    (tmp_path / "ckpt_step1000.pt").write_bytes(b"x")
    (tmp_path / "ckpt_step2000.pt").write_bytes(b"x")
    (tmp_path / "ckpt_step3000.pt").write_bytes(b"x")

    def fake_find(needle: str, *, require: tuple[str, ...] = ()):
        del require
        present = {
            "archive_task35_fm_milestones.sh": True,
            "tail -n 0 -F": True,
            "wait_task35_fm_finished_eval.sh": True,
            "wait_validate_task35_fm_milestone.sh 9000": True,
            "wait_validate_task35_fm_milestone.sh 12000": True,
        }
        return [{"pid": 1, "cmd": needle}] if present.get(needle) else []

    monkeypatch.setattr(monitor, "find_processes", fake_find)
    health = monitor.pipeline_health(
        trainer={"alive": True, "count": 1, "processes": [{"pid": 303509}]},
        checkpoint_stem=stem,
        meminfo="MemAvailable: 8388608 kB\n",
        gpu_compute_pids=[303509],
    )
    assert "waiter_6000_missing" in health["alerts"]
    assert "archiver_missing" not in health["alerts"]
    assert health["waiters"]["wait_6000"] is False
    late = monitor.pipeline_health(
        trainer={"alive": True, "count": 1, "processes": [{"pid": 303509}]},
        checkpoint_stem=stem,
        latest_step=6500,
        meminfo="MemAvailable: 8388608 kB\n",
        gpu_compute_pids=[303509],
    )
    assert "missing_archive_6000" in late["alerts"]
    assert "missing_archive_3000" in late["alerts"]  # files exist but no sha256
    (tmp_path / "ckpt_step3000.pt.sha256").write_text("abc\n")
    late_ok = monitor.pipeline_health(
        trainer={"alive": True, "count": 1, "processes": [{"pid": 303509}]},
        checkpoint_stem=stem,
        latest_step=6500,
        meminfo="MemAvailable: 8388608 kB\n",
        gpu_compute_pids=[303509],
    )
    assert "missing_archive_3000" not in late_ok["alerts"]
    assert "missing_archive_6000" in late_ok["alerts"]


def test_pipeline_health_alerts_low_ram_and_extra_gpu(tmp_path: Path, monkeypatch) -> None:
    from scripts import monitor_task35_fm_train as monitor

    monkeypatch.setattr(monitor, "find_processes", lambda needle, require=(): [{"pid": 2}])
    health = monitor.pipeline_health(
        trainer={"alive": True, "count": 1, "processes": [{"pid": 303509}]},
        checkpoint_stem=tmp_path / "ckpt",
        meminfo="MemAvailable: 1024 kB\n",
        gpu_compute_pids=[303509, 999],
    )
    assert "low_ram" in health["alerts"]
    assert "gpu_not_exclusive" in health["alerts"]


def test_monitor_markdown_does_not_repeat_pipeline_alerts() -> None:
    from scripts.monitor_task35_fm_train import render_md

    snapshot = {
        "generated_at": "t",
        "task": "peg-insert-side-v3",
        "latest_step": 1,
        "total_steps": 15000,
        "progress": 1 / 15000,
        "latest_loss": 0.2,
        "latest_grad": 1.0,
        "latest_aux_rmse_px": 10.0,
        "latest_aux_step": 1,
        "checkpoints_saved": [1000],
        "alerts": ["low_ram"],
        "windows": {},
        "aux_last_10": None,
        "milestones": [],
        "pipeline": {
            "waiters": {"archiver": True},
            "alerts": ["low_ram"],
            "mem_available_kb": 1024,
        },
    }
    text = render_md(snapshot, {"alive": True}, {"eta": None, "sec_per_step": None, "remain_steps": 0})
    assert text.count("low_ram") == 1
