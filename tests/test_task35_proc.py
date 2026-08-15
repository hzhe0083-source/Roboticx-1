from pathlib import Path

from scripts.task35_proc import find_processes, is_inspector, trainer_processes


def test_inspector_command_lines_are_ignored() -> None:
    assert is_inspector("pgrep -f train.py --task35-precision-contract")
    assert is_inspector("python3 - <<'PY'\nneedles = ('train.py',)\nPY")
    assert is_inspector("/usr/bin/python -B scripts/task35_proc.py --check trainer")
    assert is_inspector("python3 - <<'PY'\n'direct-head'\n'eval_metaworld.py'\nPY")
    assert is_inspector("/usr/bin/python -B scripts/check_task35_readiness.py")
    assert not is_inspector(
        "/home/ryan/.venvs/pytorch-gpu/bin/python -u -B train.py "
        "--task35-precision-contract --direct-head"
    )


def test_archiver_follows_log_events_not_stdin() -> None:
    text = Path("/home/ryan/Documents/robot/ORA0-task35-fullfix/scripts/archive_task35_fm_milestones.sh").read_text()
    assert "tail -n 0 -F" in text
    assert "read -r -t 30 -u 3" in text
    assert "trainer gone before all milestones were archived" in text
    assert "validate_task35_fm_checkpoint.py" not in text
    assert "verify_copy_sha" in text
    assert "archive SHA mismatch" in text
    assert "keep listening for later milestones" in text
    assert "archive_complete" in text
    assert "incomplete archive" in text
    waiter = Path("/home/ryan/Documents/robot/ORA0-task35-fullfix/scripts/wait_validate_task35_fm_milestone.sh").read_text()
    assert "loaded_modules" in waiter
    assert "CUDA_VISIBLE_DEVICES=" in waiter
    assert "milestone SHA mismatch" in waiter
    assert "wait_for_cpu_ram" in waiter
    assert "MemAvailable" in waiter


def test_find_processes_requires_all_markers() -> None:
    rows = find_processes("train.py --task35-precision-contract", require=("python", "train.py"))
    for row in rows:
        assert "python" in row["cmd"]
        assert "train.py --task35-precision-contract" in row["cmd"]
        assert not is_inspector(row["cmd"])
    live = trainer_processes()
    assert all(row["pid"] > 1 for row in live)
