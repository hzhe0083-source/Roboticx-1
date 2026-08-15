from scripts.task35_proc import find_processes, is_inspector, trainer_processes


def test_inspector_command_lines_are_ignored() -> None:
    assert is_inspector("pgrep -f train.py --task35-precision-contract")
    assert is_inspector("python3 - <<'PY'\nneedles = ('train.py',)\nPY")
    assert is_inspector("/usr/bin/python -B scripts/task35_proc.py --check trainer")
    assert not is_inspector(
        "/home/ryan/.venvs/pytorch-gpu/bin/python -u -B train.py "
        "--task35-precision-contract --direct-head"
    )


def test_find_processes_requires_all_markers() -> None:
    rows = find_processes("train.py --task35-precision-contract", require=("python", "train.py"))
    for row in rows:
        assert "python" in row["cmd"]
        assert "train.py --task35-precision-contract" in row["cmd"]
        assert not is_inspector(row["cmd"])
    live = trainer_processes()
    assert all(row["pid"] > 1 for row in live)
