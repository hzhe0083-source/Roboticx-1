"""CLI ownership and retired-route boundaries without loading training assets."""
import ast
from pathlib import Path

import pytest

from va_compound.training.config import parse_args
from va_compound.data.libero import RUN_SCHEDULE_PROFILES


@pytest.mark.parametrize("flag", ["--direct-head", "--c2-controller", "--compile-task", "--sam-rho", "--semantic-adapter"])
def test_retired_training_flags_are_rejected(flag):
    values = [flag, "0.1"] if flag == "--sam-rho" else [flag]
    with pytest.raises(SystemExit) as rejected:
        parse_args(values)
    assert rejected.value.code == 2


def test_libero_run_profiles_keep_existing_schedule():
    assert RUN_SCHEDULE_PROFILES[2] == {
        "rows": 9843, "batch_size": 8, "mixed_tasks": 2,
        "stage1_steps": 0, "suites": ["libero_10"],
        "local_task_ids": [3, 4],
        "grouping": "two_task_t8_dense_local2_deferred_v4", "epochs": 4,
    }
    assert RUN_SCHEDULE_PROFILES[40]["rows"] == 32000
    assert RUN_SCHEDULE_PROFILES[40]["stage1_steps"] == 8000


def test_production_modules_do_not_import_compatibility_entrypoints():
    root = Path(__file__).resolve().parents[1]
    paths = [*root.joinpath("va_compound").rglob("*.py"),
             *root.joinpath("scripts").rglob("*.py"),
             *root.glob("eval*.py"), root / "train_libero.py", root / "train_metaworld.py"]
    violations = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module]
            if any(module in {"train", "libero_train"} for module in modules):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations
