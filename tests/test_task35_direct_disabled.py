import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "scripts" / "run_task35_h6_matched_va.sh"


def test_matched_va_launcher_refuses_direct_without_override() -> None:
    env = os.environ.copy()
    env["TASK35_ALLOW_DIRECT"] = "0"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "direct", "10", "6", "should_not_run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "Direct training is disabled" in result.stderr
