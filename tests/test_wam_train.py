"""CPU tests for `scripts/train_wam_e7.py` (Task 6): smoke losses/gradients
and bitwise exact resume.

The script's argparse entry lives in `main(argv)`, so both tests drive it
via subprocess and parse the documented per-step loss lines
(`step=N loss=... action=... vj=... geo=... consistency=... grad=...`).
The exact-resume test additionally exercises the script's own
`--self-check-resume`, which compares the resumed step loss with
`torch.equal` (true bitwise, independent of the 6-decimal log format).

The script is implemented by a sibling agent; until it lands, tests skip
with "dependency not yet implemented".
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train_wam_e7.py"

_LOSS_RE = re.compile(r"(loss|action|vj|geo|consistency|grad)=([-+0-9.eE]+|nan|inf)")


def _load_module():
    if not SCRIPT.exists():
        pytest.skip("dependency not yet implemented: scripts/train_wam_e7.py")
    spec = importlib.util.spec_from_file_location("train_wam_e7", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        pytest.skip(f"dependency not yet implemented: {error}")
    return module


@pytest.fixture(scope="module")
def train_wam():
    _load_module()  # skip unless the script and its imports exist


def _run_cli(extra_args, timeout=1800):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra_args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _cli_failure(proc) -> str:
    return (
        f"{SCRIPT} exited {proc.returncode}:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def _parse_loss_lines(output):
    """Ordered per-step {name: float} dicts, one per printed loss line."""
    lines = []
    for line in output.splitlines():
        values = {name: float(text) for name, text in _LOSS_RE.findall(line)}
        if "loss" in values:
            lines.append(values)
    return lines


def test_smoke_losses_finite_and_grads(train_wam, tmp_path) -> None:
    out = tmp_path / "smoke.pt"
    proc = _run_cli(["--smoke", "--steps", "4", "--out", str(out)])
    assert proc.returncode == 0, _cli_failure(proc)
    lines = _parse_loss_lines(proc.stdout)
    assert len(lines) == 4, f"expected 4 per-step loss lines, got:\n{proc.stdout}"

    for values in lines:
        numbers = [values[name] for name in ("loss", "action", "vj", "geo", "grad")]
        assert torch.isfinite(torch.tensor(numbers)).all(), f"non-finite losses: {values}"
        assert values["consistency"] == 0.0  # L_consistency placeholder is 0 in M0
        # The script's own smoke gate (Task 6 Step 1) asserts non-zero
        # action/vj/geo gradients; the printed grad norm must reflect that.
        assert values["grad"] > 0.0, f"zero gradient norm printed: {values}"


def test_exact_resume_bitwise(train_wam, tmp_path) -> None:
    run_a_ckpt = tmp_path / "run_a.pt"
    run_b_ckpt = tmp_path / "run_b.pt"
    proc_a = _run_cli(
        ["--smoke", "--steps", "3", "--save-every", "2", "--out", str(run_a_ckpt)]
    )
    assert proc_a.returncode == 0, _cli_failure(proc_a)
    lines_a = _parse_loss_lines(proc_a.stdout)
    assert len(lines_a) == 3, f"expected 3 per-step loss lines, got:\n{proc_a.stdout}"
    assert run_a_ckpt.exists(), f"no checkpoint written to {run_a_ckpt}"

    proc_b = _run_cli(
        ["--smoke", "--steps", "3", "--save-every", "2",
         "--resume", str(run_a_ckpt), "--out", str(run_b_ckpt)]
    )
    assert proc_b.returncode == 0, _cli_failure(proc_b)
    lines_b = _parse_loss_lines(proc_b.stdout)
    assert len(lines_b) == 1, f"resume at step 2 should log exactly step 3:\n{proc_b.stdout}"

    # Step 3 of the uninterrupted run must reproduce bitwise from the
    # step-2 checkpoint (same printed values at the 6-decimal contract).
    for name in ("loss", "action", "vj", "geo", "consistency"):
        assert lines_a[2][name] == lines_b[0][name], (
            f"resumed {name} {lines_b[0][name]} != uninterrupted {lines_a[2][name]}"
        )

    # Independent true-bitwise gate: the script compares the resumed loss
    # with torch.equal (steps=5 so the last save is step 4 and step 5 is
    # the reference step, as required by --self-check-resume).
    self_check = _run_cli(
        ["--self-check-resume", "--smoke", "--steps", "5",
         "--save-every", "2", "--out", str(tmp_path / "self_check.pt")]
    )
    assert self_check.returncode == 0, _cli_failure(self_check)
    assert "SELF-CHECK RESUME: PASS" in self_check.stdout, self_check.stdout
