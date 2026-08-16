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


def _old_canceling_horizon_loss(pred, target, weight):
    """Buggy reduction: per-horizon (se*w)/w then /H. Span weights cancel."""
    import torch.nn.functional as F

    loss = pred.new_zeros(())
    n_h = pred.shape[1]
    for k in range(n_h):
        se = F.smooth_l1_loss(pred[:, k], target[:, k], reduction="none").mean(
            dim=tuple(range(1, pred[:, k].ndim))
        )
        w = weight[:, k]
        denom = w.sum()
        if bool(denom > 0):
            loss = loss + (se * w).sum() / denom
    return loss / float(n_h)


def test_horizon_weights_do_not_cancel(train_wam) -> None:
    """Design §4 1.0/0.5/0.25 must survive reduction (audit: old (se*w)/w then /3)."""
    module = _load_module()
    torch.manual_seed(0)
    batch, n_h, n_tok, dim = 4, 3, 16, 8
    # Constant-per-horizon residuals in the Smooth-L1 linear region (|x|>=1)
    # so each span mean is exactly |delta|-0.5.
    deltas = (3.0, 5.0, 9.0)  # means = 2.5, 4.5, 8.5
    pred = torch.zeros(batch, n_h, n_tok, dim)
    target = torch.zeros_like(pred)
    for k, delta in enumerate(deltas):
        target[:, k] = delta
    weight = torch.tensor([1.0, 0.5, 0.25]).view(1, n_h).expand(batch, n_h).contiguous()

    means = [abs(d) - 0.5 for d in deltas]
    expected = (1.0 * means[0] + 0.5 * means[1] + 0.25 * means[2]) / (1.0 + 0.5 + 0.25)
    old = _old_canceling_horizon_loss(pred, target, weight)
    got = module.horizon_weighted_smooth_l1(pred, target, weight)

    assert not torch.allclose(old, torch.as_tensor(expected, dtype=old.dtype)), (
        "sanity: old canceling formula must NOT already equal the §4 weighted mean"
    )
    assert torch.allclose(old, torch.as_tensor(sum(means) / 3.0, dtype=old.dtype))
    assert torch.allclose(got, torch.as_tensor(expected, dtype=got.dtype)), (
        f"§4 weighted mean {expected} != {float(got)} (old canceling value was {float(old)})"
    )


def test_excluded_horizon_leaves_denominator(train_wam) -> None:
    """A fully-invalid span must not be averaged in as 0 (old code still /3)."""
    module = _load_module()
    batch, n_h, n_tok, dim = 2, 3, 4, 3
    pred = torch.zeros(batch, n_h, n_tok, dim)
    target = torch.zeros_like(pred)
    target[:, 0] = 3.0   # mean 2.5
    target[:, 1] = 5.0   # mean 4.5
    target[:, 2] = 21.0  # would be mean 20.5 if included
    weight = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])

    expected = (1.0 * 2.5 + 0.5 * 4.5) / (1.0 + 0.5)
    old = _old_canceling_horizon_loss(pred, target, weight)
    got = module.horizon_weighted_smooth_l1(pred, target, weight)
    assert not torch.allclose(old, torch.as_tensor(expected, dtype=old.dtype))
    assert torch.allclose(got, torch.as_tensor(expected, dtype=got.dtype)), (
        f"excluded-horizon weighted mean {expected} != {float(got)} (old={float(old)})"
    )


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
