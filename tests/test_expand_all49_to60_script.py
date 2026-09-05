from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "expand_all49_to60.sh"


def _manifest() -> dict:
    tasks = ["assembly-v3"] + [f"task-{index:02d}-v3" for index in range(1, 49)]
    tasks[16] = "door-unlock-v3"
    return {
        "contract": "all49_canonical_raw_sources_v1",
        "sources": [{"task": task} for task in tasks],
    }


def test_expand_all49_to60_script_syntax_and_contract(tmp_path: Path) -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    manifest = tmp_path / "raw_identity.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    norm_ref = tmp_path / "normalization.pt"
    norm_ref.touch()
    out_root = tmp_path / "expand60"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PY": sys.executable,
            "RAW_IDENTITY_MANIFEST": str(manifest),
            "NORM_REF": str(norm_ref),
            "OUT_ROOT": str(out_root),
            "DRY_RUN": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("[dry-run] nice")]
    assert len(commands) == 47 * 3
    assert not out_root.exists()

    text = SCRIPT.read_text(encoding="utf-8")
    assert "MAX_JOBS=${MAX_JOBS:-4}" in text
    assert "CPUSET=${CPUSET:-0-31}" in text
    assert "EPISODE_SEED_BASE=600000" in text
    assert "COLLECTOR_RNG_BASE=900000" in text
    assert "CUDA_VISIBLE_DEVICES=" in text
    assert "MUJOCO_GL=osmesa" in text
    assert "LP_NUM_THREADS=2" in text
    assert "nice -n 10 ionice -c 3 taskset" in text
    assert "--force-perturb" in text
    assert "wait -n" in text
    assert "torch.load" in text
    assert "n_perturb_events" in text

    assert "--seed 900010" in commands[0]
    assert "--episode-seeds 601000 601001 601002" in commands[0]
    assert "601009" in commands[0]
    assert "--seed 900482" in commands[-1]
    assert "648020" in commands[-1]
    assert "648029" in commands[-1]
