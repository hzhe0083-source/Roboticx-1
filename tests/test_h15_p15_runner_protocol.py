from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_mw_all49_wam4va_h15_v1.sh"


def test_all49_runner_builds_and_trains_true_h15_p15() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    text = RUNNER.read_text(encoding="utf-8")
    prepare = text.split("prepare_data(){", 1)[1].split("\npreflight(){", 1)[0]
    prepare_full = text.split("prepare_full_data(){", 1)[1].split("\npreflight(){", 1)[0]
    preflight = text.split("preflight(){", 1)[1].split("\nrequire_idle(){", 1)[0]
    joint = text.split("run_joint(){", 1)[1]

    assert "all49_peer_h15_p15_source_v1.pt" in text
    assert "all49_peer_h15_p2_source_v1.pt" not in text
    assert "PEER_SYNC_H15_P15_CONTRACT" in prepare
    assert "planning_stride=15" in prepare
    assert "all49_peer_h15_p15_full_train_v1.pt" in text
    assert "all49_peer_h15_p15_full_split_v1.json" in text
    assert "--reuse-existing-eval" in prepare_full
    assert '--input "$SOURCE"' in prepare_full
    assert '--train-output "$FULL_TRAIN"' in prepare_full
    assert '--eval-output "$EVAL_DATA"' in prepare_full
    assert "EPOCHS=${EPOCHS:-23}" in text
    assert "EPOCHS=25" not in text
    assert '"source": 11903, "train": 10722, "eval": 1181' in preflight
    assert '(counts["train"] + global_batch - 1) // global_batch' in preflight
    assert "steps_per_epoch != 224" in preflight
    assert "expected_steps != 5152" in preflight
    assert '"decision_offsets": [0, 15, 30, 45]' in text
    assert '"world_target_offsets": [15, 30, 45, 60]' in text
    assert 'previous_action[:, 1:], actions[:, :-1, 14]' in text
    assert "World refs are not d+15 endpoints" in text
    assert "--planning-stride 15 --control-stride 15" in joint
    assert "--deployment-execution-horizon 15" in joint
    assert "--wmrm-cycle-steps 15" in joint
    assert "--flow-prefix-steps 15" in joint
    assert "--wmrm-full-language-tokens" in joint
    assert "--slot-free-policy" in joint
    assert '--va-data "$FULL_TRAIN" --world-data "$FULL_TRAIN"' in joint
    assert "--peer-shared-full-data" in joint
    assert '--world-split-manifest "$FULL_MANIFEST"' in joint
    assert "--task-sampling full" in joint
    assert "--task-sampling balanced" not in joint
    assert "--peer-batch-prefetch" in joint
    assert "--peer-batch-prefetch-depth 8" in joint
    assert "--longtraj-decode-cache-tasks 2" in joint
    assert "--disable-runtime-integrity-checks" in joint
    assert "--resume-weights" not in joint
    assert "--dino-dense-metric" not in joint
    assert "--metric-geometry-inject" not in joint
    assert "--mtvj-train-metric-head" not in joint
    assert "--mtvj-train-relation" not in joint
