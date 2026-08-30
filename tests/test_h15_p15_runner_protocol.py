from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_mw_all49_wam4va_h15_v1.sh"
MT50_RUNNER = ROOT / "scripts" / "run_mw_mt50_wam4va_h15_60ep_v2.sh"
MT50_ACCEPTANCE = ROOT / "scripts" / "run_mw_mt50_acceptance_v1.sh"
RLT_RUNNER = ROOT / "scripts" / "run_mw_rlt_h15_v1.sh"


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
    assert "EXPECTED_SOURCE_WINDOWS=${EXPECTED_SOURCE_WINDOWS:-11903}" in text
    assert "EXPECTED_TRAIN_WINDOWS=${EXPECTED_TRAIN_WINDOWS:-10722}" in text
    assert "EXPECTED_EVAL_WINDOWS=${EXPECTED_EVAL_WINDOWS:-1181}" in text
    assert "EXPECTED_TASKS=${EXPECTED_TASKS:-49}" in text
    assert "EXPECTED_EPOCHS=${EXPECTED_EPOCHS:-23}" in text
    assert "EXPECTED_RAW_CONTRACT=${EXPECTED_RAW_CONTRACT:-all49_canonical_raw_sources_v1}" in text
    assert '"source": int(sys.argv[11])' in preflight
    assert '(counts["train"] + global_batch - 1) // global_batch' in preflight
    assert "expected_steps = epochs * steps_per_epoch" in preflight
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
    assert 'PEER_BATCH_PREFETCH_DEPTH=${PEER_BATCH_PREFETCH_DEPTH:-16}' in text
    assert '--peer-batch-prefetch-depth "$PEER_BATCH_PREFETCH_DEPTH"' in joint
    assert "--longtraj-decode-cache-tasks 2" in joint
    assert "--disable-runtime-integrity-checks" in joint
    assert 'RESUME_WEIGHTS=${RESUME_WEIGHTS:-}' in text
    assert 'resume_args=(--resume-weights "$RESUME_WEIGHTS")' in joint
    assert "--dino-dense-metric" not in joint
    assert "--metric-geometry-inject" not in joint
    assert "--mtvj-train-metric-head" not in joint
    assert "--mtvj-train-relation" not in joint


def test_mt50_60ep_runner_appends_task50_and_continues_for_62_epochs() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(MT50_RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    text = MT50_RUNNER.read_text(encoding="utf-8")
    assert "mt50_full_episode_online_index_v1.json" in text
    assert "mt50_peer_h15_p15_source_60ep_v2.pt" not in text
    assert "mt50_peer_h15_p15_full_train_60ep_v2.pt" not in text
    assert "mt50_raw_canonical_identity_60ep_v2.json" in text
    assert "EPOCHS=${EPOCHS:-62}" in text
    assert "ONLINE_SAMPLES_PER_EPISODE=${ONLINE_SAMPLES_PER_EPISODE:-6}" in text
    assert "PEER_BATCH_PREFETCH_DEPTH=${PEER_BATCH_PREFETCH_DEPTH:-4}" in text
    assert "EPOCHS=10" in text
    assert "EPOCHS=50" in text
    assert "EPOCHS=3" not in text
    assert "antiforget-resume) run_joint antiforget-exact" in text
    assert "recovery) run_joint recovery" in text
    assert 'capacity16) fail "capacity16 is retired' in text
    assert "BATCH=${CAPACITY_BATCH:-20}" in text
    assert "CAPACITY_MIXED_TASKS_PER_BATCH:-5" in text
    assert '[[ "$BATCH" == 20 ]]' in text
    assert '--task-sampling mixed --mixed-tasks-per-batch "$MIXED_TASKS_PER_BATCH"' in text
    assert "--task-locality-block-batches 1 --longtraj-decode-cache-tasks 50" in text
    assert "--online-recovery-samples-per-episode 3" in text
    assert "mw_mt50_recovery25_mixed4_anchor25_pcgrad_lr1e5_from_s2015_e1_v1" in text
    assert "EXPECTED_SOURCE_EPISODES=${EXPECTED_SOURCE_EPISODES:-3420}" in text
    assert "EXPECTED_TRAIN_EPISODES=${EXPECTED_TRAIN_EPISODES:-3222}" in text
    assert "EXPECTED_EVAL_EPISODES=${EXPECTED_EVAL_EPISODES:-198}" in text
    assert "triple != expected" in text
    assert "offline_windows=0" in text
    assert "--online-episode-sampling" in text
    assert '--online-episode-samples "$ONLINE_SAMPLES_PER_EPISODE"' in text
    assert '--va-data "$ONLINE_INDEX" --world-data "$ONLINE_INDEX"' in text
    assert '--world-split-manifest "$ONLINE_INDEX"' in text
    assert "--task-sampling full" in text
    assert "--slot-free-policy" in text
    assert "--wmrm-full-language-tokens" in text
    assert "scratch_v5d16_s5152.pt" in text
    assert '--resume-weights "$BASE_CHECKPOINT"' in text
    assert "--wmrm-stage-gate-start 7" in text
    assert "wmrm_predictor_depth=7" in text
    assert "wmrm_predictor_copies=11" in text
    assert "--wmrm-predictor-copies" in text
    assert "wmrm_feature_metric=cosine" in text
    assert "peer_va8_world7_to_va16_world15_gated_capacity_v1" in text
    assert '"${CAPACITY_PHASE2_GATES:-0}" == 1' in text
    assert "CAPACITY_PHASE2_GATES requires CAPACITY_RESUME_EXPANDED=1" in text
    assert "--capacity-phase2-gates" in text
    assert 'va_layers=16' in text
    assert "from_s3224_e50_v1" in text
    assert 'run_steps=$((STEPS - completed))' in text
    assert 'resume_args=(--resume-exact "$save")' in text
    assert "resume) run_joint exact" in text


def test_mt50_acceptance_is_50_task_evomind_h15_protocol() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(MT50_ACCEPTANCE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    text = MT50_ACCEPTANCE.read_text(encoding="utf-8")
    assert "/root/private_data/ORA0/checkpoints/mw_mt50_antiforget" in text
    assert "/root/private_data/ORA0/features/all49_peer_h15_p15_eval_v1.pt" in text
    assert "/root/private_data/ORA0/runtime_libs/osmesa_jammy" in text
    assert "mt50_language_normalization_ref_v2.pt" in text
    assert "--trials-per-task 10 --episode-seed-base 4042" in text
    assert "--execution-horizon 15 --horizon 400 --mt50-benchmark" in text
    assert "--task-ids" not in text


def test_rlt_runner_uses_native_dense_h15_contract() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RLT_RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    text = RLT_RUNNER.read_text(encoding="utf-8")
    assert "CHUNK_LENGTH=${CHUNK_LENGTH:-15}" in text
    assert "REPLAY_STRIDE=${REPLAY_STRIDE:-15}" in text
    assert "PREFILL_EPISODES_PER_TASK=${PREFILL_EPISODES_PER_TASK:-0}" in text
    assert "REWARD_MODE=${REWARD_MODE:-dense}" in text
    assert '--reward-mode "$REWARD_MODE"' in text
