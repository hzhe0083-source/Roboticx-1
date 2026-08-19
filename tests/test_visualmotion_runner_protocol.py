from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_mw_hard2_wam4va_visualmotion_joint_v1.sh"
RESEARCH_RUNNER = (
    ROOT / "scripts" / "run_mw_hard2_wam4va_visualmotion_gap_ab_v1.sh"
)


def _text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _research_text() -> str:
    return RESEARCH_RUNNER.read_text(encoding="utf-8")


def _block(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_runner_bash_syntax_and_usage_fail_before_launch() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    invalid = subprocess.run(
        ["bash", str(RUNNER), "invalid-mode"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "{smoke10|pilot300|qualify50|qualify|20k}" in invalid.stderr


def test_qualification_and_formal_are_distinct_scratch_lineages() -> None:
    text = _text()
    assert "FAMILY=mw_hard2_wam4va_visualmotion_joint_v12" in text
    assert "mw_hard2_wam4va_visualmotion_joint_v11" not in text
    assert "SMOKE10_RUN_ID=${FAMILY}.smoke10" in text
    assert "PILOT300_RUN_ID=${FAMILY}.pilot300" in text
    assert "QUAL50_RUN_ID=${FAMILY}.qualification50" in text
    assert "QUAL_RUN_ID=${FAMILY}.qualification" in text
    assert "LONG_RUN_ID=${FAMILY}.long20k" in text
    assert "GATE_STEPS=(50 300 1000)" in text
    assert "qualification50" in text
    assert "for target_step in 50 300" in text
    assert "diagnostic gates do not launch 20k" in text
    preflight = _block(text, "validate_train_cli() {", "refuse_output_family() {")
    assert 'WORLD_SUPERVISION_CONTRACT != "visual_motion_constrained_v5"' in preflight
    assert '"all": 0.25' in preflight
    assert '"motion": 0.25' in preflight
    assert '"top20": 0.50' in preflight
    assert "WORLD_STAGE_AUXILIARY_DECAY != 0.25" in preflight
    assert "WORLD_NO_REGRESSION != {" in preflight
    assert '"components": ["all"]' in preflight
    assert "WORLD_STATIC_COPY_CONSTRAINT != {" in preflight
    assert '"penalty": "stage_chain_exact_hinge_v1"' in preflight
    assert '"reduction": "sum_stages_then_masked_transition_mean"' in preflight
    assert '"boundary": "copy_then_detached_min_previous_copy"' in preflight
    assert "WORLD_ACTION_RANKING != {" in preflight
    assert '"stage": "full_8stage_counterfactual_final"' in preflight
    assert '"top10_min_relative_margin": 0.05' in preflight
    assert '"top10_strong_relative_margin": 0.10' in preflight
    assert '"weight": 4.0' in preflight
    assert '"weight": 1.0' in preflight
    assert '"schedule": "both_each_valid_transition"' in preflight
    assert '"mask": "per_negative_and_both_for_strong"' in preflight
    assert '"gradient": "wrong_actions_only_detached_real_margin_v1"' in preflight
    assert "visual_motion_constrained_v4" not in text

    lineage = _block(text, "run_gated_lineage() {", "write_probe_receipt() {")
    assert "local start_step=0" in lineage
    assert "local source_checkpoint=scratch" in lineage
    assert lineage.index("run_training_segment") < lineage.index("run_heldout_gate")
    assert "stopping before further training" in lineage

    modes = text.split('case "$MODE" in', 2)[-1]
    qualify = _block(modes, "  qualify)", "  20k)")
    formal = modes.split("  20k)", 1)[1]
    assert 'refuse_output_family "$QUAL_RUN_ID"' in qualify
    assert 'run_gated_lineage "$QUAL_RUN_ID"' in qualify
    assert formal.index("verify_qualification") < formal.index(
        'refuse_output_family "$LONG_RUN_ID"'
    )
    assert formal.index('refuse_output_family "$LONG_RUN_ID"') < formal.index(
        'run_gated_lineage "$LONG_RUN_ID"'
    )
    assert formal.index('run_gated_lineage "$LONG_RUN_ID"') < formal.index(
        "1000 20000"
    )


def test_resume_steps_are_additional_and_outputs_are_unique() -> None:
    text = _text()
    training = _block(text, "run_training_segment() {", "run_heldout_gate() {")
    assert "additional_steps=$((target_step - start_step))" in training
    assert '--steps "$additional_steps"' in training
    assert 'resume_args=(--resume-exact "$source_checkpoint")' in training
    assert '[[ "$source_checkpoint" != "$save" ]]' in training
    assert "exact-resume segment modified its source checkpoint" in training
    assert "scratch -> 50 (+50) -> 300 (+250) -> 1000 (+700)" in text
    assert "1000 -> 20000 (+19000)" in text
    assert "--resume-weights" not in text
    assert "world_10k.pt" not in text
    assert "causalfix" not in text
    assert "--wam4va --wmrm-inject all" in training
    assert "--feature-autocast-bf16" in training
    assert "--task-sampling balanced" in training
    assert "--task-locality-block-batches 4" in training


def test_qualification50_is_isolated_fresh_batch3_lineage() -> None:
    text = _text()
    assert 'BATCH=${2:-${WAM4VA_BATCH_SIZE:-3}}' in text
    short = _block(text, "run_qualification50() {", "run_pilot300() {")
    assert 'milestone_checkpoint "$QUAL50_RUN_ID" 50' in short
    assert "run_training_segment 0 50 scratch" in short
    assert 'run_heldout_gate "$checkpoint" "$gate_report" "$gate_log" 50' in short
    assert "300" not in short
    modes = text.split('case "$MODE" in', 2)[-1]
    short_mode = _block(modes, "  qualify50)", "  qualify)")
    assert 'refuse_output_family "$QUAL50_RUN_ID"' in short_mode
    assert "run_qualification50" in short_mode


def test_formal_gates_precede_1k_archived_continuation() -> None:
    text = _text()
    training = _block(text, "run_training_segment() {", "run_heldout_gate() {")
    assert 'local save_step_copies=${6:-0}' in training
    assert 'save_copy_args=(--save-step-copies)' in training
    assert '"${save_copy_args[@]}"' in training
    assert 'for ((step = 2000; step <= 20000; step += 1000)); do' in text
    assert 'verify_checkpoint "$(milestone_checkpoint "$LONG_RUN_ID" 1000)" 1000' in text
    formal = text.split("  20k)", 1)[1]
    continuation_index = formal.index("run_training_segment")
    assert formal.index('run_gated_lineage "$LONG_RUN_ID"') < continuation_index
    assert formal.index("1000 20000", continuation_index) > continuation_index
    assert continuation_index < formal.index("verify_long_archives")
    assert '"logs/${LONG_RUN_ID}.train_step1000_to_step20000.log"' in formal
    assert formal.count("      1") >= 1


def test_gate_recheck_is_raw_full_heldout_and_visualmotion_only() -> None:
    text = _text()
    gate = _block(text, "verify_gate_report() {", "milestone_checkpoint() {")
    for field in (
        '"n_tasks": task_macro.get("n_tasks")',
        '"task_ids": task_macro.get("task_ids")',
        '"bootstrap": task_macro.get("bootstrap")',
        "full_heldout_evaluation=True",
        "checkpoint_world_supervision_valid=True",
        "checkpoint_world_logged_branch_valid=True",
        'protocol.get("full_heldout_evaluation") is not True',
        'protocol.get("max_transitions_per_task") is not None',
        '"world_supervision": "visual_motion_constrained_v5"',
        '"world_logged_branch": "matched_context_full_forward_v1"',
        '"world_no_regression": {',
        '"world_static_copy_constraint": {',
        '"world_action_ranking": {',
    ):
        assert field in gate
    assert "independent_full_forward_v1" not in text
    assert 'contract.get("split_manifest_sha256")' in gate


def test_checkpoint_gate_and_resume_receipt_are_bound_to_v12_v5() -> None:
    text = _text()
    checkpoint = _block(text, "verify_checkpoint() {", "verify_gate_report() {")
    gate = _block(text, "verify_gate_report() {", "milestone_checkpoint() {")
    receipt = _block(text, "write_probe_receipt() {", "run_qualification_probe() {")

    for block in (checkpoint, gate, receipt):
        assert '"world_supervision": "visual_motion_constrained_v5"' in block
        assert "visual_motion_constrained_v4" not in block
    assert '"run_family": "mw_hard2_wam4va_visualmotion_joint_v12"' in receipt
    assert (
        'receipt.get("run_family") != "mw_hard2_wam4va_visualmotion_joint_v12"'
        in receipt
    )
    assert '"penalty": "stage_chain_exact_hinge_v1"' in checkpoint
    assert '"gradient": "wrong_actions_only_detached_real_margin_v1"' in gate


def test_research_ab_runner_bash_syntax_and_fixed_batch3() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RESEARCH_RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    invalid = subprocess.run(
        ["bash", str(RESEARCH_RUNNER), "invalid-mode"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "{preflight|smoke10|pilot300|20k}" in invalid.stderr

    text = _research_text()
    assert (
        "A_FAMILY=mw_hard2_wam4va_visualmotion_oraclestgapfinal_v16" in text
    )
    assert (
        "B_FAMILY=mw_hard2_wam4va_visualmotion_oraclestgapcycle_v16" in text
    )
    assert "BATCH=3" in text
    assert "WAM4VA_BATCH_SIZE" not in text
    assert '--batch-size "$BATCH"' in text
    assert '--world-action-rank-stage "$stage_mode"' in text
    assert 'WAM4VA_RESEARCH_VARIANT:-both' in text
    assert 'final) printf \'%s\\n\' "$A_FAMILY"' in text
    assert 'cycle) printf \'%s\\n\' "$B_FAMILY"' in text


def test_research_ab_uses_one_global_lock_and_preflights_both_outputs() -> None:
    qualification = _text()
    research = _research_text()
    lock = 'exec 9>"/tmp/ora0_wam4va_visualmotion_train.lock"'
    assert lock in qualification
    assert lock in research
    assert "/tmp/ora0_${FAMILY}.lock" not in qualification
    assert "preflight_output_families()" in research
    preflight = _block(
        research,
        "preflight_output_families() {",
        "require_no_trainer() {",
    )
    assert 'for variant in "${VARIANTS[@]}"' in preflight
    assert 'refuse_output_family "$(run_id_for "$variant" "$mode")"' in preflight
    launch = research.rsplit("\nvalidate_fixed_split\n", 1)[1]
    assert launch.index('preflight_output_families "$MODE"') < launch.index(
        'run_20k_variant "$variant"'
    )


def test_research_ab_contracts_are_explicit_and_variant_local() -> None:
    text = _research_text()
    preflight = _block(text, "validate_train_cli() {", "refuse_output_family() {")
    checkpoint = _block(text, "verify_checkpoint() {", "milestone_checkpoint() {")
    assert (
        'WORLD_SUPERVISION_CONTRACT != "visual_motion_oracle_stgap_v7"'
        in preflight
    )
    assert (
        '"penalty": "copy_budget_hinge_plus_always_copy_anchor_v1"'
        in preflight
    )
    assert '"reduction": "stage_aux_weighted_masked_mean"' in preflight
    assert '"boundary": "1.00_detached_copy_each_stage"' in preflight
    assert 'for stage_mode in ("final", "cycle")' in preflight

    for value in (
        '"final_direct_matched_context"',
        '"rotating_8stage_direct_matched_context"',
        '"final_each_valid_transition"',
        '"(global_step+time_index)%num_stages"',
        '"oracle_motion_straight_through_exact_gap_v1"',
        '"diagnostic_negatives": ["zero"]',
        '"negatives": ["shuffle"]',
        '"world_supervision": "visual_motion_oracle_stgap_v7"',
    ):
        assert value in checkpoint
    assert 'arguments.get("world_action_rank_stage") != stage_mode' in checkpoint
    assert 'sampler.get("batch_size") != 3' in checkpoint


def test_research_ab_no_go_is_diagnostic_not_a_launch_gate() -> None:
    text = _research_text()
    heldout = _block(text, "run_heldout_gate() {", "run_diagnostic_milestones() {")
    lineage = _block(text, "run_diagnostic_milestones() {", "run_smoke_variant() {")
    assert '"${pipeline_status[0]}" -ne 2' in heldout
    assert "research gate NO-GO" in heldout
    assert "diagnostics preserved, continuing" in heldout
    assert "verify_gate_report" in heldout
    assert "run_heldout_gate" in lineage
    assert "exit \"$gate_status\"" not in text
    assert "verify_qualification" not in text
    assert "qualification prerequisites" not in text


def test_research_ab_20k_is_ordered_exact_resume_with_rolling_save() -> None:
    text = _research_text()
    training = _block(text, "run_training_segment() {", "verify_gate_report() {")
    formal = _block(text, "run_20k_variant() {", "validate_fixed_split")
    assert "additional_steps=$((target_step - start_step))" in training
    assert '--steps "$additional_steps" --save-every 1000' in training
    assert 'resume_args=(--resume-exact "$source_checkpoint")' in training
    assert "--resume-weights" not in training
    assert "--save-step-copies" not in text
    assert 'run_diagnostic_milestones "$run_id" "$stage_mode" 50 300 1000' in formal
    assert formal.index("run_diagnostic_milestones") < formal.index(
        '"$stage_mode" 1000 20000'
    )
    assert 'rolling="checkpoints/${run_id}.pt"' in formal
    assert 'verify_checkpoint "$save" "$target_step" "$stage_mode"' in training
    assert "run_heldout_gate" in formal
    assert (
        '"$rolling" "$final_report" "$final_gate_log" 20000 "$stage_mode"'
        in formal
    )
