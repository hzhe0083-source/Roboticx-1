from __future__ import annotations
import importlib.util, subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'scripts/run_visualmotion_static_weight_2_validation_64_v1.sh'
ANALYZER=ROOT/'scripts/analyze_visualmotion_static_weight_2_validation.py'
spec=importlib.util.spec_from_file_location('static2', ANALYZER); MOD=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(MOD)
def line(step, grad=2.0, flow=.2, world=.3, static=.01, task=None, other_static=None):
 tasks = [task] if task else ['assembly-v3', 'door-unlock-v3']
 values = {'assembly-v3': static, 'door-unlock-v3': static if other_static is None else other_static}
 section = ' | '.join(f'{name}:all=.1/.2 static={values[name]}/{values[name]}' for name in tasks)
 return f'step={step} mode=bidir_va contract=single task=assembly-v3 loss=.4 flow={flow} world={world} grad={grad} world_task[{section}]'
def checkpoint():
 return {'model': {'weight': 1}, 'optimizer_state': {'kind': 'adamw', 'state_dict': {'state': {0: {}}, 'param_groups': [{}]}},
         'sampler_state': {**MOD.EXPECTED_SAMPLER, 'epoch': 1, 'batch_cursor': 2},
         'rng_state': {'python': 1, 'numpy': 1, 'torch_cpu': 1, 'torch_cuda': []},
         'exact_run_contract': {'contract_version': 1, 'arguments': dict(MOD.STATIC2_CONTRACT_VALUES), 'model_config': {'wmrm_detach_proposal_stage_state': True}, 'optimizer': {'kind': 'adamw'}},
         'exact_resume_version': 2, 'global_step': 12074}
def test_runner_protocol_is_nonlaunching_and_pinned():
 assert subprocess.run(['bash','-n',str(RUNNER)],capture_output=True,text=True).returncode==0
 text=RUNNER.read_text()
 for token in ('EXPECTED_SOURCE_STEP=12010','TARGET_STEP=12074','UPDATES=64','MIGRATION_ID=wmrm_static_constraint_weight_4_to_2_v1','--wmrm-world-weight 1.0','--wmrm-static-constraint-weight 2.0','--resume-exact-contract-migration "$MIGRATION_ID"','require_no_active_train','require_idle_gpu','--query-compute-apps=pid,process_name,used_memory','available_kib >= 8 * 1024 * 1024'):
  assert token in text
 assert 'exec 9>"$LOCK"' in text and '--save-step-copies' not in text
 assert 'run_visualmotion_world_weight_ab' not in text
 assert 'f580caa4c1588b2a9807f9b0ab746ac54259eaaa482cea16ce5001c30a382f11' in text
def test_parser_requires_exact_64_steps_and_both_tasks():
 records=MOD.parse_log_text('\n'.join(line(s) for s in range(12011,12075)))
 assert len(records)==64 and records[0]['step']==12011
 with pytest.raises(MOD.AnalysisError,match='update steps mismatch'):
  MOD.parse_log_text('\n'.join(line(s) for s in range(12011,12074)))
def test_analyzer_hard_gate_rejects_static_spike():
 records=MOD.parse_log_text('\n'.join(line(s,static=.03 if s==12074 else .01) for s in range(12011,12075)))
 result=MOD.analyze_records(records)
 assert result['decision']=='NO-GO' and result['gates']['last32_static_max_le_0_02'] is False
def test_parser_aggregates_one_task_per_step_across_steps():
 text='\n'.join(line(s, task='assembly-v3' if s % 2 else 'door-unlock-v3') for s in range(12011,12075))
 records=MOD.parse_log_text(text)
 assert all(len(record['static_by_task']) == 1 for record in records)
 assert MOD.analyze_records(records)['decision'] == 'PASS'

def test_analyzer_rejects_static_spike_for_one_task_even_when_mean_passes():
 records=MOD.parse_log_text('\n'.join(line(s, static=.0, other_static=.03 if s==12074 else .0) for s in range(12011,12075)))
 result=MOD.analyze_records(records)
 assert result['decision']=='NO-GO' and result['gates']['last32_static_max_le_0_02'] is False
 assert result['observed']['last32_static_max_by_task']['door-unlock-v3'] == .03

def test_final_checkpoint_contract_rejects_wrong_step_and_operational_migration():
 payload=checkpoint(); payload['global_step']=12073
 with pytest.raises(MOD.AnalysisError): MOD.validate_final_checkpoint(payload)
 payload=checkpoint(); payload['exact_run_contract']['arguments']['resume_exact_contract_migration']='wmrm_static_constraint_weight_4_to_2_v1'
 with pytest.raises(MOD.AnalysisError): MOD.validate_final_checkpoint(payload)
 payload=checkpoint(); payload['exact_run_contract']['model_config']['wmrm_detach_proposal_stage_state']=False
 with pytest.raises(MOD.AnalysisError): MOD.validate_final_checkpoint(payload)

def test_final_checkpoint_contract_accepts_exact_static2_payload():
 MOD.validate_final_checkpoint(checkpoint())
