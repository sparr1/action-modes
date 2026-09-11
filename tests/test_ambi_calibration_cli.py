"""End-to-end diagnostics: pairing and file boundaries without external publication."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import evaluate_ambi_calibration as calibration
from utils.ambi_research import load_preset_matrix
from utils.ambi_benchmark import read_json
from tests.test_ambi_root_local_sac import _tiny_params, _build_cfg


def test_reference_config_has_one_axis_controls_and_fixed_scale():
    matrix = load_preset_matrix(calibration.DEFAULT_MATRIX)
    params = matrix['shared_alg_params']
    base = json.loads(Path('configs/dmcontrol/algs/td_ambi_prior_reward_qscale.json').read_text())['alg_params']
    merged = {**base, **{k:v for k,v in params.items() if v is not None}}
    for key, value in params.items():
        if value is None:
            merged.pop(key, None)
    cfg = _build_cfg(**merged)
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_rollout_horizon, cfg.inner_batch_size) == (5,512,3,512)
    assert cfg.inner_replay_capacity == 7680
    assert cfg.inner_steps_per_update == 512 and cfg.inner_update_timing == 'step'
    assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == 'random'
    assert cfg.inner_actor_loss_scale_update == 'per_action'
    assert cfg.inner_temperature == 1e-4 and cfg.inner_temperature_mode == 'fixed'
    assert cfg.inner_finite_horizon and cfg.inner_bootstrap_source == 'inner_target'
    assert cfg.inner_sac_critic_target == 'reward_only'
    assert cfg.inner_actor_entropy_mode == 'tdmpc2_scaled'
    variants = matrix['comparisons']['update_dose']['variants']
    assert variants['x2']['alg_params'] == {'inner_steps_per_update':256}
    assert variants['x4']['alg_params'] == {'inner_steps_per_update':128}
    assert matrix['checkpoint_contract']['step'] == 2000000


def test_configurable_coverage_and_round_selection():
    matrix = load_preset_matrix(calibration.DEFAULT_MATRIX)
    cfg = SimpleNamespace(inner_rounds=5)
    default = calibration.calibration_options(matrix, cfg)
    assert default['decisions'] == [0,100,200,300,400]
    assert default['solver_repetitions'] == 3 and default['rollout_repetitions'] == 4
    assert calibration.calibration_options(matrix,cfg,every_n=200)['decisions'] == [0,200,400]
    assert calibration.calibration_options(matrix,cfg,every_decision=True,max_steps=3)['decisions'] == [0,1,2]
    assert calibration.calibration_options(matrix,cfg,decisions=[3],rounds=[0,5])['rounds'] == [0,5]
    with pytest.raises(ValueError,match='only one'):
        calibration.calibration_options(matrix,cfg,decisions=[0],every_n=5)
    with pytest.raises(ValueError,match='Rounds'):
        calibration.calibration_options(matrix,cfg,rounds=[6])
    with pytest.raises(ValueError,match='Decisions'):
        calibration.calibration_options(matrix,cfg,decisions=[500])


def test_noise_and_reference_cache_are_paired_and_verified(tmp_path):
    kwargs = dict(horizon=3,tail_steps=4,rollouts=2,action_dim=1)
    first = calibration.paired_noise(55,'root',**kwargs)
    second = calibration.paired_noise(55,'root',**kwargs)
    assert np.array_equal(first[0],second[0]) and np.array_equal(first[1],second[1])
    identity = {'checkpoint':'a','root':'b','protocol':1}
    calls = []
    def compute():
        calls.append(1)
        return {'rows':[{'value':4}]}
    value, hit, path = calibration._cache_reference(tmp_path,identity,compute)
    assert not hit
    assert calibration._cache_reference(tmp_path,identity,compute)[1]
    assert len(calls)==1
    record = read_json(path)
    record['result']['rows'][0]['value'] = 5
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='corrupted'):
        calibration._cache_reference(tmp_path,identity,compute)


def _humanoid_checkpoint(tmp_path):
    import gymnasium as gym
    import domains
    from RL.AMBITDMPC2 import AMBITDMPC2
    params = _tiny_params(inner_rounds=2,inner_rollouts_per_round=2,inner_rollout_horizon=2,
        inner_replay_capacity=8,inner_batch_size=2,inner_steps_per_update=2,inner_update_timing='step',
        inner_actor_initialization='random',inner_critic_initialization='random',
        inner_critic_target_initialization='online',inner_finite_horizon=True,
        inner_temperature_mode='fixed',inner_temperature=1e-4,outer_critic_target='reward_only',
        inner_sac_critic_target='reward_only',inner_temperature_initialization='fixed')
    params.pop('inner_updates_per_round')
    env = gym.make('DMControl-v0',task='humanoid-walk',obs='state')
    config = dict(alg='AMBITDMPC2/AMBITDMPC2',env='DMControl-v0',seed=13,device='cpu',total_steps=10,alg_params=params)
    model = AMBITDMPC2('AMBITDMPC2',env,params,config,{})
    checkpoint = tmp_path/'tiny.pt'
    model.agent.save(str(checkpoint))
    env.close()
    sidecar = dict(schema_version=1,trial_run_params=config,
        experiment_params={'env_params':{'task':'humanoid-walk','obs':'state'}},
        checkpoint=dict(kind='periodic',step=10,episode=2,best_score=None,best_window=1))
    Path(str(checkpoint)+'.metadata.json').write_text(json.dumps(sidecar))
    matrix = dict(schema_version=1,base_alg_config='checkpoint',
        evaluation=dict(seeds=[101,102],controller_seed=55,max_steps=3,togo_return_rollouts=2,
                        default_presets=['init/scratch']),
        real_calibration=dict(decisions=[0,2],rounds=[0,1,2],solver_repetitions=2,rollout_repetitions=2,
                              tail_steps=4,bootstrap_resamples=20),
        comparisons={'init':{'reference':'scratch','variants':{'scratch':{'alg_params':{}},
            'inherited':{'alg_params':{'inner_actor_initialization':'prior','inner_critic_initialization':'prior'}}}}})
    matrix_path=tmp_path/'matrix.json'
    matrix_path.write_text(json.dumps(matrix))
    return checkpoint,matrix_path


@pytest.mark.skipif(os.environ.get('AMBI_RUN_REAL_DMCONTROL_TESTS') != '1',reason='opt-in real DMControl runtime')
def test_real_calibration_full_bundle_and_reference_reuse(tmp_path):
    from utils.ambi_diagnostic_series import read_diagnostic_bundle, extract_diagnostic_html_data
    checkpoint,matrix = _humanoid_checkpoint(tmp_path)
    root_bank=tmp_path/'roots.json'
    cache=tmp_path/'reference-cache'
    first=calibration.run_calibration(matrix,checkpoint,bundle_dir=tmp_path/'scratch',attempt_label='smoke',
        save_root_bank=root_bank,reference_cache=cache,benchmark_repetitions=2)
    assert first['status']=='complete'
    assert len(first['rows'])==4*2*2*3
    assert [s['actor_updates'] for s in first['summaries']]==[0,2,4]
    assert first['timing']['outer_state_unchanged']
    assert first['timing']['branch_simulator_decisions']==4*2*6 + 4*2*3*2*6
    assert first['timing']['prior_reference_cache_hits']==0
    assert read_diagnostic_bundle(tmp_path/'scratch')==first
    assert extract_diagnostic_html_data((tmp_path/'scratch/report.html').read_text())==first
    second=calibration.run_calibration(matrix,checkpoint,preset='init/inherited',
        bundle_dir=tmp_path/'inherited',attempt_label='smoke',root_bank=root_bank,reference_cache=cache)
    assert second['timing']['prior_reference_cache_hits']==4
    assert second['timing']['branch_simulator_decisions']==4*2*3*2*6
    assert second['timing']['root_collection_seconds']==0
    for row in first['rows']:
        assert row['metrics']['model_return']==pytest.approx(row['metrics']['model_prefix_reward']+row['metrics']['model_bootstrap'])
        assert row['metrics']['real_mc_return']==pytest.approx(row['metrics']['real_prefix_reward']+row['metrics']['real_tail_contribution'])
        assert row['metrics']['bootstrap_prediction_error']==pytest.approx(row['metrics']['real_bootstrap']-row['metrics']['real_tail_contribution'])
        assert row['endpoint_action'] is not None
    roots=read_json(root_bank)['roots']
    assert len(roots)==4 and roots[0]['snapshot']['sha256']
    bad_bank=read_json(root_bank)
    bad_bank['roots'][0]['observation'][0]+=1
    bad_bank['id']=calibration.canonical_hash({k:v for k,v in bad_bank.items() if k!='id'})
    corrupted=tmp_path/'mismatched-observation.json'
    corrupted.write_text(json.dumps(bad_bank))
    with pytest.raises(ValueError,match='observation differs'):
        calibration.load_root_bank(corrupted,bad_bank['checkpoint_sha256'],bad_bank['protocol'])


def test_short_tail_cannot_claim_original_cutoff_returns():
    with pytest.raises(ValueError,match='too short'):
        calibration.calibration_options({},SimpleNamespace(inner_rounds=5,inner_rollout_horizon=3),
                                        tail_steps=100)


def test_disabled_benchmark_does_not_run_a_solve():
    assert calibration._benchmark_model_probes(None,None,55,32,0)==[]


def test_model_probe_episode_rounds_export_with_original_checkpoint_metadata(tmp_path):
    import evaluate_ambi_checkpoint as evaluator
    from tests.test_ambi_benchmark_evaluation import checkpoint_matrix
    from utils.ambi_diagnostic_series import record_from_model_bundle, write_diagnostic_bundle, read_diagnostic_bundle
    checkpoint,matrix_path=checkpoint_matrix.__wrapped__(tmp_path)
    matrix=read_json(matrix_path)
    matrix['evaluation']['togo_return_rollouts']=2
    matrix_path.write_text(json.dumps(matrix))
    output=tmp_path/'episodes'
    result=evaluator.evaluate_matrix(matrix_path,checkpoint,selectors=['budget/sac'],bundle_dir=output)
    run=read_json(output/'manifest.json')['runs'][0]
    assert len(run['togo_probe_rows'])==2*3*3
    assert [row['actor_updates'] for row in result['results'][0]['togo_round_summaries']]==[0,1,2]
    for episode in result['results'][0]['episodes']:
        assert len(episode['togo_round_summaries'])==3
    record=record_from_model_bundle(output,'budget/sac','model-smoke',bootstrap_resamples=20)
    assert record['status']=='complete'
    assert record['identity']['scope']=='controller_episode'
    assert record['identity']['checkpoint']['step']==10
    write_diagnostic_bundle(tmp_path/'export',record)
    assert read_diagnostic_bundle(tmp_path/'export')==record
