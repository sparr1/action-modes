"""The six-run screen preserves its references and changes only gSDE scope."""
import json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
from RL.AMBITDMPC2 import AMBITDMPC2
from slurm.gsde_noise_scopes_oscar import MANIFEST, recipe, select_plan, parity_cases

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('index', range(6))
def test_resolved_gsde_campaign_budget_and_objectives(index):
    name, manifest, run = recipe(index, MANIFEST)
    params = deepcopy(run['alg_params'])
    params.update(device='cpu', wandb=False)
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = SimpleNamespace(observation_type='state',
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(67,), dtype=np.float32),
        action_space=gym.spaces.Box(-1., 1., shape=(21,), dtype=np.float32),
        spec=SimpleNamespace(max_episode_steps=500))
    algorithm.run_params = {**run, 'device':'cpu'}
    algorithm.custom_params = params
    cfg = algorithm._build_cfg(params)
    assert cfg.actor_distribution == 'gsde' and cfg.gsde_feature_source == 'actor'
    assert cfg.gsde_log_std_init == -3. and cfg.gsde_mean_clip == 2.
    assert cfg.inner_gsde_noise_scope == ('round','solve','real_steps')[index % 3]
    assert cfg.inner_gsde_resample_steps == cfg.gsde_execution_resample_steps == 8
    mode, alpha = ('tdmpc2_scaled', .0001) if index < 3 else ('squashed', .0021)
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == mode
    assert cfg.ent_coef == cfg.inner_temperature == alpha
    assert cfg.inner_temperature_mode == 'inherit_outer'
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_rounds == 8 and cfg.inner_rollout_horizon == 3
    assert cfg.inner_rollouts_per_round == 128 and cfg.inner_batch_size == 256
    assert cfg.inner_actor_updates_per_action == 32 and cfg.inner_critic_updates_per_action == 128
    assert cfg.steps == 1000000 and cfg.seed == 55
    assert run['checkpoint_every'] == manifest['checkpoint_every'] == 25000
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == 'reward_only'
    assert cfg.inner_actor_loss_scale_scope == 'action' and cfg.inner_actor_loss_scale_update == 'per_update'
    assert cfg.inner_finite_horizon and cfg.inner_solve_interval == 1
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
    assert cfg.critic_ensemble_backend == 'vmap' and cfg.critic_geometry_diagnostics
    assert cfg.critic_geometry_every == 1000


def test_only_scope_and_publication_identity_change_within_each_baseline():
    for start in (0,3):
        rows=[]
        for index in range(start,start+3):
            _, _, run = recipe(index, MANIFEST)
            params=deepcopy(run['alg_params'])
            for key in ('inner_gsde_noise_scope','wandb_run_name','wandb_tags'):
                params.pop(key)
            rows.append(params)
        assert rows[0] == rows[1] == rows[2]


def test_resource_plan_preserves_fastest_useful_throughput():
    rows=[{'index':0,'physical_cpus':cores,'measured':{'seconds_per_decision':seconds},
           'checkpoint_bytes':100,'peak_rss_kib':2*2**20} for cores,seconds in ((1,.2),(2,.199))]
    assert select_plan(rows)['cpus'] == 1
    rows[1]['measured']['seconds_per_decision'] = .15
    assert select_plan(rows)['cpus'] == 2


def test_cuda_evidence_rejects_skipped_tests(tmp_path):
    path=tmp_path/'junit.xml'
    cases=''.join(f'<testcase name="test_cuda_case_{i}" />' for i in range(7))
    path.write_text('<testsuites><testsuite>'+cases+'</testsuite></testsuites>')
    assert len(parity_cases(path)) == 7
    path.write_text('<testsuites><testsuite>'+cases.replace('name="test_cuda_case_0" />','name="test_cuda_case_0"><skipped /></testcase>')+'</testsuite></testsuites>')
    with pytest.raises(AssertionError): parity_cases(path)
