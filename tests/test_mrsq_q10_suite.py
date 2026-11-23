"""Task-specific Q10 manifests preserve their Q5 training recipes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from slurm.dog_stand_mrsq import REDUCTIONS
from slurm import mrsq_q10_suite
from tests.test_ambi_root_local_sac import _build_cfg


CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'configs/dmcontrol'
TASKS = ('humanoid-walk', 'humanoid-run', 'dog-trot', 'dog-run', 'humanoid-stand', 'dog-walk')
CAMPAIGN = 'mrsq-q10-suite-20261122'


def read_config(directory, basename):
    def unique_keys(items):
        data = dict(items)
        assert len(data) == len(items), f'Duplicate JSON key in {basename}.json'
        return data

    path = CONFIG_ROOT / directory / f'{basename}.json'
    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


@pytest.mark.parametrize('task', TASKS)
def test_q10_matches_task_q5_except_critic_count_and_publication_labels(task):
    q5_name = f"ambi_{task.replace('-', '_')}_mrsq_q5_seed55"
    q10_name = q5_name.replace('_q5_', '_q10_')
    q5 = read_config('algs', q5_name)
    q10 = read_config('algs', q10_name)
    expected = deepcopy(q5)
    q5_params = q5['alg_params']
    run_name = q5_params['wandb_run_name'].replace('Q5', 'Q10')
    tag_changes = {'q5': 'q10', 'mrsq-q5-suite-20261118': CAMPAIGN, 'oscar-l40s': 'oscar',
                   'dog-walk-mrsq-q5-20261118': CAMPAIGN}
    tags = [tag_changes.get(tag, tag) for tag in q5_params['wandb_tags']]
    if task == 'dog-walk':
        run_name += ' | fresh 1M attempt'
        tags.append('fresh-1m-attempt')
    expected['alg_params'].update(
        num_q=10,
        q_pair_size=10,
        wandb_run_name=run_name,
        wandb_group=f'{CAMPAIGN}-{task}',
        wandb_tags=tags,
    )
    assert q10 == expected

    q5_manifest = read_config('experiments', q5_name)
    q10_manifest = read_config('experiments', q10_name)
    descriptions = ('name', 'study_type', 'study_note', 'description')
    protocol = {key: value for key, value in q10_manifest.items() if key not in descriptions}
    assert protocol == {key: value for key, value in dict(q5_manifest, configs=[q10_name]).items()
                        if key not in descriptions}
    assert q10_manifest['name'] == q5_manifest['name'].replace('Q5', 'Q10')
    assert q10_manifest['env_params'] == {'task': task, 'obs': 'state', 'render_mode': None}
    assert q10_manifest['trials'] == 1 and q10['seed'] == 55
    assert q10['total_steps'] == q10_manifest['overrides_alg']['total_steps'] == 1_000_000
    assert q10['checkpoint_every'] == q10_manifest['checkpoint_every'] == 25_000
    assert q10['save_strat'] == q10_manifest['save_strat'] == 'all'

    cfg = _build_cfg(**q10['alg_params'])
    assert cfg.num_q == cfg.q_pair_size == 10
    assert all(getattr(cfg, name) == reduction for name, reduction in REDUCTIONS.items())
    assert cfg.inner_execution_action == cfg.inner_behavior_action == 'policy_sample'
    assert cfg.critic_ensemble_backend == 'vmap' and cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.critic_geometry_batch_size == 64
    assert (cfg.critic_geometry_obs_noise_std, cfg.critic_geometry_action_noise_std) == (0.01, 0.02)


@pytest.mark.parametrize('task,environment_test', [
    ('dog-walk', 'tests/test_dog_walk_mrsq.py::test_real_dog_walk_two_control_steps'),
    ('humanoid-stand', 'tests/test_mrsq_q5_suite.py::test_real_task_action_repeat[humanoid-stand]'),
])
def test_new_tasks_route_real_environment_gate(task, environment_test, tmp_path, monkeypatch):
    report = tmp_path / 'parity.xml'

    def planned_test_run(command, *, cwd, env, check):
        family = task.split('-')[0]
        assert command[-3:] == [f'tests/test_mrsq_q10_suite.py::test_cuda_q10_family_parity[{family}]',
                               environment_test, f'--junitxml={report}']
        assert env['AMBI_RUN_REAL_DMCONTROL_TESTS'] == '1' and check
        report.write_text('<testsuite><testcase/><testcase/></testsuite>')

    monkeypatch.setattr(mrsq_q10_suite.subprocess, 'run', planned_test_run)
    mrsq_q10_suite.verify_parity(task, report)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires scheduled CUDA GPU')
@pytest.mark.parametrize('family', ('dog', 'humanoid'))
def test_cuda_q10_family_parity(family, monkeypatch):
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from tests import test_inner_entropy_graph as checks

    original = checks.make_agent
    obs_dim, action_dim = {'dog': (223, 38), 'humanoid': (67, 21)}[family]

    def make_agent(entropy, scope, update, *, device, compiled):
        cfg = deepcopy(original(entropy, scope, update).cfg)
        cfg.action_dim = action_dim
        cfg.action_dims = [action_dim]
        cfg.obs_shape = {'state': (obs_dim,)}
        cfg.obs_shapes = [{'state': (obs_dim,)}]
        cfg.ent_coef = cfg.inner_temperature = .0021
        cfg.target_entropy = cfg.inner_target_entropy = -float(action_dim)
        cfg.num_q = cfg.q_pair_size = 10
        for name, reduction in REDUCTIONS.items():
            setattr(cfg, name, reduction)
        cfg.device = device
        cfg.compile = cfg.compile_strict = compiled
        agent = AMBITDMPC2Agent(cfg)
        agent.model.eval()
        agent.actor_loss_scale.fill_(4.)
        return agent

    monkeypatch.setattr(checks, 'make_agent', make_agent)
    checks.test_cuda_capture_matches_uncaptured_normalized_solve('squashed', 'action', 'per_update')
