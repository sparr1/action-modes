"""Task changes must not silently change the linked run's learning recipe."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


@pytest.mark.parametrize('task', ['stand', 'run'])
def test_humanoid_task_is_only_scientific_change(task, monkeypatch):
    source = read('configs/dmcontrol/algs/ambi_posttanh_prior_outer_adapt_boom_x0p01.json')
    target = read(f'configs/dmcontrol/algs/ambi_boom_x0p01_humanoid_{task}.json')
    for config in (source, target):
        config['alg_params'] = {k: v for k, v in config['alg_params'].items() if not k.startswith('wandb')}
    assert target == source
    original = read('configs/dmcontrol/experiments/ambi_posttanh_prior_strength.json')
    manifest = read(f'configs/dmcontrol/experiments/ambi_boom_x0p01_humanoid_{task}.json')
    for key in ('overrides_alg', 'trials', 'logs', 'save_trials', 'checkpoint_every', 'save_strat', 'log_info', 'log_type'):
        assert manifest[key] == original[key]
    assert manifest['configs'] == [f'ambi_boom_x0p01_humanoid_{task}']
    assert manifest['env_params'] == {**original['env_params'], 'task': f'humanoid-{task}'}
    from slurm import posttanh_prior_fixes_oscar as base
    from slurm import boom_humanoid_oscar as launch
    monkeypatch.setattr(base, 'MANIFEST', ROOT / f'configs/dmcontrol/experiments/ambi_boom_x0p01_humanoid_{task}.json')
    monkeypatch.setattr(base, 'CAMPAIGN_ROOT', launch.CAMPAIGN / task)
    launch.verify_recipe(task)


def test_registered_tasks_have_separate_output_ownership():
    from slurm.posttanh_prior_fixes_oscar import CAMPAIGNS
    assert CAMPAIGNS['humanoid_stand'][1] != CAMPAIGNS['humanoid_run'][1]
    for task in ('stand', 'run'):
        manifest, output = CAMPAIGNS[f'humanoid_{task}']
        assert manifest == f'ambi_boom_x0p01_humanoid_{task}.json'
        assert output.endswith('/' + task)
