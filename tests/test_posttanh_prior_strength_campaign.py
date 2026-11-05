"""Keep the strength sweep scientifically paired with the prior outer-adapt runs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from tests.test_ambi_root_local_sac import _build_cfg

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/dmcontrol/experiments/ambi_posttanh_prior_strength.json'


@pytest.mark.parametrize('method,key,original', [
    ('boom', 'outer_boom_coef', .42),
    ('tdmpc_square', 'outer_behavior_policy_kl_coef', 1.),
])
@pytest.mark.parametrize('suffix,multiplier', [('0p1', .1), ('0p01', .01)])
def test_coefficient_is_the_only_scientific_change(method, key, original, suffix, multiplier):
    source_name = f'ambi_posttanh_prior_outer_adapt_{method}'
    source = json.loads((ROOT/'configs/dmcontrol/algs'/f'{source_name}.json').read_text())
    target = json.loads((ROOT/'configs/dmcontrol/algs'/f'{source_name}_x{suffix}.json').read_text())
    expected = deepcopy(source)
    expected['alg_params'][key] = original * multiplier
    for value in (expected, target):
        value['alg_params'] = {k:v for k,v in value['alg_params'].items() if not k.startswith('wandb')}
    assert target == expected
    cfg = _build_cfg(**target['alg_params'])
    assert getattr(cfg, key) == pytest.approx(original * multiplier)
    assert cfg.inner_actor_loss_scale_scope == 'action'
    assert cfg.inner_actor_loss_scale_update == 'per_update'
    assert cfg.inner_cuda_graph and cfg.compile and cfg.compile_strict
    assert cfg.critic_ensemble_backend == 'vmap' and cfg.critic_geometry_diagnostics
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == 'squashed'
    assert cfg.ent_coef == .0021 and cfg.outer_behavior_policy_kl_q_threshold == 2.


def test_exact_four_cells_and_original_budget():
    m = json.loads(MANIFEST.read_text())
    assert m['configs'] == [f'ambi_posttanh_prior_outer_adapt_{method}_x{suffix}'
                           for method in ('boom', 'tdmpc_square') for suffix in ('0p1', '0p01')]
    assert m['trials'] == 1 and m['overrides_alg']['seed'] == 55
    assert m['overrides_alg']['total_steps'] == 1000000 and m['checkpoint_every'] == 25000
