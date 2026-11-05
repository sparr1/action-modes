import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm.bernoulli_checkpoint_campaign import MATRIX, SELECTOR, validate_result
from utils.ambi_research import resolve_preset
from tests.test_ambi_research_presets import _build_cfg


@pytest.mark.parametrize('mode', ['outer_frozen', 'outer_adapt'])
def test_checkpoint_budget_and_entropy_survive_bernoulli_override(mode):
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / f'configs/dmcontrol/algs/ambi_inner_entropy_posttanh_{mode}.json').read_text())
    context = SimpleNamespace(source=Path('metadata.json'), trial_run_params={**saved, 'env':'DMControl-v0'},
                              experiment_params={'env_params':{'task':'humanoid-walk','obs':'state','render_mode':None}})
    resolved = resolve_preset(MATRIX, SELECTOR, checkpoint_context=context)
    cfg = _build_cfg(resolved)
    assert cfg.inner_critic_bernoulli_p == .5
    assert cfg.inner_cuda_graph is False
    assert cfg.inner_actor_scope == cfg.inner_critic_scope == 'action'
    assert cfg.inner_critic_updates_per_action == 128
    assert cfg.inner_actor_updates_per_action == 32
    assert cfg.inner_model_step_budget == 3072
    assert cfg.inner_actor_loss_scale_update == saved['alg_params']['inner_actor_loss_scale_update']
    assert cfg.inner_actor_entropy_mode == saved['alg_params']['inner_actor_entropy_mode']
    assert resolved['evaluation']['execution_backend'] == 'native'
    allowed = set(json.loads(MATRIX.read_text())['shared_alg_params']) | {'inner_critic_bernoulli_p'}
    for key, value in saved['alg_params'].items():
        if key not in allowed:
            assert resolved['algorithm_config']['alg_params'][key] == value


def test_result_gate_rejects_missing_transfer_and_outer_mutation():
    metrics = {'inner_critic_bernoulli_p':.5,'inner_critic_bernoulli_applied':.998,
               'inner_critic_optimizer_steps':128.,'inner_actor_optimizer_steps':32.,
               'inner_model_steps':3072.,'inner_compile_fallback':0.}
    result = {'selector':SELECTOR,'outer_state_unchanged':True,'outer_updates_before':0,
              'outer_updates_after':0,'environment_seeds':[101],
              'episodes':[{'length':500,'model_metrics':metrics}]}
    assert validate_result({'results':[result]},[101],500) is result
    metrics['inner_critic_bernoulli_applied'] = 0
    with pytest.raises(ValueError, match='bernoulli_applied'):
        validate_result({'results':[result]},[101],500)
    result['outer_state_unchanged'] = False
    with pytest.raises(ValueError, match='Frozen state'):
        validate_result({'results':[result]},[101],500)
