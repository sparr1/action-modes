"""Task-only recipe and Dog-shaped gSDE CUDA graph correctness."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from tests.test_ambi_root_local_sac import _build_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_dog_walk_preserves_completed_sampled_gsde8_recipe():
    folder = ROOT / 'configs/dmcontrol'
    original = json.loads((folder / 'algs/ambi_gsde_squashed_real_steps.json').read_text())
    target = json.loads((folder / 'algs/ambi_gsde8_dog_walk.json').read_text())
    for config in (original, target):
        config['alg_params'] = {k:v for k,v in config['alg_params'].items() if not k.startswith('wandb')}
    assert original == target
    manifest = json.loads((folder / 'experiments/ambi_gsde8_dog_walk.json').read_text())
    assert manifest['env_params'] == {'task':'dog-walk', 'obs':'state', 'render_mode':None}
    assert manifest['configs'] == ['ambi_gsde8_dog_walk'] and manifest['trials'] == 1
    cfg = _build_cfg(**target['alg_params'])
    assert target['total_steps'] == manifest['overrides_alg']['total_steps'] == 1000000
    assert target['seed'] == 55 and target['checkpoint_every'] == manifest['checkpoint_every'] == 25000
    assert cfg.inner_execution_action == 'policy_sample'
    assert cfg.inner_gsde_noise_scope == 'real_steps'
    assert cfg.inner_gsde_resample_steps == cfg.gsde_execution_resample_steps == 8
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires scheduled CUDA GPU')
def test_cuda_dog_gsde_graph_matches_uncaptured():
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
    from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph, _gsde_inputs_for_action
    from tests.test_ambi_gsde_graph import _cuda_agent
    from tests.test_ambi_distillation_collection_graph import _assert_close, _inner_snapshot
    from tests.test_ambi_inner_decoupling import _clone_tree
    torch._dynamo.reset()
    try:
        cfg = deepcopy(_cuda_agent('real_steps', 'round').cfg)
        cfg.obs_shape = {'state': (223,)}; cfg.obs_shapes = [cfg.obs_shape]
        cfg.action_dim = 38; cfg.action_dims = [38]
        cfg.inner_gsde_resample_steps = cfg.gsde_execution_resample_steps = 8
        cfg.outer_actor_entropy_mode = cfg.inner_actor_entropy_mode = 'squashed'
        cfg.outer_critic_target = cfg.inner_sac_critic_target = 'reward_only'
        cfg.ent_coef = cfg.inner_temperature = .0021
        source, target = AMBITDMPC2Agent(cfg), AMBITDMPC2Agent(cfg)
        target.load(deepcopy(source.checkpoint_state()))
        source.model.eval(); target.model.eval()
        root = source.model.encode(torch.zeros(1, 223, device='cuda')).detach()
        eager_worker, graph_worker = InnerImprovementEngine(source), InnerImprovementEngine(target)
        eager, graph = WholeInnerGraph(eager_worker, root, capture=False), WholeInnerGraph(graph_worker, root, capture=True)
        for index in range(9):
            controllers = (source.inner_engine, target.inner_engine)
            for controller in controllers:
                controller._begin_gsde_action(t0=index == 0, eval_mode=False)
            noises = [_gsde_inputs_for_action(controller, root) for controller in controllers]
            expected = _clone_tree(eager(root, gsde_collection=noises[0][0], gsde_execution=noises[0][1]))
            actual = _clone_tree(graph(root, gsde_collection=noises[1][0], gsde_execution=noises[1][1]))
            torch.cuda.synchronize()
            _assert_close(actual, expected)
            _assert_close(_inner_snapshot(graph_worker), _inner_snapshot(eager_worker))
            torch.testing.assert_close(eager.rng_state, graph.rng_state, rtol=0, atol=0)
            for controller in controllers:
                controller._gsde_execution_uses += 1
                controller._finish_gsde_action()
    finally:
        torch._dynamo.reset()
