"""The launch pair differs only in the selected entropy and its unit conversion."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prior_entropy_pair_and_checkpoint_budget():
    manifest = json.loads((ROOT / 'configs/dmcontrol/experiments/ambi_prior_entropy.json').read_text())
    assert manifest['trials'] == 1 and len(manifest['configs']) == 2
    assert manifest['overrides_alg']['total_steps'] == 1_000_000
    assert manifest['checkpoint_every'] == 25_000 and manifest['save_strat'] == 'all'
    assert manifest['env_params']['task'] == 'humanoid-walk'
    configs = [json.loads((ROOT / f'configs/dmcontrol/algs/{name}.json').read_text()) for name in manifest['configs']]
    for config in configs:
        assert config['seed'] == 55 and config['total_steps'] == 1_000_000
        assert config['checkpoint_every'] == 25_000 and config['save_strat'] == 'all'
        p = config['alg_params']
        assert p['inner_operator'] == 'none' and not p['mpc']
        assert p['inner_rounds'] == p['inner_rollouts_per_round'] == p['inner_updates_per_round'] == 0
        assert p['outer_critic_target'] == 'reward_only' and p['aux_return_mode'] == 'off'
        assert p['sac_actor_loss_scale_mode'] == 'tdmpc2_percentile_range'
        assert p['sac_actor_loss_scale_tau'] == .01
        assert p['num_q'] == 5 and p['q_pair_size'] == 2
        assert p['log_std_mapping'] == 'tdmpc2_tanh'
        assert p['compile'] and p['compile_strict'] and p['critic_ensemble_backend'] == 'vmap'
        assert not p['critic_geometry_diagnostics']
    pre, post = [c['alg_params'] for c in configs]
    assert pre['outer_actor_entropy_mode'] == 'tdmpc2_scaled'
    assert post['outer_actor_entropy_mode'] == 'squashed'
    assert pre['ent_coef'] == .0001 and abs(post['ent_coef'] - 21 * pre['ent_coef']) < 1e-12
    different = {k for k in pre if pre[k] != post[k]}
    assert different == {'outer_actor_entropy_mode', 'ent_coef', 'wandb_run_name', 'wandb_tags'}
