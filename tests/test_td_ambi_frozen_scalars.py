"""Prior checkpoints retain scalar/target state across fresh frozen SAC roots."""

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_inner_decoupling import _assert_tree_equal


CONFIGS = Path(__file__).resolve().parents[1] / "configs/dmcontrol/algs"
FAMILIES = ("reward_qscale", "entropy_qscale", "entropy_autotemp", "reward_autotemp")


def _agent(family, *, adaptive=None):
    params = json.loads(
        (CONFIGS / f"td_ambi_prior_{family}.json").read_text()
    )["alg_params"]
    if adaptive is not None:
        # Receive only the full learner's inner settings. Outer objective and
        # portable checkpoint identity must remain those of the actual bank.
        full_family = family + "_frozen" if family.endswith("qscale") else family
        full = json.loads(
            (CONFIGS / f"td_ambi_full_{full_family}.json").read_text()
        )["alg_params"]
        params.update({key: value for key, value in full.items() if key.startswith("inner_")})
        params.update(
            inner_temperature_mode=(
                "auto" if adaptive and family.endswith("autotemp") else "inherit_outer"
            ),
            inner_temperature_initialization="inherit_outer",
            inner_target_entropy=-441.0,
            inner_temperature_grad_clip_norm=None,
            inner_actor_loss_scale_update=(
                "per_update" if adaptive and family.endswith("qscale") else "per_action"
            ),
            # Full J5/N512/H3 execution is exercised by the campaign tests.
            # This isolated load/reset test uses two updates per root.
            inner_rounds=1, inner_rollout_horizon=2,
            inner_rollouts_per_round=8, inner_batch_size=8,
            inner_replay_capacity=16, inner_steps_per_update=8,
            inner_execution_action="mean",
        )
    params.update(
        device="cpu", compile=False, wandb=False, model_size=None,
        enc_dim=32, mlp_dim=32, latent_dim=16, num_enc_layers=2,
        batch_size=4,
    )
    cfg = _build_cfg(**params)
    cfg.action_dim = 21
    cfg.obs_shape = {"state": (67,)}
    return AMBITDMPC2Agent(cfg)


def _assert_fresh_optimizer(optimizer):
    assert optimizer is not None
    for state in optimizer.state.values():
        assert state["step"].item() == 0
        assert not state["exp_avg"].count_nonzero()
        assert not state["exp_avg_sq"].count_nonzero()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("adaptive", (False, True))
def test_prior_checkpoint_scalars_targets_and_optimizers_reset_at_each_root(
    family, adaptive, monkeypatch, tmp_path,
):
    source = _agent(family)
    scaled = family.endswith("qscale")
    # Populate saved outer Adam moments so frozen-state equality also covers
    # real optimizer state, while the receiving action-local Adam starts fresh.
    horizon, batch = source.cfg.train_unroll_horizon, source.cfg.batch_size
    source._update(
        torch.randn(horizon + 1, batch, 67),
        torch.randn(horizon, batch, 21).tanh(),
        torch.rand(horizon, batch, 1),
        torch.zeros(horizon, batch, 1),
    )
    with torch.no_grad():
        if scaled:
            source.actor_loss_scale.fill_(20.949)
        else:
            # Early learned-temperature checkpoints are below the old 1e-6
            # guard; preserve their actual coefficient on load and per root.
            source.log_ent_coef.fill_(math.log(9e-8))
        for parameter in source.model._target_Qs.parameters():
            parameter.add_(0.015)
    checkpoint = tmp_path / "prior_25000"
    saved = deepcopy(source.checkpoint_state())
    torch.save(saved, checkpoint)
    restored = _agent(family, adaptive=adaptive)
    restored.load(checkpoint)
    _assert_tree_equal(restored.model.state_dict(), saved["model"])
    torch.testing.assert_close(restored.alpha, source.alpha, rtol=0, atol=0)
    if scaled:
        _assert_tree_equal(restored._actor_loss_scale_state(), saved["actor_loss_scale_state"])
    else:
        torch.testing.assert_close(restored.log_ent_coef, saved["log_ent_coef"], rtol=0, atol=0)

    frozen = deepcopy(restored.checkpoint_state())
    engine = restored.inner_engine
    original_prepare = engine._prepare_workspace
    original_policy = engine._sac_policy_step
    inherited = []
    scale_sequences = []
    alpha_sequences = []

    def prepare(**kwargs):
        original_prepare(**kwargs)
        state = engine.state
        _assert_tree_equal(state.actor.state_dict(), source.model._pi.state_dict())
        _assert_tree_equal(state.critic.state_dict(), source.model._Qs.state_dict())
        _assert_tree_equal(state.critic_target.state_dict(), source.model._target_Qs.state_dict())
        _assert_fresh_optimizer(state.actor_optim)
        _assert_fresh_optimizer(state.critic_optim)
        assert state.replay.size == 0
        torch.testing.assert_close(engine.alpha.reshape(()), source.alpha.reshape(()), rtol=0, atol=0)
        inherited.append(engine.alpha.detach().clone())
        if adaptive and not scaled:
            _assert_fresh_optimizer(state.temperature_optim)
            assert engine._resolved_inner_target_entropy() == -441.0
        else:
            assert state.temperature_optim is None
        scale_sequences.append([])
        alpha_sequences.append([])

    def policy(batch, **kwargs):
        scale = kwargs["actor_loss_scale"]
        if scaled:
            assert scale.data_ptr() != restored.actor_loss_scale.data_ptr()
            scale_sequences[-1].append(scale.clone())
        alpha_sequences[-1].append(engine.alpha.detach().clone())
        result = original_policy(batch, **kwargs)
        if scaled:
            scale_sequences[-1].append(scale.clone())
        alpha_sequences[-1].append(engine.alpha.detach().clone())
        return result

    monkeypatch.setattr(engine, "_prepare_workspace", prepare)
    monkeypatch.setattr(engine, "_sac_policy_step", policy)
    for root in range(2):
        action = restored.act(torch.zeros(67), t0=(root == 0), collect_diagnostics=False)
        assert torch.isfinite(action).all()
        assert engine.state.actor is engine.state.critic is engine.state.replay is None
        _assert_tree_equal(restored.checkpoint_state(), frozen)

    assert len(inherited) == 2
    for scales, alphas in zip(scale_sequences, alpha_sequences):
        assert len(alphas) == 4
        if scaled:
            torch.testing.assert_close(scales[0], source.actor_loss_scale, rtol=0, atol=0)
            if adaptive:
                assert not torch.equal(scales[-1], scales[0])
            else:
                for scale in scales:
                    torch.testing.assert_close(scale, scales[0], rtol=0, atol=0)
        if adaptive and not scaled:
            assert not torch.equal(alphas[-1], alphas[0])
        else:
            for alpha in alphas:
                torch.testing.assert_close(alpha, alphas[0], rtol=0, atol=0)
