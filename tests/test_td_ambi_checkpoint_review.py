"""A saved study learner must reproduce its next update, not just tensor shapes."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree


CONFIGS = Path(__file__).resolve().parents[1] / "configs/dmcontrol/algs"
CASES = (
    "td_ambi_prior_reward_qscale",
    "td_ambi_prior_entropy_qscale",
    "td_ambi_prior_entropy_autotemp",
    "td_ambi_prior_reward_autotemp",
    "td_ambi_full_entropy_autotemp",
)


def _agent(name):
    params = json.loads((CONFIGS / f"{name}.json").read_text())["alg_params"]
    # Preserve study equations, Q support, optimizers and 21-action policy.
    # Small hidden layers/batches make serialization/continuation a CPU check.
    params.update(
        device="cpu", compile=False, wandb=False, model_size=None,
        enc_dim=32, mlp_dim=32, latent_dim=16, num_enc_layers=2,
        batch_size=4,
    )
    cfg = _build_cfg(**params)
    cfg.action_dim = 21
    cfg.obs_shape = {"state": (67,)}
    return AMBITDMPC2Agent(cfg)


def _batch(agent):
    horizon, batch = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    return (
        torch.randn(horizon + 1, batch, 67),
        torch.randn(horizon, batch, 21).tanh(),
        torch.rand(horizon, batch, 1),
        torch.zeros(horizon, batch, 1),
    )


@pytest.mark.parametrize("name", CASES)
def test_study_checkpoint_reproduces_next_outer_update(tmp_path, name):
    torch.manual_seed(321)
    source = _agent(name)
    batch = _batch(source)
    # Build nonzero critic/actor moments and learned scale or temperature state.
    source._update(*batch)
    source._update(*batch)
    source.prepare_training_resume_boundary()
    snapshot = _clone_tree(source.training_state_dict())
    checkpoint = tmp_path / "study.pt"
    torch.save(snapshot, checkpoint)
    restored = AMBITDMPC2Agent(deepcopy(source.cfg))
    restored.load_training_state_dict(torch.load(checkpoint, weights_only=False))
    _assert_tree_equal(restored.training_state_dict(), snapshot)

    # RNG belongs to the enclosing training checkpoint, not the agent payload.
    # Give both continuations the same draws, including active critic dropout.
    torch.manual_seed(654)
    expected_metrics = source._update(*batch)
    torch.manual_seed(654)
    actual_metrics = restored._update(*batch)
    _assert_tree_equal(actual_metrics, expected_metrics)
    source.prepare_training_resume_boundary()
    restored.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())
    if source.actor_loss_scale_enabled:
        assert source.actor_loss_scale.item() >= 1.0
    assert torch.isfinite(source.alpha).all()
