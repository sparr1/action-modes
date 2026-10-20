"""Independent isolation checks for the entropy diagnostic experiment."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from evaluate_ambi_checkpoint import _outer_state_digest
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_component_model
from utils.ambi_entropy_probe import probe_entropy_root


def _fixture(representation="scalar", **overrides):
    options = dict(
        aux_return_mode="sac",
        inner_critic_source="aux_return",
        inner_horizon_critic_source="aux_return",
        inner_sac_critic_target="reward_only",
        inner_finite_horizon=True,
        inner_rollout_horizon=3,
        train_unroll_horizon=3,
        inner_rounds=2,
        inner_rollouts_per_round=4,
        inner_replay_capacity=24,
        inner_critic_updates_per_round=2,
        inner_actor_updates_per_round=2,
        inner_batch_size=4,
        inner_critic_dropout_enabled=True,
        dropout=0.2,
        critic_ensemble_backend="vmap",
        q_representation=representation,
        num_q=2 if representation == "scalar" else 3,
    )
    options.update(overrides)
    model = _tiny_component_model(**options)
    with torch.no_grad():
        for module in (model.agent.model._aux_return_Qs, model.agent.model._Qs):
            for parameter in module.parameters():
                parameter.add_(0.025 * torch.randn_like(parameter))
    return model


def _snapshot(model):
    agent = model.agent
    return {
        "outer": _outer_state_digest(model),
        "modes": [module.training for module in agent.model.modules()],
        "flags": [parameter.requires_grad for parameter in agent.model.parameters()],
        "gradients": [None if parameter.grad is None else parameter.grad.clone()
                      for parameter in agent.model.parameters()],
        "engine_id": id(agent.inner_engine),
        "engine_rng": deepcopy(agent.inner_engine.rng.training_state_dict()),
        "engine_index": agent.inner_engine.action_index,
        "torch_rng": torch.random.get_rng_state().clone(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def _assert_unchanged(model, before):
    after = _snapshot(model)
    for key in before.keys() - {"numpy_rng"}:
        _assert_tree_equal(after[key], before[key])
    assert before["numpy_rng"][0] == after["numpy_rng"][0]
    np.testing.assert_array_equal(before["numpy_rng"][1], after["numpy_rng"][1])
    assert before["numpy_rng"][2:] == after["numpy_rng"][2:]


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_probe_preserves_live_state_rng_modes_flags_and_existing_gradients(representation):
    model = _fixture(representation)
    try:
        observation = np.asarray([0.2, -0.1, 0.7], dtype=np.float32)
        model.agent.act(torch.as_tensor(observation), t0=True, eval_mode=True)
        # Existing gradients and mixed flags are not included in state_dict.
        # A diagnostic must preserve them too, including when called mid-analysis.
        parameters = list(model.agent.model.parameters())
        parameters[0].requires_grad_(False)
        parameters[1].grad = torch.full_like(parameters[1], 0.125)
        before = _snapshot(model)
        probe_entropy_root(model, observation, seed=173, actor_steps=3,
                           anchor_samples=8, exact_engine=True)
        _assert_unchanged(model, before)
    finally:
        model.env.close()


def test_failed_probe_restores_outer_modes_flags_and_rng(monkeypatch):
    model = _fixture()
    try:
        before = _snapshot(model)
        original = model.agent.model.pi
        calls = 0

        def interrupted_policy(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected entropy probe failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(model.agent.model, "pi", interrupted_policy)
        with pytest.raises(RuntimeError, match="injected entropy probe failure"):
            probe_entropy_root(model, np.asarray([0.2, -0.1, 0.7], dtype=np.float32),
                               seed=173, actor_steps=3, anchor_samples=8,
                               exact_engine=True)
        _assert_unchanged(model, before)
    finally:
        model.env.close()
