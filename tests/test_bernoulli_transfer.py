"""Scientific contracts for scalar Bernoulli copying between inner SAC solves."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_aux_critic_transfer import assert_optimizer_reset, critic_params
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params
from utils.transfer_campaign import (
    arm_initialization, bernoulli_state, evaluate_episode, make_transfer_generators,
    needs_donor, validate_arm,
)


def module_with_buffers():
    module = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.LayerNorm(32))
    module.register_buffer("counter", torch.tensor(3))
    module.register_buffer("floating_buffer", torch.tensor([2., 4.]))
    return module


@pytest.mark.parametrize("probability", [0., .5, 1.])
def test_masks_select_exact_scalar_values_include_bias_and_norm_and_preserve_buffers(probability):
    module = module_with_buffers()
    prior = deepcopy(module.state_dict())
    donor = {name: value + 2 for name, value in prior.items()}
    saved_prior, saved_donor = deepcopy(prior), deepcopy(donor)
    generator = torch.Generator().manual_seed(79)
    before = generator.get_state().clone()
    metrics = {}
    actual = bernoulli_state(prior, donor, probability,
        parameter_names=dict(module.named_parameters()), generator=generator, metrics=metrics)
    parameters = dict(module.named_parameters())
    for name, value in actual.items():
        if name not in parameters or probability == 0:
            torch.testing.assert_close(value, prior[name], rtol=0, atol=0)
        elif probability == 1:
            torch.testing.assert_close(value, donor[name], rtol=0, atol=0)
        else:
            assert torch.all((value == prior[name]) | (value == donor[name]))
            # All biases and LayerNorm affine tensors have 32 scalars, enough
            # to observe both outcomes with this deterministic mask seed.
            assert torch.any(value == prior[name]) and torch.any(value == donor[name])
        assert value.data_ptr() != prior[name].data_ptr()
        assert value.data_ptr() != donor[name].data_ptr()
    selected = sum(torch.count_nonzero(actual[name] == donor[name]).item() for name in parameters)
    count = sum(value.numel() for value in parameters.values())
    assert metrics["retained_fraction"] == selected / count
    delta = torch.cat([(actual[name] - prior[name]).flatten() for name in parameters])
    base = torch.cat([prior[name].flatten() for name in parameters])
    assert metrics["relative_parameter_delta_l2"] == pytest.approx((delta.norm() / base.norm()).item())
    assert torch.equal(before, generator.get_state()) == (probability in (0., 1.))
    for value in actual.values():
        value.zero_()
    _assert_tree_equal(prior, saved_prior)
    _assert_tree_equal(donor, saved_donor)


def test_private_component_masks_resample_reproduce_and_do_not_consume_other_rngs():
    engine = SimpleNamespace(device=torch.device("cpu"))
    first = make_transfer_generators(engine, controller_seed=55, episode_seed=101)
    repeated = make_transfer_generators(engine, controller_seed=55, episode_seed=101)
    other_episode = make_transfer_generators(engine, controller_seed=55, episode_seed=102)
    prior, donor = {"w": torch.zeros(1024)}, {"w": torch.ones(1024)}
    global_state = torch.random.get_rng_state().clone()
    critic_state = first["critic"].get_state().clone()
    def sample(generator):
        return bernoulli_state(prior, donor, .5, parameter_names={"w"}, generator=generator)["w"]
    one, two = sample(first["actor"]), sample(first["actor"])
    assert not torch.equal(one, two)
    assert torch.equal(one, sample(repeated["actor"]))
    assert torch.equal(two, sample(repeated["actor"]))
    assert not torch.equal(one, sample(other_episode["actor"]))
    assert not torch.equal(one, sample(repeated["critic"]))
    torch.testing.assert_close(critic_state, first["critic"].get_state(), rtol=0, atol=0)
    torch.testing.assert_close(global_state, torch.random.get_rng_state(), rtol=0, atol=0)


@pytest.mark.parametrize("arm", [
    {"actor_bernoulli_p": -.1}, {"critic_bernoulli_p": 1.1},
    {"actor_bernoulli_p": float("nan")}, {"actor_bernoulli_p": True},
    {"actor_bernoulli_p": .5, "actor_rho": 0.},
    {"critic_bernoulli_p": .5, "critic_rho": 1.},
    {"actor_bernoulli_p": .5, "full_state": True},
])
def test_invalid_probabilities_and_ambiguous_transfer_rules_are_rejected(arm):
    with pytest.raises(ValueError):
        validate_arm(arm)


def test_partial_copy_requires_private_generator_and_validates_before_drawing():
    prior, donor = {"w": torch.zeros(8)}, {"w": torch.ones(8)}
    with pytest.raises(ValueError, match="isolated generator"):
        bernoulli_state(prior, donor, .5, parameter_names={"w"})
    generator = torch.Generator().manual_seed(11)
    saved = generator.get_state().clone()
    with pytest.raises(ValueError, match="Incompatible"):
        bernoulli_state(prior, {"w": torch.ones(9)}, .5,
                        parameter_names={"w"}, generator=generator)
    torch.testing.assert_close(saved, generator.get_state(), rtol=0, atol=0)
    assert not needs_donor({"actor_bernoulli_p": 0.})
    assert needs_donor({"critic_bernoulli_p": .5})


@pytest.mark.parametrize("components", [("actor",), ("critic",), ("actor", "critic")])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_successive_solves_copy_only_weights_reset_learning_state_and_leave_prior_frozen(components, horizon):
    wrapped = _model_from_params(critic_params(inner_critic_scope="action",
        inner_rounds=1, inner_rollout_horizon=horizon))
    try:
        engine = wrapped.agent.inner_engine
        arm = {f"{name}_bernoulli_p": .5 for name in components}
        frozen = _clone_tree(wrapped.agent.checkpoint_state())
        with engine.diagnostic_initialization(target="online"):
            wrapped.agent.act(torch.ones(3), t0=True, eval_mode=True)
        donor = engine.export_diagnostic_state(include_optimizers=True, include_replay=True)
        donor_before = _clone_tree(donor)
        # A deliberately stale target makes accidental target carry detectable.
        for value in donor["modules"]["critic_target"].values():
            value.add_(12.)
        generators = make_transfer_generators(engine, controller_seed=55, episode_seed=101)
        learner_rng = _clone_tree(engine.rng.training_state_dict())
        global_rng = torch.random.get_rng_state().clone()
        metrics = {}
        options = arm_initialization(engine, arm, donor,
            transfer_generators=generators, transfer_metrics=metrics)
        _assert_tree_equal(engine.rng.training_state_dict(), learner_rng)
        torch.testing.assert_close(global_rng, torch.random.get_rng_state(), rtol=0, atol=0)
        assert set(options) == {"allow_compile", "actor_prior_kl_coef", "critic_prior_l2_coef", "target", *components}
        with engine.diagnostic_initialization(**options):
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=False)
            engine._apply_diagnostic_initialization()
            for name in ("actor", "critic"):
                selected = getattr(engine.state, name)
                expected = options.get(name, getattr(engine, f"_{name}_base").state_dict())
                _assert_tree_equal(selected.state_dict(), expected)
                assert_optimizer_reset(getattr(engine.state, f"{name}_optim"), selected.parameters())
                assert getattr(engine.state, f"{name}_steps") == 0
                assert getattr(engine.state, f"{name}_lifetime_steps") == 0
            _assert_tree_equal(engine.state.critic_target.state_dict(), engine.state.critic.state_dict())
            assert_optimizer_reset(engine.state.temperature_optim, [engine.state.log_alpha])
            assert engine.state.replay.size == 0 and engine._diagnostic_previous_replay is None
            torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha(), rtol=0, atol=0)
        for name in components:
            assert 0 < metrics[f"inner_{name}_bernoulli_retained_fraction"] < 1
            assert metrics[f"inner_{name}_bernoulli_applied"] == 1
            _assert_tree_equal(donor["modules"][name], donor_before["modules"][name])
        _assert_tree_equal(wrapped.agent.checkpoint_state(), frozen)
    finally:
        wrapped.close()


@pytest.mark.parametrize("probability", [0., .5, 1.])
def test_episode_reset_reproduces_masks_actions_and_fresh_first_solve(probability):
    wrapped = _model_from_params(critic_params(inner_critic_scope="action", inner_rounds=1))
    try:
        arm = {"actor_bernoulli_p": probability, "critic_bernoulli_p": probability}
        trajectories = []
        for _ in range(2):
            rows = []
            evaluate_episode(wrapped, wrapped.env, arm, episode_seed=101,
                controller_seed=55, max_steps=3, on_step=rows.append, smoke=True)
            trajectories.append(rows)
            for name in ("actor", "critic"):
                assert rows[0]["metrics"][f"inner_{name}_bernoulli_applied"] == 0
                assert rows[0]["metrics"][f"inner_{name}_relative_parameter_delta_l2"] == 0
                assert rows[1]["metrics"][f"inner_{name}_bernoulli_applied"] == (probability > 0)
        for first, second in zip(*trajectories):
            assert first["metrics"] == second["metrics"]
            assert np.array_equal(first["action"], second["action"])
            assert first["reward"] == second["reward"]
    finally:
        wrapped.close()


@pytest.mark.parametrize("probability", [0., 1.])
def test_bernoulli_endpoints_match_existing_deterministic_controllers(probability):
    wrapped = _model_from_params(critic_params(inner_critic_scope="action", inner_rounds=1))
    try:
        trajectories = []
        for method in ("rho", "bernoulli_p"):
            rows = []
            evaluate_episode(wrapped, wrapped.env, {f"actor_{method}": probability, f"critic_{method}": probability},
                episode_seed=101, controller_seed=55, max_steps=3, on_step=rows.append, smoke=True)
            trajectories.append(rows)
        for first, second in zip(*trajectories):
            assert np.array_equal(first["action"], second["action"])
            assert first["reward"] == second["reward"]
            assert all(second["metrics"][key] == value for key, value in first["metrics"].items())
    finally:
        wrapped.close()


def test_compiled_and_eager_masked_solves_preserve_identical_learning_rng(monkeypatch):
    torch._dynamo.reset()
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original_compile(fn, backend="eager", **kwargs))
    eager = _model_from_params(critic_params(inner_critic_scope="action", inner_rounds=1))
    compiled = _model_from_params(critic_params(inner_critic_scope="action", inner_rounds=1,
                                              compile=True, compile_strict=True))
    try:
        compiled.agent.model.load_state_dict(eager.agent.model.state_dict())
        arm = {"actor_bernoulli_p": .5, "critic_bernoulli_p": .5}
        trajectories = []
        for model in (eager, compiled):
            rows = []
            evaluate_episode(model, model.env, arm, episode_seed=101, controller_seed=55,
                             max_steps=3, on_step=rows.append, smoke=True)
            trajectories.append(rows)
        for first, second in zip(*trajectories):
            assert np.array_equal(first["action"], second["action"])
            assert first["metrics"] == second["metrics"]
        _assert_tree_equal(eager.agent.inner_engine.export_diagnostic_state(),
                           compiled.agent.inner_engine.export_diagnostic_state())
        _assert_tree_equal(eager.agent.inner_engine.rng.training_state_dict(),
                           compiled.agent.inner_engine.rng.training_state_dict())
    finally:
        eager.close()
        compiled.close()
        torch._dynamo.reset()
