"""Weight-only transfer across live SAC decisions, including outer updates."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from RL.tdmpc2_core.common.bernoulli_transfer import reset_critic_with_bernoulli_
from tests.test_ambi_inner_initialization import _snapshot
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_model
from tests.test_pave_integration import _batch


@pytest.fixture
def models():
    opened = []

    def create(**overrides):
        options = dict(inner_rounds=1, inner_updates_per_round=2,
                       inner_critic_bernoulli_p=.5)
        options.update(overrides)
        model = _tiny_model(**options)
        opened.append(model)
        return model

    yield create
    for model in opened:
        model.close()
        model.env.close()


def _parameters(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def _oracle(old, prior, probability, generator):
    """Independent scalar selection oracle; no interpolation or delta rebasing."""
    selected, retained = {}, 0
    for name, value in old.items():
        if probability in (0., 1.):
            mask = torch.full_like(value, bool(probability), dtype=torch.bool)
        else:
            mask = torch.rand(value.shape, generator=generator) < probability
        selected[name] = torch.where(mask, value, prior[name])
        retained += mask.sum().item()
    return selected, retained / sum(value.numel() for value in old.values())


@pytest.mark.parametrize("probability", [0., .5, 1.])
def test_scalar_copy_includes_affine_parameters_and_refreshes_buffers(probability):
    prior = nn.Sequential(nn.Linear(32, 16), nn.BatchNorm1d(16))
    old = deepcopy(prior)
    with torch.no_grad():
        for parameter in prior.parameters():
            parameter.fill_(-7.)
        for parameter in old.parameters():
            parameter.fill_(3.)
        old[1].running_mean.fill_(9.)
    base = deepcopy(prior.state_dict())
    identities = [id(value) for value in old.parameters()]
    generator = torch.Generator().manual_seed(700)
    initial_rng = generator.get_state().clone()
    expected, fraction = _oracle(_parameters(old), _parameters(prior), probability,
                                 torch.Generator().manual_seed(700))
    result = reset_critic_with_bernoulli_(old, prior, probability,
                                        generator=generator, measure=True)
    _assert_tree_equal(_parameters(old), expected)
    _assert_tree_equal(dict(old.named_buffers()), dict(prior.named_buffers()))
    _assert_tree_equal(prior.state_dict(), base)
    assert identities == [id(value) for value in old.parameters()]
    assert float(result) == pytest.approx(fraction)
    if probability in (0., 1.):
        assert torch.equal(generator.get_state(), initial_rng)
    else:
        assert .4 < fraction < .6
        assert all(set(value.unique().tolist()) == {-7., 3.} for value in old.parameters())


def test_invalid_layout_is_rejected_before_mutation_or_rng_draws():
    old = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    prior = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    before = deepcopy(old.state_dict())
    generator = torch.Generator().manual_seed(6)
    rng = generator.get_state().clone()
    with pytest.raises(ValueError, match="Incompatible"):
        reset_critic_with_bernoulli_(old, prior, .5, generator=generator)
    _assert_tree_equal(old.state_dict(), before)
    assert torch.equal(generator.get_state(), rng)


def _assert_fresh_optim(optimizer):
    for state in optimizer.state.values():
        assert all(torch.count_nonzero(value) == 0 for value in state.values())


@pytest.mark.parametrize("probability", [0., .5, 1.])
@pytest.mark.parametrize("backend", ["loop", "vmap"])
@pytest.mark.parametrize("source", ["sac", "aux_return"])
def test_next_solve_selects_previous_critic_or_latest_outer_prior(
    models, monkeypatch, probability, backend, source,
):
    options = dict(inner_critic_bernoulli_p=probability, critic_ensemble_backend=backend)
    if source == "aux_return":
        options.update(aux_return_mode="return_actor", inner_actor_source="return_actor",
                       inner_critic_source="aux_return", inner_horizon_actor_source="return_actor",
                       inner_horizon_critic_source="aux_return")
    agent = models(**options).agent
    engine = agent.inner_engine
    seen = []
    prepare = engine._prepare_workspace

    def capture(**kwargs):
        prepare(**kwargs)
        state = engine.state
        seen.append((_parameters(state.critic), [id(p) for p in state.critic.parameters()]))
        _assert_tree_equal(_parameters(state.critic_target), _parameters(state.critic))
        _assert_tree_equal(state.actor.state_dict(), engine._actor_base.state_dict())
        for optimizer in (state.actor_optim, state.critic_optim, state.temperature_optim):
            _assert_fresh_optim(optimizer)
        assert state.replay.size == 0
        torch.testing.assert_close(engine.alpha, engine._actor_owner.alpha.reshape(()), rtol=0, atol=0)

    monkeypatch.setattr(engine, "_prepare_workspace", capture)
    before_prior = _parameters(engine._critic_base)
    initial_mask_rng = (engine.rng.generator("critic_bernoulli").get_state().clone()
                        if probability > 0 else None)
    agent.act(torch.zeros(3), t0=True)
    _assert_tree_equal(seen[0][0], before_prior)
    donor = _parameters(engine._action_pool.critic)
    if probability > 0:
        assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 0.
        assert torch.equal(engine.rng.generator("critic_bernoulli").get_state(), initial_mask_rng)

    # The full training path changes the prior between successive inner solves.
    agent._update(*_batch(agent))
    current_prior = _parameters(engine._critic_base)
    assert any(not torch.equal(current_prior[key], before_prior[key]) for key in current_prior)
    outer_before = deepcopy(agent.model.state_dict())
    oracle_rng = torch.Generator()
    if probability > 0:
        oracle_rng.set_state(engine.rng.generator("critic_bernoulli").get_state())
    expected, retained = _oracle(donor, current_prior, probability, oracle_rng)
    global_rng = torch.random.get_rng_state().clone()
    agent.act(torch.tensor([.2, -.1, .3]))
    _assert_tree_equal(seen[1][0], expected)
    assert seen[1][1] == seen[0][1]
    _assert_tree_equal(agent.model.state_dict(), outer_before)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    if probability > 0:
        assert torch.equal(engine.rng.generator("critic_bernoulli").get_state(), oracle_rng.get_state())
        metrics = agent.last_inner_metrics
        assert metrics["inner_critic_bernoulli_applied"] == 1.
        assert metrics["inner_critic_bernoulli_retained_fraction"] == pytest.approx(retained)
        assert metrics["inner_critic_transfer_outer_updates"] == 1.
    else:
        assert not any("bernoulli" in key for key in agent.last_inner_metrics)


@pytest.mark.parametrize("reset", ["episode", "t0", "evaluation", "clear", "mode"])
def test_no_donor_crosses_an_episode_or_evaluation_boundary(models, reset):
    agent = models().agent
    engine = agent.inner_engine
    agent.act(torch.zeros(3), t0=True)
    agent.act(torch.zeros(3))
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 1.
    if reset == "episode":
        agent.reset()
    elif reset == "evaluation":
        engine.reset_for_evaluation(707, reuse_action_pool=True)
    elif reset == "clear":
        engine.clear_all()
    agent.act(torch.zeros(3), t0=reset == "t0", eval_mode=reset == "mode")
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 0.
    agent.act(torch.zeros(3), eval_mode=reset == "mode")
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 1.


def test_failed_solve_cannot_supply_a_donor(models, monkeypatch):
    agent = models().agent
    agent.act(torch.zeros(3))
    engine = agent.inner_engine
    original = engine._act_rl

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected solve failure")

    monkeypatch.setattr(engine, "_act_rl", fail)
    with pytest.raises(RuntimeError, match="injected solve failure"):
        agent.act(torch.zeros(3))
    monkeypatch.setattr(engine, "_act_rl", original)
    agent.act(torch.zeros(3))
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 0.


def test_diagnostics_do_not_change_transfer_or_learning(models):
    observed, quiet = models().agent, models().agent
    for index in range(3):
        expected = observed.act(torch.zeros(3), t0=index == 0, collect_diagnostics=True)
        actual = quiet.act(torch.zeros(3), t0=index == 0, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        left, right = _snapshot(observed.inner_engine), _snapshot(quiet.inner_engine)
        # Diagnostic-only streams can advance; all learner and mask streams agree.
        left.pop("rng")
        right.pop("rng")
        _assert_tree_equal(left, right)
        for name in ("critic_bernoulli", "collection", "replay", "gradient_policy", "execution"):
            assert torch.equal(observed.inner_engine.rng.generator(name).get_state(),
                               quiet.inner_engine.rng.generator(name).get_state())
        assert "inner_critic_bernoulli_retained_fraction" not in quiet.last_inner_metrics


@pytest.mark.parametrize("value", [True, -.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_probability_validation_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError), match="inner_critic_bernoulli_p"):
        _build_cfg(inner_critic_bernoulli_p=value)


@pytest.mark.parametrize("setting", [
    {"inner_critic_scope": "episode"},
    {"inner_critic_initialization": "random"},
    {"inner_bootstrap_source": "outer_target"},
    {"inner_critic_target_initialization": "outer_target"},
    {"inner_actor_writeback_coef": .5},
    {"inner_rounds": 0},
])
def test_incompatible_lifecycles_fail_during_configuration(setting):
    with pytest.raises(ValueError):
        _build_cfg(inner_critic_bernoulli_p=.5, **setting)


@pytest.mark.parametrize("settings", [{}, {"inner_critic_initialization": "random"},
                                    {"inner_critic_adaptation": "lora_rl", "inner_critic_lora_rank": 2}])
def test_native_tdambi_does_not_silently_accept_sac_transfer(settings):
    from tests.test_tdambi_initialization import _native_cfg

    with pytest.raises(ValueError, match="inner_critic_bernoulli_p"):
        _native_cfg(inner_critic_bernoulli_p=.5, **settings)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_online_return_actor_recipe_accepts_each_requested_horizon(horizon):
    cfg = _build_cfg(
        inner_critic_bernoulli_p=.5, inner_rounds=6, inner_rollout_horizon=horizon,
        inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
        inner_rollouts_per_round=128, inner_batch_size=256, inner_replay_capacity=3072,
        aux_return_mode="return_actor", inner_actor_source="return_actor",
        inner_critic_source="aux_return", inner_horizon_actor_source="return_actor",
        inner_horizon_critic_source="aux_return",
    )
    assert cfg.inner_critic_updates_per_action == 96
    assert cfg.inner_actor_updates_per_action == 24
