"""Head initialization is opt-in scientific state, including at empty boundaries."""

from copy import deepcopy

import pytest
import torch

from tests.test_actor_transfer_solve_cadence import learner_snapshot
from tests.test_aux_critic_transfer import critic_params
from tests.test_aux_critic_hidden_transfer import body_state
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _model_from_params
from utils.resume_identity import scientific_trial_parameters


@pytest.fixture
def models():
    opened = []

    def create(critic="return", **overrides):
        wrapper = _model_from_params(critic_params(critic, **overrides))
        opened.append(wrapper)
        return wrapper.agent

    yield create
    for wrapper in opened:
        wrapper.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
def test_explicit_retain_preserves_checkpoint_semantics_and_lineage(models, critic):
    implicit = models(critic)
    explicit = models(critic, inner_critic_transfer_head="retain")
    _assert_tree_equal(implicit.training_state_dict(), explicit.training_state_dict())
    inner = implicit.inner_engine.training_state_dict()
    assert inner["version"] == 6 and "critic_transfer_head_spec" not in inner
    assert "critic_transfer_head" not in implicit._critic_target_spec()["inner_solve"]

    trial = {"alg": "AMBITDMPC2/AMBITDMPC2", "seed": 55,
             "alg_params": critic_params(critic)}
    original = scientific_trial_parameters(trial)
    trial["alg_params"]["inner_critic_transfer_head"] = "retain"
    assert scientific_trial_parameters(trial) == original
    trial["alg_params"]["inner_critic_transfer_head"] = "random"
    assert scientific_trial_parameters(trial) != original


@pytest.mark.parametrize("critic", ["soft", "return"])
def test_random_head_resume_reproduces_private_rng_and_next_episode(models, critic):
    source = models(critic, inner_critic_transfer_head="random")
    for index in range(2):
        source.act(torch.tensor([.2, -.1 * index, .3]), t0=index == 0,
                   collect_diagnostics=False)
    source.prepare_training_resume_boundary()
    payload = deepcopy(source.training_state_dict())
    assert payload["inner"]["version"] == 6
    assert payload["inner"]["critic_transfer_head_spec"] == {
        "mode": "random", "protocol_version": 1,
        "weight_initializer": "xavier_uniform", "bias_initializer": "zeros",
        "reset_timing": "every_solve_including_first",
    }
    assert payload["outer"]["critic_target_spec"]["inner_solve"]["critic_transfer_head"] == "random"
    assert payload["inner"]["workspace"]["critic"] is None
    assert payload["inner"]["workspace"]["critic_target"] is None
    assert payload["inner"]["workspace"]["counters"]["critic_lifetime_steps"] == 0

    direct = models(critic, inner_critic_transfer_head="random").inner_engine
    direct.load_training_state_dict(payload["inner"])
    _assert_tree_equal(direct.training_state_dict(), payload["inner"])
    restored = models(critic, inner_critic_transfer_head="random")
    global_rng = torch.random.get_rng_state().clone()
    restored.load_training_state_dict(payload)
    _assert_tree_equal(restored.training_state_dict(), payload)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    source.reset()
    restored.reset()

    original_prepare = restored.inner_engine._prepare_workspace
    preparations = 0

    def checked_prepare(*, t0):
        nonlocal preparations
        original_prepare(t0=t0)
        if preparations == 0:
            _assert_tree_equal(body_state(restored.inner_engine.state.critic),
                               body_state(restored.inner_engine._critic_base))
            assert restored.inner_engine.state.critic_lifetime_steps == 0
        preparations += 1

    restored.inner_engine._prepare_workspace = checked_prepare
    for observation in (torch.tensor([.3, -.2, .1]), torch.zeros(3)):
        expected = source.act(observation, collect_diagnostics=False)
        actual = restored.act(observation, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(learner_snapshot(restored.inner_engine),
                           learner_snapshot(source.inner_engine))
        _assert_tree_equal(restored.inner_engine.rng.training_state_dict(),
                           source.inner_engine.rng.training_state_dict())
    source.prepare_training_resume_boundary()
    restored.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())


@pytest.mark.parametrize("source_mode,target_mode", [("retain", "random"), ("random", "retain")])
@pytest.mark.parametrize("direct", [False, True])
def test_cross_mode_exact_resume_rejected_before_mutation(models, source_mode, target_mode, direct):
    source = models(inner_critic_transfer_head=source_mode)
    target = models(inner_critic_transfer_head=target_mode)
    if direct:
        source, target = source.inner_engine, target.inner_engine
    payload = deepcopy(source.training_state_dict())
    before = deepcopy(target.training_state_dict())
    global_rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="incompatible|does not match"):
        target.load_training_state_dict(payload)
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("source_mode,target_mode", [("retain", "random"), ("random", "retain")])
def test_frozen_weight_transfer_accepts_changed_head_mode_and_clears_inner_state(models, source_mode, target_mode):
    source = models(inner_critic_transfer_head=source_mode)
    target = models(inner_critic_transfer_head=target_mode)
    target.act(torch.ones(3), t0=True, eval_mode=True)
    assert target.inner_engine.state.critic is not None
    source_weights = deepcopy(source.checkpoint_state())
    target.load(source_weights)
    _assert_tree_equal(target.model.state_dict(), source_weights["model"])
    assert target.inner_engine.state.critic is None
    assert target.inner_engine._held_actor is None
    assert target.cfg.inner_critic_transfer_head == target_mode


@pytest.mark.parametrize("tamper", ["missing", "extra", "weight", "bias", "timing", "version", "mode"])
@pytest.mark.parametrize("direct", [False, True])
def test_head_spec_corruption_rejected_before_mutation(models, tamper, direct):
    target = models(inner_critic_transfer_head="random")
    if direct:
        target = target.inner_engine
    before = deepcopy(target.training_state_dict())
    payload = deepcopy(before)
    inner = payload if direct else payload["inner"]
    spec = inner["critic_transfer_head_spec"]
    if tamper == "missing":
        inner.pop("critic_transfer_head_spec")
    elif tamper == "extra":
        spec["head_scope"] = "episode"
    else:
        key, value = {
            "weight": ("weight_initializer", "kaiming_uniform"),
            "bias": ("bias_initializer", "random"),
            "timing": ("reset_timing", "subsequent_solves_only"),
            "version": ("protocol_version", True),
            "mode": ("mode", "retain"),
        }[tamper]
        spec[key] = value
    global_rng = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        target.load_training_state_dict(payload)
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
