"""Backend changes permit portable weights but cannot silently resume dropout."""

from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.TDMPC2 import TDMPC2Baseline
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_pave_integration import _batch
from tests.test_tdambi_checkpoint import make_tdambi, tiny_native_params


@pytest.fixture
def models():
    wrappers = []

    def create(**overrides):
        options = {"inner_rounds": 1, "inner_updates_per_round": 1,
                   "ent_coef": .1, "dropout": .2}
        wrapper = _tiny_model(**(options | overrides))
        wrappers.append(wrapper)
        return wrapper.agent

    yield create
    for wrapper in wrappers:
        wrapper.close()
        wrapper.env.close()


@pytest.mark.parametrize("backend", ["loop", "vmap"])
@pytest.mark.parametrize("aux_mode", ["off", "sac", "return_actor"])
def test_backend_routes_online_targets_and_inner_clones(models, backend, aux_mode):
    agent = models(critic_ensemble_backend=backend, aux_return_mode=aux_mode)
    attributes = ["_Qs", "_target_Qs"]
    if aux_mode != "off":
        attributes.extend(["_aux_return_Qs", "_target_aux_return_Qs"])
    for attribute in attributes:
        critic = getattr(agent.model, attribute)
        assert critic.backend == backend
        assert deepcopy(critic).backend == backend
    with agent.inner_engine.rng.fork("initialization"):
        agent.inner_engine._prepare_workspace(t0=True)
    assert agent.inner_engine.state.critic.backend == backend
    assert agent.inner_engine.state.critic_target.backend == backend
    assert agent.checkpoint_state()["critic_target_spec"]["critic_ensemble_backend"] == backend
    assert "critic_ensemble_backend" not in agent.model.critic_signature


@pytest.mark.parametrize("source_backend,target_backend", [("loop", "vmap"), ("vmap", "loop")])
@pytest.mark.parametrize("aux_mode", ["off", "sac"])
@pytest.mark.parametrize("historical", [False, True])
def test_portable_transfer_preserves_weights_optimizer_slots_and_parameter_ownership(
    models, source_backend, target_backend, aux_mode, historical,
):
    source = models(critic_ensemble_backend=source_backend, aux_return_mode=aux_mode)
    source._update(*_batch(source))
    checkpoint = deepcopy(source.checkpoint_state())
    if historical:
        checkpoint["critic_target_spec"].pop("critic_ensemble_backend")
    target = models(critic_ensemble_backend=target_backend, aux_return_mode=aux_mode)
    parameters_before = {name: id(parameter) for name, parameter in target.model.named_parameters()}
    target.load(checkpoint)
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    _assert_tree_equal(target.optim.state_dict(), source.optim.state_dict())
    assert {name: id(parameter) for name, parameter in target.model.named_parameters()} == parameters_before
    assert target.model._Qs.backend == target_backend
    assert target._critic_target_spec()["critic_ensemble_backend"] == target_backend


@pytest.mark.parametrize("source_backend,target_backend", [("loop", "vmap"), ("vmap", "loop")])
@pytest.mark.parametrize("aux_mode", ["off", "sac"])
def test_exact_backend_change_rejected_before_live_state_or_rng_mutation(models, source_backend, target_backend, aux_mode):
    source = models(critic_ensemble_backend=source_backend, aux_return_mode=aux_mode)
    source._update(*_batch(source))
    source.prepare_training_resume_boundary()
    saved = deepcopy(source.training_state_dict())
    target = models(critic_ensemble_backend=target_backend, aux_return_mode=aux_mode)
    before = deepcopy(target.training_state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="critic-target specification"):
        target.load_training_state_dict(saved)
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("target_backend", ["loop", "vmap"])
def test_missing_saved_backend_means_historical_loop(models, target_backend):
    source = models(critic_ensemble_backend="loop")
    source._update(*_batch(source))
    source.prepare_training_resume_boundary()
    saved = deepcopy(source.training_state_dict())
    saved["outer"]["critic_target_spec"].pop("critic_ensemble_backend")
    saved["inner"].pop("critic_ensemble_backend")
    target = models(critic_ensemble_backend=target_backend)
    if target_backend == "vmap":
        before = deepcopy(target.training_state_dict())
        with pytest.raises(ValueError, match="critic_ensemble_backend"):
            target.load_training_state_dict(saved)
        _assert_tree_equal(target.training_state_dict(), before)
    else:
        target.load_training_state_dict(saved)
        expected = deepcopy(saved)
        expected["outer"]["critic_target_spec"]["critic_ensemble_backend"] = "loop"
        expected["inner"]["critic_ensemble_backend"] = "loop"
        _assert_tree_equal(target.training_state_dict(), expected)
    assert "critic_ensemble_backend" not in saved["outer"]["critic_target_spec"]
    assert "critic_ensemble_backend" not in saved["inner"]


@pytest.mark.parametrize("source_backend,target_backend", [("loop", "vmap"), ("vmap", "loop")])
def test_direct_inner_backend_change_rejected_transactionally(models, source_backend, target_backend):
    source = models(critic_ensemble_backend=source_backend)
    target = models(critic_ensemble_backend=target_backend)
    saved = deepcopy(source.inner_engine.training_state_dict())
    before = deepcopy(target.inner_engine.training_state_dict())
    global_rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="critic_ensemble_backend.*incompatible"):
        target.inner_engine.load_training_state_dict(saved)
    _assert_tree_equal(target.inner_engine.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("target_backend", ["loop", "vmap"])
def test_direct_inner_missing_backend_accepts_only_historical_loop(models, target_backend):
    source = models(critic_ensemble_backend="loop")
    saved = deepcopy(source.inner_engine.training_state_dict())
    saved.pop("critic_ensemble_backend")
    target = models(critic_ensemble_backend=target_backend)
    if target_backend == "loop":
        target.inner_engine.load_training_state_dict(saved)
        _assert_tree_equal(target.inner_engine.training_state_dict(),
                           {**saved, "critic_ensemble_backend": "loop"})
    else:
        before = deepcopy(target.inner_engine.training_state_dict())
        with pytest.raises(ValueError, match="critic_ensemble_backend.*incompatible"):
            target.inner_engine.load_training_state_dict(saved)
        _assert_tree_equal(target.inner_engine.training_state_dict(), before)
    assert "critic_ensemble_backend" not in saved


@pytest.mark.parametrize("backend", ["loop", "vmap"])
def test_direct_inner_resume_preserves_persistent_replay_optimizer_and_rng(models, backend):
    options = {f"inner_{component}_scope": "run" for component in (
        "actor", "critic", "temperature", "replay", "actor_optimizer",
        "critic_optimizer", "temperature_optimizer",
    )}
    source = models(critic_ensemble_backend=backend, **options)
    source.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source.inner_engine.prepare_training_resume_boundary()
    saved = deepcopy(source.inner_engine.training_state_dict())
    assert saved["critic_ensemble_backend"] == backend
    for key in ("critic", "critic_optim", "actor_optim", "replay"):
        assert saved["workspace"][key] is not None
    target = models(critic_ensemble_backend=backend, **options)
    target.load(deepcopy(source.checkpoint_state()))
    target.inner_engine.load_training_state_dict(saved)
    _assert_tree_equal(target.inner_engine.training_state_dict(), saved)
    global_rng = torch.random.get_rng_state().clone()
    expected = source.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    actual = target.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    source.inner_engine.prepare_training_resume_boundary()
    target.inner_engine.prepare_training_resume_boundary()
    _assert_tree_equal(target.inner_engine.training_state_dict(), source.inner_engine.training_state_dict())


@pytest.mark.parametrize("backend", ["loop", "vmap"])
@pytest.mark.parametrize("aux_mode", ["off", "sac"])
def test_same_backend_exact_resume_reproduces_next_update_with_dropout_and_pave(models, backend, aux_mode):
    settings = dict(critic_ensemble_backend=backend, aux_return_mode=aux_mode,
                    outer_pave_mpr_coef=.005, outer_pave_vfc_coef=.005,
                    outer_pave_curvature_coef=.0005, outer_pave_batch_size=3)
    source = models(**settings)
    source._update(*_batch(source))
    source.prepare_training_resume_boundary()
    saved = deepcopy(source.training_state_dict())
    restored = models(**settings)
    restored.load_training_state_dict(saved)
    _assert_tree_equal(restored.training_state_dict(), saved)
    rng = torch.random.get_rng_state().clone()
    expected = source._update(*_batch(source))
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng)
    actual = restored._update(*_batch(restored))
    _assert_tree_equal(actual, expected)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    source.prepare_training_resume_boundary()
    restored.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())


def test_native_tdmpc2_stays_loop_and_native_weights_load_into_default_tdambi():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    native = adapted = None
    try:
        native = TDMPC2Baseline(
            "TDMPC2", env, tiny_native_params(),
            {"seed": 3, "device": "cpu", "env": "Pendulum-v1", "total_steps": 10}, {},
        )
        adapted = make_tdambi()
        assert native.agent.model._Qs.backend == "loop"
        assert native.agent.model._target_Qs.backend == "loop"
        assert not hasattr(native.cfg, "critic_ensemble_backend")
        assert adapted.cfg.critic_ensemble_backend == "vmap"
        adapted.agent.load(deepcopy(native.agent.checkpoint_state()))
        assert adapted.agent.model._Qs.backend == "vmap"
        assert adapted.agent.model._target_Qs.backend == "vmap"
        # Native policy-bound buffers can be integer-valued, while AMBI stores
        # their equivalent floating-point values. Every learned tensor is exact.
        _assert_tree_equal(dict(adapted.agent.model.named_parameters()),
                           dict(native.agent.model.named_parameters()))
        for name, value in native.agent.model.named_buffers():
            torch.testing.assert_close(dict(adapted.agent.model.named_buffers())[name],
                                       value, rtol=0, atol=0, check_dtype=False)
    finally:
        if adapted is not None:
            adapted.close()
            adapted.env.close()
        if native is not None:
            native.close()
        env.close()
