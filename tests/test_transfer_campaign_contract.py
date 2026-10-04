"""Independent scientific contracts for successive-decision transfer arms."""

from copy import deepcopy

import pytest
import torch

from tests.test_aux_critic_transfer import critic_params
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params
from utils.ambi_benchmark import solver_seed
from utils.transfer_campaign import arm_initialization


def _module_after_solve(engine, name):
    live = getattr(engine.state, name)
    return live if live is not None else getattr(engine._action_pool, name)


@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("mode", ["fresh", "actor", "critic"])
def test_donor_injection_matches_native_control_lifecycles(horizon, mode):
    """The new mechanism controls retain the existing action and RNG semantics."""
    options = dict(inner_rollout_horizon=horizon, inner_rounds=2,
                   inner_actor_scope="episode" if mode == "actor" else "action",
                   inner_critic_scope="episode" if mode == "critic" else "action")
    native = _model_from_params(critic_params(**options))
    injected = _model_from_params(critic_params(**dict(
        options, inner_actor_scope="action", inner_critic_scope="action")))
    try:
        injected.agent.load(_clone_tree(native.agent.checkpoint_state()))
        frozen = _clone_tree(injected.agent.checkpoint_state())
        arm = {"actor_rho": float(mode == "actor"), "critic_rho": float(mode == "critic")}
        donor = None
        for episode_seed in (101, 102):
            donor = None
            for model in (native, injected):
                model.agent.inner_engine.reset_for_evaluation(
                    solver_seed(55, "episode", episode_seed), reuse_action_pool=True)
            for decision in range(3):
                observation = torch.tensor([1., .13 * decision, -.2])
                expected = native.agent.act(observation, t0=decision == 0, eval_mode=True)
                engine = injected.agent.inner_engine
                kwargs = arm_initialization(engine, arm, donor)
                with engine.diagnostic_initialization(**kwargs):
                    actual = injected.agent.act(observation, t0=decision == 0, eval_mode=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name in ("actor", "critic", "critic_target"):
                    _assert_tree_equal(
                        _module_after_solve(engine, name).state_dict(),
                        _module_after_solve(native.agent.inner_engine, name).state_dict())
                _assert_tree_equal(engine.rng.training_state_dict(),
                                   native.agent.inner_engine.rng.training_state_dict())
                if mode != "fresh":
                    donor = {"modules": {mode: deepcopy(_module_after_solve(engine, mode).state_dict())}}
            _assert_tree_equal(injected.agent.checkpoint_state(), frozen)
    finally:
        native.close()
        injected.close()


def test_full_state_carry_restores_lagged_target_adam_and_temperature():
    model = _model_from_params(critic_params(inner_critic_scope="action", inner_rounds=2))
    try:
        engine = model.agent.inner_engine
        with engine.diagnostic_initialization(target="online"):
            model.agent.act(torch.ones(3), t0=True, eval_mode=True)
        donor = engine.export_diagnostic_state(include_optimizers=True, include_replay=True)
        frozen = _clone_tree(model.agent.checkpoint_state())
        assert any(not torch.equal(value, donor["modules"]["critic"][name])
                   for name, value in donor["modules"]["critic_target"].items())
        original = engine._apply_diagnostic_initialization
        inspected = []

        def inspect():
            original()
            for name, weights in donor["modules"].items():
                _assert_tree_equal(getattr(engine.state, name).state_dict(), weights)
            for name, optimizer in donor["optimizers"].items():
                _assert_tree_equal(getattr(engine.state, name + "_optim").state_dict(), optimizer)
                assert getattr(engine.state, name + "_steps") == 0
                assert getattr(engine.state, name + "_lifetime_steps") == donor["lifetimes"][name]
            torch.testing.assert_close(engine.state.log_alpha, donor["temperature"]["log_alpha"],
                                       rtol=0, atol=0)
            assert engine.state.replay.size == 0
            _assert_tree_equal(engine._diagnostic_previous_replay.training_state_dict(),
                               donor["replay"]["buffer"])
            inspected.append(True)

        engine._apply_diagnostic_initialization = inspect
        with engine.diagnostic_initialization(learner_state=donor, replay_fraction=.25):
            model.agent.act(torch.tensor([.8, .2, -.1]), t0=False, eval_mode=True)
        assert inspected == [True]
        carried = engine.export_diagnostic_state(include_optimizers=True, include_replay=True)
        for name in ("actor", "critic", "temperature"):
            assert carried["lifetimes"][name] == 2 * donor["lifetimes"][name]
        # The next handoff contains this solve's data, not recursively accumulated replay.
        assert carried["replay"]["buffer"]["state"]["next_sample_id"] == donor["replay"]["buffer"]["state"]["next_sample_id"]
        _assert_tree_equal(model.agent.checkpoint_state(), frozen)
    finally:
        model.close()


def test_previous_replay_preserves_boundary_and_termination_without_relabelling():
    model = _model_from_params(critic_params(inner_critic_scope="action",
        inner_rollout_horizon=3, inner_batch_size=8, inner_replay_capacity=32))
    try:
        engine = model.agent.inner_engine
        previous = engine._new_replay()

        def insert(buffer, marker, terminated, horizon_end):
            buffer.add_batch(
                torch.full((4, model.cfg.latent_dim), marker),
                torch.full((4, model.cfg.action_dim), marker / 10.),
                torch.full((4, 1), marker + 1.),
                torch.full((4, model.cfg.latent_dim), marker + 2.),
                torch.full((4, 1), terminated),
                horizon_end=torch.full((4, 1), horizon_end))

        insert(previous, 2., 0., 1.)
        saved = {"contract": engine._diagnostic_contract(), "buffer": previous.training_state_dict()}
        with engine.diagnostic_initialization(replay=saved, replay_fraction=.25):
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=True)
            engine._apply_diagnostic_initialization()
            insert(engine.state.replay, 3., 1., 0.)
            before = _clone_tree(engine.state.replay.training_state_dict())
            batch = engine._sample_batch()
            old = batch["previous_solve"]
            assert old.sum().item() == 2
            torch.testing.assert_close(batch["z"][old], torch.full((2, model.cfg.latent_dim), 2.))
            assert torch.all(batch["terminated"][old] == 0)
            assert torch.all(batch["horizon_end"][old] == 1)
            assert torch.all(batch["terminated"][~old] == 1)
            assert torch.all(batch["horizon_end"][~old] == 0)
            assert not any("target" in name for name in batch)
            _assert_tree_equal(engine.state.replay.training_state_dict(), before)
            _assert_tree_equal(engine._diagnostic_previous_replay.training_state_dict(), saved["buffer"])
        assert engine._diagnostic_previous_replay is None

        incompatible = deepcopy(saved)
        incompatible["contract"]["inner_rollout_horizon"] = 1
        with engine.diagnostic_initialization(replay=incompatible):
            with pytest.raises(ValueError, match="horizon/objective"):
                engine._apply_diagnostic_initialization()
    finally:
        model.close()
