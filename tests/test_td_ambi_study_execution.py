"""Execute the study's real update dose and lifetimes, beyond config fields.

Networks are narrowed for CPU; action dimension, distributional support,
population, replay, batches, horizon, rounds and optimizer settings are kept.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

# Match normal startup: MuJoCo's macOS platform probe runs before threaded
# numerical work, rather than forking lazily after the inner runtime tests.
import main as training_main
from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_inner_decoupling import _assert_tree_equal


ROOT = Path(__file__).resolve().parents[1]
FULL = (
    "reward_qscale_frozen", "reward_qscale_adaptive",
    "entropy_qscale_frozen", "entropy_qscale_adaptive",
    "entropy_autotemp", "reward_autotemp",
)
PRIOR = ("reward_qscale", "entropy_qscale", "entropy_autotemp", "reward_autotemp")


def _run_config(category, case):
    return json.loads((ROOT / f"configs/dmcontrol/algs/td_ambi_{category}_{case}.json").read_text())


def _agent(category, case, *, full_wrapper=False):
    run = _run_config(category, case)
    params = deepcopy(run["alg_params"])
    params.update(
        device="cpu", model_size=None, enc_dim=32, mlp_dim=32,
        latent_dim=16, num_enc_layers=2, simnorm_dim=8,
        compile=False, wandb=False,
    )
    # Resolve against Humanoid's 21-dimensional action space and 500-decision
    # episodes rather than overriding action_dim after Pendulum resolution.
    env = gym.Env()
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32)
    env.action_space = gym.spaces.Box(-1.0, 1.0, (21,), dtype=np.float32)
    env.spec = SimpleNamespace(max_episode_steps=500)
    if full_wrapper:
        return AMBITDMPC2("study-test", env, params, {**run, "device": "cpu"}, {})
    wrapper = object.__new__(AMBITDMPC2)
    wrapper.env = env
    wrapper.run_params = {**run, "device": "cpu"}
    wrapper.custom_params = params
    cfg = wrapper._build_cfg(params)
    return AMBITDMPC2Agent(cfg)


def _assert_finite_tree(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_finite_tree(item)


def _assert_optimizer_reset(optimizer):
    assert optimizer is not None
    for state in optimizer.state.values():
        assert state["step"].item() == 0
        assert not state["exp_avg"].count_nonzero()
        assert not state["exp_avg_sq"].count_nonzero()


@pytest.mark.parametrize("case", FULL)
def test_full_study_executes_18_paired_steps_and_reinherits_next_decision(case, monkeypatch):
    agent = _agent("full", case)
    engine = agent.inner_engine
    cfg = agent.cfg
    automatic = case == "entropy_autotemp"
    scaled = "qscale" in case
    adaptive = case.endswith("adaptive")
    assert cfg.action_dim == 21
    if automatic:
        assert cfg.inner_target_entropy == -441
    if scaled:
        agent.actor_loss_scale.fill_(7.25)
    if agent.log_ent_coef is not None:
        with torch.no_grad():
            agent.log_ent_coef.fill_(np.log(0.0004))
    # Make online and target weights distinguishable, so copying online Q into
    # the inner target cannot accidentally satisfy the initialization check.
    with torch.no_grad():
        for parameter in agent.model._target_Qs.parameters():
            parameter.add_(0.01)

    original_prepare = engine._prepare_workspace
    original_critic = engine._sac_critic_step
    original_policy = engine._sac_policy_step
    original_targets = engine._maybe_update_targets
    seen = {"events": [], "scale": [], "alpha": [], "batch": None}

    def prepare(**kwargs):
        original_prepare(**kwargs)
        state = engine.state
        _assert_tree_equal(state.actor.state_dict(), agent.model._pi.state_dict())
        _assert_tree_equal(state.critic.state_dict(), agent.model._Qs.state_dict())
        _assert_tree_equal(state.critic_target.state_dict(), agent.model._target_Qs.state_dict())
        _assert_optimizer_reset(state.actor_optim)
        _assert_optimizer_reset(state.critic_optim)
        for key in ("lr", "eps", "betas", "weight_decay"):
            assert state.actor_optim.param_groups[0][key] == agent.pi_optim.param_groups[0][key]
            assert state.critic_optim.param_groups[0][key] == agent.optim.param_groups[3][key]
        assert state.replay.size == 0
        torch.testing.assert_close(engine.alpha.reshape(()), agent.alpha.reshape(()), rtol=1e-6, atol=0)
        if automatic:
            _assert_optimizer_reset(state.temperature_optim)
            for key in ("lr", "eps", "betas", "weight_decay"):
                assert state.temperature_optim.param_groups[0][key] == agent.ent_coef_optim.param_groups[0][key]
        else:
            assert state.temperature_optim is None

    def critic(batch, alpha, **kwargs):
        update = len(seen["alpha"])
        assert engine.state.replay.size == (update + 1) * 512
        assert batch["z"].shape == (512, cfg.latent_dim)
        assert batch["action"].shape == (512, 21)
        assert engine.state.actor_steps == engine.state.critic_steps == update
        seen["events"].append("critic")
        seen["alpha"].append(alpha.detach().clone())
        seen["batch"] = batch
        if scaled and case.startswith("entropy"):
            seen["critic_scale"] = kwargs["actor_loss_scale"].clone()
        result = original_critic(batch, alpha, **kwargs)
        _assert_finite_tree(result)
        return result

    def policy(batch, **kwargs):
        assert batch is seen["batch"], "Critic and actor must use the paired minibatch"
        assert kwargs["update_actor"] is True
        assert kwargs["update_temperature"] is automatic
        seen["events"].append("actor")
        scale = kwargs["actor_loss_scale"]
        if scaled:
            assert scale.data_ptr() != agent.actor_loss_scale.data_ptr()
            if case.startswith("entropy"):
                torch.testing.assert_close(scale, seen["critic_scale"], rtol=0, atol=0)
            seen["scale"].append(scale.clone())
        else:
            assert scale is None
        result = original_policy(batch, **kwargs)
        _assert_finite_tree(result)
        return result

    def targets(**kwargs):
        assert kwargs == {"critic_updated": True, "actor_updated": True}
        seen["events"].append("target")
        expected = [
            old.detach() * 0.99 + online.detach() * 0.01
            for old, online in zip(engine.state.critic_target.parameters(), engine.state.critic.parameters())
        ]
        result = original_targets(**kwargs)
        for actual, reference in zip(engine.state.critic_target.parameters(), expected):
            torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-7)
        # act() expires action-local state before returning. Capture its last
        # live values without disabling that production cleanup path.
        if engine.state.actor_steps == 18:
            seen["final"] = SimpleNamespace(**vars(engine.state))
            seen["final_alpha"] = engine.alpha.detach().clone()
        return result

    monkeypatch.setattr(engine, "_prepare_workspace", prepare)
    monkeypatch.setattr(engine, "_sac_critic_step", critic)
    monkeypatch.setattr(engine, "_sac_policy_step", policy)
    monkeypatch.setattr(engine, "_maybe_update_targets", targets)

    for decision in range(2):
        if decision:
            # New outer training values must replace the previous local values.
            with torch.no_grad():
                if scaled:
                    agent.actor_loss_scale.fill_(12.5)
                if agent.log_ent_coef is not None:
                    agent.log_ent_coef.fill_(np.log(0.0007))
                for parameter in agent.model._pi.parameters():
                    parameter.add_(0.0001)
                for parameter in agent.model._target_Qs.parameters():
                    parameter.add_(0.0002)
        outer = deepcopy(agent.checkpoint_state())
        rng = torch.random.get_rng_state().clone()
        inherited_alpha = agent.alpha.detach().clone()
        inherited_scale = agent.actor_loss_scale.detach().clone() if scaled else None
        seen.update(events=[], scale=[], alpha=[], batch=None)
        action = agent.act(torch.zeros(67), t0=(decision == 0), collect_diagnostics=False)
        state = seen["final"]
        assert engine.state.actor is engine.state.critic is engine.state.replay is None
        assert action.shape == (21,)
        assert torch.isfinite(action).all() and (action.abs() <= 1).all()
        assert seen["events"] == ["critic", "actor", "target"] * 18
        assert state.replay.size == 9216
        assert state.critic_steps == state.actor_steps == state.critic_target_steps == 18
        assert state.temperature_steps == (18 if automatic else 0)
        for optimizer in (state.actor_optim, state.critic_optim):
            assert {item["step"].item() for item in optimizer.state.values()} == {18}
            _assert_finite_tree(optimizer.state_dict())
        _assert_finite_tree(state.actor.state_dict())
        _assert_finite_tree(state.critic.state_dict())
        _assert_finite_tree(state.critic_target.state_dict())
        _assert_tree_equal(agent.checkpoint_state(), outer)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(seen["alpha"][0].reshape(()), inherited_alpha.reshape(()), rtol=1e-6, atol=0)
        if automatic:
            assert {item["step"].item() for item in state.temperature_optim.state.values()} == {18}
            assert not torch.equal(seen["final_alpha"].reshape(()), inherited_alpha.reshape(()))
            _assert_finite_tree(state.temperature_optim.state_dict())
        else:
            for alpha in seen["alpha"]:
                torch.testing.assert_close(alpha.reshape(()), inherited_alpha.reshape(()), rtol=1e-6, atol=0)
        if scaled:
            torch.testing.assert_close(seen["scale"][0], inherited_scale, rtol=0, atol=0)
            final_scale = torch.as_tensor(agent.last_inner_metrics["inner_actor_loss_scale"])
            if adaptive:
                assert not torch.equal(final_scale.reshape(1), inherited_scale)
            else:
                for scale in seen["scale"] + [final_scale.reshape(1)]:
                    torch.testing.assert_close(scale, inherited_scale, rtol=0, atol=0)


@pytest.mark.parametrize("case", PRIOR)
def test_prior_study_executes_stochastic_outer_policy_without_imagination(case, monkeypatch):
    agent = _agent("prior", case)
    engine = agent.inner_engine
    if agent.actor_loss_scale_enabled:
        agent.actor_loss_scale.fill_(19.5)
    outer = deepcopy(agent.checkpoint_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("Prior-only action unexpectedly entered imagined work")

    monkeypatch.setattr(engine, "_prepare_workspace", forbidden)
    monkeypatch.setattr(engine, "_act_rl", forbidden)
    monkeypatch.setattr(engine, "_act_mppi", forbidden)
    monkeypatch.setattr(agent.model, "next", forbidden)
    monkeypatch.setattr(agent.model, "reward", forbidden)
    monkeypatch.setattr(agent.model, "Q", forbidden)
    actions = [agent.act(torch.zeros(67), t0=(i == 0), collect_diagnostics=False) for i in range(3)]
    assert all(action.shape == (21,) for action in actions)
    assert not torch.equal(actions[0], actions[1])
    assert agent.last_inner_metrics["inner_model_steps"] == 0
    assert agent.last_inner_metrics["inner_actor_optimizer_steps"] == 0
    assert agent.last_inner_metrics["inner_critic_optimizer_steps"] == 0
    _assert_tree_equal(agent.checkpoint_state(), outer)


@pytest.mark.parametrize("category", ("prior", "full"))
def test_main_resolves_complete_study_manifests_and_checkpoint_policy(category, monkeypatch, tmp_path):
    from tests.test_main_checkpointing import _Env, _Model

    path = ROOT / f"configs/dmcontrol/experiments/td_ambi_{category}_study.json"
    manifest = json.loads(path.read_text())
    captured = []

    def initialize(algorithm, params, env, *, full_run_params, experiment_params):
        model = _Model()
        model.set_logger = lambda logger: None
        captured.append((algorithm, deepcopy(params), deepcopy(full_run_params), model))
        return model, False, "AMBITDMPC2"

    monkeypatch.setattr(training_main, "build_env", lambda *args, **kwargs: _Env())
    monkeypatch.setattr(training_main, "initialize_alg", initialize)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--run", str(path), "--alg-dir", str(ROOT / "configs/dmcontrol/algs"),
        "--log-dir", str(tmp_path),
    ])
    training_main.main()
    assert len(captured) == (4 if category == "prior" else 6)
    assert [run["name"] for _, _, run, _ in captured] == manifest["configs"]
    for algorithm, params, run, model in captured:
        source = json.loads((ROOT / f'configs/dmcontrol/algs/{run["name"]}.json').read_text())
        assert algorithm == "AMBITDMPC2/AMBITDMPC2"
        assert params == source["alg_params"]
        assert run["seed"] == 55
        assert run["total_steps"] == 2_000_000
        assert model.learn_calls == [{"total_timesteps": 2_000_000}]
        assert model.save_calls == []
        assert len(model.checkpoint_calls) == 1
        call = model.checkpoint_calls[0]
        assert call["save_freq"] == 25_000
        assert call["save_strat"] == ("all",)
        assert call["trial_run_params"]["alg_params"] == params


@pytest.mark.parametrize(
    "category,case",
    [("prior", case) for case in PRIOR] + [("full", case) for case in FULL],
)
def test_study_periodic_checkpoint_captures_scale_or_alpha_with_networks(category, case, tmp_path):
    model = _agent(category, case, full_wrapper=True)
    agent = model.agent
    try:
        model.set_checkpointing(25_000, tmp_path, "prior", save_strat="all")
        model._global_step = 24_999
        model._maybe_checkpoint()
        model.flush_checkpoints()
        assert not list(tmp_path.iterdir())
        expected = []
        for step in (25_000, 50_000):
            model._global_step = step
            agent.num_updates = model._num_updates = step // 2
            with torch.no_grad():
                if agent.actor_loss_scale_enabled:
                    agent.actor_loss_scale.fill_(step / 1000)
                if agent.log_ent_coef is not None:
                    agent.log_ent_coef.fill_(np.log(1.0 / step))
            expected.append(deepcopy(agent.checkpoint_state()))
            model._maybe_checkpoint()
            # Mutate immediately after queuing to detect deferred/live aliases.
            with torch.no_grad():
                next(agent.model.parameters()).add_(0.001)
                if agent.actor_loss_scale_enabled:
                    agent.actor_loss_scale.fill_(999.0)
                if agent.log_ent_coef is not None:
                    agent.log_ent_coef.fill_(-1.0)
        model.flush_checkpoints()
        for step, state in zip((25_000, 50_000), expected):
            saved = torch.load(tmp_path / f"prior_{step}", weights_only=False)
            _assert_tree_equal(saved, state)
            if "qscale" in case:
                assert "actor_loss_scale_state" in saved
            else:
                assert "log_ent_coef" in saved
            assert (tmp_path / f"prior_{step}.metadata.json").is_file()
    finally:
        model._checkpoint_writer.shutdown()
        model.env.close()
