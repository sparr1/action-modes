"""Small CPU training-harness tests; never publish or submit experiments."""
import copy
import json
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2


class TinyEnv(gym.Env):
    def __init__(self, fail_at=None):
        self.observation_space = gym.spaces.Box(-10., 10., shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.actions = []
        self.resets = 0
        self.fail_at = fail_at
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.state = self.np_random.uniform(-.1, .1, size=3).astype(np.float32)
        return self.state.copy(), {}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        if len(self.actions) == self.fail_at:
            raise RuntimeError("injected simulator failure")
        self.state[:2] += .01 * action
        self.state[2] += .02
        return self.state.copy(), float(1. - .1 * np.square(action).sum()), False, False, {}


def build(tmp_path, enabled=True, *, fail_at=None, **overrides):
    env = gym.wrappers.TimeLimit(TinyEnv(fail_at=fail_at), max_episode_steps=4)
    params = dict(
        device="cpu", model_size=None, enc_dim=16, mlp_dim=32, latent_dim=16,
        num_enc_layers=2, num_q=2, simnorm_dim=8, q_num_bins=11,
        batch_size=2, train_unroll_horizon=2, outer_planning_horizon=2,
        inner_rollout_horizon=2, buffer_size=100, seed_steps=4, pretrain_steps=3,
        utd=1, compile=False, episodic=False, discount=.99, dropout=.01,
        wandb=False, mpc=False, inner_operator="none", inner_rounds=0,
        inner_rollouts_per_round=0, inner_updates_per_round=0,
        ent_coef="auto_1.0", target_entropy=-2., log_std_min=-10., log_std_max=2.,
        outer_q_actor_reduction="mean_pair", outer_q_target_reduction="min_pair",
        outer_policy_diagnostics=enabled, outer_policy_diagnostics_early_every=2,
        outer_policy_diagnostics_early_until=4, outer_policy_diagnostics_every=3,
        outer_policy_diagnostics_states=3, outer_policy_diagnostics_samples=4,
    )
    params.update(overrides)
    learner = AMBITDMPC2("AMBI", env, params,
                         run_params={"seed": 55, "device": "cpu", "total_steps": 8})
    learner.set_checkpointing(100, tmp_path, "test", save_strat="all")
    return learner, env.unwrapped


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def run_and_snapshot(learner, env):
    replay_draws = 0
    original_sample = learner.buffer.sample

    def sample(*args, **kwargs):
        nonlocal replay_draws
        replay_draws += 1
        return original_sample(*args, **kwargs)

    learner.buffer.sample = sample
    learner.learn(total_timesteps=8)
    assert replay_draws == learner._num_updates
    return dict(
        model=copy.deepcopy(learner.agent.model.state_dict()),
        optim=copy.deepcopy(learner.agent.optim.state_dict()),
        actor_optim=copy.deepcopy(learner.agent.pi_optim.state_dict()),
        alpha_optim=copy.deepcopy(learner.agent.ent_coef_optim.state_dict()),
        alpha=learner.agent.alpha.clone(),
        gradients=[None if p.grad is None else p.grad.clone() for p in learner.agent.model.parameters()],
        actions=np.asarray(env.actions), resets=env.resets,
        rng=(random.getstate(), np.random.get_state(), torch.get_rng_state()),
        env_rng=copy.deepcopy(env.np_random.bit_generator.state),
        modes=[m.training for m in learner.agent.model.modules()],
        updates=learner._num_updates, replay_draws=replay_draws,
    )


def test_recording_preserves_complete_harness_and_exposes_pretraining(tmp_path):
    off, off_env = build(tmp_path / "off", False)
    off_state = run_and_snapshot(off, off_env)
    on, on_env = build(tmp_path / "on", True)
    on_state = run_and_snapshot(on, on_env)
    equal(off_state, on_state)
    recorder = on._outer_policy_recorder
    rows = recorder.rows
    assert rows[0]["source"] == "initial_observation"
    assert rows[0]["phase"] == "initialization"
    assert rows[0]["updates_completed"] == rows[0]["env_step"] == 0
    bank = [row for row in rows if row["source"] == "reference_bank"]
    assert [(row["updates_completed"], row["env_step"]) for row in bank] == [
        (0, 5), (2, 5), (3, 5), (4, 6), (6, 8),
    ]
    assert bank[0]["phase"] == "pretrain_before"
    assert bank[2]["phase"] == "pretrain_after"
    assert bank[-1]["phase"] == "final"
    assert bank[0]["observation_count"] == 3
    assert all(row["histograms"]["log_std_pooled"]["count"] == 6 for row in bank)
    learner_rows = [row for row in rows if row["source"] == "learner"]
    assert [(row["actor_updates_before"], row["actor_updates_after"]) for row in learner_rows] == [(1,2),(2,3),(3,4),(5,6)]
    sources = {row["source"] for row in rows}
    assert {"executed_uniform", "executed_prior"} <= sources
    assert json.loads((recorder.directory / "manifest.json").read_text())["status"] == "complete"
    persisted = [json.loads(line) for line in (recorder.directory / "events.jsonl").read_text().splitlines()]
    assert persisted == rows
    equal(off.cfg.outer_policy_diagnostics, False)


def test_error_retains_incomplete_trace_and_cleans_force(tmp_path):
    learner, _ = build(tmp_path, fail_at=7)
    with pytest.raises(RuntimeError, match="injected simulator failure"):
        learner.learn(total_timesteps=8)
    assert learner.agent._outer_policy_diagnostics_force is False
    manifest = json.loads((learner._outer_policy_recorder.directory / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert not any(row["phase"] == "final" and row["source"] == "reference_bank" for row in learner._outer_policy_recorder.rows)
    assert sum(row["metrics"]["decision_count"] for row in learner._outer_policy_recorder.rows
               if row["source"].startswith("executed_")) == 6


@pytest.mark.parametrize("key,value", [
    ("outer_policy_diagnostics", "yes"),
    ("outer_policy_diagnostics_early_every", 0),
    ("outer_policy_diagnostics_every", 1.5),
    ("outer_policy_diagnostics_states", 0),
    ("outer_policy_diagnostics_samples", True),
    ("outer_policy_diagnostics_seed", -1),
    ("outer_policy_diagnostics_early_until", -1),
])
def test_invalid_diagnostic_config_rejected(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        build(tmp_path, **{key: value})


def test_diagnostics_reject_wandb_transport_that_drops_pretraining(tmp_path):
    with pytest.raises(ValueError, match="wandb_event_indexed"):
        build(tmp_path, wandb=True)


def test_diagnostics_require_prior_collection(tmp_path):
    with pytest.raises(ValueError, match="prior-only"):
        build(tmp_path, inner_operator="sac")


def test_recorder_resume_roundtrip_does_not_resample_or_republish(tmp_path):
    from utils.outer_policy_diagnostics import OuterPolicyDiagnostics
    learner, env = build(tmp_path / "first")
    run_and_snapshot(learner, env)
    recorder = learner._outer_policy_recorder
    state = recorder.state_dict()
    validated = OuterPolicyDiagnostics.validate_state(state, learner.cfg)
    rng = torch.get_rng_state().clone()
    resumed = OuterPolicyDiagnostics(learner.cfg, tmp_path / "resumed", state=validated)
    assert torch.equal(rng, torch.get_rng_state())
    restored = resumed.state_dict()
    equal({key: value for key, value in state.items() if key != "timing"},
          {key: value for key, value in restored.items() if key != "timing"})
    # Restoring the local bundle itself adds serialization work.
    assert restored["timing"]["serialization_seconds"] >= state["timing"]["serialization_seconds"]
    resumed.probe(learner.agent, env_step=8, updates=6, phase="final", run=None)
    assert resumed.rows == recorder.rows
    bad = copy.deepcopy(state)
    bad["noise"] = torch.zeros(1)
    with pytest.raises(ValueError, match="noise"):
        OuterPolicyDiagnostics.validate_state(bad, learner.cfg)


def test_wrapper_checkpoint_retains_diagnostic_resume_state(tmp_path):
    learner, env = build(tmp_path / "first")
    run_and_snapshot(learner, env)
    learner._reset_wandb_window()
    learner._prepare_resume_boundary()
    state = learner._training_resume_algorithm_state()
    expected = learner._outer_policy_recorder.state_dict()
    equal(state["outer_policy_diagnostics"], expected)
    restored, _ = build(tmp_path / "restored")
    restored._load_training_resume_algorithm_state(state)
    equal(restored._outer_policy_recorder_state, expected)
    disabled, _ = build(tmp_path / "disabled", False)
    assert "outer_policy_diagnostics" not in disabled._training_resume_algorithm_state()
    with pytest.raises(ValueError, match="fields"):
        disabled._preflight_training_resume_algorithm_state(state)
