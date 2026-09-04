import copy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_tdmpc2_mppi_action_mc as evaluator
from render_checkpoint import RenderContext


class _FakeWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sentinel", torch.tensor([3.0]))


class _FakeModel:
    def __init__(self, *, horizon=4):
        cfg = SimpleNamespace(
            mpc=True,
            episode_length=horizon,
            action_dim=1,
            iterations=8,
            outer_planning_horizon=3,
            num_samples=16,
            num_elites=4,
            num_pi_trajs=2,
        )
        self.agent = SimpleNamespace(
            cfg=cfg,
            device=torch.device("cpu"),
            model=_FakeWorldModel(),
            num_updates=17,
            discount=0.9,
            _prev_mean=torch.full((3, 1), 7.0),
            last_plan_metrics={"original": 5.0},
            _resume_boundary_prepared=False,
        )
        self.cfg = cfg
        self._predict_t0 = False
        self.close_calls = 0

    def reset(self):
        self.agent._prev_mean.zero_()
        self.agent.last_plan_metrics = {}
        self._predict_t0 = True

    def predict(self, observation, *, deterministic, episode_start):
        assert deterministic is True
        assert np.asarray(observation).shape == (2,)
        if episode_start:
            self.reset()
        self._predict_t0 = False
        if self.cfg.mpc:
            value = float(torch.rand(()))
            self.agent._prev_mean.fill_(value)
            self.agent.last_plan_metrics = {"planner_value_mean": value}
            return np.array([1.0], dtype=np.float32), None
        self.agent.last_plan_metrics = {}
        return np.array([0.0], dtype=np.float32), None

    def close(self):
        self.close_calls += 1


class _FakeEnv:
    def __init__(self, *, horizon=4, reward_offset=0.0):
        self.horizon = horizon
        self.reward_offset = reward_offset
        self.action_space = SimpleNamespace(seed=lambda seed: None)
        self.observation_space = SimpleNamespace(seed=lambda seed: None)
        self.close_calls = 0
        self.reset_seeds = []
        self.seed = 0
        self.step_index = 0

    def reset(self, *, seed):
        self.seed = int(seed)
        self.step_index = 0
        self.reset_seeds.append(self.seed)
        return np.array([float(self.seed), 0.0], dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        reward = float(np.asarray(action).reshape(-1)[0]) + self.reward_offset
        return (
            np.array([float(self.seed), float(self.step_index)], dtype=np.float32),
            reward,
            False,
            self.step_index == self.horizon,
            {},
        )

    def close(self):
        self.close_calls += 1


def _context(tmp_path):
    return RenderContext(
        trial_run_params={
            "alg": "TDMPC2/TDMPC2Baseline",
            "env": "DMControl-v0",
            "seed": 1,
            "alg_params": {"obs": "state", "iterations": 6},
            "resolved_runtime": {
                "horizons": {
                    "train_unroll_horizon": 3,
                    "outer_planning_horizon": 3,
                }
            },
        },
        experiment_params={
            "env_params": {"task": "humanoid-walk", "obs": "state"}
        },
        source=tmp_path / "metadata.json",
    )


def _predicted_gain(*args, **kwargs):
    del args, kwargs
    return {
        "target_q_mppi_mean_all": 2.0,
        "target_q_policy_prior_mean_all": 1.0,
        "target_q_mppi_minus_policy_prior": 1.0,
        "policy_prior_to_mppi_action_l2": 1.0,
        "policy_prior_action_at_mppi_state": [0.0],
        "diagnostic_seconds": 0.0,
    }


def _behavior_payload(checkpoint, *, episodes=2, horizon=4, reward=1.0):
    rows = []
    for episode in range(episodes):
        environment_seed = 101 + episode
        cumulative = 0.0
        steps = []
        for step in range(horizon):
            cumulative += reward
            steps.append(
                {
                    "step": step,
                    "action": [1.0],
                    "planner": {"planner_value_mean": 2.0},
                    "predicted_action_gain": _predicted_gain(),
                    "reward": reward,
                    "cumulative_return": cumulative,
                    "terminated": False,
                    "truncated": step == horizon - 1,
                }
            )
        rows.append(
            {
                "episode": episode + 1,
                "environment_seed": environment_seed,
                "native_mppi": {
                    "controller": "native_mppi",
                    "controller_seed": evaluator._namespaced_seed(
                        12345, "native_mppi", environment_seed
                    ),
                    "return": cumulative,
                    "length": horizon,
                    "terminated": False,
                    "truncated": True,
                    "capped": False,
                    "steps": steps,
                },
            }
        )
    return {
        "schema_version": 1,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "algorithm": "TDMPC2/TDMPC2Baseline",
        "environment": "DMControl-v0",
        "protocol": {
            "controllers": ["policy_prior_mean", "native_mppi"],
            "controller_seed_base": 12345,
            "environment_seed_first": 101,
            "environment_seed_last": 100 + episodes,
            "max_steps": None,
        },
        "episodes": rows,
    }


def _patch_runtime(monkeypatch, tmp_path, model, envs):
    context = _context(tmp_path)
    queue = list(envs)
    monkeypatch.setattr(evaluator, "resolve_checkpoint_path", lambda path: Path(path))
    monkeypatch.setattr(
        evaluator, "resolve_render_context", lambda *args, **kwargs: context
    )
    monkeypatch.setattr(evaluator, "_backend_for", lambda algorithm: "tdmpc2")
    monkeypatch.setattr(
        evaluator,
        "_prepare_run_params",
        lambda *args, **kwargs: (
            copy.deepcopy(context.trial_run_params),
            copy.deepcopy(context.experiment_params),
        ),
    )
    monkeypatch.setattr(evaluator, "build_env", lambda *args, **kwargs: queue.pop(0))
    monkeypatch.setattr(evaluator, "_initialize_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(evaluator, "_predicted_action_gain", _predicted_gain)


def test_parser_and_anchor_defaults_are_stable(tmp_path):
    args = evaluator.build_parser().parse_args(
        [str(tmp_path / "checkpoint"), "--output", str(tmp_path / "out.json")]
    )

    assert args.episodes == 12
    assert args.block_size == 25
    assert args.action_draws == 4
    first = evaluator._anchor_steps(500, 25, seed_base=12345, environment_seed=101)
    second = evaluator._anchor_steps(500, 25, seed_base=12345, environment_seed=101)
    assert first == second
    assert len(first) == 20
    assert all(row["block_start"] <= row["step"] < row["block_stop_exclusive"] for row in first)


def test_behavior_json_realized_action_mc_is_paired_and_restores_state(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint bytes")
    behavior_path = tmp_path / "paired.json"
    behavior_path.write_text(json.dumps(_behavior_payload(checkpoint)))
    output = tmp_path / "same-state.json"
    model = _FakeModel()
    envs = [_FakeEnv(), _FakeEnv()]
    _patch_runtime(monkeypatch, tmp_path, model, envs)

    original_mean = model.agent._prev_mean.clone()
    original_metrics = copy.deepcopy(model.agent.last_plan_metrics)
    random.seed(701)
    np.random.seed(702)
    torch.manual_seed(703)
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()

    payload = evaluator.evaluate_tdmpc2_mppi_action_mc(
        checkpoint,
        output=output,
        behavior_json=behavior_path,
        episodes=2,
        seed=101,
        block_size=2,
        bootstrap_samples=100,
        device="cpu",
    )

    assert json.loads(output.read_text()) == payload
    assert payload["behavior_source"]["path"] == str(behavior_path.resolve())
    assert payload["behavior_source"]["sha256"] == hashlib.sha256(
        behavior_path.read_bytes()
    ).hexdigest()
    assert payload["protocol"]["effective_action_draws"] == 1
    assert payload["frozen_state"][
        "global_rng_streams_present_at_entry_restored"
    ] is True
    assert payload["summary"]["anchors_per_episode"] == 2
    assert payload["compute_accounting"] == {
        "total_anchors": 4,
        "counterfactual_branches": 8,
        "prefix_reconstructions": 8,
        "environment_decisions_in_prefix_plus_suffix_branches": 32,
        "environment_decisions_generating_behavior": 0,
        "native_mppi_planner_calls_in_this_evaluator": 0,
        "native_mppi_model_transitions_in_this_evaluator": 0,
        "recorded_behavior_planner_calls_reused_without_rerun": 8,
        "target_q_diagnostic_calls": 4,
    }
    assert payload["summary"]["undiscounted_mc_gain"]["mean"] == 1.0
    assert payload["summary"]["discounted_mc_gain"]["mean"] == 1.0
    assert payload["summary"]["undiscounted_mc_gain"][
        "conditional_episode_cluster_bootstrap_95_interval"
    ] == [1.0, 1.0]
    assert len(payload["block_summary"]) == 2
    assert all(
        len(anchor["mppi_action_draws"]) == 1
        for episode in payload["episodes"]
        for anchor in episode["anchors"]
    )
    first_draw = payload["episodes"][0]["anchors"][0]["mppi_action_draws"][0]
    assert first_draw["planner_seed"] is None
    assert first_draw["behavior_episode_stream_seed"] == evaluator._namespaced_seed(
        12345, "native_mppi", 101
    )
    assert first_draw["branch"]["initial_reward"] == 1.0
    assert first_draw[
        "source_policy_prior_action_matches_recomputed_baseline_exactly"
    ] is True
    assert first_draw["source_predicted_action_gain_comparison"][
        "all_scientific_fields_exact"
    ] is True
    assert payload["block_summary"][0]["undiscounted_mc_gain_median"] == 1.0
    assert payload["block_summary"][0][
        "undiscounted_mc_gain_positive_fraction"
    ] == 1.0
    assert all(
        anchor["mppi_action_draws"][0]["prefix_replay"][
            "rewards_checked_exactly"
        ]
        == anchor["step"]
        for episode in payload["episodes"]
        for anchor in episode["anchors"]
    )
    torch.testing.assert_close(model.agent._prev_mean, original_mean)
    assert model.agent.last_plan_metrics == original_metrics
    assert model.cfg.mpc is True
    assert model.close_calls == 1
    assert [env.close_calls for env in envs] == [1, 1]
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)


def test_rng_comparison_ignores_new_cuda_streams_but_checks_preexisting_ones():
    base = evaluator._capture_global_rng()
    current_new_cuda = (base[0], base[1], base[2], (torch.tensor([1], dtype=torch.uint8),))
    assert evaluator._rng_state_matches(current_new_cuda, base)

    expected_existing = (
        base[0],
        base[1],
        base[2],
        (torch.tensor([1], dtype=torch.uint8),),
    )
    assert evaluator._rng_state_matches(current_new_cuda, expected_existing)
    changed_existing = (
        base[0],
        base[1],
        base[2],
        (torch.tensor([2], dtype=torch.uint8),),
    )
    assert not evaluator._rng_state_matches(changed_existing, expected_existing)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload.update(checkpoint_sha256="bad"), "checkpoint SHA-256"),
        (
            lambda payload: payload["episodes"][0]["native_mppi"].update(capped=True),
            "uncapped",
        ),
        (
            lambda payload: payload["episodes"][0]["native_mppi"]["steps"][0].update(
                action=[float("nan")]
            ),
            "finite values",
        ),
    ],
)
def test_behavior_json_validation_rejects_bad_provenance_and_values(
    tmp_path, mutate, match
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint bytes")
    payload = _behavior_payload(checkpoint)
    mutate(payload)

    with pytest.raises(evaluator.TDMPC2MPPIEvaluationError, match=match):
        evaluator._validate_behavior_json(
            payload,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            algorithm="TDMPC2/TDMPC2Baseline",
            environment="DMControl-v0",
            first_seed=101,
            episodes=2,
            controller_seed=12345,
            episode_length=4,
            action_dim=1,
        )


def test_prefix_mismatch_fails_without_output_and_restores_rng(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint bytes")
    behavior_path = tmp_path / "paired.json"
    behavior_path.write_text(json.dumps(_behavior_payload(checkpoint)))
    output = tmp_path / "never-written.json"
    model = _FakeModel()
    envs = [_FakeEnv(), _FakeEnv(reward_offset=0.25)]
    _patch_runtime(monkeypatch, tmp_path, model, envs)
    original_mean = model.agent._prev_mean.clone()
    random.seed(801)
    np.random.seed(802)
    torch.manual_seed(803)
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()

    with pytest.raises(
        evaluator.TDMPC2MPPIEvaluationError,
        match="prefix replay|Recorded MPPI action",
    ):
        evaluator.evaluate_tdmpc2_mppi_action_mc(
            checkpoint,
            output=output,
            behavior_json=behavior_path,
            episodes=2,
            seed=101,
            block_size=2,
            bootstrap_samples=10,
            device="cpu",
        )

    assert not output.exists()
    torch.testing.assert_close(model.agent._prev_mean, original_mean)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)


def test_saved_prior_action_must_match_recomputed_baseline_exactly(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint bytes")
    payload = _behavior_payload(checkpoint, episodes=1)
    for step in payload["episodes"][0]["native_mppi"]["steps"]:
        step["predicted_action_gain"]["policy_prior_action_at_mppi_state"] = [0.25]
    behavior_path = tmp_path / "paired.json"
    behavior_path.write_text(json.dumps(payload))
    model = _FakeModel()
    _patch_runtime(monkeypatch, tmp_path, model, [_FakeEnv(), _FakeEnv()])

    with pytest.raises(
        evaluator.TDMPC2MPPIEvaluationError,
        match="policy-prior baseline action",
    ):
        evaluator.evaluate_tdmpc2_mppi_action_mc(
            checkpoint,
            output=tmp_path / "never-written.json",
            behavior_json=behavior_path,
            episodes=1,
            seed=101,
            block_size=2,
            bootstrap_samples=10,
            device="cpu",
        )


def test_generated_behavior_uses_four_draws_per_anchor(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint bytes")
    output = tmp_path / "generated.json"
    model = _FakeModel()
    envs = [_FakeEnv(), _FakeEnv(), _FakeEnv()]
    _patch_runtime(monkeypatch, tmp_path, model, envs)

    payload = evaluator.evaluate_tdmpc2_mppi_action_mc(
        checkpoint,
        output=output,
        episodes=1,
        seed=101,
        controller_seed=0,
        block_size=2,
        action_draws=4,
        bootstrap_samples=10,
        device="cpu",
    )

    assert payload["behavior_source"]["mode"] == "generated_native_mppi"
    assert payload["protocol"]["effective_action_draws"] == 4
    assert all(
        len(anchor["mppi_action_draws"]) == 4
        for anchor in payload["episodes"][0]["anchors"]
    )
    first_anchor_draws = payload["episodes"][0]["anchors"][0]["mppi_action_draws"]
    assert first_anchor_draws[0]["planner_seed"] is None
    assert first_anchor_draws[0][
        "behavior_episode_stream_seed"
    ] == evaluator._namespaced_seed(0, "native_mppi", 101)
    assert all(row["planner_seed"] is not None for row in first_anchor_draws[1:])
    assert payload["protocol"]["action_draw_breakdown"] == {
        "recorded_behavior_actions": 0,
        "untouched_generated_behavior_stream_actions": 1,
        "fresh_namespaced_observational_actions": 3,
    }
    assert payload["summary"]["undiscounted_mc_gain"]["mean"] == 1.0
