import copy
import importlib

import gymnasium as gym
import numpy as np
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from tests.resume_test_support import BoundaryEnv, _model as resume_model
from tests.test_tdmpc2_correctness import make_model, tiny_params
from utils.wandb_utils import EventIndexedWandbRun


class CaptureRun:
    def __init__(self):
        self.rows = []

    def log(self, payload, *, step):
        assert step == len(self.rows)
        self.rows.append(dict(payload))


class ConstantEpisodeEnv(BoundaryEnv):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(-2., 2., shape=(2,), dtype=np.float32)

    def step(self, action):
        observation, _reward, terminated, truncated, info = super().step(action)
        return observation, 1., terminated, truncated, info


@pytest.mark.parametrize("algorithm", [TDMPC2Baseline, AMBITDMPC2])
@pytest.mark.parametrize("logging", [True, False])
def test_shared_loop_observes_execution_once_and_excludes_reset_deltas(
    monkeypatch, algorithm, logging
):
    module = importlib.import_module("RL.TDMPC2")
    raw_run = CaptureRun()
    run = EventIndexedWandbRun(raw_run, wandb_module=None)
    monkeypatch.setattr(module, "init_wandb", lambda *_args, **_kwargs: run if logging else None)
    monkeypatch.setattr(module, "finish_wandb", lambda *_args: None)
    model = make_model(
        algorithm, ConstantEpisodeEnv(),
        tiny_params(seed_steps=100, episode_length=2, train_unroll_horizon=1,
                    wandb=logging, wandb_step_every=6, mpc=False), total_steps=6,
    )
    # The first action deliberately exceeds bounds: actual execution clips it.
    actions = iter([[2., 0.], [-1., 0.], [1., 0.], [1., 0.], [-1., 0.], [1., 0.]])
    monkeypatch.setattr(model, "_random_action_norm", lambda: np.asarray(next(actions), dtype=np.float32))
    monkeypatch.setattr(model.agent, "act", lambda *_args, **_kwargs: pytest.fail("Unexpected actor forward"))
    model.learn(total_timesteps=6)
    state = model._training_diagnostics.state_dict()
    assert state["episodes"] == 3
    assert state["recent_returns"] == [2., 2., 2.]
    assert model._training_diagnostics.episode_metrics()["episode/return_window20_count"] == 3
    if not logging:
        assert raw_run.rows == []
        return
    episode_rows = [row for row in raw_run.rows if "episode/return" in row]
    assert [row["env_step"] for row in episode_rows] == [2, 4, 6]
    assert [row["episode/return_diagnostic_episodes"] for row in episode_rows] == [1, 2, 3]
    assert [row["episode/return_window20_count"] for row in episode_rows] == [1, 2, 3]
    assert "train/executed_action_samples" not in episode_rows[0]
    aggregate = episode_rows[-1]
    prefix = "train/executed_action_"
    assert aggregate[prefix + "samples"] == 6
    assert aggregate[prefix + "finite_coordinates"] == 12
    assert aggregate[prefix + "transition_samples"] == 3
    assert aggregate[prefix + "transition_coordinates"] == 6
    assert aggregate[prefix + "abs_ge_0p99_fraction"] == .5
    assert aggregate[prefix + "exact_saturation_fraction"] == .5
    assert aggregate[prefix + "sign_flip_fraction"] == pytest.approx(1 / 3)
    assert aggregate[prefix + "bound_flip_fraction"] == pytest.approx(1 / 3)
    assert aggregate[prefix + "delta_rms"] == pytest.approx(2 * np.sqrt(2) / 3)
    # Reading episode payloads repeatedly cannot advance return coverage.
    model._episode_payload_extras()
    model._episode_payload_extras()
    assert model._training_diagnostics.state_dict() == state


@pytest.mark.parametrize("algorithm", [TDMPC2Baseline, AMBITDMPC2])
def test_wrapper_restores_return_history_and_accepts_legacy_without_fabrication(algorithm):
    source = resume_model(BoundaryEnv(), algorithm=algorithm, total_steps=100)
    target = resume_model(BoundaryEnv(), algorithm=algorithm, total_steps=100)
    legacy_target = resume_model(BoundaryEnv(), algorithm=algorithm, total_steps=100)
    try:
        # Build a valid boundary without running a training campaign.
        for value in [10.] * 20 + [4.] * 20:
            source._training_diagnostics.complete_episode(value)
        source._global_step = source._wandb_window_start_step = 80
        source._episode_idx = 40
        source._resume_phase = "between_episodes_before_reset"
        source.agent.prepare_training_resume_boundary()
        saved = source.training_state_dict()
        assert saved["version"] == 5
        target.load_training_state_dict(saved)
        assert target._training_diagnostics.state_dict() == saved["training_diagnostics"]
        for value in [3., 2., 1.]:
            source._training_diagnostics.complete_episode(value)
            target._training_diagnostics.complete_episode(value)
            assert target._episode_payload_extras() == source._episode_payload_extras()
        legacy = copy.deepcopy(saved)
        legacy["version"] = 4
        del legacy["training_diagnostics"]
        legacy_target.load_training_state_dict(legacy)
        assert legacy_target._episode_idx == 40
        metrics = legacy_target._episode_payload_extras()
        assert metrics["episode/return_diagnostic_episodes"] == 0
        assert metrics["episode/return_window20_count"] == 0
        assert "episode/return_window20_mean" not in metrics
    finally:
        for model in (source, target, legacy_target):
            model._checkpoint_writer.shutdown()


def test_corrupt_diagnostics_rejected_before_committing_agent_state(monkeypatch):
    source, target = resume_model(BoundaryEnv()), resume_model(BoundaryEnv())
    try:
        saved = source.training_state_dict()
        saved["training_diagnostics"]["recent_returns"] = [float("nan")]
        monkeypatch.setattr(target.agent, "_commit_training_state_candidate",
                            lambda *_args: pytest.fail("Committed corrupt trainer state"))
        with pytest.raises(ValueError, match="diagnostics"):
            target.load_training_state_dict(saved)
        assert target._global_step == 0
        assert target._training_diagnostics.state_dict()["episodes"] == 0
    finally:
        source._checkpoint_writer.shutdown()
        target._checkpoint_writer.shutdown()
