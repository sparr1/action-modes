import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from tests.resume_test_support import (
    BoundaryEnv,
    _FakeWandb,
    _assert_tree_equal_nan,
    _model,
    _replay_state,
    _session,
)
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_tdmpc2_correctness import tiny_params
from utils.resume_runtime import register_test_resume_environment
from utils.resume_training import RESUME_COMPLETE, RESUME_HANDOFF


class OneStepBoundaryEnv(BoundaryEnv):
    """One-transition episodes expose every early learner phase as a boundary."""

    def __init__(self, *, on_first_done=None):
        super().__init__(on_first_done=on_first_done)
        self.spec = SimpleNamespace(
            id="OneStepBoundaryResume-v0", max_episode_steps=1
        )

    def step(self, action):
        self._episode_step += 1
        action = np.asarray(action, dtype=np.float32).copy()
        reward = float(self._episode_step + 0.25 * action[0])
        observation = self.np_random.normal(size=3).astype(np.float32)
        self.trace.append((action, reward, observation.copy(), True))
        self._done_count += 1
        if self._done_count == 1 and self.on_first_done is not None:
            self.on_first_done()
        return observation, reward, False, True, {}


register_test_resume_environment(OneStepBoundaryEnv, episode_steps=1)


def _assert_trace_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_step, expected_step in zip(actual, expected):
        np.testing.assert_array_equal(actual_step[0], expected_step[0])
        assert actual_step[1] == expected_step[1]
        np.testing.assert_array_equal(actual_step[2], expected_step[2])
        assert actual_step[3] is expected_step[3]


def _scientific_wandb_history(run):
    return [
        {key: value for key, value in row.items() if not key.startswith("time/")}
        for row in run.history
    ]


def test_trainer_checkpoint_requires_a_flushed_metric_boundary():
    model = _one_step_model(OneStepBoundaryEnv())
    model._reset_wandb_window()
    model._wandb_train_window.add_weighted("train/pending", 1.0)
    try:
        with pytest.raises(RuntimeError, match="empty W&B metric window"):
            model.training_state_dict()
    finally:
        model._checkpoint_writer.shutdown()


def test_primary_resume_failure_aborts_wandb_without_handoff(
    monkeypatch, tmp_path
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    model = _one_step_model(OneStepBoundaryEnv())
    session = _session(
        tmp_path / "failed-segment", mode="new", segment="part-1"
    )
    aborts = []
    original_abort = session.abort_wandb

    def abort_wandb(learner, error):
        aborts.append(error)
        original_abort(learner, error)

    def fail_step(_action):
        raise LookupError("primary training failure")

    monkeypatch.setattr(session, "abort_wandb", abort_wandb)
    monkeypatch.setattr(model.env, "step", fail_step)
    try:
        with pytest.raises(LookupError, match="primary training failure"):
            model.learn(total_timesteps=4, resume_session=session)
    finally:
        session.close()

    assert len(aborts) == 1
    assert fake_wandb.runs
    assert next(iter(fake_wandb.runs.values())).finish_count == 1
    assert not (session.store.root / "HANDOFF.json").exists()
    assert not (session.store.root / "DONE").exists()


def _one_step_model(env):
    return _model(
        env,
        total_steps=4,
        episode_length=1,
        seed_steps=1,
        pretrain_steps=1,
        buffer_size=4,
    )


def _replay_boundary_summary(model):
    metadata = model.buffer.training_state_metadata()
    return {
        "step": model._global_step,
        "pretrained": model._pretrained,
        "updates": model._num_updates,
        "episodes": model.buffer.num_eps,
        "resident_transitions": model.buffer.num_transitions,
        "total_transitions": model.buffer.total_transitions,
        "rows": model.buffer.size,
        "writer_cursor": metadata["torchrl"]["writer"]["_cursor"],
    }


def test_baseline_repeated_resumes_span_seed_pretrain_update_wrap_and_target(
    monkeypatch, tmp_path
):
    """One short lineage crosses every outer-training boundary in order."""

    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    continuous_env = OneStepBoundaryEnv()
    continuous_model = _one_step_model(continuous_env)
    continuous_session = _session(
        tmp_path / "continuous-baseline-matrix",
        mode="new",
        segment="continuous",
    )
    continuous_run_id = continuous_session.lineage_metadata[
        "initial_wandb_run_id"
    ]
    try:
        assert (
            continuous_model.learn(
                total_timesteps=4, resume_session=continuous_session
            )
            == RESUME_COMPLETE
        )
    finally:
        continuous_session.close()

    lineage = tmp_path / "split-baseline-matrix"
    split_trace = []
    boundary_summaries = []
    split_model = None
    for segment_index in range(4):
        holder = {}

        def request_drain():
            holder["session"]._drain_requested = True

        env = OneStepBoundaryEnv(
            on_first_done=request_drain if segment_index < 3 else None
        )
        split_model = _one_step_model(env)
        session = _session(
            lineage,
            mode="new" if segment_index == 0 else "required",
            segment=f"part-{segment_index + 1}",
        )
        if segment_index == 0:
            split_run_id = session.lineage_metadata["initial_wandb_run_id"]
        holder["session"] = session
        try:
            status = split_model.learn(total_timesteps=4, resume_session=session)
            expected = RESUME_HANDOFF if segment_index < 3 else RESUME_COMPLETE
            assert status == expected
            boundary_summaries.append(_replay_boundary_summary(split_model))
            split_trace.extend(env.trace)
        finally:
            session.close()

    assert boundary_summaries == [
        {
            "step": 1,
            "pretrained": False,
            "updates": 0,
            "episodes": 1,
            "resident_transitions": 1,
            "total_transitions": 1,
            "rows": 2,
            "writer_cursor": 2,
        },
        {
            "step": 2,
            "pretrained": True,
            "updates": 1,
            "episodes": 2,
            "resident_transitions": 2,
            "total_transitions": 2,
            "rows": 4,
            "writer_cursor": 0,
        },
        {
            "step": 3,
            "pretrained": True,
            "updates": 2,
            "episodes": 3,
            "resident_transitions": 2,
            "total_transitions": 3,
            "rows": 4,
            "writer_cursor": 2,
        },
        {
            "step": 4,
            "pretrained": True,
            "updates": 3,
            "episodes": 4,
            "resident_transitions": 2,
            "total_transitions": 4,
            "rows": 4,
            "writer_cursor": 0,
        },
    ]
    _assert_trace_equal(split_trace, continuous_env.trace)
    _assert_tree_equal(
        _clone_tree(split_model.agent.training_state_dict()),
        _clone_tree(continuous_model.agent.training_state_dict()),
    )
    _assert_tree_equal_nan(
        _clone_tree(_replay_state(split_model.buffer)),
        _clone_tree(_replay_state(continuous_model.buffer)),
    )
    assert split_model._episode_idx == continuous_model._episode_idx == 4
    assert split_model._num_updates == continuous_model._num_updates == 3
    split_history = _scientific_wandb_history(fake_wandb.runs[split_run_id])
    continuous_history = _scientific_wandb_history(
        fake_wandb.runs[continuous_run_id]
    )
    assert split_history == continuous_history
    assert len(continuous_history) == 4
    assert [row["episode/index"] for row in continuous_history] == [0, 1, 2, 3]
    assert all("episode/return" in row for row in continuous_history)
    done = json.loads((lineage / "DONE").read_text(encoding="utf-8"))
    assert done["global_step"] == 4


_MIXED_SCOPES = {
    "inner_actor_scope": "run",
    "inner_critic_scope": "episode",
    "inner_temperature_scope": "action",
    "inner_replay_scope": "run",
    "inner_actor_optimizer_scope": "run",
    "inner_critic_optimizer_scope": "action",
    "inner_temperature_optimizer_scope": "action",
}


def _mixed_lifetime_model(env):
    params = tiny_params(
        episode_length=2,
        total_steps=6,
        seed_steps=1,
        pretrain_steps=1,
        utd=1,
        train_unroll_horizon=1,
        outer_planning_horizon=1,
        batch_size=1,
        buffer_size=16,
        wandb=True,
        wandb_mode="online",
        wandb_step_every=2,
        mpc=False,
        dropout=0.0,
    )
    for legacy_key in (
        "inner_adaptation",
        "inner_iterations",
        "inner_rollouts",
        "inner_horizon",
        "inner_updates_per_iteration",
        "inner_tau",
    ):
        params.pop(legacy_key, None)
    params.update(
        inner_operator="sac",
        inner_model_step_budget=1,
        inner_rounds=1,
        inner_rollout_horizon=1,
        inner_critic_updates_per_action=1,
        inner_actor_updates_per_action=1,
        inner_temperature_updates_per_action=1,
        inner_batch_size=1,
        inner_replay_capacity=4,
        inner_actor_adaptation="clone",
        inner_critic_adaptation="clone",
        inner_temperature_mode="auto",
        inner_temperature_initialization="fixed",
        **_MIXED_SCOPES,
    )
    model = AMBITDMPC2(
        AMBITDMPC2.__name__,
        env,
        params,
        {
            "seed": 7,
            "device": "cpu",
            "env": "BoundaryResume-v0",
            "total_steps": 6,
        },
        {},
    )
    model.enable_training_resume(total_timesteps=6)
    return model


def _active_ambi_model(env, lifetime, *, compiled=False):
    if lifetime == "mixed":
        return _mixed_lifetime_model(env)
    return _model(
        env,
        algorithm=AMBITDMPC2,
        inner_scope=lifetime,
        total_steps=6,
        compile=compiled,
        compile_strict=False,
    )


_PERSISTENT_WORKSPACE_FIELDS = {
    "actor",
    "actor_anchor",
    "actor_target",
    "critic",
    "critic_anchor",
    "critic_target",
    "actor_optim",
    "critic_optim",
    "log_alpha",
    "alpha_fixed",
    "temperature_optim",
    "replay",
}


@pytest.mark.parametrize(
    ("lifetime", "compiled"),
    [
        ("action", False),
        ("episode", False),
        ("run", False),
        ("mixed", False),
        ("run", True),
    ],
    ids=["action", "episode", "run", "mixed", "run-compiled"],
)
def test_ambi_resume_after_inner_actions_preserves_lifetime_inventory_and_result(
    monkeypatch, tmp_path, lifetime, compiled
):
    """The handoff occurs only after persistent AMBI state has been exercised."""

    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    compile_calls = []
    if compiled:
        def fake_compile(function, **kwargs):
            compile_calls.append((function, kwargs))
            random.random()
            np.random.random()
            torch.rand(())
            return function

        monkeypatch.setattr(torch, "compile", fake_compile)

    continuous_env = BoundaryEnv()
    continuous_model = _active_ambi_model(
        continuous_env, lifetime, compiled=compiled
    )
    continuous_session = _session(
        tmp_path / f"continuous-active-{lifetime}",
        mode="new",
        segment="continuous",
        total_steps=6,
    )
    try:
        assert (
            continuous_model.learn(
                total_timesteps=6, resume_session=continuous_session
            )
            == RESUME_COMPLETE
        )
    finally:
        continuous_session.close()

    first_env = BoundaryEnv()
    first_model = _active_ambi_model(first_env, lifetime, compiled=compiled)
    first_session = _session(
        tmp_path / f"split-active-{lifetime}",
        mode="new",
        segment="part-1",
        total_steps=6,
    )
    original_prepare = first_model._prepare_resume_boundary

    def drain_after_second_episode():
        original_prepare()
        if first_model._global_step == 4:
            first_session._drain_requested = True

    first_model._prepare_resume_boundary = drain_after_second_episode
    try:
        assert (
            first_model.learn(total_timesteps=6, resume_session=first_session)
            == RESUME_HANDOFF
        )
        assert first_model._global_step == 4
        inner_state = first_model.agent.training_state_dict()["inner"]
        assert inner_state["action_index"] > 0
        workspace = inner_state["workspace"]
        present = {
            key
            for key in _PERSISTENT_WORKSPACE_FIELDS
            if workspace[key] is not None
        }
        if lifetime in {"action", "episode"}:
            assert present == set()
        elif lifetime == "run":
            assert present == {
                "actor",
                "actor_anchor",
                "critic",
                "critic_anchor",
                "critic_target",
                "actor_optim",
                "critic_optim",
                "log_alpha",
                "temperature_optim",
                "replay",
            }
        else:
            assert present == {"actor", "actor_anchor", "actor_optim", "replay"}
    finally:
        first_session.close()

    second_env = BoundaryEnv()
    second_model = _active_ambi_model(second_env, lifetime, compiled=compiled)
    second_session = _session(
        tmp_path / f"split-active-{lifetime}",
        mode="required",
        segment="part-2",
        total_steps=6,
    )
    try:
        assert (
            second_model.learn(total_timesteps=6, resume_session=second_session)
            == RESUME_COMPLETE
        )
    finally:
        second_session.close()

    _assert_trace_equal(first_env.trace + second_env.trace, continuous_env.trace)
    _assert_tree_equal(
        _clone_tree(second_model.agent.training_state_dict()),
        _clone_tree(continuous_model.agent.training_state_dict()),
    )
    _assert_tree_equal_nan(
        _clone_tree(_replay_state(second_model.buffer)),
        _clone_tree(_replay_state(continuous_model.buffer)),
    )
    assert second_model._inner_steps_total == continuous_model._inner_steps_total
    assert second_model._inner_updates_total == continuous_model._inner_updates_total
    assert second_model._global_step == continuous_model._global_step == 6
    assert second_model._episode_idx == continuous_model._episode_idx == 3
    assert second_model._num_updates == continuous_model._num_updates == 5
    assert bool(compile_calls) is compiled
    # Three two-step episodes submit nine replay rows to a six-row ring. The
    # first episode is therefore evicted after the resume boundary.
    assert second_model.buffer.size == continuous_model.buffer.size == 6
    assert second_model.buffer.total_transitions == 6
    assert second_model.buffer.num_transitions == 4
