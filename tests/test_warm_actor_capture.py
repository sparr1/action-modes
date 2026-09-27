"""Scientific invariants for warm-history capture and local interventions."""

import copy
import json

import numpy as np
import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_actor_transfer_evaluation import transfer_matrix, uniform_transfer_matrix
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params
from tests.test_aux_actor_transfer import _params
from utils import warm_actor_capture as capture
from utils.ambi_real_calibration import SimulatorSnapshot


def _model():
    return _model_from_params(_params(inner_first_action_rounds=None, inner_rounds=2))


def _pendulum_snapshot(env):
    return SimulatorSnapshot.capture({
        "state": env.unwrapped.state.copy(), "elapsed": env._elapsed_steps,
        "rng": copy.deepcopy(env.unwrapped.np_random.bit_generator.state),
    })


def _restore_pendulum(env, snapshot, *, continuing=False):
    assert not continuing
    state = snapshot.state()
    env.unwrapped.state = state["state"].copy()
    env.unwrapped.np_random.bit_generator.state = state["rng"]
    env._elapsed_steps = state["elapsed"]
    return env.unwrapped._get_obs()


def test_snapshot_capture_and_cold_branch_do_not_change_source_learning_or_rng(tmp_path):
    plain, source, cold = _model(), _model(), _model()
    try:
        source.agent.model.load_state_dict(plain.agent.model.state_dict())
        cold.agent.model.load_state_dict(plain.agent.model.state_dict())
        for decision in range(3):
            observation = np.array([1., .1, -.3], dtype=np.float32)
            a = plain.predict(observation, deterministic=True, episode_start=decision == 0)[0]
            engine = source.agent.inner_engine
            before_rng = engine.rng.training_state_dict()
            trace = capture.make_trace(source, 55, 101, decision, capture=True, rollouts=2)
            b = source.predict(observation, deterministic=True,
                               episode_start=decision == 0, trace=trace)[0]
            np.testing.assert_array_equal(a, b)
            after_rng = _clone_tree(engine.rng.training_state_dict())
            actor_before = _clone_tree(engine.state.actor.state_dict())
            _, cold_trace = capture.captured_cold_solve(
                cold, observation, before_rng, controller_seed=55,
                seed=101, decision=decision, rollouts=2)
            _assert_tree_equal(after_rng, engine.rng.training_state_dict())
            _assert_tree_equal(actor_before, engine.state.actor.state_dict())
            _assert_tree_equal(plain.agent.inner_engine.rng.training_state_dict(), after_rng)
            assert [s.round_index for s in trace.actor_snapshots] == [0, 1, 2]
            _assert_tree_equal(cold_trace.actor_snapshots[0].make_policy().state_dict(),
                               cold.agent.inner_engine._actor_base.state_dict())
            root_dir = tmp_path / str(decision)
            item = capture.save_actor_snapshot(trace.actor_snapshots[-1], root_dir, "warm")
            frozen = capture.load_actor_snapshot(root_dir / "root.json", item)
            _assert_tree_equal(frozen.make_policy().state_dict(), actor_before)
            # Deserializations own their storage; diagnostic mutation is isolated.
            with torch.no_grad():
                next(frozen.make_policy().parameters()).add_(10.)
            _assert_tree_equal(frozen.make_policy().state_dict(), actor_before)
    finally:
        plain.close(); source.close(); cold.close()


def test_corrupt_actor_is_rejected_before_deserialization(tmp_path):
    model = _model()
    try:
        snapshot = capture.prior_snapshot(model)
        actor = capture.save_actor_snapshot(snapshot, tmp_path, "prior")
        (tmp_path / actor["path"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum"):
            capture.load_actor_snapshot(tmp_path / "root.json", actor)
    finally:
        model.close()


def test_installed_actor_survives_and_other_state_resets_with_paired_rng():
    model = _model()
    try:
        trace = capture.make_trace(model, 55, 101, 0, capture=True, rollouts=2)
        observation = np.array([1., .1, -.3], dtype=np.float32)
        model.predict(observation, deterministic=True, episode_start=True, trace=trace)
        snapshot = trace.actor_snapshots[1]
        engine = model.agent.inner_engine
        actions = []
        for _ in range(2):
            capture.install_carried_actor(model, snapshot, future_seed=471,
                                           decision_index=25, lifetime_updates=123)
            assert engine.action_index == 26
            _assert_tree_equal(engine.state.actor.state_dict(), snapshot.make_policy().state_dict())
            assert engine.state.critic is engine.state.replay is engine.state.log_alpha is None
            checked = capture.make_trace(model, 55, 101, 26, capture=True, rollouts=2)
            actions.append(model.predict(observation, deterministic=True,
                                         episode_start=False, trace=checked)[0])
            _assert_tree_equal(checked.actor_snapshots[0].make_policy().state_dict(),
                               snapshot.make_policy().state_dict())
            initial = checked.events[0]["metrics"]
            assert initial["inner_actor_lifetime_updates_initial"] == 123
            assert initial["inner_actor_transferred"] == 1
            for component in ("actor", "critic", "temperature"):
                assert initial[f"{component}_optimizer_steps_initial"] == 0
        np.testing.assert_array_equal(actions[0], actions[1])
    finally:
        model.close()


def test_full_capture_matches_existing_uniform_j_evaluator(transfer_matrix, tmp_path, monkeypatch):
    checkpoint, matrix = transfer_matrix
    uniform_transfer_matrix(matrix)
    monkeypatch.setattr(capture, "capture_simulator_snapshot", _pendulum_snapshot)
    baseline = evaluator.evaluate_matrix(matrix, checkpoint, seeds=[101],
                                         bundle_dir=tmp_path / "baseline")["results"][0]
    output = tmp_path / "capture"
    manifest = capture.capture_episode(
        checkpoint=checkpoint, matrix_path=matrix, selector="transfer/warm", seed=101,
        output=output, decisions=[0, 1, 2], max_steps=3, device="cpu")
    assert manifest["status"] == "complete"
    assert manifest["source_return"] == baseline["episodes"][0]["return"]
    assert manifest["source_steps"] == 3
    root = json.loads((output / "decision-1/root.json").read_text())
    assert root["actor_lifetime_updates_before"] == 1
    assert {(a["family"], a["round"]) for a in root["actors"]} == {
        ("warm", 0), ("warm", 1), ("cold", 0), ("cold", 1), ("prior", 0)}
    assert root["matrix_sha256"] == capture.sha256_file(matrix)
    assert root["checkpoint_sha256"] == capture.sha256_file(checkpoint)
    for actor in root["actors"]:
        capture.load_actor_snapshot(output / "decision-1/root.json", actor)
    resumed = capture.capture_episode(
        checkpoint=checkpoint, matrix_path=matrix, selector="transfer/warm", seed=101,
        output=output, decisions=[0, 1, 2], max_steps=3, device="cpu")
    assert resumed["reused_complete_capture"]
    assert resumed["source_return"] == manifest["source_return"]
    with pytest.raises(FileExistsError):
        capture.capture_episode(checkpoint=checkpoint, matrix_path=matrix,
            selector="transfer/warm", seed=101, output=output, decisions=[0], max_steps=3)


def test_replanning_uses_original_cutoff_and_replans_after_first_action(monkeypatch):
    model = _model()
    try:
        env = model.env
        observation, _ = env.reset(seed=101)
        env.step(np.array([0.], dtype=np.float32))
        observation = env.unwrapped._get_obs()
        snapshot = _pendulum_snapshot(env)
        actor = capture.prior_snapshot(model)
        root = {"snapshot": snapshot.to_dict(), "observation": observation.tolist(),
                "episode_max_steps": 5, "decision_index": 1}
        monkeypatch.setattr(capture, "restore_simulator_snapshot", _restore_pendulum)
        calls = []
        original = model.predict
        def predict(*args, **kwargs):
            calls.append(kwargs["episode_start"])
            return original(*args, **kwargs)
        monkeypatch.setattr(model, "predict", predict)
        first = capture.run_replanning_branch(model, env, root, actor, future_seed=777)
        second = capture.run_replanning_branch(model, env, root, actor, future_seed=777)
        assert first == second
        assert first["steps"] == 4
        assert first["truncated"] and not first["terminated"]
        assert calls == [False] * 6
        assert first["return_semantics"] == "remaining_episode_undiscounted"
        assert first["real_mc_return"] == sum(first["rewards"])
    finally:
        model.close()
