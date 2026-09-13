"""Active prior diagnostics survive exact segmented training without live W&B."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from tests.resume_test_support import BoundaryEnv, _FakeWandb, _model, _replay_state, _session
from utils.resume_runtime import capture_environment_state, capture_global_rng_state
from utils.resume_training import RESUME_COMPLETE, RESUME_HANDOFF
from utils.wandb_resume import WandbEventBuffer


TOTAL_STEPS = 6


def _equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            _equal(item, reference)
    else:
        assert actual == expected


class _LocalArtifact:
    def __init__(self, name, *, type, metadata):
        self.name = name
        self.type = type
        self.metadata = copy.deepcopy(metadata)
        self.files = {}

    def add_dir(self, directory):
        root = Path(directory)
        self.files = {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()
        }


class _ArtifactWandb(_FakeWandb):
    Artifact = _LocalArtifact

    def __init__(self):
        super().__init__()
        self.sessions = {}

    def init(self, *, id=None, resume=None, **kwargs):
        run = super().init(id=id, resume=resume, **kwargs)
        run.id = id
        if not hasattr(run, "artifacts"):
            run.artifacts = []
            run.log_artifact = lambda artifact: self._publish_artifact(run, artifact)
        return run

    def _publish_artifact(self, run, artifact):
        # Artifact attachment must follow the durable target generation and
        # publication of the metric journal contained in that generation.
        session = self.sessions[run.id]
        generation = session.last_generation
        assert generation.metadata["global_step"] == TOTAL_STEPS
        assert (session.store.root / "LATEST").read_text().strip() == generation.generation_id
        envelope = torch.load(
            generation.files_for_role("trainer")[0], map_location="cpu", weights_only=False,
        )
        rows = [json.loads(line) for line in artifact.files["events.jsonl"].splitlines()]
        saved = envelope["trainer"]["algorithm_state"]["outer_policy_diagnostics"]
        assert rows == saved["rows"]
        assert len(run.history) == len(envelope["wandb"]["events"])
        run.artifacts.append(artifact)
        return artifact


def _prior(env, directory):
    model = _model(
        env, algorithm=AMBITDMPC2, total_steps=TOTAL_STEPS,
        inner_operator="none", pretrain_steps=3,
        # The shared resume fixture supplies these legacy inner controls.
        inner_iterations=0, inner_rollouts=0, inner_updates_per_iteration=0, inner_horizon=1,
        outer_q_actor_reduction="mean_pair", outer_q_target_reduction="min_pair",
        outer_critic_target="entropy_augmented", outer_actor_entropy_mode="squashed",
        sac_actor_loss_scale_mode="none", ent_coef="auto_1.0", target_entropy=-1.0,
        log_std_mapping="direct_clamp", log_std_min=-10.0, log_std_max=2.0,
        outer_policy_diagnostics=True, outer_policy_diagnostics_early_every=1,
        outer_policy_diagnostics_early_until=8, outer_policy_diagnostics_every=2,
        outer_policy_diagnostics_states=2, outer_policy_diagnostics_samples=3,
        outer_policy_diagnostics_seed=12345, wandb_event_indexed=True,
    )
    model.set_checkpointing(100, directory, "prior", save_strat="all")
    assert model.cfg.inner_operator == "none"
    return model


def _snapshot(model):
    return dict(
        agent=copy.deepcopy(model.agent.training_state_dict()),
        replay=copy.deepcopy(_replay_state(model.buffer)),
        rng=copy.deepcopy(capture_global_rng_state()),
        environment=copy.deepcopy(capture_environment_state(model.env)),
        gradients=[None if parameter.grad is None else parameter.grad.clone()
                   for parameter in model.agent.model.parameters()],
        modes=[module.training for module in model.agent.model.modules()],
        global_step=model._global_step, episodes=model._episode_idx,
        updates=model._num_updates,
    )


def _diagnostic_state(model):
    state = model._outer_policy_recorder.state_dict()
    for row in state["rows"]:
        row.pop("diagnostic_collection_seconds", None)
    return {key: value for key, value in state.items() if key != "timing"}


def _scientific_history(run):
    return [
        {key: value for key, value in row.items()
         if not key.startswith(("time/", "outer_diag/time/"))}
        for row in run.history
    ]


def test_active_diagnostics_resume_matches_continuous_training(monkeypatch, tmp_path):
    fake = _ArtifactWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    continuous_env = BoundaryEnv()
    continuous = _prior(continuous_env, tmp_path / "continuous-output")
    session = _session(tmp_path / "continuous", mode="new", segment="whole", total_steps=TOTAL_STEPS)
    continuous_id = session.lineage_metadata["initial_wandb_run_id"]
    fake.sessions[continuous_id] = session
    try:
        assert continuous.learn(total_timesteps=TOTAL_STEPS, resume_session=session) == RESUME_COMPLETE
        continuous_state = _snapshot(continuous)
    finally:
        session.close()

    holder = {}
    first_env = BoundaryEnv(on_first_done=lambda: setattr(holder["session"], "_drain_requested", True))
    first = _prior(first_env, tmp_path / "first-output")
    lineage = tmp_path / "split"
    first_session = _session(lineage, mode="new", segment="first", total_steps=TOTAL_STEPS)
    split_id = first_session.lineage_metadata["initial_wandb_run_id"]
    fake.sessions[split_id] = holder["session"] = first_session
    try:
        assert first.learn(total_timesteps=TOTAL_STEPS, resume_session=first_session) == RESUME_HANDOFF
        assert first._global_step == 2
        assert first._num_updates == 3
        first_rows = copy.deepcopy(first._outer_policy_recorder.rows)
        assert fake.runs[split_id].artifacts == []
    finally:
        first_session.close()

    second_env = BoundaryEnv()
    second = _prior(second_env, tmp_path / "second-output")
    second_session = _session(lineage, mode="required", segment="second", total_steps=TOTAL_STEPS)
    fake.sessions[split_id] = second_session
    try:
        assert second.learn(total_timesteps=TOTAL_STEPS, resume_session=second_session) == RESUME_COMPLETE
        split_state = _snapshot(second)
        envelope = torch.load(
            second_session.last_generation.files_for_role("trainer")[0],
            map_location="cpu", weights_only=False,
        )
    finally:
        second_session.close()

    _equal(first_env.trace + second_env.trace, continuous_env.trace)
    _equal(split_state, continuous_state)
    assert second._num_updates == continuous._num_updates == 7
    _equal(_diagnostic_state(second), _diagnostic_state(continuous))
    recorder = second._outer_policy_recorder
    assert recorder.rows[:len(first_rows)] == first_rows
    assert [row["event_index"] for row in recorder.rows] == list(range(len(recorder.rows)))
    local_rows = [json.loads(line) for line in (recorder.directory / "events.jsonl").read_text().splitlines()]
    assert local_rows == recorder.rows

    split_run, continuous_run = fake.runs[split_id], fake.runs[continuous_id]
    assert split_run.finish_count == 2
    _equal(_scientific_history(split_run), _scientific_history(continuous_run))
    journal = WandbEventBuffer.from_records(envelope["wandb"]["events"])
    assert len(journal.events) == len(split_run.history)
    for event, remote in zip(journal.events, split_run.history):
        assert remote == {"_step": event.event_index, **event.wandb_payload()}
    bank_rows = [row for row in split_run.history
                 if row.get("outer_diag/source") == "reference_bank" and row["env_step"] == 2]
    assert [row["outer_diag/updates_completed"] for row in bank_rows] == [0, 1, 2, 3]
    assert len({row["_step"] for row in bank_rows}) == 4

    assert len(split_run.artifacts) == len(continuous_run.artifacts) == 1
    artifact = split_run.artifacts[0]
    assert artifact.type == "outer-policy-diagnostics"
    assert [json.loads(line) for line in artifact.files["events.jsonl"].splitlines()] == recorder.rows
    assert artifact.files["reference.json"] == continuous_run.artifacts[0].files["reference.json"]
    manifest = json.loads(artifact.files["manifest.json"])
    assert manifest["status"] == "complete"
    assert manifest["row_count"] == len(recorder.rows)
    assert len({(row["source"], row["updates_completed"]) for row in recorder.rows
                if row["source"] == "reference_bank"}) == len([
                    row for row in recorder.rows if row["source"] == "reference_bank"
                ])


def test_resumable_failure_keeps_incomplete_local_diagnostics(monkeypatch, tmp_path):
    fake = _ArtifactWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    env = BoundaryEnv()
    original_step = env.step

    def fail_in_second_episode(action):
        if len(env.trace) == 2:
            raise RuntimeError("injected resumed-diagnostic simulator failure")
        return original_step(action)

    monkeypatch.setattr(env, "step", fail_in_second_episode)
    model = _prior(env, tmp_path / "failed-output")
    session = _session(tmp_path / "failed", mode="new", segment="first", total_steps=TOTAL_STEPS)
    run_id = session.lineage_metadata["initial_wandb_run_id"]
    fake.sessions[run_id] = session
    try:
        with pytest.raises(RuntimeError, match="injected resumed-diagnostic simulator failure"):
            model.learn(total_timesteps=TOTAL_STEPS, resume_session=session)
    finally:
        session.close()
    recorder = model._outer_policy_recorder
    manifest = json.loads((recorder.directory / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["row_count"] == len(recorder.rows)
    assert not any(row["phase"] == "final" and row["source"] == "reference_bank" for row in recorder.rows)
    assert [row["event_index"] for row in recorder.rows] == list(range(len(recorder.rows)))
    assert fake.runs[run_id].artifacts == []
    assert fake.runs[run_id].finish_count == 1
    assert model.agent._outer_policy_diagnostics_force is False


def test_target_recovery_retries_artifact_without_more_training(monkeypatch, tmp_path):
    fake = _ArtifactWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    original_publish = fake._publish_artifact
    attempts = []

    def fail_first_attachment(run, artifact):
        session = fake.sessions[run.id]
        generation = session.last_generation
        assert generation.metadata["global_step"] == TOTAL_STEPS
        assert (session.store.root / "LATEST").read_text().strip() == generation.generation_id
        assert not (session.store.root / "DONE").exists()
        attempts.append(generation.generation_id)
        if len(attempts) == 1:
            raise RuntimeError("injected final artifact attachment failure")
        return original_publish(run, artifact)

    monkeypatch.setattr(fake, "_publish_artifact", fail_first_attachment)
    first = _prior(BoundaryEnv(), tmp_path / "first-output")
    lineage = tmp_path / "artifact-recovery"
    first_session = _session(lineage, mode="new", segment="first", total_steps=TOTAL_STEPS)
    run_id = first_session.lineage_metadata["initial_wandb_run_id"]
    fake.sessions[run_id] = first_session
    try:
        with pytest.raises(RuntimeError, match="injected final artifact attachment failure"):
            first.learn(total_timesteps=TOTAL_STEPS, resume_session=first_session)
        failed_generation = first_session.last_generation
        envelope = torch.load(
            failed_generation.files_for_role("trainer")[0],
            map_location="cpu", weights_only=False,
        )
        assert failed_generation.metadata["global_step"] == TOTAL_STEPS
        assert not (first_session.store.root / "DONE").exists()
        assert fake.runs[run_id].artifacts == []
        assert len(attempts) == 1
        original_reference = (first._outer_policy_recorder.directory / "reference.json").read_bytes()
    finally:
        first_session.close()

    recovered_env = BoundaryEnv()
    recovered = _prior(recovered_env, tmp_path / "recovered-output")

    def forbidden(*args, **kwargs):
        raise AssertionError("Target recovery must not reset, step, or update the learner")

    monkeypatch.setattr(recovered_env, "reset", forbidden)
    monkeypatch.setattr(recovered_env, "step", forbidden)
    monkeypatch.setattr(recovered.agent, "update", forbidden)
    recovery_session = _session(lineage, mode="required", segment="recovery", total_steps=TOTAL_STEPS)
    fake.sessions[run_id] = recovery_session
    try:
        assert recovered.learn(total_timesteps=TOTAL_STEPS, resume_session=recovery_session) == RESUME_COMPLETE
        final_generation = recovery_session.last_generation
        assert final_generation.generation_id != failed_generation.generation_id
        done = json.loads((recovery_session.store.root / "DONE").read_text())
        assert done["generation_id"] == final_generation.generation_id
        assert done["global_step"] == done["target_step"] == TOTAL_STEPS
        assert attempts == [failed_generation.generation_id, final_generation.generation_id]
    finally:
        recovery_session.close()

    saved = envelope["trainer"]["algorithm_state"]["outer_policy_diagnostics"]
    restored = recovered._outer_policy_recorder.state_dict()
    for key in ("bank", "noise", "rows", "emitted", "executed"):
        _equal(restored[key], saved[key])
    assert recovered_env.trace == []
    assert recovered._global_step == TOTAL_STEPS
    assert recovered._num_updates == first._num_updates == 7
    _equal(recovered.agent.training_state_dict(), first.agent.training_state_dict())
    _equal(_replay_state(recovered.buffer), _replay_state(first.buffer))
    _equal(capture_global_rng_state(), envelope["global_rng"])
    _equal(capture_environment_state(recovered_env), envelope["environment"])
    run = fake.runs[run_id]
    assert run.finish_count == 2
    assert len(run.artifacts) == 1
    artifact = run.artifacts[0]
    assert artifact.files["reference.json"] == original_reference
    artifact_rows = [json.loads(line) for line in artifact.files["events.jsonl"].splitlines()]
    assert artifact_rows == saved["rows"]
    assert [row["event_index"] for row in artifact_rows] == list(range(len(artifact_rows)))
    manifest = json.loads(artifact.files["manifest.json"])
    assert manifest["status"] == "complete"
    assert manifest["row_count"] == len(artifact_rows)
