"""Outer telemetry preserves scientific state and complete local measurements."""

from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_wandb_event_indexed import _Wandb, _initialize
from utils.outer_policy_diagnostics import OuterPolicyDiagnostics, _host_packet


def _cfg(**overrides):
    values = dict(
        outer_policy_diagnostics_states=3, seed_steps=4,
        outer_policy_diagnostics_seed=83, outer_policy_diagnostics_samples=2,
        action_dim=2, log_std_min=-10., log_std_max=2., rho=.5,
        obs="state", obs_shape={"state": (3,)},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _PolicyModel(torch.nn.Module):
    """A differentiable squashed policy whose probe deliberately consumes RNG."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 2), torch.nn.LayerNorm(2), torch.nn.Dropout(.5),
        )
        self.failure = False
        self.observed_noise = []

    def encode(self, observation):
        random.random()
        np.random.random()
        torch.rand(3)
        return self.encoder(observation)

    def pi(self, z, *, noise):
        assert not self.training
        self.observed_noise.append(noise.detach().clone())
        if self.failure:
            raise RuntimeError("injected policy failure")
        log_std = torch.full_like(z, -.5)
        pre_action = z + log_std.exp() * noise
        action = pre_action.tanh()
        log_prob = (-.5 * (noise.square() + np.log(2 * np.pi)) - log_std).sum(-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - pre_action - torch.nn.functional.softplus(-2 * pre_action))).sum(-1, keepdim=True)
        return action, dict(
            mean=z.tanh(), pre_tanh_mean=z, log_std=log_std,
            pre_tanh_action=pre_action, log_prob=log_prob, entropy=-log_prob,
        )


def _agent():
    model = _PolicyModel()
    # Mixed child modes and nonempty optimizer state catch careless restoration.
    model.train()
    model.encoder[1].eval()
    optim = torch.optim.Adam(model.parameters())
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optim.step()
    return SimpleNamespace(model=model, device=torch.device("cpu"), alpha=torch.tensor(.5),
                           target_entropy=-2., optim=optim)


def _rng():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_rng_equal(before):
    after = _rng()
    assert after[0] == before[0]
    assert after[1][0] == before[1][0]
    np.testing.assert_array_equal(after[1][1], before[1][1])
    assert after[1][2:] == before[1][2:]
    torch.testing.assert_close(after[2], before[2], rtol=0, atol=0)


def _assert_tree_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def _fill_bank(recorder):
    for step in range(5):
        recorder.observe(torch.tensor([step, step + .1, step + .2]), step)


@pytest.mark.parametrize("failure", [False, True])
def test_probe_preserves_rng_parameters_gradients_optimizer_and_mixed_modes(tmp_path, failure):
    agent = _agent()
    agent.model.failure = failure
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    _fill_bank(recorder)
    scientific = deepcopy(agent.model.state_dict())
    gradients = [parameter.grad.clone() for parameter in agent.model.parameters()]
    optimizer = deepcopy(agent.optim.state_dict())
    modes = [module.training for module in agent.model.modules()]
    rng = _rng()
    if failure:
        with pytest.raises(RuntimeError, match="injected policy failure"):
            recorder.probe(agent, env_step=5, updates=0, phase="pretrain_before", run=None)
        assert recorder.rows == []
        assert recorder.emitted == set()
    else:
        recorder.probe(agent, env_step=5, updates=0, phase="pretrain_before", run=None)
        assert recorder.rows[0]["observation_count"] == 3
        assert recorder.rows[0]["metrics"]["pooled_coordinate_count"] == 12
    _assert_rng_equal(rng)
    _assert_tree_equal(agent.model.state_dict(), scientific)
    _assert_tree_equal(agent.optim.state_dict(), optimizer)
    _assert_tree_equal([parameter.grad for parameter in agent.model.parameters()], gradients)
    assert [module.training for module in agent.model.modules()] == modes


@pytest.mark.parametrize("mapping", ["direct_clamp", "tdmpc2_tanh"])
def test_real_squashed_policy_probe_preserves_checkpoint_and_inner_rng(tmp_path, mapping):
    from tests.test_ambi_inner_decoupling import _model

    wrapper = _model(inner_operator="none", ent_coef="auto_1.0", target_entropy=-1.,
                     inner_critic_updates_per_action=0, inner_actor_updates_per_action=0,
                     inner_rounds=0, inner_model_step_budget=0,
                     log_std_mapping=mapping, log_std_min=-10., log_std_max=2.,
                     outer_actor_entropy_mode="squashed")
    try:
        agent = wrapper.agent
        recorder = OuterPolicyDiagnostics(_cfg(action_dim=1), tmp_path)
        _fill_bank(recorder)
        checkpoint = deepcopy(agent.checkpoint_state())
        inner_rng = deepcopy(agent.inner_engine.rng.training_state_dict())
        modes = [module.training for module in agent.model.modules()]
        before = _rng()
        recorder.probe(agent, env_step=5, updates=0, phase="pretrain_before", run=None)
        _assert_rng_equal(before)
        _assert_tree_equal(checkpoint, agent.checkpoint_state())
        _assert_tree_equal(inner_rng, agent.inner_engine.rng.training_state_dict())
        assert [module.training for module in agent.model.modules()] == modes
        row = recorder.rows[0]
        assert row["parameter_coordinate_count"] == 3
        assert row["sampled_coordinate_count"] == 6
        assert row["metrics"]["alpha"] == 1
        assert row["metrics"]["entropy_target"] == -1
    finally:
        wrapper.env.close()


def test_bank_and_noise_are_deterministic_private_and_immutable(tmp_path):
    before = _rng()
    first = OuterPolicyDiagnostics(_cfg(), tmp_path / "first")
    second = OuterPolicyDiagnostics(_cfg(), tmp_path / "second")
    _assert_rng_equal(before)
    assert first.indices == [0, 2, 4]
    _assert_tree_equal(first.noise, second.noise)
    assert not torch.equal(first.noise[0], first.noise[1])
    third = OuterPolicyDiagnostics(_cfg(outer_policy_diagnostics_seed=84), tmp_path / "third")
    assert not torch.equal(first.noise, third.noise)
    observation = torch.tensor([1., 2., 3.])
    first.observe(observation, 0)
    observation.fill_(99)
    first.observe(observation, 0)
    first.observe(observation, 1)
    assert sorted(first.bank) == [0]
    torch.testing.assert_close(first.bank[0], torch.tensor([1., 2., 3.]))


def test_initial_and_bank_probes_reuse_exact_noise_with_distinct_phase_counts(tmp_path):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    agent = _agent()
    recorder.probe(agent, observation=torch.zeros(3), env_step=0, updates=0,
                   phase="initial", run=None)
    _fill_bank(recorder)
    recorder.probe(agent, env_step=5, updates=0, phase="pretrain_before", run=None)
    recorder.learner({"metrics": {"loss": torch.tensor(2.)}}, env_step=5, updates=1,
                     phase="pretrain", run=None)
    recorder.probe(agent, env_step=5, updates=2500, phase="pretrain_after", run=None)
    recorder.probe(agent, env_step=5, updates=2500, phase="duplicate_boundary", run=None)
    assert [(row["source"], row["phase"], row["env_step"], row["updates_completed"])
            for row in recorder.rows] == [
        ("initial_observation", "initial", 0, 0),
        ("reference_bank", "pretrain_before", 5, 0),
        ("learner", "pretrain", 5, 1),
        ("reference_bank", "pretrain_after", 5, 2500),
    ]
    assert [row["event_index"] for row in recorder.rows] == list(range(4))
    _assert_tree_equal(agent.model.observed_noise[0], recorder.noise[:1].reshape(-1, 2))
    _assert_tree_equal(agent.model.observed_noise[1], agent.model.observed_noise[2])
    before = len(recorder.rows)
    recorder.finish(agent, env_step=5, updates=2500, run=None)
    assert len(recorder.rows) == before
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["row_count"] == 4


def test_host_packet_packs_once_and_preserves_histogram_counts(monkeypatch):
    calls = []
    original = torch.Tensor.cpu

    def counted_cpu(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    packet = dict(metrics={"n": torch.tensor(6), "mean": torch.tensor(.25)},
                  histograms={"log_std": {"counts": torch.tensor([1, 2, 3]),
                                          "edges": torch.tensor([-10., -6., -2., 2.]),
                                          "count": 6}}, extra=(np.int64(2), "fixed"))
    host = _host_packet(packet)
    assert calls == [9]
    assert host["metrics"] == {"n": 6, "mean": .25}
    assert host["histograms"]["log_std"]["counts"] == [1, 2, 3]
    assert sum(host["histograms"]["log_std"]["counts"]) == 6
    assert json.loads(json.dumps(host, allow_nan=False)) == host


def test_executed_actions_keep_uniform_and_prior_denominators_separate(tmp_path):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    recorder.action([1., -.99], prior=False)
    recorder.action([0., .5], prior=False)
    recorder.action([-1., .2], prior=True)
    recorder.flush_actions(env_step=6, updates=1, phase="online", run=None)
    by_source = {row["source"]: row["metrics"] for row in recorder.rows}
    uniform, prior = by_source["executed_uniform"], by_source["executed_prior"]
    assert uniform["coordinate_count"] == 4
    assert uniform["decision_count"] == 2
    assert uniform["action_exact_saturation_fraction"] == .25
    assert uniform["action_near_saturation_fraction"] == .5
    assert uniform["action_mean"] == pytest.approx(np.mean([1., -.99, 0., .5]))
    assert uniform["action_std"] == pytest.approx(np.std([1., -.99, 0., .5]))
    assert prior["coordinate_count"] == 2
    assert prior["decision_count"] == 1
    assert prior["action_exact_saturation_fraction"] == .5
    recorder.flush_actions(env_step=6, updates=1, phase="online", run=None)
    assert len(recorder.rows) == 2


def test_resume_state_is_immutable_and_keeps_local_events_and_probe_identity(tmp_path):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path / "original")
    agent = _agent()
    _fill_bank(recorder)
    recorder.probe(agent, env_step=5, updates=0, phase="pretrain_before", run=None)
    state = recorder.state_dict()
    restored = OuterPolicyDiagnostics(_cfg(), tmp_path / "restored", state=state)
    restored_state = restored.state_dict()
    for key in state.keys() - {"timing"}:
        _assert_tree_equal(restored_state[key], state[key])
    assert restored_state["timing"]["serialization_seconds"] >= state["timing"]["serialization_seconds"]
    restored.probe(agent, env_step=5, updates=0, phase="duplicate", run=None)
    assert len(restored.rows) == 1
    state["noise"].fill_(88)
    state["bank"][0].fill_(88)
    state["rows"][0]["phase"] = "mutated"
    assert recorder.rows == restored.rows
    assert restored.rows[0]["phase"] == "pretrain_before"
    assert not torch.any(restored.noise == 88)
    assert not torch.any(restored.bank[0] == 88)


@pytest.mark.parametrize(("critic_target", "q_units"), [
    (None, "decoded soft Q, not reward-only return"),
    ("entropy_augmented", "decoded soft Q, not reward-only return"),
    ("reward_only", "decoded predicted reward-only return, not measured return"),
])
def test_finish_artifact_contains_identical_final_manifest_trace_and_reference(
    tmp_path, critic_target, q_units,
):
    overrides = {} if critic_target is None else {"outer_critic_target": critic_target}
    recorder = OuterPolicyDiagnostics(_cfg(**overrides), tmp_path)
    agent = _agent()
    wandb = _Wandb()
    run = _initialize(wandb)
    _fill_bank(recorder)
    recorder.probe(agent, env_step=5, updates=0, phase="pretrain_before", run=run)
    recorder.finish(agent, env_step=5, updates=2500, run=run)
    assert len(wandb.run.artifacts) == 1
    artifact = wandb.run.artifacts[0]
    for name, content in artifact.files.items():
        assert (tmp_path / name).read_bytes() == content
    manifest = json.loads(artifact.files["manifest.json"])
    assert manifest["status"] == "complete"
    assert manifest["q_units"] == q_units
    assert manifest["config"].get("outer_critic_target") == critic_target
    assert hashlib.sha256(artifact.files["reference.json"]).hexdigest() == manifest["reference_sha256"]
    trace = [json.loads(line) for line in artifact.files["events.jsonl"].splitlines()]
    assert trace == recorder.rows
    for row in trace:
        histogram = row["histograms"]["log_std_pooled"]
        assert sum(histogram["counts"]) == histogram["count"] == row["parameter_coordinate_count"]
        assert row["metrics"]["pooled_coordinate_count"] == row["sampled_coordinate_count"]
    assert [step for step, _ in wandb.run.rows] == list(range(len(wandb.run.rows)))
    assert all(row["env_step"] == 5 for _, row in wandb.run.rows)


def test_failed_finish_marks_incomplete_without_a_policy_probe(tmp_path):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    agent = _agent()
    agent.model.failure = True
    recorder.action([1., 0.], prior=True)
    recorder.finish(agent, env_step=7, updates=2, run=None, failed=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["row_count"] == 1
    assert recorder.rows[0]["source"] == "executed_prior"
    assert agent.model.observed_noise == []


def test_existing_trace_requires_explicit_resume_state(tmp_path):
    OuterPolicyDiagnostics(_cfg(), tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        OuterPolicyDiagnostics(_cfg(), tmp_path)


@pytest.mark.parametrize("learner_failed", [False, True])
def test_final_history_failure_preserves_all_local_rows_and_final_status(tmp_path, learner_failed):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    agent = _agent()
    _fill_bank(recorder)
    recorder.action([1., 0.], prior=True)
    wandb = _Wandb()
    run = _initialize(wandb)

    def fail_history(_payload, *, step):
        # The local bundle is already authoritative when publication starts.
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["status"] == ("incomplete" if learner_failed else "complete")
        raise OSError("injected history outage")

    wandb.run.log = fail_history
    with pytest.raises(OSError, match="history outage"):
        recorder.finish(agent, env_step=5, updates=3, run=run, failed=learner_failed)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert rows == recorder.rows
    assert manifest["row_count"] == len(rows) == (1 if learner_failed else 2)
    assert [row["source"] for row in rows].count("executed_prior") == 1
    assert recorder.executed == {}
    assert wandb.run.artifacts == []


def test_final_probe_error_keeps_primary_error_and_writes_incomplete_manifest(tmp_path):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    agent = _agent()
    agent.model.failure = True
    _fill_bank(recorder)
    rng = _rng()
    with pytest.raises(RuntimeError, match="injected policy failure"):
        recorder.finish(agent, env_step=5, updates=3, run=None)
    _assert_rng_equal(rng)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["row_count"] == 0


def test_partial_action_publication_does_not_duplicate_locally_committed_windows(tmp_path):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    agent = _agent()
    _fill_bank(recorder)
    recorder.action([1., 0.], prior=False)
    recorder.action([-1., .2], prior=True)
    wandb = _Wandb()
    run = _initialize(wandb)
    original_log = wandb.run.log

    def fail_uniform(payload, *, step):
        if payload.get("outer_diag/source") == "executed_uniform":
            raise OSError("injected uniform history failure")
        return original_log(payload, step=step)

    wandb.run.log = fail_uniform
    with pytest.raises(OSError, match="uniform history"):
        recorder.probe(agent, env_step=5, updates=0, phase="pretrain_before", run=run)
    assert ("reference_bank", 0) in recorder.emitted
    assert set(recorder.executed) == {"executed_prior"}
    recorder.finish(agent, env_step=5, updates=0, run=None, failed=True)
    assert [row["source"] for row in recorder.rows] == [
        "reference_bank", "executed_uniform", "executed_prior",
    ]
    assert sum(row["metrics"]["decision_count"] for row in recorder.rows[1:]) == 2
    resumed = OuterPolicyDiagnostics(_cfg(), tmp_path / "resumed", state=recorder.state_dict())
    resumed.probe(agent, env_step=5, updates=0, phase="duplicate", run=None)
    assert resumed.rows == recorder.rows


def test_failed_probe_publication_still_records_dedup_identity(tmp_path):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    agent = _agent()
    _fill_bank(recorder)
    wandb = _Wandb()
    run = _initialize(wandb)

    def fail(_payload, *, step):
        raise OSError("probe history failed")

    wandb.run.log = fail
    with pytest.raises(OSError, match="probe history failed"):
        recorder.probe(agent, env_step=5, updates=0, phase="pretrain_before", run=run)
    assert recorder.emitted == {("reference_bank", 0)}
    assert len(recorder.rows) == 1
    recorder.probe(agent, env_step=5, updates=0, phase="duplicate", run=None)
    assert len(recorder.rows) == 1


def _consume_global_rng():
    random.random()
    np.random.random(3)
    torch.rand(3)


@pytest.mark.parametrize("failure", [False, True])
def test_learner_publication_preserves_global_rng_even_on_error(tmp_path, failure):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    wandb = _Wandb()
    run = _initialize(wandb)
    original = wandb.run.log

    def publishing(payload, *, step):
        _consume_global_rng()
        if failure:
            raise OSError("random publisher failed")
        return original(payload, step=step)

    wandb.run.log = publishing
    packet = {"metrics": {"loss": torch.tensor(1.)}}
    before = _rng()
    if failure:
        with pytest.raises(OSError, match="random publisher failed"):
            recorder.learner(packet, env_step=5, updates=1, phase="pretraining", run=run)
    else:
        recorder.learner(packet, env_step=5, updates=1, phase="pretraining", run=run)
    _assert_rng_equal(before)


@pytest.mark.parametrize("stage", ["history", "artifact"])
@pytest.mark.parametrize("failure", [False, True])
def test_final_publication_preserves_global_rng_and_immutable_bundle(tmp_path, stage, failure):
    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    _fill_bank(recorder)
    agent = _agent()
    wandb = _Wandb()
    run = _initialize(wandb)
    original_log, original_artifact = wandb.run.log, wandb.run.log_artifact

    def logging(payload, *, step):
        _consume_global_rng()
        if failure and stage == "history":
            raise OSError("random final publisher failed")
        return original_log(payload, step=step)

    def artifact_upload(artifact):
        _consume_global_rng()
        if failure and stage == "artifact":
            raise OSError("random final publisher failed")
        return original_artifact(artifact)

    wandb.run.log, wandb.run.log_artifact = logging, artifact_upload
    before = _rng()
    if failure:
        with pytest.raises(OSError, match="random final publisher failed"):
            recorder.finish(agent, env_step=5, updates=2, run=run)
    else:
        recorder.finish(agent, env_step=5, updates=2, run=run)
    _assert_rng_equal(before)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    receipt_path = tmp_path.with_name(tmp_path.name + "_publication.json")
    if not failure or stage == "artifact":
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == ("failed" if failure else "queued")
        assert receipt["enqueue_seconds"] >= 0
        assert receipt["manifest_sha256"] == hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    for artifact in wandb.run.artifacts:
        assert all((tmp_path / name).read_bytes() == content for name, content in artifact.files.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA publication RNG check requires CUDA")
def test_publication_preserves_active_cuda_rng_on_error(tmp_path):
    recorder = OuterPolicyDiagnostics(_cfg(device="cuda:0"), tmp_path)
    torch.rand(1, device="cuda:0")
    before = torch.cuda.get_rng_state(0).clone()
    wandb = _Wandb()
    run = _initialize(wandb)

    def fail(_payload, *, step):
        torch.rand(3, device="cuda:0")
        raise OSError("CUDA publisher failed")

    wandb.run.log = fail
    with pytest.raises(OSError, match="CUDA publisher failed"):
        recorder.learner({"metrics": {}}, env_step=5, updates=1, phase="training", run=run)
    torch.testing.assert_close(torch.cuda.get_rng_state(0), before, rtol=0, atol=0)


def test_deferred_resume_attachment_adds_no_uncommitted_history(tmp_path):
    from utils.wandb_resume import WandbResumeContext

    recorder = OuterPolicyDiagnostics(_cfg(), tmp_path)
    _fill_bank(recorder)
    agent = _agent()
    wandb = _Wandb()
    run = _initialize(wandb, resume_context=WandbResumeContext.new(run_id="seed-55"))
    recorder.finish(agent, env_step=5, updates=3, run=run, publish_artifact=False)
    assert wandb.run.artifacts == []
    assert wandb.run.rows == []
    run.publish_committed(run.checkpoint_state())
    committed = deepcopy(run.checkpoint_state())
    history = deepcopy(wandb.run.rows)
    before = _rng()
    elapsed = recorder.attach(run, env_step=5, updates=3)
    _assert_rng_equal(before)
    assert elapsed >= 0
    assert run.checkpoint_state() == committed
    assert wandb.run.rows == history
    assert len(wandb.run.artifacts) == 1
    artifact = wandb.run.artifacts[0]
    assert all((tmp_path / name).read_bytes() == content for name, content in artifact.files.items())
    receipt = json.loads(tmp_path.with_name(tmp_path.name + "_publication.json").read_text())
    assert receipt["status"] == "queued"
    assert receipt["enqueue_seconds"] == elapsed
    assert receipt["updates_completed"] == 3
    assert receipt["run_id"] == "seed-55"
