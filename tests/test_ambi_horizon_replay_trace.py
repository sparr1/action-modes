"""Remaining-horizon replay ownership and finite-prefix probe semantics."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer
from RL.tdmpc2_core.inner_trace import InnerActionTrace, evaluate_outer_tail


def _replay(horizon=3, capacity=4):
    return LatentReplayBuffer(capacity, 1, 1, "cpu", store_horizon=True,
                              remaining_horizon_max=horizon)


def _append(replay, remaining, *, packed=False, validate=True):
    h = torch.as_tensor(remaining, dtype=torch.float32).reshape(-1, 1)
    z = torch.arange(len(h), dtype=torch.float32).reshape(-1, 1)
    fields = (z, z, z, z + 1, torch.zeros_like(z))
    boundary = (h == 1).float()
    if packed:
        replay.add_packed(torch.cat((*fields, boundary, h), -1), validate=validate)
    else:
        replay.add_batch(*fields, horizon_end=boundary, remaining_horizon=h, validate=validate)


@pytest.mark.parametrize("packed", [False, True])
def test_horizon_rows_wrap_sample_and_restore_with_their_transitions(packed):
    replay = _replay()
    _append(replay, [3, 2, 1, 3, 2, 1], packed=packed)
    batch = replay.sample(4, indices=torch.arange(4))
    original_h = torch.tensor([3., 2., 1., 3., 2., 1.]).unsqueeze(1)
    torch.testing.assert_close(batch["remaining_horizon"], original_h[batch["sample_ids"]])
    torch.testing.assert_close(batch["horizon_end"], (batch["remaining_horizon"] == 1).float())
    state = replay.training_state_dict()
    assert state["version"] == 4 and state["remaining_horizon_max"] == 3
    restored = _replay()
    restored.load_training_state_dict(state)
    simple = _replay()
    simple.load_state_dict(state["state"])
    for candidate in (restored, simple):
        torch.testing.assert_close(candidate._storage, replay._storage)
        assert (candidate.pos, candidate.full, candidate.next_sample_id) == (replay.pos, replay.full, replay.next_sample_id)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("bad", [0, -1, 4, 1.5, float("nan"), float("inf")])
def test_public_append_rejects_invalid_horizon_before_mutation(packed, bad):
    replay = _replay()
    _append(replay, [3, 2, 1, 3])
    before = replay._storage.clone()
    with pytest.raises(ValueError, match="remaining_horizon"):
        _append(replay, [bad], packed=packed)
    torch.testing.assert_close(replay._storage, before)
    assert replay.next_sample_id == 4


def test_horizon_end_consistency_and_required_rows_are_validated():
    replay = _replay()
    zero = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="requires remaining_horizon"):
        replay.add_batch(zero, zero, zero, zero, zero, horizon_end=zero)
    with pytest.raises(ValueError, match="horizon_end must equal"):
        replay.add_batch(zero, zero, zero, zero, zero, horizon_end=zero,
                         remaining_horizon=torch.ones(1, 1))
    assert replay.size == 0


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("corruption", ["horizon", "fraction", "boundary", "mode"])
def test_restore_rejects_mismatched_semantics_before_mutation(exact, corruption):
    replay = _replay()
    _append(replay, [3, 2, 1, 3])
    saved = deepcopy(replay.training_state_dict())
    physical = saved["state"]
    if corruption == "horizon":
        saved["remaining_horizon_max"] = 2
        physical["remaining_horizon_max"] = 2
    elif corruption == "fraction":
        physical["remaining_horizon"][0] = 1.5
    elif corruption == "boundary":
        physical["horizon_end"][0] = 1
    else:
        del saved["remaining_horizon_max"]
        del physical["remaining_horizon_max"]
    before = replay._storage.clone()
    with pytest.raises((TypeError, ValueError)):
        if exact:
            replay.load_training_state_dict(saved)
        else:
            replay.load_state_dict(physical)
    torch.testing.assert_close(replay._storage, before)
    assert replay.next_sample_id == 4


def test_legacy_replay_contract_and_explicit_trusted_append():
    plain = LatentReplayBuffer(2, 1, 1, "cpu")
    boundary = LatentReplayBuffer(2, 1, 1, "cpu", store_horizon=True)
    assert plain.training_state_dict()["version"] == 1
    assert boundary.training_state_dict()["version"] == 3
    assert "remaining_horizon" not in boundary.state_dict()
    assert "remaining_horizon_max" not in boundary.training_state_dict()
    replay = _replay()
    _append(replay, [3, 2, 1], packed=True, validate=False)
    with pytest.raises((ValueError, TypeError)):
        boundary.load_training_state_dict(replay.training_state_dict())


class _ProbeModel(torch.nn.Module):
    def __init__(self, *, terminate=False):
        super().__init__()
        self._pi = torch.nn.Linear(1, 1)
        self._Qs = _ProbeCritic()
        self.calls = []
        self.terminate = terminate

    def pi(self, z, *, policy=None, noise, remaining_horizon=None, **kwargs):
        self.calls.append((policy, None if remaining_horizon is None else remaining_horizon.clone()))
        if policy is not None and getattr(policy, "horizon_conditioning_horizon", None) is not None:
            assert remaining_horizon is not None
            return remaining_horizon.to(z.dtype), {}
        assert remaining_horizon is None
        return torch.zeros_like(z), {}

    @staticmethod
    def joint_input(z, action):
        return torch.cat((z, action), -1)

    @staticmethod
    def reward_from_joint(joint):
        return joint[:, 1:]

    @staticmethod
    def decode_reward(reward):
        return reward

    @staticmethod
    def next_from_joint(joint):
        return joint[:, :1] + 1

    def termination(self, z):
        return (z >= 1).float()

    def Q(self, z, action, **kwargs):
        return kwargs["qs"](self.joint_input(z, action))


class _ProbeCritic(torch.nn.Module):
    @staticmethod
    def _forward_eager(joint):
        return torch.full_like(joint[:, :1], 10.)


def _probe_fixture():
    model = _ProbeModel()
    actor = torch.nn.Linear(1, 1)
    actor.horizon_conditioning_horizon = 3
    cfg = SimpleNamespace(action_dim=1, episodic=False, inner_rollout_horizon=3,
        inner_horizon_conditioning="one_hot", mppi_terminal_q_reduction="mean_all",
        inner_termination_threshold=.5, inner_log_std_mapping="direct_clamp",
        inner_log_std_min=-3., inner_log_std_max=1.)
    state = SimpleNamespace(actor_steps=0, critic_steps=0, temperature_steps=0, replay=None)
    engine = SimpleNamespace(model=model, cfg=cfg, agent=SimpleNamespace(discount=.5),
                             state=state, device=torch.device("cpu"))
    return engine, actor


@pytest.mark.parametrize("episodic", [False, True])
def test_outer_tail_decrements_horizon_keeps_prior_tail_and_masks_termination(episodic):
    engine, actor = _probe_fixture()
    engine.cfg.episodic = episodic
    before_rng = torch.random.get_rng_state().clone()
    result = evaluate_outer_tail(engine, torch.zeros(1, 1), actor, torch.zeros(4, 2, 1))
    expected = 3. if episodic else 3. + .5 * 2. + .25 * 1. + .125 * 10.
    torch.testing.assert_close(result["total"], torch.full((2, 1), expected))
    assert [int(h[0]) for _, h in engine.model.calls[:-1]] == [3, 2, 1]
    assert engine.model.calls[-1] == (None, None)
    torch.testing.assert_close(torch.random.get_rng_state(), before_rng)


def test_probe_rejects_legacy_and_wrong_horizon_before_model_calls():
    engine, actor = _probe_fixture()
    for mode, horizon, message in [("legacy", 3, "legacy"), ("outer_tail", 2, "probe horizon")]:
        trace = InnerActionTrace(probes=True, probe_mode=mode, probe_horizon=horizon)
        with pytest.raises(ValueError, match=message):
            trace.probe(engine, torch.zeros(1, 1), actor)
    with pytest.raises(ValueError, match="probe horizon"):
        evaluate_outer_tail(engine, torch.zeros(1, 1), actor, torch.zeros(3, 2, 1))
    assert not engine.model.calls


def test_frozen_actor_snapshot_preserves_horizon_metadata_and_independence():
    engine, actor = _probe_fixture()
    trace = InnerActionTrace(capture_actors=True)
    trace.begin()
    trace.capture_actor(engine, actor)
    snapshot = trace.actor_snapshots[0]
    assert snapshot.horizon_conditioning_horizon == 3
    assert trace.events[0]["horizon_conditioning_horizon"] == 3
    expected = snapshot.make_policy().weight.clone()
    with torch.no_grad():
        actor.weight.add_(10)
    torch.testing.assert_close(snapshot.make_policy().weight, expected)
    with pytest.raises(ValueError, match="metadata"):
        replace(snapshot, horizon_conditioning_horizon=2).make_policy()


def test_checkpoint_evaluator_rejects_legacy_bank_before_environment(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    resolved = {"algorithm_config": {"alg_params": {"inner_horizon_conditioning": "one_hot"}}}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"guard must run before loading")
    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must precede environment creation.")
    monkeypatch.setattr(evaluator, "_make_env", forbidden)
    with pytest.raises(ValueError, match="legacy bank probes"):
        evaluator.evaluate_preset(resolved, checkpoint, [101], controller_seed=55,
                                  root_bank={"roots": []})
    # Ordinary full episodes and outer-tail bank probes remain supported.
    evaluator._validate_horizon_probe_selection(
        [resolved], root_bank_requested=False, togo_return_rollouts=0,
    )
    evaluator._validate_horizon_probe_selection(
        [resolved], root_bank_requested=True, togo_return_rollouts=2,
    )


def test_conditioned_outer_tail_trace_preserves_actual_solver_and_rng():
    from tests.test_ambi_horizon_diagnostics import snapshot as _snapshot
    from tests.test_ambi_latency_contract import _assert_tree_equal
    from tests.test_ambi_root_local_sac import _tiny_model
    options = dict(inner_horizon_conditioning="one_hot", inner_finite_horizon=True,
                   inner_rollout_horizon=3, train_unroll_horizon=3,
                   inner_rounds=2, inner_updates_per_round=1, inner_replay_capacity=24)
    ordinary = _tiny_model(**options)
    observed = _tiny_model(**options)
    try:
        for decision in range(2):
            root = torch.full((3,), decision * .1)
            expected = ordinary.agent.act(root, t0=decision == 0, collect_diagnostics=False)
            trace = InnerActionTrace(probes=True, probe_mode="outer_tail", probe_horizon=3,
                                     probe_rollouts=2, probe_seed=31, capture_actors=True)
            actual = observed.agent.act(root, t0=decision == 0, collect_diagnostics=False, trace=trace)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(ordinary.agent), _snapshot(observed.agent))
            assert len(trace.actor_snapshots) == 3
            assert all(snapshot.horizon_conditioning_horizon == 3 for snapshot in trace.actor_snapshots)
            assert len([event for event in trace.events if event["phase"] == "probe"]) == 3
    finally:
        ordinary.env.close()
        observed.env.close()
