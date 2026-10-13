"""Safe bank persistence, exact sequence boundaries, and frozen collection."""

from copy import deepcopy
import json
import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from utils.pave_probe_data import (
    PaveProbeBank, collect_prior_bank, export_buffer_bank, load_probe_bank,
    sample_probe_windows, save_probe_bank, valid_window_starts,
)


def _provenance():
    return {"source_kind": "training_replay", "env_signature": {"id": "test-env"},
            "source_checkpoint_sha256": "a" * 64}


def _arrays():
    observations = np.array([[0, 1], [1, 2], [2, 3], [10, 11], [11, 12]], dtype=np.float32)
    return dict(observations=observations, next_observations=observations + 1,
                actions=np.array([[.1], [.2], [.3], [.4], [.5]], dtype=np.float32),
                rewards=np.arange(5, dtype=np.float32)[:, None],
                terminated=np.array([[0], [0], [1], [0], [0]], dtype=bool),
                truncated=np.array([[0], [0], [0], [0], [1]], dtype=bool),
                episode_ids=np.array([4, 4, 4, 9, 9]), episode_steps=np.array([0, 1, 2, 0, 1]))


def _bank(**changes):
    return PaveProbeBank(**(_arrays() | changes), provenance=_provenance())


def _rng_snapshot():
    return torch.random.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()


def _assert_rng(snapshot):
    torch.testing.assert_close(torch.random.get_rng_state(), snapshot[0], rtol=0, atol=0)
    actual = np.random.get_state()
    assert actual[0] == snapshot[1][0] and actual[2:] == snapshot[1][2:]
    np.testing.assert_array_equal(actual[1], snapshot[1][1])
    assert random.getstate() == snapshot[2]


def test_round_trip_safe_numeric_archive_and_collision_does_not_replace(tmp_path):
    bank = _bank()
    path = tmp_path / "bank.npz"
    assert save_probe_bank(path, bank) == path
    before = path.read_bytes()
    restored = load_probe_bank(path)
    for name in _arrays():
        np.testing.assert_array_equal(getattr(restored, name), getattr(bank, name))
    assert restored.provenance == bank.provenance
    assert len(restored) == 5
    assert not restored.observations.flags.writeable
    with pytest.raises(FileExistsError):
        save_probe_bank(path, _bank(actions=np.zeros((5, 1))))
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".pave-bank-*"))
    with np.load(path, allow_pickle=False) as raw:
        assert all(raw[key].dtype.kind != "O" for key in raw.files)


@pytest.mark.parametrize("change,match", [
    ({"observations": np.zeros((5, 2, 1))}, "observations"),
    ({"next_observations": np.zeros((4, 2))}, "next_observations"),
    ({"actions": np.full((5, 1), 1.01)}, "normalized"),
    ({"actions": np.full((5, 1), np.nan)}, "finite"),
    ({"rewards": np.zeros(5)}, "rewards"),
    ({"terminated": np.full((5, 1), 2)}, "boolean"),
    ({"terminated": np.array([[1], [0], [1], [0], [0]])}, "boundary"),
    ({"episode_ids": np.array([1, 2, 1, 2, 2])}, "contiguous block"),
    ({"episode_ids": np.array([1., 1., 1., 2., 2.])}, "integers"),
    ({"episode_steps": np.array([0, 2, 3, 0, 1])}, "advance"),
    ({"next_observations": np.zeros((5, 2))}, "contiguous"),
])
def test_invalid_banks_reject_bad_data_and_boundaries(change, match):
    with pytest.raises(ValueError, match=match):
        _bank(**change)


@pytest.mark.parametrize("change", [{"schema_version": 2}, {"source_kind": "unknown"},
                                   {"env_signature": None}, {"source_checkpoint_sha256": "oops"}])
def test_invalid_provenance_rejected(change):
    with pytest.raises(ValueError):
        PaveProbeBank(**_arrays(), provenance=_provenance() | change)


def test_pickle_payload_and_non_scalar_provenance_are_rejected(tmp_path):
    path = tmp_path / "unsafe.npz"
    np.savez(path, **(_arrays() | {"actions": np.array([object()], dtype=object)}),
             provenance_json=np.asarray(json.dumps(_provenance())))
    with pytest.raises(ValueError, match="Object arrays"):
        load_probe_bank(path)
    np.savez(path, **_arrays(), provenance_json=np.array([json.dumps(_provenance())]))
    with pytest.raises(ValueError, match="scalar"):
        load_probe_bank(path)


def test_windows_are_uniform_seeded_paired_and_include_terminal_rows_without_crossing():
    bank = _bank()
    np.testing.assert_array_equal(valid_window_starts(bank, 2), [0, 1, 3])
    before = _rng_snapshot()
    batch = sample_probe_windows(bank, horizon=2, batch_size=6000, seed=123)
    _assert_rng(before)
    repeated = sample_probe_windows(bank, horizon=2, batch_size=6000, seed=123)
    for actual, expected in zip(repeated, batch):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    obs, actions, rewards, terminated = batch
    assert obs.shape == (3, 6000, 2) and actions.shape == (2, 6000, 1)
    assert rewards.shape == terminated.shape == (2, 6000, 1)
    values, counts = obs[0, :, 0].unique(return_counts=True)
    assert values.tolist() == [0, 1, 10]
    assert all(1700 < count < 2300 for count in counts)
    for first, reward, terminal in ((0, 0, 0), (1, 1, 1), (10, 3, 0)):
        columns = obs[0, :, 0] == first
        assert (obs[-1, columns, 0] == first + 2).all()
        assert (rewards[0, columns, 0] == reward).all()
        assert (terminated[-1, columns, 0] == terminal).all()
        torch.testing.assert_close(actions[0, columns, 0], torch.full_like(actions[0, columns, 0], (reward + 1) / 10))
    # The final truncated transition is bootstrap eligible, but never crosses
    # into another episode; true terminal transitions remain masked.
    with pytest.raises(ValueError, match="no valid"):
        sample_probe_windows(bank, horizon=4, batch_size=2, seed=1)


class CollectionEnv(gym.Env):
    def __init__(self, *, termination=None, fail=False):
        self.action_space = gym.spaces.Box(-2., 2., shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-100., 100., shape=(3,), dtype=np.float32)
        self.termination = termination
        self.fail = fail
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.origin = self.np_random.uniform(-1, 1, 3).astype(np.float32)
        return self.origin.copy(), {}

    def step(self, action):
        np.random.rand()
        random.random()
        if self.fail:
            raise RuntimeError("collection failed")
        self.actions.append(action.copy())
        self.t += 1
        done = self.termination is not None and self.t == self.termination
        return self.origin + self.t, 1., done, False, {}


def test_collection_uses_actual_stochastic_prior_without_learning_and_preserves_state(monkeypatch):
    wrapper = _tiny_model()
    env = CollectionEnv()
    try:
        for owner, names in ((wrapper, ("learn", "predict")),
                             (wrapper.agent, ("update", "_update"))):
            for name in names:
                monkeypatch.setattr(owner, name, lambda *_a, **_k: pytest.fail("Training/inner policy called"))
        wrapper.agent.model.train()
        wrapper.agent.model._Qs[0].eval()
        modes = [(module, module.training) for module in wrapper.agent.model.modules()]
        before = deepcopy(wrapper.agent.checkpoint_state())
        rng = _rng_snapshot()
        bank = collect_prior_bank(wrapper, env, seeds=[7, 8], max_steps=3,
                                  controller_seed=91, provenance=_provenance())
        _assert_rng(rng)
        _assert_tree_equal(wrapper.agent.checkpoint_state(), before)
        assert all(module.training == training for module, training in modes)
        assert len(bank) == 6 and bank.provenance["source_kind"] == "stochastic_prior_proxy"
        assert bank.provenance["collection"]["forced_cap_truncations"] == 2
        np.testing.assert_array_equal(bank.truncated[:, 0], [0, 0, 1, 0, 0, 1])
        assert not bank.terminated.any()
        np.testing.assert_allclose(np.asarray(env.actions), 2 * bank.actions, rtol=0, atol=2e-7)
        repeated = collect_prior_bank(wrapper, env, seeds=[7, 8], max_steps=3,
                                      controller_seed=91, provenance=_provenance())
        np.testing.assert_array_equal(repeated.actions, bank.actions)
        changed = collect_prior_bank(wrapper, env, seeds=[7, 8], max_steps=3,
                                     controller_seed=92, provenance=_provenance())
        assert not np.array_equal(changed.actions, bank.actions)
    finally:
        wrapper.env.close()
        env.close()


def test_collection_keeps_true_termination_and_restores_rng_modes_on_failure():
    wrapper = _tiny_model()
    try:
        env = CollectionEnv(termination=2)
        bank = collect_prior_bank(wrapper, env, seeds=[7], max_steps=3,
                                  controller_seed=91, provenance=_provenance())
        assert len(bank) == 2 and bank.terminated[-1, 0] and not bank.truncated.any()
        assert bank.provenance["collection"]["forced_cap_truncations"] == 0
        modes = [(module, module.training) for module in wrapper.agent.model.modules()]
        rng = _rng_snapshot()
        with pytest.raises(RuntimeError, match="collection failed"):
            collect_prior_bank(wrapper, CollectionEnv(fail=True), seeds=[7], max_steps=3,
                               controller_seed=91, provenance=_provenance())
        _assert_rng(rng)
        assert all(module.training == training for module, training in modes)
    finally:
        wrapper.env.close()


def test_collection_copies_environment_reused_observation_storage():
    class ReusedEnv(CollectionEnv):
        def reset(self, **kwargs):
            self.values = np.zeros(3, dtype=np.float32)
            return self.values, {}

        def step(self, action):
            self.values += 1
            return self.values, 0., False, False, {}

    wrapper = _tiny_model()
    try:
        bank = collect_prior_bank(wrapper, ReusedEnv(), seeds=[1], max_steps=3,
                                  controller_seed=7, provenance=_provenance())
        np.testing.assert_array_equal(bank.observations[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(bank.next_observations[:, 0], [1, 2, 3])
    finally:
        wrapper.env.close()


def test_replay_export_obeys_ring_eviction_and_successor_row_action_convention():
    wrapper = _tiny_model(buffer_size=7)
    try:
        for offset, terminate in ((0, True), (10, False)):
            n = 5
            observations = (torch.arange(n) + offset).float()[:, None].expand(n, 3).clone()
            wrapper.buffer.add(TensorDict({
                "obs": observations, "action": torch.tensor([[float("nan")], [.1], [.2], [.3], [.4]]),
                "reward": torch.tensor([float("nan"), 1., 2., 3., 4.]),
                "terminated": torch.tensor([0., 0., 0., 0., float(terminate)]),
            }, batch_size=[n]))
        before = wrapper.buffer._buffer._storage[:].clone()
        cursor = wrapper.buffer._buffer._writer._cursor
        rng = _rng_snapshot()
        bank = export_buffer_bank(wrapper.buffer, provenance=_provenance())
        _assert_rng(rng)
        assert cursor == wrapper.buffer._buffer._writer._cursor
        for key in before.keys():
            torch.testing.assert_close(before[key], wrapper.buffer._buffer._storage[:][key], equal_nan=True)
        np.testing.assert_array_equal(bank.observations[:, 0], [3, 10, 11, 12, 13])
        np.testing.assert_array_equal(bank.next_observations[:, 0], [4, 11, 12, 13, 14])
        np.testing.assert_allclose(bank.actions[:, 0], [.4, .1, .2, .3, .4])
        np.testing.assert_array_equal(bank.terminated[:, 0], [1, 0, 0, 0, 0])
        assert not bank.truncated.any()
        assert bank.provenance["export"]["truncation_flags"] == "not_stored_by_source_buffer"
        np.testing.assert_array_equal(valid_window_starts(bank, 2), [1, 2, 3])
    finally:
        wrapper.env.close()
