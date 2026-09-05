"""Real-transition sampling and critic-only replay mixing contracts."""

from copy import deepcopy
from types import SimpleNamespace
import warnings

import pytest
import torch
from tensordict import TensorDict

from evaluate_ambi_checkpoint import _initialize_frozen_model
from RL.tdmpc2_core.common.buffer import Buffer
from tests.test_ambi_root_local_sac import _tiny_model


def _buffer(capacity=12):
    return Buffer(SimpleNamespace(
        device="cpu", buffer_size=capacity, steps=1000, batch_size=1,
        train_unroll_horizon=2, multitask=False, obs="state",
        obs_shape={"state": (3,)}, obs_dtype="float32", action_dim=1,
    ), resumable=True)


def _episode(start=0, rows=4, terminated=True):
    ids = torch.arange(start, start + rows, dtype=torch.float32)
    done = torch.zeros(rows)
    done[-1] = float(terminated)
    return TensorDict({
        "obs": ids[:, None].expand(-1, 3).clone(),
        "action": (ids / 1000)[:, None],
        "reward": ids + 1000,
        "terminated": done,
    }, batch_size=[rows])


def test_real_sampling_alignment_eviction_timeouts_and_rng():
    replay = _buffer(capacity=6)
    replay.add(_episode(0))
    replay.add(_episode(100, terminated=False))
    assert replay.num_sampleable_transitions == 4
    generator = torch.Generator().manual_seed(7)
    global_rng = torch.random.get_rng_state().clone()
    obs, action, reward, next_obs, terminated, task = replay.sample_transitions(
        512, generator=generator,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    assert task is None
    assert set(zip(obs[:, 0].tolist(), next_obs[:, 0].tolist())) == {
        (2., 3.), (100., 101.), (101., 102.), (102., 103.),
    }
    torch.testing.assert_close(action * 1000, next_obs[:, :1])
    torch.testing.assert_close(reward, next_obs[:, :1] + 1000)
    torch.testing.assert_close(terminated, (next_obs[:, :1] == 3).float())


def test_real_index_cache_reused_invalidated_and_rebuilt_after_restore():
    replay = _buffer(capacity=6)
    replay.add(_episode())
    generator = torch.Generator().manual_seed(5)
    replay.sample_transitions(3, generator=generator)
    initial = replay._transition_index_cache[torch.device("cpu")]
    replay.sample_transitions(3, generator=generator)
    assert replay._transition_index_cache[torch.device("cpu")] is initial
    replay.add(_episode(100))
    assert not replay._transition_index_cache
    replay.sample_transitions(3, generator=generator)
    assert replay._transition_index_cache[torch.device("cpu")] is not initial
    restored = _buffer(capacity=6)
    restored.add(_episode(500))
    restored.sample_transitions(3, generator=generator)
    restored.load_training_state_shards(
        replay.training_state_metadata(),
        list(replay.iter_training_state_shards(max_rows=2)),
    )
    assert not restored._transition_index_cache
    assert restored.num_sampleable_transitions == replay.num_sampleable_transitions
    for left, right in zip(
        replay.sample_transitions(20, generator=torch.Generator().manual_seed(9)),
        restored.sample_transitions(20, generator=torch.Generator().manual_seed(9)),
    ):
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_real_sampling_empty_and_bulk_load():
    replay = _buffer(capacity=6)
    with pytest.raises(ValueError, match="resident transition"):
        replay.sample_transitions(2, generator=torch.Generator())
    replay.load(torch.stack((_episode(0), _episode(100))))
    assert replay.num_sampleable_transitions == 4
    obs, _, _, next_obs, _, _ = replay.sample_transitions(100, generator=torch.Generator())
    assert torch.all(next_obs[:, 0] == obs[:, 0] + 1)


@pytest.mark.parametrize("fraction,count", [(0., 0), (.125, 1), (.5, 2), (1., 4)])
def test_mixing_keeps_actor_batch_and_encoder_frozen(monkeypatch, fraction, count):
    model = _tiny_model(inner_finite_horizon=True, inner_outer_replay_fraction=fraction)
    try:
        engine = model.agent.inner_engine
        engine.outer_replay_buffer = _buffer()
        engine.outer_replay_buffer.add(_episode())
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        engine._collect_round(torch.zeros(1, model.cfg.latent_dim))
        batch = engine._sample_batch()
        original = {key: value.clone() for key, value in batch.items()}
        calls = []
        def encode(obs, task):
            assert not torch.is_grad_enabled()
            calls.append(obs.clone())
            return obs[:, :1].expand(-1, model.cfg.latent_dim) + 50
        monkeypatch.setattr(engine.model, "encode", encode)
        mixed, metrics = engine._mix_outer_critic_batch(batch)
        for key in batch:
            torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)
        if count == 0:
            assert mixed is batch and metrics == {} and not calls
            return
        assert len(calls) == 1 and calls[0].shape[0] == 2 * count
        assert len(mixed["z"]) == 4
        assert metrics["outer_replay_fraction"] == count / 4
        assert metrics["outer_replay_samples"] == count
        assert not mixed["horizon_end"][-count:].any()
        torch.testing.assert_close(mixed["next_z"][-count:], mixed["z"][-count:] + 1)
        torch.testing.assert_close(mixed["reward"][-count:], mixed["next_z"][-count:, :1] + 950)
        assert not mixed["z"].requires_grad
    finally:
        model.env.close()


def test_empty_replay_warns_once_then_recovers():
    model = _tiny_model(inner_outer_replay_fraction=.5)
    try:
        engine = model.agent.inner_engine
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        engine._collect_round(torch.zeros(1, model.cfg.latent_dim))
        batch = engine._sample_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                mixed, metrics = engine._mix_outer_critic_batch(batch)
                assert mixed is batch
                assert metrics["outer_replay_fraction"] == 0
                assert metrics["outer_replay_available"] == 0
        assert len(caught) == 1 and "usable outer replay" in str(caught[0].message)
        model.buffer.add(_episode())
        _, metrics = engine._mix_outer_critic_batch(batch)
        assert metrics["outer_replay_fraction"] == .5
        assert metrics["outer_replay_available"] == 1
    finally:
        model.env.close()


@pytest.mark.parametrize("mode", ["none", "shared_mixture", "separate_critics"])
def test_joint_updates_keep_policy_states_imagined_and_report_real_samples(monkeypatch, mode):
    model = _tiny_model(
        inner_explorer_mode=mode, inner_outer_replay_fraction=1.,
        inner_finite_horizon=True, inner_steps_per_update=4,
        inner_updates_per_round=None, inner_rounds=1,
    )
    try:
        model.buffer.add(_episode(100))
        engine = model.agent.inner_engine
        observed = []
        method = {
            "none": "_sac_policy_step", "shared_mixture": "_shared_mixture_policy_step",
            "separate_critics": "_separate_policy_step",
        }[mode]
        original = getattr(engine, method)
        def policy_step(batch, **kwargs):
            observed.append(batch["z"].clone())
            return original(batch, **kwargs)
        monkeypatch.setattr(engine, method, policy_step)
        before = deepcopy(model.agent.model.state_dict())
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_outer_replay_samples"] == 4
        assert metrics["inner_outer_replay_fraction"] == 1
        assert metrics["inner_critic_optimizer_steps"] == 1
        assert metrics["inner_actor_optimizer_steps"] == 1
        assert observed
        replay_z = engine._action_pool.replay.z[:engine._action_pool.replay.size]
        for states in observed:
            assert ((states[:, None, :] == replay_z[None, :, :]).all(-1).any(-1)).all()
        for key, value in model.agent.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    finally:
        model.env.close()


def test_frozen_checkpoint_evaluation_rejects_mixing_before_loading():
    resolved = {"algorithm_config": {"alg_params": {"inner_outer_replay_fraction": .5}}}
    with pytest.raises(ValueError, match="no real replay"):
        _initialize_frozen_model(resolved, None, "unused.pt", 3)
