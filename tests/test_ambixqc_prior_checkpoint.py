"""Prior-only collection and frozen portable XQC checkpoint contracts."""

from copy import deepcopy
import json

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBIXQC import AMBIXQC
from test_ambixqc_core import _batch, _tiny_params, _tree_equal


def _wrapper(*, action_high=2.0, **overrides):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    env.action_space = gym.spaces.Box(-action_high, action_high, (1,), np.float32)
    params = _tiny_params(**overrides)
    return AMBIXQC(
        "AMBIXQC", env, params,
        {"alg": "AMBIXQC/AMBIXQC", "alg_params": params, "seed": 3,
         "device": params["device"], "env": "Pendulum-v1", "total_steps": 10},
        {},
    )


@pytest.fixture
def wrappers():
    created = []

    def make(**overrides):
        model = _wrapper(**overrides)
        created.append(model)
        return model

    yield make
    for model in created:
        model.flush_checkpoints()
        model.env.close()


def test_prior_training_keeps_outer_learning_warmup_and_checkpoint_metadata(
    wrappers, monkeypatch, tmp_path
):
    model = wrappers(inner_operator="none")
    agent = model.agent
    before = agent.frozen_outer_state()

    def forbidden(*args, **kwargs):
        pytest.fail("Prior collection must never construct or run an inner learner.")

    monkeypatch.setattr(agent.inner_engine, "act", forbidden)
    monkeypatch.setattr(agent.xqc_controller, "clone_for_inner", forbidden)
    actor_calls = []
    original_sample = agent.xqc_controller.sample_action

    def sample(*args, **kwargs):
        actor_calls.append(kwargs["deterministic"])
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(agent.xqc_controller, "sample_action", sample)
    action_metrics = []
    original_act = agent.act

    def act(*args, **kwargs):
        result = original_act(*args, **kwargs)
        action_metrics.append(dict(agent.last_inner_metrics))
        return result

    monkeypatch.setattr(agent, "act", act)
    model.learn(total_timesteps=10)
    assert actor_calls == [False] * 5  # Inclusive unchanged random warmup.
    assert model._global_step == 10
    assert agent.num_updates == model._num_updates == 6
    assert agent.xqc_workspace.update_step == 6
    assert agent.reward_normalizer.count == 10
    assert model.buffer.total_transitions == 10
    assert agent.inner_engine.action_index == 0
    assert agent.inner_engine._workspace_pool is None
    assert agent.inner_engine._replay_pool is None
    assert not _tree_equal(before["module"], agent.frozen_outer_state()["module"])
    for key in ("inner_active", "inner_model_steps", "inner_critic_optimizer_steps",
                "inner_actor_optimizer_steps", "inner_temperature_optimizer_steps"):
        assert all(metrics[key] == 0 for metrics in action_metrics)
    assert model.cfg.inner_rounds == 1
    assert model.cfg.inner_rollout_horizon == 2
    assert model.cfg.inner_replay_capacity == 4
    for key in ("inner_model_step_budget", "inner_expected_update_slots",
                "inner_nominal_updates_per_round", "inner_nominal_transitions_per_round",
                "inner_critic_updates_per_action", "inner_actor_updates_per_action",
                "inner_temperature_updates_per_action", "inner_nominal_critic_utd"):
        assert getattr(model.cfg, key) == 0
    path = model.save(tmp_path, "prior.pt")
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["semantic_signature"]["collection_operator"] == "none"
    metadata = json.loads((tmp_path / "prior.pt.metadata.json").read_text())
    assert metadata["checkpoint"]["step"] == 10
    assert metadata["trial_run_params"]["alg_params"]["inner_operator"] == "none"
    restored = wrappers(inner_operator="none").load(path)
    assert _tree_equal(restored.agent.checkpoint_state(), checkpoint)


def test_prior_actions_use_running_bn_and_dedicated_rng(wrappers):
    model = wrappers(inner_operator="none")
    agent = model.agent
    agent._update(*_batch(agent))
    observation, _ = model.env.reset(seed=13)
    with torch.no_grad():
        z = agent.model.encode(model._obs_to_tensor(observation).unsqueeze(0))
        expected, _ = agent.xqc_controller.sample_action(z, deterministic=True)
    expected = model._unscale_action(expected[0].cpu().numpy())
    outer_before = agent.frozen_outer_state()
    global_rng = torch.get_rng_state().clone()
    first = model.predict(observation, deterministic=True)[0]
    second = model.predict(observation, deterministic=True)[0]
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(first, second)
    sampled = [model.predict(observation, deterministic=False)[0] for _ in range(3)]
    assert any(not np.array_equal(sampled[0], value) for value in sampled[1:])
    assert _tree_equal(outer_before, agent.frozen_outer_state())
    assert torch.equal(global_rng, torch.get_rng_state())


def test_prior_collection_does_not_log_inner_action_counts_or_time(wrappers):
    model = wrappers(inner_operator="none")
    observation, _ = model.env.reset(seed=13)
    model.predict(observation, deterministic=False)
    model._record_action_metrics(planned=True, action_seconds=0.25)
    payload = model._wandb_train_window.pop()
    assert payload["train/inner_active"] == 0
    assert payload["train/inner_actions"] == 0
    assert payload["train/inner_steps"] == 0
    assert payload["train/inner_updates"] == 0
    assert model._wandb_inner_seconds == 0
    assert model._wandb_inner_actions == 0


def test_frozen_load_changes_only_inner_semantics_and_records_provenance(wrappers):
    source = wrappers(inner_operator="none")
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    target = wrappers(
        inner_rounds=2, inner_updates_per_round=4,
        inner_replay_capacity=8, inner_actor_lr=1e-4, inner_critic_lr=2e-4,
        inner_reward_normalization="action_local_imagined",
    )
    before = target.agent.frozen_outer_state()
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.frozen_outer_state())
    target.load(saved, frozen_evaluation=True)
    assert _tree_equal(source.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["saved_semantic_signature"]["collection_operator"] == "none"
    assert provenance["evaluated_semantic_signature"]["collection_operator"] == "xqc"
    assert provenance["evaluated_semantic_signature"]["inner_schedule"]["rounds"] == 2
    assert provenance["frozen_evaluation"] is True
    with pytest.raises(RuntimeError, match="cannot train"):
        target.learn(total_timesteps=10)
    with pytest.raises(RuntimeError, match="cannot update"):
        target.agent._update(*_batch(target.agent))


@pytest.mark.parametrize("overrides", [
    {"xqc_actor_net_arch": [12, 12]}, {"xqc_critic_net_arch": [12, 12]},
    {"xqc_policy_delay": 2}, {"xqc_tau": 0.1}, {"discount": 0.9},
    {"xqc_actor_lr": 0.02}, {"lr": 0.02}, {"rho": 0.8},
    {"action_high": 3.0}, {"train_unroll_horizon": 3},
])
def test_frozen_load_rejects_outer_or_action_contract_changes(wrappers, overrides):
    saved = deepcopy(wrappers(inner_operator="none").agent.checkpoint_state())
    target = wrappers(**overrides)
    before = target.agent.frozen_outer_state()
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved, frozen_evaluation=True)
    assert _tree_equal(before, target.agent.frozen_outer_state())


def test_v1_checkpoints_mean_xqc_and_keep_strict_loading(wrappers):
    source = wrappers()
    source.agent._update(*_batch(source.agent))
    legacy = deepcopy(source.agent.checkpoint_state())
    legacy["checkpoint_version"] = 1
    legacy["semantic_signature"].pop("collection_operator")
    legacy["semantic_signature"].pop("action_contract")
    same = wrappers().load(legacy)
    assert _tree_equal(source.agent.frozen_outer_state(), same.agent.frozen_outer_state())
    prior = wrappers(inner_operator="none")
    with pytest.raises(ValueError, match="semantics"):
        prior.load(legacy)
    prior.load(legacy, frozen_evaluation=True)
    provenance = prior.agent.checkpoint_evaluation_provenance
    assert provenance["checkpoint_version"] == 1
    assert provenance["saved_semantic_signature"]["collection_operator"] == "xqc"
    assert provenance["saved_semantic_signature"]["action_contract"] is None


@pytest.mark.parametrize("normalization", ["frozen_real_scale", "action_local_imagined"])
def test_seeded_episodes_reset_warmup_and_pool_without_mutating_outer(
    wrappers, normalization
):
    source = wrappers(inner_operator="none")
    source.agent.observe_reward(3.0, False, False)
    source.agent._update(*_batch(source.agent))
    model = wrappers(inner_updates_per_round=4, inner_reward_normalization=normalization)
    model.load(deepcopy(source.agent.checkpoint_state()), frozen_evaluation=True)
    agent = model.agent
    before = agent.frozen_outer_state()
    observation, _ = model.env.reset(seed=10)
    model.predict(observation)  # Unscored warmup allocates a local learner.
    pool = agent.inner_engine._workspace_pool

    def episode(seed, reuse):
        model.reset_for_evaluation(seed, reuse_action_pool=reuse)
        outputs = []
        for _ in range(3):
            outputs.append(model.predict(observation)[0])
            agent.observe_reward(100.0, True, False)
            assert agent.last_inner_metrics["inner_critic_optimizer_steps"] == 4
            assert agent.last_inner_metrics["inner_actor_optimizer_steps"] == 2
            assert agent.last_inner_metrics["inner_temperature_optimizer_steps"] == 2
        assert agent.inner_engine.action_index == 3
        return np.stack(outputs)

    first = episode(101, True)
    assert agent.inner_engine._workspace_pool is pool
    episode(333, True)  # Other configuration/episode activity cannot contaminate it.
    repeated = episode(101, True)
    fresh = episode(101, False)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, fresh)
    assert not np.array_equal(first[0], first[1])  # Inner adaptation is stochastic.
    assert _tree_equal(before, agent.frozen_outer_state())


def test_frozen_preflight_rejects_bad_inner_settings_and_rng_atomically(wrappers):
    source = wrappers(inner_operator="none")
    target = wrappers()
    state = deepcopy(source.agent.checkpoint_state())
    before = target.agent.frozen_outer_state()
    invalid = deepcopy(state)
    invalid["semantic_signature"]["inner_schedule"]["rounds"] = -1
    with pytest.raises(ValueError, match="rounds"):
        target.load(invalid, frozen_evaluation=True)
    invalid = deepcopy(state)
    invalid["inner"]["rng"]["streams"]["execution"] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises((ValueError, RuntimeError)):
        target.load(invalid, frozen_evaluation=True)
    assert _tree_equal(before, target.agent.frozen_outer_state())


def test_cpu_frozen_load_validates_cuda_philox_and_resets_backend_rng(wrappers):
    saved = deepcopy(wrappers(inner_operator="none").agent.checkpoint_state())
    # Synthetic CUDA generator wire state: uint64 seed and aligned uint64 offset.
    cuda_rng = torch.tensor(list((42).to_bytes(8, "little") + (4).to_bytes(8, "little")), dtype=torch.uint8)
    saved["outer_generator"] = cuda_rng.clone()
    saved["inner"]["rng"]["device_type"] = "cuda"
    for name in saved["inner"]["rng"]["streams"]:
        saved["inner"]["rng"]["streams"][name] = cuda_rng.clone()
    target = wrappers(inner_operator="none")
    with pytest.raises(ValueError, match="device type is incompatible"):
        target.load(saved)
    target.load(saved, frozen_evaluation=True)
    assert target.agent.checkpoint_evaluation_provenance["cross_device_rng_reset"]
    target.reset_for_evaluation(101)
    before = target.agent.frozen_outer_state()
    obs, _ = target.env.reset(seed=101)
    target.predict(obs)
    assert _tree_equal(before, target.agent.frozen_outer_state())
    malformed = deepcopy(saved)
    malformed["outer_generator"] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="generator state is invalid"):
        target.load(malformed, frozen_evaluation=True)
    assert _tree_equal(before, target.agent.frozen_outer_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_real_cross_device_frozen_checkpoint_round_trip(wrappers):
    cpu = wrappers(inner_operator="none")
    cpu.agent._update(*_batch(cpu.agent))
    cuda = wrappers(device="cuda").load(
        deepcopy(cpu.agent.checkpoint_state()), frozen_evaluation=True
    )
    cuda.reset_for_evaluation(101)
    before = cuda.agent.frozen_outer_state()
    obs, _ = cuda.env.reset(seed=101)
    cuda.predict(obs)
    assert _tree_equal(before, cuda.agent.frozen_outer_state())
    restored = wrappers().load(deepcopy(cuda.agent.checkpoint_state()), frozen_evaluation=True)
    restored.reset_for_evaluation(101)
    restored.predict(obs)
    assert restored.agent.checkpoint_evaluation_provenance["cross_device_rng_reset"]
