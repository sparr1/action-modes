import io
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.common.buffer import Buffer
from RL.tdmpc2_core.common.inner_utils import InnerRNG
from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer
from tests.test_ambi_inner_decoupling import (
    _assert_tree_equal,
    _clone_tree,
    _model as ambi_model,
)
from tests.test_tdmpc2_correctness import make_model, tiny_params
from tests.resume_test_support import BoundaryEnv, _model as resume_model
from utils.resume_runtime import UnsupportedResumeEnvironment


def _episode(rows, offset=0):
    return TensorDict(
        {
            "obs": torch.arange(
                offset, offset + rows * 3, dtype=torch.float32
            ).reshape(rows, 3),
            "action": torch.linspace(-0.5, 0.5, rows).reshape(rows, 1),
            "reward": torch.arange(rows, dtype=torch.float32),
            "terminated": torch.zeros(rows, dtype=torch.float32),
        },
        batch_size=[rows],
    )


def _replay_cfg(*, obs="state"):
    return SimpleNamespace(
        device="cpu",
        buffer_size=7,
        steps=30,
        batch_size=1,
        train_unroll_horizon=1,
        multitask=False,
        obs=obs,
        obs_shape={obs: (3,)},
        obs_dtype="float32",
        action_dim=1,
    )


def _contains_tensor(value):
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _swap_same_shaped_parameter_ids(optimizer, serialized):
    """Permute serialized IDs without making a shape mismatch reveal it."""
    group_index, first, second = next(
        (group_index, first, second)
        for group_index, group in enumerate(optimizer.param_groups)
        for first in range(len(group["params"]))
        for second in range(first + 1, len(group["params"]))
        if group["params"][first].shape == group["params"][second].shape
    )
    parameter_ids = serialized["param_groups"][group_index]["params"]
    parameter_ids[first], parameter_ids[second] = (
        parameter_ids[second],
        parameter_ids[first],
    )


def _corrupt_optimizer_state(optimizer, serialized, corruption):
    if corruption == "permuted-ids":
        _swap_same_shaped_parameter_ids(optimizer, serialized)
        return "parameter identifiers/order"
    if corruption == "missing-state":
        serialized["state"].pop(next(iter(serialized["state"])))
        return "state inventory"
    if corruption == "hyperparameter":
        serialized["param_groups"][0]["lr"] *= 2
        return "hyperparameters"
    if corruption == "step":
        state = next(iter(serialized["state"].values()))
        if torch.is_tensor(state["step"]):
            state["step"].add_(1)
        else:
            state["step"] += 1
        return "step does not match"
    raise AssertionError(f"Unknown optimizer corruption {corruption!r}.")


def test_baseline_exact_training_state_roundtrip_and_transfer_checkpoint_unchanged():
    source = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
        tiny_params(),
        total_steps=12,
    )
    source.learn(total_timesteps=12)
    source.agent._prev_mean.fill_(0.25)
    source.agent.last_plan_metrics = {"score": torch.tensor(3.0)}
    saved = _clone_tree(source.agent.training_state_dict())

    restored = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
        tiny_params(),
        total_steps=12,
    )
    restored.agent.load_training_state_dict(saved)
    _assert_tree_equal(restored.agent.training_state_dict(), saved)

    # The portable model-checkpoint contract must remain deliberately partial.
    assert set(source.agent.checkpoint_state()) == {
        "observation_spec",
        "model",
        "num_updates",
    }


def test_baseline_training_state_mismatch_fails_before_live_model_mutation():
    model = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
    )
    before = _clone_tree(model.agent.model.state_dict())
    saved = _clone_tree(model.agent.training_state_dict())
    key = next(
        name
        for name, value in saved["model"].items()
        if value.ndim and value.shape[0] > 1
    )
    saved["model"][key] = saved["model"][key][:-1]

    with pytest.raises(ValueError, match="incompatible before load"):
        model.agent.load_training_state_dict(saved)
    _assert_tree_equal(model.agent.model.state_dict(), before)


def test_baseline_optimizer_mismatch_fails_before_any_live_state_mutation():
    model = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
    )
    pristine = _clone_tree(model.agent.training_state_dict())
    invalid = _clone_tree(pristine)
    model_key = next(
        key for key, value in invalid["model"].items() if value.is_floating_point()
    )
    invalid["model"][model_key].add_(1.0)
    invalid["optim"]["param_groups"] = []

    with pytest.raises(ValueError, match="parameter-group layout"):
        model.agent.load_training_state_dict(invalid)

    _assert_tree_equal(model.agent.training_state_dict(), pristine)


@pytest.mark.parametrize(
    "corruption",
    ("permuted-ids", "missing-state", "hyperparameter", "step"),
)
def test_baseline_trained_optimizer_inventory_is_exact_and_transactional(corruption):
    model = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
        tiny_params(),
        total_steps=12,
    )
    model.learn(total_timesteps=12)
    pristine = _clone_tree(model.agent.training_state_dict())
    invalid = _clone_tree(pristine)
    model_key = next(
        key for key, value in invalid["model"].items() if value.is_floating_point()
    )
    invalid["model"][model_key].add_(1.0)

    expected = _corrupt_optimizer_state(
        model.agent.optim, invalid["optim"], corruption
    )

    with pytest.raises(ValueError, match=expected):
        model.agent.load_training_state_dict(invalid)

    _assert_tree_equal(model.agent.training_state_dict(), pristine)


@pytest.mark.parametrize("corruption", ("model-dtype", "scale-percentiles"))
def test_baseline_exact_tensor_policy_is_transactional(corruption):
    model = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
    )
    pristine = _clone_tree(model.agent.training_state_dict())
    invalid = _clone_tree(pristine)
    model_key = next(
        key for key, value in invalid["model"].items() if value.is_floating_point()
    )
    if corruption == "model-dtype":
        invalid["model"][model_key] = invalid["model"][model_key].to(torch.float64)
        expected = "dtype_mismatches"
    else:
        invalid["model"][model_key].add_(1.0)
        invalid["scale"]["percentiles"][0] = 1
        expected = "percentiles differ"

    with pytest.raises(ValueError, match=expected):
        model.agent.load_training_state_dict(invalid)

    _assert_tree_equal(model.agent.training_state_dict(), pristine)


@pytest.mark.parametrize(
    "corruption",
    ("permuted-ids", "missing-state", "hyperparameter", "step"),
)
def test_ambi_outer_optimizer_inventory_is_exact_and_transactional(corruption):
    model = ambi_model()
    agent = model.agent
    observations = torch.randn(
        model.cfg.train_unroll_horizon + 1,
        model.cfg.batch_size,
        3,
    )
    actions = torch.randn(
        model.cfg.train_unroll_horizon,
        model.cfg.batch_size,
        1,
    ).tanh()
    rewards = torch.randn_like(actions)
    agent._update(observations, actions, rewards, torch.zeros_like(rewards))
    agent.prepare_training_resume_boundary()

    pristine = _clone_tree(agent.training_state_dict())
    invalid = _clone_tree(pristine)
    model_key = next(
        key
        for key, value in invalid["outer"]["model"].items()
        if value.is_floating_point()
    )
    invalid["outer"]["model"][model_key].add_(1.0)
    optimizer_state = invalid["outer"]["optim"]
    expected = _corrupt_optimizer_state(agent.optim, optimizer_state, corruption)

    with pytest.raises(ValueError, match=expected):
        agent.load_training_state_dict(invalid)

    _assert_tree_equal(agent.training_state_dict(), pristine)


@pytest.mark.parametrize(
    "corruption",
    ("model-dtype", "entropy-shape", "fixed-entropy-value"),
)
def test_ambi_outer_tensor_contract_is_exact_and_transactional(corruption):
    model = ambi_model(**({"ent_coef": 0.2} if corruption == "fixed-entropy-value" else {}))
    model.agent.prepare_training_resume_boundary()
    pristine = _clone_tree(model.agent.training_state_dict())
    invalid = _clone_tree(pristine)
    model_key = next(
        key
        for key, value in invalid["outer"]["model"].items()
        if value.is_floating_point()
    )
    if corruption == "model-dtype":
        invalid["outer"]["model"][model_key] = invalid["outer"]["model"][
            model_key
        ].to(torch.float64)
        expected = "dtype_mismatches"
    elif corruption == "entropy-shape":
        invalid["outer"]["model"][model_key].add_(1.0)
        invalid["outer"]["log_ent_coef"] = torch.zeros(2)
        expected = "log_ent_coef.*shape"
    else:
        invalid["outer"]["model"][model_key].add_(1.0)
        invalid["outer"]["fixed_ent_coef"].add_(0.1)
        expected = "fixed entropy coefficient"

    with pytest.raises(ValueError, match=expected):
        model.agent.load_training_state_dict(invalid)

    _assert_tree_equal(model.agent.training_state_dict(), pristine)


@pytest.mark.parametrize("lifetime_profile", ("run", "mixed"))
@pytest.mark.parametrize(
    "corruption",
    ("permuted-ids", "missing-state", "hyperparameter", "step"),
)
def test_ambi_inner_optimizer_inventory_is_exact_and_transactional(
    lifetime_profile,
    corruption,
):
    if lifetime_profile == "run":
        overrides = {
            "inner_actor_scope": "run",
            "inner_critic_scope": "run",
            "inner_temperature_scope": "run",
            "inner_replay_scope": "run",
            "inner_actor_optimizer_scope": "run",
            "inner_critic_optimizer_scope": "run",
            "inner_temperature_optimizer_scope": "run",
        }
    else:
        overrides = {
            "inner_actor_scope": "run",
            "inner_actor_optimizer_scope": "run",
            "inner_critic_scope": "episode",
            "inner_critic_optimizer_scope": "episode",
            "inner_temperature_scope": "action",
            "inner_temperature_optimizer_scope": "action",
            "inner_replay_scope": "run",
        }

    source = ambi_model(**overrides)
    source.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source.agent.prepare_training_resume_boundary()
    optimizer = source.agent.inner_engine.state.actor_optim
    assert optimizer is not None
    invalid = _clone_tree(source.agent.training_state_dict()["inner"])
    optimizer_state = invalid["workspace"]["actor_optim"]
    expected = _corrupt_optimizer_state(optimizer, optimizer_state, corruption)

    target = ambi_model(**overrides)
    pristine = _clone_tree(target.agent.inner_engine.training_state_dict())
    with pytest.raises(ValueError, match=expected):
        target.agent.inner_engine.load_training_state_dict(invalid)

    _assert_tree_equal(target.agent.inner_engine.training_state_dict(), pristine)


@pytest.mark.parametrize("mode", ("auto", "fixed"))
@pytest.mark.parametrize("corruption", ("shape", "dtype"))
def test_ambi_inner_temperature_tensor_contract_is_transactional(mode, corruption):
    overrides = {
        "inner_temperature_mode": mode,
        "inner_temperature_scope": "run",
        "inner_temperature_optimizer_scope": "run",
    }
    if mode == "auto":
        overrides.update(
            inner_temperature_initialization="fixed",
            inner_temperature_updates_per_action=1,
        )
        field = "log_alpha"
    else:
        field = "alpha_fixed"

    source = ambi_model(**overrides)
    source.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source.agent.prepare_training_resume_boundary()
    invalid = _clone_tree(source.agent.training_state_dict()["inner"])
    value = invalid["workspace"][field]
    if corruption == "shape":
        invalid["workspace"][field] = value.reshape(1)
        expected = "shape"
    else:
        invalid["workspace"][field] = value.to(torch.float64)
        expected = "dtype"

    target = ambi_model(**overrides)
    pristine = _clone_tree(target.agent.inner_engine.training_state_dict())
    with pytest.raises(ValueError, match=expected):
        target.agent.inner_engine.load_training_state_dict(invalid)

    _assert_tree_equal(target.agent.inner_engine.training_state_dict(), pristine)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_baseline_training_state_is_cpu_serialized_and_loads_on_visible_cuda():
    source_device = "cuda:0"
    target_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    source = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
        tiny_params(device=source_device),
        total_steps=12,
    )
    source.learn(total_timesteps=12)

    stream = io.BytesIO()
    torch.save(source.agent.training_state_dict(), stream)
    stream.seek(0)
    cpu_state = torch.load(stream, map_location="cpu", weights_only=False)
    assert all(
        tensor.device.type == "cpu"
        for tensor in cpu_state["model"].values()
        if torch.is_tensor(tensor)
    )

    restored = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
        tiny_params(device=target_device),
        total_steps=12,
    )
    restored.agent.load_training_state_dict(cpu_state)
    expected_device = torch.device(target_device)
    assert all(parameter.device == expected_device for parameter in restored.agent.parameters())
    for optimizer in (restored.agent.optim, restored.agent.pi_optim):
        for parameter_state in optimizer.state.values():
            assert all(
                value.device == expected_device
                for value in parameter_state.values()
                if torch.is_tensor(value)
            )
    action = restored.agent.act(torch.zeros(3, device=expected_device), t0=True)
    assert action.device.type == "cpu"
    assert torch.isfinite(action).all()


def test_resume_enable_rejects_unreviewed_environment_before_collection():
    model = make_model(
        TDMPC2Baseline,
        gym.make("Pendulum-v1", max_episode_steps=5),
        tiny_params(episode_length=5),
        total_steps=10,
    )

    with pytest.raises(UnsupportedResumeEnvironment, match="reviewed"):
        model.enable_training_resume(total_timesteps=10)

    assert model.buffer.resumable_storage is False
    assert model.buffer.size == 0


def test_wrapper_builds_and_commits_ambi_agent_candidate_once(monkeypatch):
    source = resume_model(
        BoundaryEnv(),
        algorithm=AMBITDMPC2,
        inner_scope="run",
    )
    target = resume_model(
        BoundaryEnv(),
        algorithm=AMBITDMPC2,
        inner_scope="run",
    )
    try:
        saved = _clone_tree(source.training_state_dict())
        calls = []
        original = target.agent._preflight_training_state_dict

        def counted_preflight(state):
            calls.append(state)
            return original(state)

        monkeypatch.setattr(
            target.agent,
            "_preflight_training_state_dict",
            counted_preflight,
        )
        target.load_training_state_dict(saved)
        assert len(calls) == 1
        assert calls[0] is saved["agent"]
    finally:
        source._checkpoint_writer.shutdown()
        target._checkpoint_writer.shutdown()


def test_wrapper_rejects_corrupt_ambi_agent_without_live_mutation():
    source = resume_model(
        BoundaryEnv(),
        algorithm=AMBITDMPC2,
        inner_scope="run",
    )
    target = resume_model(
        BoundaryEnv(),
        algorithm=AMBITDMPC2,
        inner_scope="run",
    )
    try:
        invalid = _clone_tree(source.training_state_dict())
        model_key = next(
            key
            for key, value in invalid["agent"]["outer"]["model"].items()
            if value.is_floating_point()
        )
        invalid["agent"]["outer"]["model"][model_key].add_(1.0)
        invalid["agent"]["outer"]["log_ent_coef"] = torch.zeros(2)
        pristine = _clone_tree(target.agent.training_state_dict())

        with pytest.raises(ValueError, match="log_ent_coef.*shape"):
            target.load_training_state_dict(invalid)

        _assert_tree_equal(target.agent.training_state_dict(), pristine)
        assert target._global_step == 0
        assert target._num_updates == 0
    finally:
        source._checkpoint_writer.shutdown()
        target._checkpoint_writer.shutdown()


def test_inner_rng_restores_every_named_and_phase_stream_exactly():
    source = InnerRNG(seed=41, device="cpu")
    for name in source.STREAMS:
        torch.rand(5, generator=source.generator(name))
        with source.fork(name):
            torch.rand(3)
    saved = _clone_tree(source.training_state_dict())
    restored = InnerRNG(seed=999, device="cpu")
    restored.load_training_state_dict(saved)

    for name in source.STREAMS:
        torch.testing.assert_close(
            torch.rand(8, generator=source.generator(name)),
            torch.rand(8, generator=restored.generator(name)),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            source.phase_generators[name].get_state(),
            restored.phase_generators[name].get_state(),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "episodes",
    [(), ((3, 0),), ((3, 0), (4, 20), (3, 40))],
    ids=("empty", "partial", "wrapped"),
)
def test_replay_shards_roundtrip_physical_state_with_bounded_export(episodes):
    source = Buffer(_replay_cfg(), resumable=True)
    for rows, offset in episodes:
        source.add(_episode(rows, offset))

    metadata = source.training_state_metadata()
    # Capacity-sized tensors never leak into the metadata object.
    assert not _contains_tensor(metadata)
    shards = list(source.iter_training_state_shards(max_rows=2))
    assert all(
        all(value.shape[0] <= 2 for value in shard["fields"].values())
        for shard in shards
    )
    assert sum(shard["stop"] - shard["start"] for shard in shards) == source.size

    restored = Buffer(_replay_cfg(), resumable=True)
    restored.load_training_state_shards(metadata, iter(shards))
    assert restored.size == source.size
    assert restored.num_eps == source.num_eps
    assert restored.num_transitions == source.num_transitions
    assert restored.total_transitions == source.total_transitions
    assert list(restored._resident_episode_rows) == list(source._resident_episode_rows)

    if source.size:
        source_storage = source._buffer.state_dict()["_storage"]["_storage"]
        restored_storage = restored._buffer.state_dict()["_storage"]["_storage"]
        for key, value in source_storage.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(
                    restored_storage[key][: source.size],
                    value[: source.size],
                    rtol=0,
                    atol=0,
                )
        rng_state = torch.random.get_rng_state()
        source_sample = source.sample()
        torch.random.set_rng_state(rng_state)
        restored_sample = restored.sample()
        _assert_tree_equal(source_sample, restored_sample)


def test_replay_shard_export_never_calls_capacity_cloning_state_dict(monkeypatch):
    source = Buffer(_replay_cfg(), resumable=True)
    source.add(_episode(3))

    def fail_full_state_dict(*_args, **_kwargs):
        raise AssertionError("capacity-sized replay state_dict must not be called")

    monkeypatch.setattr(type(source._buffer), "state_dict", fail_full_state_dict)
    monkeypatch.setattr(
        type(source._buffer._storage._storage),
        "state_dict",
        fail_full_state_dict,
    )

    metadata = source.training_state_metadata()
    shard = source.training_state_shard(0, max_rows=2)
    assert metadata["storage_rows"] == 3
    assert shard["start"] == 0
    assert shard["stop"] == 2


def test_replay_shard_failure_is_transactional_and_resume_opt_in_is_strict():
    source = Buffer(_replay_cfg(), resumable=True)
    source.add(_episode(3))
    metadata = source.training_state_metadata()
    shards = list(source.iter_training_state_shards(max_rows=2))
    target = Buffer(_replay_cfg(), resumable=True)
    with pytest.raises(ValueError, match="ended before"):
        target.load_training_state_shards(metadata, shards[:-1])
    assert target.size == 0

    rgb = Buffer(_replay_cfg(obs="rgb"))
    with pytest.raises(NotImplementedError, match="state observations only"):
        rgb.enable_resumable_storage()
    legacy = Buffer(_replay_cfg())
    legacy.add(_episode(3))
    with pytest.raises(RuntimeError, match="before the first episode"):
        legacy.enable_resumable_storage()


def test_replay_restore_rejects_writer_cursor_accounting_corruption():
    source = Buffer(_replay_cfg(), resumable=True)
    source.add(_episode(3))
    source.add(_episode(4, 20))
    source.add(_episode(3, 40))
    target = Buffer(_replay_cfg(), resumable=True)

    state = _clone_tree(source.training_state_metadata())
    state["torchrl"]["writer"]["_cursor"] = (
        state["torchrl"]["writer"]["_cursor"] + 1
    ) % source.capacity
    shards = list(source.iter_training_state_shards(max_rows=2))
    with pytest.raises(ValueError, match="writer cursor is inconsistent"):
        target.load_training_state_shards(state, iter(shards))

    assert target.size == 0


@pytest.mark.parametrize("corruption", ("shape", "dtype"))
def test_replay_restore_rejects_configured_field_schema_before_allocation(corruption):
    source = Buffer(_replay_cfg(), resumable=True)
    source.add(_episode(3))
    metadata = _clone_tree(source.training_state_metadata())
    observation_spec = next(
        spec for spec in metadata["field_specs"] if spec["name"] == "obs"
    )
    if corruption == "shape":
        observation_spec["shape"] = [4]
    else:
        observation_spec["dtype"] = "torch.float64"

    target = Buffer(_replay_cfg(), resumable=True)
    with pytest.raises(ValueError, match="field specifications are incompatible"):
        target.load_training_state_shards(
            metadata,
            iter(source.iter_training_state_shards(max_rows=2)),
        )

    assert target.size == 0
    assert not hasattr(target, "_buffer")


def test_latent_replay_strict_training_state_preserves_wrapped_next_sample():
    source = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    z = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    source.add_batch(
        z,
        torch.arange(6, dtype=torch.float32).reshape(6, 1),
        torch.ones(6, 1),
        z + 1,
        torch.zeros(6, 1),
    )
    saved = _clone_tree(source.training_state_dict())
    restored = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    restored.load_training_state_dict(saved)
    generator_state = torch.Generator().manual_seed(13).get_state()
    source_generator = torch.Generator().set_state(generator_state)
    restored_generator = torch.Generator().set_state(generator_state)
    _assert_tree_equal(
        source.sample(9, generator=source_generator),
        restored.sample(9, generator=restored_generator),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pos", 3, "write position is inconsistent"),
        ("full", False, "full flag is inconsistent"),
        ("next_sample_id", 5, "write position is inconsistent"),
    ],
)
def test_latent_replay_rejects_ring_metadata_corruption(field, value, message):
    source = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    z = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    source.add_batch(
        z,
        torch.arange(6, dtype=torch.float32).reshape(6, 1),
        torch.ones(6, 1),
        z + 1,
        torch.zeros(6, 1),
    )
    state = _clone_tree(source.training_state_dict())
    state["state"][field] = value
    target = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")

    with pytest.raises(ValueError, match=message):
        target.load_training_state_dict(state)

    assert target.pos == 0
    assert target.full is False
    assert target.next_sample_id == 0


def test_latent_replay_rejects_storage_size_inconsistent_with_sample_ids():
    source = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    z = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    source.add_batch(
        z,
        torch.arange(2, dtype=torch.float32).reshape(2, 1),
        torch.ones(2, 1),
        z + 1,
        torch.zeros(2, 1),
    )
    state = _clone_tree(source.training_state_dict())
    state["state"]["z"] = state["state"]["z"][:-1]

    with pytest.raises(ValueError, match="different row counts"):
        LatentReplayBuffer(
            4, latent_dim=2, action_dim=1, device="cpu"
        ).load_training_state_dict(state)


@pytest.mark.parametrize(
    "corruption",
    ("extra-key", "field-shape", "field-dtype", "sample-id"),
)
def test_latent_replay_exact_preflight_is_transactional(corruption):
    source = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    z = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    source.add_batch(
        z,
        torch.arange(6, dtype=torch.float32).reshape(6, 1),
        torch.ones(6, 1),
        z + 1,
        torch.zeros(6, 1),
    )
    invalid = _clone_tree(source.training_state_dict())
    if corruption == "extra-key":
        invalid["state"]["unexpected"] = 1
        expected = "incompatible schema"
    elif corruption == "field-shape":
        invalid["state"]["z"] = invalid["state"]["z"].reshape(2, 4)
        expected = "wrong width"
    elif corruption == "field-dtype":
        invalid["state"]["z"] = invalid["state"]["z"].to(torch.float64)
        expected = "wrong dtype"
    else:
        invalid["state"]["sample_id"][0].add_(1)
        expected = "sample_id is inconsistent"

    target = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    target_z = torch.tensor([[100.0, 101.0], [102.0, 103.0]])
    target.add_batch(
        target_z,
        torch.tensor([[10.0], [11.0]]),
        torch.ones(2, 1),
        target_z + 1,
        torch.zeros(2, 1),
    )
    pristine = _clone_tree(target.training_state_dict())

    with pytest.raises((TypeError, ValueError), match=expected):
        target.load_training_state_dict(invalid)

    _assert_tree_equal(target.training_state_dict(), pristine)


@pytest.mark.parametrize("adaptation", ["clone", "lora"])
def test_ambi_exact_training_state_restores_all_run_scoped_workspace(adaptation):
    overrides = dict(
        inner_actor_adaptation=adaptation,
        inner_critic_adaptation=adaptation,
        inner_temperature_mode="auto",
        inner_temperature_initialization="fixed",
        inner_temperature_updates_per_action=1,
        inner_actor_scope="run",
        inner_critic_scope="run",
        inner_temperature_scope="run",
        inner_replay_scope="run",
        inner_actor_optimizer_scope="run",
        inner_critic_optimizer_scope="run",
        inner_temperature_optimizer_scope="run",
        inner_mppi_warm_start_scope="run",
    )
    source = ambi_model(**overrides)
    source.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source.agent.prepare_training_resume_boundary()
    saved = _clone_tree(source.agent.training_state_dict())

    restored = ambi_model(**overrides)
    restored.agent.load_training_state_dict(saved)
    _assert_tree_equal(restored.agent.training_state_dict(), saved)
    assert "inner" not in source.agent.checkpoint_state()

    episode_index = source.agent.inner_engine.episode_index
    source.agent.reset()
    restored.agent.reset()
    assert source.agent.inner_engine.episode_index == episode_index
    assert restored.agent.inner_engine.episode_index == episode_index

    source_action = source.agent.act(
        torch.full((3,), 0.2), t0=True, collect_diagnostics=False
    )
    restored_action = restored.agent.act(
        torch.full((3,), 0.2), t0=True, collect_diagnostics=False
    )
    torch.testing.assert_close(source_action, restored_action, rtol=0, atol=0)


def test_ambi_episode_state_must_expire_before_snapshot_and_mppi_run_state_restores():
    episode_scoped = ambi_model(
        inner_actor_scope="episode",
        inner_critic_scope="episode",
        inner_temperature_scope="episode",
        inner_replay_scope="episode",
        inner_actor_optimizer_scope="episode",
        inner_critic_optimizer_scope="episode",
        inner_temperature_optimizer_scope="episode",
    )
    episode_scoped.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    with pytest.raises(RuntimeError, match="resume boundary"):
        episode_scoped.agent.training_state_dict()
    episode_scoped.agent.prepare_training_resume_boundary()
    state = episode_scoped.agent.training_state_dict()["inner"]["workspace"]
    assert all(
        state[key] is None
        for key in (
            "actor",
            "critic",
            "actor_optim",
            "critic_optim",
            "replay",
        )
    )

    mppi = ambi_model(
        inner_operator="mppi",
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0,
        inner_temperature_updates_per_action=0,
        inner_mppi_warm_start_scope="run",
    )
    mppi.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    mppi.agent.prepare_training_resume_boundary()
    saved = _clone_tree(mppi.agent.training_state_dict())
    assert saved["inner"]["mppi_prev_mean"] is not None
    restored = ambi_model(
        inner_operator="mppi",
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0,
        inner_temperature_updates_per_action=0,
        inner_mppi_warm_start_scope="run",
    )
    restored.agent.load_training_state_dict(saved)
    _assert_tree_equal(restored.agent.training_state_dict(), saved)


@pytest.mark.parametrize(
    "tamper",
    ("outer-counter-divergence", "inner-newer-than-outer"),
)
def test_ambi_exact_state_rejects_impossible_outer_versions_transactionally(tamper):
    model = ambi_model()
    if tamper == "inner-newer-than-outer":
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    model.agent.prepare_training_resume_boundary()
    pristine = _clone_tree(model.agent.training_state_dict())
    invalid = _clone_tree(pristine)
    if tamper == "outer-counter-divergence":
        invalid["outer"]["outer_version"] += 1
        expected = "outer_version == num_updates"
    else:
        invalid["inner"]["workspace"]["counters"]["outer_version"] = (
            invalid["outer"]["outer_version"] + 1
        )
        expected = "cannot be newer"

    with pytest.raises(ValueError, match=expected):
        model.agent.load_training_state_dict(invalid)
    _assert_tree_equal(model.agent.training_state_dict(), pristine)


@pytest.mark.parametrize("divergence", ("outer", "inner"))
def test_ambi_capture_refuses_impossible_live_versions(divergence):
    model = ambi_model()
    if divergence == "inner":
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    model.agent.prepare_training_resume_boundary()
    if divergence == "outer":
        model.agent.outer_version += 1
        expected = "outer_version must equal num_updates"
    else:
        model.agent.inner_engine.state.outer_version = model.agent.outer_version + 1
        expected = "inner workspace is newer"
    with pytest.raises(RuntimeError, match=expected):
        model.agent.training_state_dict()


def test_ambi_mixed_lifetimes_resume_from_a_canonical_transient_free_boundary():
    overrides = dict(
        inner_actor_scope="run",
        inner_actor_optimizer_scope="action",
        inner_critic_scope="episode",
        inner_critic_optimizer_scope="episode",
        inner_temperature_mode="auto",
        inner_temperature_initialization="fixed",
        inner_temperature_updates_per_action=1,
        inner_temperature_scope="run",
        inner_temperature_optimizer_scope="episode",
        inner_replay_scope="episode",
    )
    source = ambi_model(**overrides)
    source.agent.act(torch.zeros(3), t0=True, collect_diagnostics=True)
    engine = source.agent.inner_engine
    assert engine.state.actor is not None
    assert engine.state.actor_optim is None
    assert engine._action_pool.actor_optim is not None
    assert engine.state.critic is not None
    assert engine.state.log_alpha is not None
    assert engine.state.temperature_optim is not None
    assert engine.state.replay is not None
    assert any(
        getattr(engine.state, name) != 0
        for name in engine._ACTION_TRANSIENT_COUNTER_FIELDS
    )

    source.agent.prepare_training_resume_boundary()
    assert engine._nondefault_workspace_fields(engine._action_pool) == []
    assert all(
        getattr(engine.state, name) == 0
        for name in engine._ACTION_TRANSIENT_COUNTER_FIELDS
    )
    assert engine.state.sampled_ids == []
    assert engine.state.actor is not None
    assert engine.state.actor_optim is None
    assert engine.state.actor_lifetime_steps > 0
    assert engine.state.critic is None
    assert engine.state.critic_optim is None
    assert engine.state.log_alpha is not None
    assert engine.state.temperature_optim is None
    assert engine.state.replay is None

    saved = _clone_tree(source.agent.training_state_dict())
    restored = ambi_model(**overrides)
    restored.agent.load_training_state_dict(saved)
    _assert_tree_equal(restored.agent.training_state_dict(), saved)

    # Boundary preparation advances the episode lifecycle exactly once. The
    # next environment reset consumes the marker but does not advance it again.
    episode_index = engine.episode_index
    source.agent.reset()
    restored.agent.reset()
    assert engine.episode_index == episode_index
    assert restored.agent.inner_engine.episode_index == episode_index
    observation = torch.full((3,), 0.125)
    torch.testing.assert_close(
        source.agent.act(observation, t0=True, collect_diagnostics=False),
        restored.agent.act(observation, t0=True, collect_diagnostics=False),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"inner_actor_adaptation": "lora"},
        {"inner_replay_capacity": 17},
        {
            "inner_actor_scope": "episode",
            "inner_actor_optimizer_scope": "episode",
        },
    ),
)
def test_standalone_inner_resume_rejects_structural_changes(changes):
    persistent = dict(
        inner_actor_scope="run",
        inner_critic_scope="run",
        inner_temperature_scope="run",
        inner_replay_scope="run",
        inner_actor_optimizer_scope="run",
        inner_critic_optimizer_scope="run",
        inner_temperature_optimizer_scope="run",
    )
    source = ambi_model(**persistent)
    source.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source.agent.prepare_training_resume_boundary()
    saved = _clone_tree(source.agent.training_state_dict()["inner"])

    target = ambi_model(**(persistent | changes))
    before_rng = _clone_tree(target.agent.inner_engine.rng.training_state_dict())
    with pytest.raises(ValueError):
        target.agent.inner_engine.load_training_state_dict(saved)
    assert target.agent.inner_engine.state.actor is None
    _assert_tree_equal(
        target.agent.inner_engine.rng.training_state_dict(),
        before_rng,
    )


def test_inner_resume_rejects_missing_run_scoped_components_before_mutation():
    persistent = dict(
        inner_actor_scope="run",
        inner_critic_scope="run",
        inner_temperature_scope="run",
        inner_replay_scope="run",
        inner_actor_optimizer_scope="run",
        inner_critic_optimizer_scope="run",
        inner_temperature_optimizer_scope="run",
    )
    source = ambi_model(**persistent)
    source.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source.agent.prepare_training_resume_boundary()
    saved = _clone_tree(source.agent.training_state_dict()["inner"])
    required = (
        "actor",
        "actor_anchor",
        "critic",
        "critic_anchor",
        "critic_target",
        "actor_optim",
        "critic_optim",
        "alpha_fixed",
        "replay",
    )
    assert all(saved["workspace"][field] is not None for field in required)

    target = ambi_model(**persistent)
    before_rng = _clone_tree(target.agent.inner_engine.rng.training_state_dict())
    for field in required:
        invalid = _clone_tree(saved)
        invalid["workspace"][field] = None
        with pytest.raises(ValueError, match="workspace inventory is incomplete"):
            target.agent.inner_engine.load_training_state_dict(invalid)
        assert target.agent.inner_engine.state.actor is None
        _assert_tree_equal(
            target.agent.inner_engine.rng.training_state_dict(),
            before_rng,
        )


def test_inner_resume_rejects_missing_run_scoped_mppi_warm_start():
    config = dict(
        inner_operator="mppi",
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0,
        inner_temperature_updates_per_action=0,
        inner_mppi_warm_start_scope="run",
    )
    source = ambi_model(**config)
    source.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source.agent.prepare_training_resume_boundary()
    invalid = _clone_tree(source.agent.training_state_dict()["inner"])
    assert invalid["mppi_prev_mean"] is not None
    invalid["mppi_prev_mean"] = None

    target = ambi_model(**config)
    with pytest.raises(ValueError, match="MPPI warm-start inventory"):
        target.agent.inner_engine.load_training_state_dict(invalid)
