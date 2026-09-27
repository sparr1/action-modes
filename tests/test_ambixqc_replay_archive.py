"""Replay preservation must observe, rather than perturb, AMBI-XQC training."""

from copy import deepcopy
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBIXQC import AMBIXQC
from utils.replay_archive import ReplayArchiveError, load_checkpoint_replay


def _model(*, total_steps=13, inner_operator="xqc"):
    return AMBIXQC(
        "AMBIXQC",
        gym.make("Pendulum-v1", max_episode_steps=5),
        {
            "device": "cpu", "model_size": None,
            "enc_dim": 16, "mlp_dim": 16, "latent_dim": 8,
            "num_enc_layers": 2, "simnorm_dim": 4,
            "num_bins": 5, "vmin": -5, "vmax": 5,
            "batch_size": 2, "train_unroll_horizon": 2,
            "buffer_size": 9, "seed_steps": 4, "pretrain_steps": 1,
            "utd": 1, "compile": False, "episodic": False,
            "discount": 0.99, "wandb": False,
            "xqc_actor_net_arch": [8, 8], "xqc_critic_net_arch": [8, 8],
            "xqc_num_atoms": 11, "xqc_vmin": -2, "xqc_vmax": 2,
            "xqc_optimizer_backend": "single_tensor",
            "inner_operator": inner_operator,
            "inner_rounds": 1, "inner_rollouts_per_round": 2,
            "inner_rollout_horizon": 2, "inner_updates_per_round": 1,
            "inner_batch_size": 2, "inner_replay_capacity": 4,
        },
        {"seed": 3, "device": "cpu", "env": "test", "total_steps": total_steps},
        {},
    )


def _assert_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def _resident_rows(buffer):
    if not buffer.size:
        return None
    storage = buffer._buffer._storage._storage
    cursor = int(buffer._buffer._writer._cursor)
    indices = torch.arange(buffer.size)
    if buffer.size == buffer.capacity:
        indices = (indices + cursor) % buffer.capacity
    return {key: value[indices].clone() for key, value in storage.items()}


def _view_rows(view):
    fragments = [fragment["fields"] for fragment in view.fragments]
    if not fragments:
        return None
    return {
        key: torch.cat([fragment[key] for fragment in fragments])
        for key in fragments[0].keys()
    }


def _rng_state(model):
    return deepcopy({
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "outer": model.agent._outer_generator.get_state(),
        "inner": model.agent.inner_engine.rng.training_state_dict(),
        "environment": model.env.unwrapped.np_random.bit_generator.state,
        "actions": model.env.action_space.np_random.bit_generator.state,
    })


@pytest.mark.parametrize("inner_operator", ["none", "xqc"])
def test_archive_matches_checkpoint_replay_without_changing_training(tmp_path, inner_operator):
    results = []
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        model = _model(inner_operator=inner_operator)
        model.set_checkpointing(
            2, directory, "model", save_strat=("all", "best", "latest")
        )
        if enabled:
            model.enable_replay_archive(directory, "model")
        expected_at_steps = {}
        original_checkpoint = model._maybe_checkpoint

        def checkpoint():
            if model._global_step % 2 == 0:
                expected_at_steps[model._global_step] = _resident_rows(model.buffer)
            return original_checkpoint()

        model._maybe_checkpoint = checkpoint
        try:
            model.learn(total_timesteps=13)
            expected_at_steps[13] = _resident_rows(model.buffer)
            # The archive is closed after learn, but final explicit saves and
            # checkpoint-file reuse must retain a valid replay reference.
            final = Path(model.save(directory, "model_final"))
            repeated = Path(model.save(directory, "model_final_again"))
            assert final.read_bytes() == repeated.read_bytes()
            results.append({
                "state": deepcopy(model.agent.checkpoint_state()),
                "rng": _rng_state(model),
                "replay": _resident_rows(model.buffer),
                "accounting": model.buffer._accounting_state(),
            })
            assert model.buffer.total_transitions == 10
            assert model._episode_len == 3
            assert not model.buffer.resumable_storage
            assert model.agent.num_updates > 0
            if enabled:
                for sidecar in directory.glob("*.metadata.json"):
                    checkpoint_path = Path(str(sidecar).removesuffix(".metadata.json"))
                    metadata = json.loads(sidecar.read_text())
                    step = metadata["checkpoint"]["step"]
                    replay = load_checkpoint_replay(checkpoint_path)
                    _assert_equal(_view_rows(replay), expected_at_steps[step])
                    assert "replay" in metadata
                rows = _view_rows(load_checkpoint_replay(final))
                # Capacity nine keeps the last three rows of episode zero and
                # all six rows of episode one; the partial third episode is out.
                assert rows["episode"].tolist() == [0] * 3 + [1] * 6
                assert torch.isnan(rows["reward"][3])
                assert torch.isfinite(rows["reward"][:3]).all()
                assert torch.isfinite(rows["reward"][4:]).all()
                assert rows["reward"][torch.isfinite(rows["reward"])].abs().max() > 1
                assert rows["action"][torch.isfinite(rows["action"])].abs().max() <= 1
                before_sample = _rng_state(model)
                replay = load_checkpoint_replay(final)
                first = replay.sample_sequences(batch_size=7, horizon=1, seed=20)
                second = replay.sample_sequences(batch_size=7, horizon=1, seed=20)
                _assert_equal(first, second)
                _assert_equal(_rng_state(model), before_sample)
                assert first[0].shape == (2, 7, 3)
                assert first[1].shape == (1, 7, 1)
                assert first[4] is None
            else:
                assert not list(directory.glob("*.replay"))
                assert "replay" not in json.loads(Path(str(final) + ".metadata.json").read_text())
        finally:
            model.close_replay_archive()
            model.env.close()
    # Includes learned parameters, BN buffers, optimizer/temperature state,
    # reward normalization, and both global and named private random streams.
    _assert_equal(results[0], results[1])


def test_explicit_save_without_periodic_policy_preserves_empty_replay(tmp_path):
    model = _model(total_steps=3, inner_operator="none")
    try:
        model.enable_replay_archive(tmp_path, "final_only")
        model.learn(total_timesteps=3)
        checkpoint = model.save(tmp_path, "final_only")
        assert _view_rows(load_checkpoint_replay(checkpoint)) is None
        assert model.buffer.num_eps == 0
    finally:
        model.close_replay_archive()
        model.env.close()


def test_legacy_periodic_tuple_attaches_replay_metadata(tmp_path):
    model = _model(total_steps=6, inner_operator="none")
    try:
        model._checkpointing = (6, tmp_path, "legacy")
        model.enable_replay_archive(tmp_path, "legacy")
        model.learn(total_timesteps=6)
        _assert_equal(
            _view_rows(load_checkpoint_replay(tmp_path / "legacy_6")),
            _resident_rows(model.buffer),
        )
    finally:
        model.close_replay_archive()
        model.env.close()


def test_archive_must_be_enabled_before_collection(tmp_path):
    model = _model(total_steps=3, inner_operator="none")
    try:
        model.learn(total_timesteps=3)
        with pytest.raises(RuntimeError, match="before collection"):
            model.enable_replay_archive(tmp_path, "late")
        assert not (tmp_path / "late.replay").exists()
    finally:
        model.env.close()


def test_failed_archive_snapshot_preserves_existing_checkpoint(tmp_path, monkeypatch):
    model = _model(total_steps=3, inner_operator="none")
    archive = model.enable_replay_archive(tmp_path, "model")
    try:
        checkpoint = Path(model.save(tmp_path, "model_latest"))
        sidecar = Path(f"{checkpoint}.metadata.json")
        original_model, original_metadata = checkpoint.read_bytes(), sidecar.read_bytes()

        def fail_snapshot(*args, **kwargs):
            raise ReplayArchiveError("injected manifest publication failure")

        monkeypatch.setattr(archive, "checkpoint_reference", fail_snapshot)
        model._global_step = 1
        with pytest.raises(ReplayArchiveError, match="manifest publication failure"):
            model.save(tmp_path, "model_latest")
        model.flush_checkpoints()
        assert checkpoint.read_bytes() == original_model
        assert sidecar.read_bytes() == original_metadata
        assert load_checkpoint_replay(checkpoint).metadata["step"] == 0
    finally:
        model._checkpoint_writer.shutdown()
        model.close_replay_archive()
        model.env.close()


def test_training_failure_closes_replay_writer_and_preserves_primary_error(tmp_path, monkeypatch):
    model = _model(total_steps=6, inner_operator="none")
    archive = model.enable_replay_archive(tmp_path, "failure")
    closed = []
    original_close = archive.close

    def close():
        original_close()
        closed.append(True)
        raise OSError("archive close failed")

    def fail_update(_buffer):
        raise RuntimeError("training failed")

    monkeypatch.setattr(archive, "close", close)
    monkeypatch.setattr(model.agent, "update", fail_update)
    try:
        with pytest.raises(RuntimeError, match="training failed") as error:
            model.learn(total_timesteps=6)
        assert closed == [True]
        assert any("archive close failed" in note for note in error.value.__notes__)
        assert model.buffer.num_eps == 1
    finally:
        original_close()
        model.env.close()
