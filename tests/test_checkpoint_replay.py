"""Retained model/replay pairs remain exact under asynchronous publication."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from RL.tdmpc2_core.common.buffer import Buffer
from tests.resume_test_support import BoundaryEnv, _FakeWandb, _model, _session, _assert_tree_equal_nan
from tests.test_ambi_alpha_carry_training_state import _training_model
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from utils.checkpoint_replay import load_checkpoint_replay, prepare_checkpoint_replay
from utils.resume_training import RESUME_COMPLETE


def _assert_buffer_equal(left, right):
    _assert_tree_equal(left.training_state_metadata(), right.training_state_metadata())
    for actual, expected in zip(left.iter_training_state_shards(max_rows=2),
                                right.iter_training_state_shards(max_rows=2)):
        _assert_tree_equal_nan(actual, expected)


def test_periodic_replay_retains_empty_partial_and_wrapped_buffers_without_rng_changes(tmp_path, monkeypatch):
    model = _training_model("run")
    model.set_checkpointing(3, tmp_path, "model", checkpoint_replay=True)
    original = model._maybe_checkpoint
    verified = []

    def check():
        rng = torch.random.get_rng_state().clone()
        original()
        model.flush_checkpoints()
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        if model._global_step % 3:
            return
        target = tmp_path / f"model_{model._global_step}"
        restored = Buffer(model.cfg, resumable=True)
        loaded = load_checkpoint_replay(target, restored)
        assert loaded["step"] == model._global_step
        _assert_buffer_equal(restored, model.buffer)
        if model._last_terminated or model._last_truncated:
            assert loaded["pending_episode"] is None
        else:
            pending = loaded["pending_episode"]
            assert pending["episode_index"] == model._episode_idx
            assert pending["length"] == model._episode_len
            assert pending["return"] == model._episode_return
            for key, value in pending["rows"].items():
                torch.testing.assert_close(value, model._episode_staging[key][:model._episode_len + 1],
                                           rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        verified.append(model._global_step)

    monkeypatch.setattr(model, "_maybe_checkpoint", check)
    try:
        model.learn(total_timesteps=15)
        assert verified == [3, 6, 9, 12, 15]
        assert len(list(tmp_path.glob("model_*.replay/manifest.json"))) == 5
        # Older saved replay is unchanged after subsequent online collection.
        old = Buffer(model.cfg, resumable=True)
        loaded = load_checkpoint_replay(tmp_path / "model_6", old)
        assert old.total_transitions == 5
        assert loaded["pending_episode"]["length"] == 1
    finally:
        model.close()
        model.env.close()


def test_replay_shards_own_only_resident_rows_and_publish_before_sidecars(tmp_path, monkeypatch):
    model = _training_model("run")
    model.set_checkpointing(5, tmp_path, "model", checkpoint_replay=True)
    try:
        model.learn(total_timesteps=15)
        target = model._checkpointing.explicit_target(tmp_path / "manual", step=15, episode=3)
        prepared = prepare_checkpoint_replay(model.buffer, (target,), step=15, max_rows=2)
        shard_paths = sorted(prepared.staging.glob("shard-*.pt"))
        assert len(shard_paths) == 8
        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu")
            assert shard["stop"] - shard["start"] <= 2
            for value in shard["fields"].values():
                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        state = deepcopy(model.agent.checkpoint_state())
        model._checkpoint_writer.enqueue_many(state, (target,), before_metadata=prepared)
        model.flush_checkpoints()
        restored = Buffer(model.cfg, resumable=True)
        load_checkpoint_replay(target.path, restored)
        _assert_buffer_equal(model.buffer, restored)
        assert not prepared.staging.exists()
        metadata = json.loads(tmp_path.joinpath("manual.metadata.json").read_text())
        assert metadata["replay"]["step"] == 15
    finally:
        model.close()
        model.env.close()


@pytest.mark.parametrize("corrupt", ["model", "shard", "step"])
def test_mismatched_replay_pair_rejects_before_mutating_destination(tmp_path, corrupt):
    model = _training_model("run")
    model.set_checkpointing(5, tmp_path, "model", checkpoint_replay=True)
    try:
        model.learn(total_timesteps=10)
        path = tmp_path / "model_10"
        if corrupt == "model":
            with path.open("ab") as stream:
                stream.write(b"changed")
        elif corrupt == "shard":
            with Path(f"{path}.replay/shard-000000.pt").open("ab") as stream:
                stream.write(b"changed")
        else:
            metadata_path = Path(f"{path}.metadata.json")
            metadata = json.loads(metadata_path.read_text())
            metadata["checkpoint"]["step"] = 11
            metadata_path.write_text(json.dumps(metadata))
        restored = Buffer(model.cfg, resumable=True)
        before = restored.training_state_metadata()
        with pytest.raises(ValueError):
            load_checkpoint_replay(path, restored)
        _assert_tree_equal(restored.training_state_metadata(), before)
    finally:
        model.close()
        model.env.close()


def test_exact_resume_retains_periodic_pairs_and_copies_best_companion(tmp_path, monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    model = _model(BoundaryEnv(), total_steps=4)
    model.set_checkpointing(2, tmp_path / "portable", "model",
                           save_strat=["all", "best", "latest"], checkpoint_replay=True)
    session = _session(tmp_path / "lineage", mode="new", segment="whole", total_steps=4)
    try:
        assert model.learn(total_timesteps=4, resume_session=session) == RESUME_COMPLETE
        for step in (2, 4):
            loaded = load_checkpoint_replay(tmp_path / "portable" / f"model_{step}",
                                            Buffer(model.cfg, resumable=True))
            assert loaded["step"] == step
        generation = session.store._current
        best = generation.files_for_role("model_best")[0]
        loaded = load_checkpoint_replay(best, Buffer(model.cfg, resumable=True))
        assert loaded["step"] in (2, 4)
        assert generation.files_for_role("model_best_replay")
    finally:
        session.close()
        model.close()
        model.env.close()
