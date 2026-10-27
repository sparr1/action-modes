import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from tests.resume_test_support import (
    BoundaryEnv, _FakeWandb, _assert_tree_equal_nan, _model, _replay_state,
)
import utils.extend_completed_training as extension
import utils.resume_identity as identity_module
from utils.resume_identity import lineage_identity
from utils.resume_lineage import LineageConcurrentWriterError, LineageCorruptionError, LineageStore
from utils.resume_runtime import environment_contract
from utils.resume_training import RESUME_COMPLETE, TrainingResumeSession
from utils.oscar_resume_launcher import DoneVerificationError


@pytest.fixture
def setup(monkeypatch, tmp_path):
    source_id = {"commit": "a" * 40, "dirty": False, "tracked_diff_sha256": "e" * 64,
                 "untracked_count": 0, "untracked_sha256": "e" * 64}
    monkeypatch.setattr(identity_module, "source_identity", lambda _: copy.deepcopy(source_id))
    monkeypatch.setattr(extension, "source_identity", lambda _: copy.deepcopy(source_id))
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return tmp_path, fake


@contextmanager
def _session(path, identity, *, mode, target, segment):
    session = TrainingResumeSession.open(
        path, mode=mode, scientific_identity=identity, total_steps=target,
        checkpoint_minutes=10000, drain_after_seconds=None, segment_id=segment,
    )
    try:
        yield session
    finally:
        session.close()


def _complete(setup, *, algorithm=TDMPC2Baseline):
    root, fake = setup
    scope = {"inner_scope": "run"} if algorithm is AMBITDMPC2 else {}
    env = BoundaryEnv()
    model = _model(env, algorithm=algorithm, total_steps=4, **scope)
    run = {**model.run_params, "name": "tiny", "alg": (
        "AMBITDMPC2/AMBITDMPC2" if algorithm is AMBITDMPC2 else "TDMPC2/TDMPC2Baseline"
    ), "alg_params": copy.deepcopy(model.custom_params), "resume_environment": environment_contract(env)}
    experiment = {"configs": ["tiny"], "trials": 1, "save_trials": "none",
                  "overrides_alg": {"total_steps": 4}}
    identity = lineage_identity(trial_run_params=run, experiment_params=experiment, repo_root=root)
    source = root / "source"
    with _session(source, identity, mode="new", target=4, segment="first") as session:
        assert model.learn(total_timesteps=4, resume_session=session) == RESUME_COMPLETE
        generation = session.last_generation
    return source, model, env, generation, identity


def _migrate(root, source, **kwargs):
    return extension.extend_completed_training(
        source_lineage=source, total_steps=kwargs.pop("total_steps", 8), training_repo=root,
        destination=kwargs.pop("destination", root / "scratch" / "continued"), **kwargs,
    )


def _hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file() and path.name != ".lineage.lock"}


@pytest.mark.parametrize("algorithm", [TDMPC2Baseline, AMBITDMPC2])
def test_extension_preserves_all_state_then_matches_uninterrupted_training(setup, algorithm):
    root, fake = setup
    source, initial_model, first_env, old_generation, _ = _complete(setup, algorithm=algorithm)
    old_envelope = torch.load(old_generation.files_for_role("trainer")[0], weights_only=False)
    source_hashes = _hashes(source)
    old_history = copy.deepcopy(fake.runs[old_envelope["wandb"]["run_id"]].history)
    result = _migrate(root, source)
    assert _hashes(source) == source_hashes
    assert not (Path(result["lineage_dir"]) / "DONE").exists()
    assert len(fake.runs) == 1  # Offline migration never initializes or writes W&B.
    _assert_tree_equal_nan(fake.runs[result["wandb_run_id"]].history, old_history)
    with LineageStore.open(result["lineage_dir"], mode="required") as store:
        new_generation = store.load()
        new_envelope = torch.load(new_generation.files_for_role("trainer")[0], weights_only=False)
        expected = copy.deepcopy(old_envelope)
        expected["generation_id"] = new_generation.generation_id
        expected["trainer"]["total_steps"] = 8
        # Includes optimizer/alpha/inner state, environment RNG, W&B and counters.
        _assert_extended_envelope_equal(new_envelope, expected)
        for old_file, new_file in zip(old_generation.files, new_generation.files):
            assert old_file.path == new_file.path
            if old_file.role != "trainer":
                assert old_file.sha256 == new_file.sha256
        new_identity = store.lineage_metadata["scientific_identity"]

    run = json.loads((Path(result["alg_dir"]) / "tiny.json").read_text())
    experiment = json.loads(Path(result["run_config"]).read_text())
    resolved = {"name": "tiny", **run, **experiment["overrides_alg"]}
    assert lineage_identity(trial_run_params=resolved, experiment_params=experiment, repo_root=root) == new_identity
    assert resolved["alg_params"]["buffer_size"] == 4
    scope = {"inner_scope": "run"} if algorithm is AMBITDMPC2 else {}
    resumed_env = BoundaryEnv()
    resumed = algorithm("tiny", resumed_env, resolved["alg_params"], resolved, experiment)
    resumed.enable_training_resume(total_timesteps=8)
    with _session(result["lineage_dir"], new_identity, mode="required", target=8, segment="second") as session:
        assert resumed.learn(total_timesteps=8, resume_session=session) == RESUME_COMPLETE
    assert len(fake.runs) == 1
    history = fake.runs[result["wandb_run_id"]].history
    _assert_tree_equal_nan(history[:len(old_history)], old_history)
    assert [row["_step"] for row in history] == list(range(len(history)))
    assert len(history) > len(old_history)

    continuous_env = BoundaryEnv()
    continuous = _model(continuous_env, algorithm=algorithm, total_steps=8, buffer_size=4, **scope)
    with _session(root / "continuous", new_identity, mode="new", target=8, segment="full") as session:
        assert continuous.learn(total_timesteps=8, resume_session=session) == RESUME_COMPLETE
    _assert_tree_equal_nan(resumed.agent.training_state_dict(), continuous.agent.training_state_dict())
    _assert_tree_equal_nan(_replay_state(resumed.buffer), _replay_state(continuous.buffer))
    assert resumed._num_updates == continuous._num_updates
    for actual, expected in zip(first_env.trace + resumed_env.trace, continuous_env.trace):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
        np.testing.assert_array_equal(actual[2], expected[2])


def _assert_extended_envelope_equal(actual, expected):
    # The shared helper handles tensor trees; NumPy RNG state contains arrays.
    for key in actual:
        if key == "global_rng":
            for rng_key in actual[key]:
                if rng_key == "numpy":
                    left, right = actual[key][rng_key], expected[key][rng_key]
                    assert left[0] == right[0] and left[2:] == right[2:]
                    np.testing.assert_array_equal(left[1], right[1])
                else:
                    _assert_tree_equal_nan(actual[key][rng_key], expected[key][rng_key])
        else:
            _assert_tree_equal_nan(actual[key], expected[key])


def test_same_request_is_idempotent_and_competing_destination_is_refused(setup):
    root, _ = setup
    source, *_ = _complete(setup)
    first = _migrate(root, source)
    assert _migrate(root, source) == first
    with pytest.raises(extension.ExtensionError, match="different continuation claim"):
        _migrate(root, source, destination=root / "duplicate")
    with pytest.raises(extension.ExtensionError, match="different continuation claim"):
        _migrate(root, source, total_steps=10)


def test_failed_publication_keeps_source_and_retries_same_claim(setup, monkeypatch):
    root, _ = setup
    source, *_ = _complete(setup)
    original = _hashes(source)
    real_rename = extension.os.rename
    with monkeypatch.context() as patch:
        def fail_rename(src, dst):
            if Path(dst).name == "continued":
                raise OSError("injected publication failure")
            return real_rename(src, dst)
        patch.setattr(extension.os, "rename", fail_rename)
        with pytest.raises(OSError, match="injected"):
            _migrate(root, source)
    assert _hashes(source) == original
    assert not (root / "scratch" / "continued").exists()
    result = _migrate(root, source)
    assert Path(result["lineage_dir"]).exists()


def test_post_publication_claim_failure_recovers_without_new_generation(setup, monkeypatch):
    root, _ = setup
    source, *_ = _complete(setup)
    write_json = extension._atomic_json
    with monkeypatch.context() as patch:
        def fail_committed_claim(path, payload):
            if path.name.endswith(".extension-claim.json") and payload.get("status") == "committed":
                raise OSError("injected claim finalization failure")
            return write_json(path, payload)
        patch.setattr(extension, "_atomic_json", fail_committed_claim)
        with pytest.raises(OSError, match="claim finalization"):
            _migrate(root, source)
    destination = root / "scratch" / "continued"
    generations = sorted((destination / "generations").iterdir())
    result = _migrate(root, source)
    assert result["generation_id"] == generations[0].name
    assert sorted((destination / "generations").iterdir()) == generations


@pytest.mark.parametrize("target", [4, 2, 7])
def test_rejects_shorter_or_nonboundary_target_without_claim(setup, target):
    root, _ = setup
    source, *_ = _complete(setup)
    with pytest.raises(extension.ExtensionError):
        _migrate(root, source, total_steps=target)
    assert not source.with_name(source.name + ".extension-claim.json").exists()


def test_incomplete_locked_corrupt_or_different_source_is_rejected(setup, monkeypatch):
    root, _ = setup
    source, _, _, generation, _ = _complete(setup)
    with LineageStore.open(source, mode="required"):
        with pytest.raises(LineageConcurrentWriterError):
            _migrate(root, source)
    with monkeypatch.context() as patch:
        patch.setattr(extension, "source_identity", lambda _: {"commit": "different"})
        with pytest.raises(extension.ExtensionError, match="exactly match"):
            _migrate(root, source)
    done = (source / "DONE").read_bytes()
    (source / "DONE").unlink()
    with pytest.raises(DoneVerificationError):
        _migrate(root, source)
    (source / "DONE").write_bytes(done)
    generation.files_for_role("replay")[0].write_bytes(b"damaged")
    with pytest.raises(LineageCorruptionError):
        _migrate(root, source)
    assert not source.with_name(source.name + ".extension-claim.json").exists()
