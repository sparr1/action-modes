"""Prior-KL changes only the requested regularizer and preserves pairing."""
from copy import deepcopy
import json

import numpy as np
import pytest

from evaluate_ambi_prior_kl import validate_saved_bank
from tests.test_ambi_entropy_probe_invariants import _fixture, _snapshot, _assert_unchanged
from utils.ambi_benchmark import canonical_hash, capture_root
from utils.ambi_entropy_probe import probe_entropy_root
from utils.ambi_prior_kl import probe_prior_kl_root


def test_kl_probe_preserves_baseline_and_disables_temperature_without_changing_first_phase():
    wrapper = _fixture("distributional")
    try:
        observation = np.asarray([.2, -.1, .7], dtype=np.float32)
        before = _snapshot(wrapper)
        reference = probe_entropy_root(wrapper, observation, seed=23, actor_steps=3, anchor_samples=8)
        result = probe_prior_kl_root(wrapper, observation, seed=23, actor_steps=3, anchor_samples=8)
        _assert_unchanged(wrapper, before)
        full = result["full_solves"]
        for key in ("final_actor_sha256", "final_critic_sha256", "final_private_rng_sha256"):
            assert full["baseline_entropy"][key] == reference["exact_engine"][key]
        np.testing.assert_array_equal(full["baseline_entropy"]["action"], reference["exact_engine"]["action"])
        assert full["prior_kl"]["beta"] == pytest.approx(float(wrapper.agent.alpha.detach()))
        assert full["prior_kl"]["temperature_steps"] == 0
        assert not full["prior_kl"]["temperature_allocated"]
        assert full["baseline_entropy"]["temperature_steps"] == 4
        assert full["prior_kl"]["actor_steps"] == full["baseline_entropy"]["actor_steps"] == 4
        for row in full["prior_kl"]["anchors_initial"].values():
            assert row["kl_to_prior"] == pytest.approx(0., abs=1e-7)
        first = full["prior_kl"]["actor_updates"][0]
        # Equivalent float32 distributions can leave roundoff in exp/log's
        # derivative cancellation; this is well below a meaningful actor step.
        assert first["gradient_projections"]["kl_gradient_norm"] == pytest.approx(0., abs=1e-6)
        assert first["metrics"]["actor_loss"] == pytest.approx(-first["metrics"]["actor_q_mean"], abs=1e-7)
        assert result["paired_first_phase"]
        fixed = result["fixed_replay_arms"]
        for key in ("paired_sample_id", "initial_actor_sha256", "frozen_critic_sha256", "fixed_replay_sha256"):
            assert fixed["baseline_entropy"][key] == fixed["prior_kl"][key]
        assert fixed["prior_kl"]["temperature_steps"] == 0
        assert fixed["baseline_entropy"]["temperature_steps"] == 3
    finally:
        wrapper.env.close()


def _saved(tmp_path):
    resolved = {"algorithm_config": {"alg_params": {"inner_rounds": 8}}}
    source = {"git_commit": "original", "files": {"code.py": "abc"}}
    source["sha256"] = canonical_hash(source["files"])
    identity = dict(checkpoint_sha256="checkpoint", metadata_sha256="metadata", profile_sha256="profile",
                    source_sha256=source["sha256"], resolved_sha256=canonical_hash(resolved))
    bank = dict(kind="ambi_entropy_matched_prefix_roots", environment_seed=101, identity=identity,
                roots=[capture_root(np.asarray([.5], dtype=np.float32), 101, 0, 0.)], transitions=[])
    bank["id"] = canonical_hash(bank)
    profile = dict(checkpoints={"test": {"checkpoint_sha256": "checkpoint", "metadata_sha256": "metadata"}},
                   source_git_commit="original", source_profile_sha256="profile", decisions=[0],
                   root_banks={"test": {"101": {"id": bank["id"]}}})
    path = tmp_path / "root-bank.json"
    path.write_text(json.dumps(bank))
    (tmp_path / "source.json").write_text(json.dumps(source))
    return path, profile, resolved, bank, source


def test_saved_bank_keeps_original_identity_and_rejects_rehashed_replacement(tmp_path):
    path, profile, resolved, bank, _ = _saved(tmp_path)
    loaded, _ = validate_saved_bank(path, profile=profile, checkpoint_id="test", environment_seed=101, resolved=resolved)
    assert loaded == bank
    modified = deepcopy(bank)
    modified["roots"][0]["observation"] = [1.]
    modified["id"] = canonical_hash({k: v for k, v in modified.items() if k != "id"})
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="pinned original bank"):
        validate_saved_bank(path, profile=profile, checkpoint_id="test", environment_seed=101, resolved=resolved)


def test_saved_bank_rejects_new_source_or_resolved_identity(tmp_path):
    path, profile, resolved, _, source = _saved(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_saved_bank(path, profile=profile, checkpoint_id="test", environment_seed=101, resolved={})
    source["git_commit"] = "new"
    (tmp_path / "source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="pinned original source"):
        validate_saved_bank(path, profile=profile, checkpoint_id="test", environment_seed=101, resolved=resolved)
