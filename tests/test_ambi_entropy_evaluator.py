"""Scientific protocol and immutable-worker checks without a simulator or W&B."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from evaluate_ambi_entropy import (
    _file_sha256, checkpoint_matrix, critic_probe, expected_rows, load_study,
    metric_semantics, state_digest, validate_worker,
)
from RL.AMBITDMPC2 import AMBITDMPC2
from utils.ambi_benchmark import atomic_json, canonical_hash
from utils.ambi_research import resolve_preset
from utils.checkpoint_context import CheckpointContext


ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = ROOT / "configs/research/ambi_entropy_prior_h1.json"
ARMS = ["off", "prior_recipe", "squashed_matched"]


@pytest.fixture
def study():
    return load_study(STUDY_PATH)


def test_reference_study_declares_full_paired_panel(study):
    assert study["episode_seeds"] == list(range(101, 121))
    assert study["decisions"] == [0, 100, 200, 300, 400]
    assert study["actor_updates"] == [0, 1, 4, 16]
    assert (study["rollouts"], study["batch_size"], study["critic_updates"]) == (128, 256, 32)
    assert (study["solver_repetitions"], study["rollout_repetitions"]) == (3, 4)
    assert (study["tail_steps"], study["model_probe_rollouts"]) == (1000, 32)
    assert study["bootstrap_resamples"] == 2000
    assert [(c["id"], c["step"]) for c in study["checkpoints"]] == [
        ("mey3rxj8", 200000), ("jirflxz1", 500000), ("u13m14st", 500000)]
    rows = [r for c in study["checkpoints"]
            for r in expected_rows(study, c, study["episode_seeds"], ARMS)]
    assert len(rows) == 86400
    assert len({canonical_hash(r) for r in rows}) == len(rows)
    assert {r["prefix_action_rule"] for r in rows} == {"sampled", "mean"}
    assert all(len(c["sha256"]) == len(c["metadata_sha256"]) == 64
               for c in study["checkpoints"])


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("kind", "other"),
    ("episode_seeds", []), ("episode_seeds", [101, 101]),
    ("episode_seeds", [True]), ("decisions", [-1]),
    ("decisions", [500]), ("actor_updates", [1, 4]),
    ("actor_updates", [0, 16, 4]), ("actor_updates", [0, 1, 1]),
    ("rollouts", 0), ("batch_size", True), ("critic_updates", 1.5),
    ("solver_repetitions", 0), ("rollout_repetitions", -1),
    ("tail_steps", 498), ("model_probe_rollouts", 0),
    ("bootstrap_resamples", 0), ("horizon", 3), ("max_steps", 1000),
    ("prefix_action_rules", ["mean"]),
])
def test_study_rejects_invalid_or_incomplete_protocol(study, tmp_path, key, value):
    study[key] = value
    path = tmp_path / "study.json"
    atomic_json(path, study)
    with pytest.raises(ValueError):
        load_study(path)


def test_study_rejects_duplicate_checkpoint_identity(study, tmp_path):
    study["checkpoints"].append(deepcopy(study["checkpoints"][0]))
    path = tmp_path / "study.json"
    atomic_json(path, study)
    with pytest.raises(ValueError, match="Duplicate checkpoint"):
        load_study(path)


@pytest.mark.parametrize("source_id", ["u13m14st", "mey3rxj8", "jirflxz1"])
def test_checkpoint_matrix_preserves_outer_and_policy_compatibility(study, tmp_path, source_id):
    # Minimal saved contexts represent the missing legacy fields as well as
    # both new recipes. No downloaded checkpoint or current training preset is
    # needed to exercise the real resolver and configuration validation.
    old = source_id == "u13m14st"
    scaled_q = source_id == "mey3rxj8"
    saved = dict(model_size=5, obs="state", discount=.99, compile=False,
                 outer_critic_target="entropy_augmented" if old else "reward_only",
                 sac_actor_loss_scale_mode="tdmpc2_percentile_range" if scaled_q else "none",
                 outer_q_actor_reduction="min_pair" if old else "mean_pair",
                 outer_q_target_reduction="min_pair", ent_coef=.0001 if scaled_q else "auto",
                 log_std_min=-20 if old else -10, log_std_max=2,
                 inner_operator="none", inner_updates_per_round=0,
                 inner_critic_updates_per_action=0, inner_actor_updates_per_action=0,
                 inner_temperature_updates_per_action=0)
    if not old:
        saved.update(log_std_mapping="tdmpc2_tanh", outer_actor_entropy_mode="tdmpc2_scaled",
                     target_entropy=-441)
    run = dict(alg="AMBITDMPC2/AMBITDMPC2", env="DMControl-v0", seed=55,
               device="cpu", total_steps=2000000, alg_params=saved)
    context = CheckpointContext(run, {"env_params": {"task": "humanoid-walk", "obs": "state"}},
                                tmp_path / "checkpoint.metadata.json")
    before = deepcopy(context)
    source = next(c for c in study["checkpoints"] if c["id"] == source_id)
    matrix = checkpoint_matrix(study, source, context)
    resolved = resolve_preset(STUDY_PATH, "entropy/shared_fit", matrix, checkpoint_context=context)
    params = resolved["algorithm_config"]["alg_params"]
    assert context == before
    assert resolved["environment"] == {"id": "DMControl-v0", "params": context.experiment_params["env_params"]}
    assert all(params[k] == v for k, v in saved.items() if not k.startswith("inner_"))
    assert "mppi_terminal_q_reduction" not in matrix["shared_alg_params"]
    assert all(params.get(k) is None for k in ("inner_log_std_min", "inner_log_std_max", "inner_log_std_mapping"))
    assert "inner_updates_per_round" not in params

    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=500)
    algorithm.env.action_space = gym.spaces.Box(-1, 1, (21,), dtype=np.float32)
    algorithm.env.observation_space = gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32)
    algorithm.run_params = resolved["algorithm_config"]
    algorithm.custom_params = params
    try:
        cfg = algorithm._build_cfg(params)
    finally:
        algorithm.env.close()
    assert cfg.log_std_mapping == cfg.inner_log_std_mapping == ("direct_clamp" if old else "tdmpc2_tanh")
    assert cfg.log_std_min == cfg.inner_log_std_min == (-20 if old else -10)
    assert cfg.log_std_max == cfg.inner_log_std_max == 2
    assert cfg.outer_actor_entropy_mode == ("squashed" if old else "tdmpc2_scaled")
    assert cfg.outer_critic_target == saved["outer_critic_target"]
    assert cfg.sac_actor_loss_scale_mode == saved["sac_actor_loss_scale_mode"]
    assert cfg.ent_coef == saved["ent_coef"]
    assert cfg.inner_q_actor_reduction == cfg.outer_q_actor_reduction == saved["outer_q_actor_reduction"]
    assert cfg.inner_q_target_reduction == cfg.outer_q_target_reduction == "min_pair"
    assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == "prior"
    assert cfg.inner_sac_critic_target == "reward_only"
    assert cfg.inner_actor_loss_scale_update == "per_action"
    assert cfg.inner_temperature == 0 and cfg.inner_temperature_updates_per_action == 0
    assert (cfg.inner_model_step_budget, cfg.inner_critic_updates_per_action,
            cfg.inner_actor_updates_per_action) == (128, 32, 16)


def test_soft_q_labels_do_not_claim_reward_prediction_or_change_real_rewards():
    rename = {
        "model_return": "hybrid_model_score", "model_bootstrap": "hybrid_model_bootstrap",
        "real_endpoint_q": "soft_endpoint_q", "real_bootstrap": "soft_bootstrap",
        "real_bootstrapped_return": "hybrid_real_prefix_score",
        "bootstrap_prediction_error": "soft_bootstrap_minus_measured_reward_tail",
        "total_prediction_error": "hybrid_score_minus_real_reward_return",
        "model_dynamics_prediction_error": "hybrid_model_minus_real_prefix_score",
        "model_gain_vs_prior": "hybrid_model_gain_vs_prior",
        "real_bootstrapped_gain_vs_prior": "hybrid_real_prefix_gain_vs_prior",
    }
    raw = {key: float(i) for i, key in enumerate([
        "real_mc_return", "real_gain_vs_prior", "real_prefix_reward", "real_tail_return",
        "discounted_cutoff_return", "undiscounted_cutoff_return"])}
    metrics = {**raw, **{key: i + 100. for i, key in enumerate(rename)}}
    metrics.update({"prior_" + key: value + 1000 for key, value in list(metrics.items())})
    before = deepcopy(metrics)
    converted = metric_semantics(metrics, reward_only=False)
    assert metrics == before
    assert len(converted) == len(metrics)
    for prefix in ("", "prior_"):
        for key, value in raw.items():
            assert converted[prefix + key] == metrics[prefix + key]
        for key, name in rename.items():
            assert prefix + key not in converted
            assert converted[prefix + name] == metrics[prefix + key]
    assert metric_semantics(metrics, reward_only=True) == before


def test_expected_rows_preserves_shard_ownership_and_gaussian_control(study):
    source = study["checkpoints"][0]
    rows = expected_rows(study, source, [107], ARMS + ["gaussian_control"])
    assert len(rows) == 1920
    assert {r["episode_seed"] for r in rows} == {107}
    assert {r["root_id"] for r in rows} == {f"seed-107-decision-{d}" for d in study["decisions"]}
    for update in study["actor_updates"]:
        off = {canonical_hash({k: v for k, v in r.items() if k != "arm"})
               for r in rows if r["arm"] == "off" and r["actor_updates"] == update}
        for arm in ARMS[1:] + ["gaussian_control"]:
            other = {canonical_hash({k: v for k, v in r.items() if k != "arm"})
                     for r in rows if r["arm"] == arm and r["actor_updates"] == update}
            assert other == off


def _seal_worker(directory):
    atomic_json(directory / "checksums.json", {
        p.name: _file_sha256(p) for p in directory.iterdir() if p.name != "checksums.json"}, overwrite=True)


def _write_rows(directory, rows):
    with gzip.open(directory / "measurements.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


@pytest.fixture
def sealed_worker(tmp_path, study):
    directory = tmp_path / "worker"
    directory.mkdir()
    source = study["checkpoints"][0]
    record = dict(complete=True, smoke=False, outer_state_unchanged=True,
                  study_sha256=canonical_hash(study), source=source, episode_seed=101,
                  arm_names=ARMS + (["gaussian_control"] if source.get("gaussian_control") else []), rows=1)
    rows = [dict(mc_complete=True, truncated=False, metrics={"real_mc_return": 3.})]
    atomic_json(directory / "results.json", record)
    for name in ("root-bank.json", "matrix.json", "checkpoint.metadata.json", "arms.json"):
        atomic_json(directory / name, {})
    atomic_json(directory / "study.json", study)
    _write_rows(directory, rows)
    _seal_worker(directory)
    return directory, record, rows


def test_valid_worker_seal_round_trip(sealed_worker, study):
    directory, record, rows = sealed_worker
    assert validate_worker(directory, study=study, source=study["checkpoints"][0], episode_seed=101) == (record, rows)


@pytest.mark.parametrize("field,value", [
    ("complete", False), ("smoke", True), ("outer_state_unchanged", False),
    ("study_sha256", "other"), ("source", {"id": "other"}),
    ("episode_seed", 102), ("arm_names", ["off", "prior_recipe"]),
])
def test_worker_rejects_wrong_source_or_protocol_even_with_valid_seal(sealed_worker, study, field, value):
    directory, record, _ = sealed_worker
    record[field] = value
    atomic_json(directory / "results.json", record, overwrite=True)
    _seal_worker(directory)
    with pytest.raises(ValueError, match="source, protocol, coverage or frozen-state"):
        validate_worker(directory, study=study, source=study["checkpoints"][0], episode_seed=101)


@pytest.mark.parametrize("mutation", ["missing_mc", "mc_false", "truncated", "count"])
def test_worker_rejects_incomplete_real_continuations(sealed_worker, study, mutation):
    directory, _, rows = sealed_worker
    if mutation == "missing_mc":
        rows[0].pop("mc_complete")
    elif mutation == "mc_false":
        rows[0]["mc_complete"] = False
    elif mutation == "truncated":
        rows[0]["truncated"] = True
    else:
        rows.append(deepcopy(rows[0]))
    _write_rows(directory, rows)
    _seal_worker(directory)
    with pytest.raises(ValueError, match="Incomplete worker"):
        validate_worker(directory, study=study, source=study["checkpoints"][0], episode_seed=101)


def test_worker_rejects_tampering_and_unsealed_required_file(sealed_worker, study):
    directory, _, _ = sealed_worker
    (directory / "matrix.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="checksum mismatch: matrix.json"):
        validate_worker(directory, study=study, source=study["checkpoints"][0], episode_seed=101)
    _seal_worker(directory)
    checksums = json.loads((directory / "checksums.json").read_text())
    checksums.pop("matrix.json")
    atomic_json(directory / "checksums.json", checksums, overwrite=True)
    with pytest.raises(ValueError, match="missing required"):
        validate_worker(directory, study=study, source=study["checkpoints"][0], episode_seed=101)


def test_worker_seal_cannot_reference_outside_directory(sealed_worker, study):
    directory, _, _ = sealed_worker
    checksums = json.loads((directory / "checksums.json").read_text())
    checksums["../outside.json"] = "not-read"
    atomic_json(directory / "checksums.json", checksums, overwrite=True)
    with pytest.raises(ValueError, match="checksum mismatch: ../outside.json"):
        validate_worker(directory, study=study, source=study["checkpoints"][0], episode_seed=101)


def test_critic_probe_uses_supplied_noise_pair_and_restores_mixed_modes():
    critic = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
    critic.train()
    critic[0].eval()
    modes = [m.training for m in critic.modules()]
    noise = np.array([[.2, -.3], [.4, -.5]], dtype=np.float32)
    indices = torch.tensor([1, 3])
    calls = []

    def pi(z, **kwargs):
        assert kwargs["policy"] == "actor"
        assert kwargs["log_std_min"] == -20
        torch.testing.assert_close(kwargs["noise"], torch.as_tensor(noise))
        return z + kwargs["noise"], None

    def q(z, action, **kwargs):
        assert not any(m.training for m in critic.modules())
        assert kwargs["reduction"] == "min_pair"
        assert kwargs["pair_indices"] is indices and kwargs["trusted_pair_indices"]
        calls.append(action.clone())
        return action.sum(-1, keepdim=True)

    model = SimpleNamespace(agent=SimpleNamespace(model=SimpleNamespace(pi=pi, Q=q)),
                            cfg=SimpleNamespace(inner_q_actor_reduction="min_pair"))
    result = critic_probe(model, critic, torch.ones(1, 2), "actor", {"log_std_min": -20}, noise, indices)
    np.testing.assert_allclose(result, [1.9, 1.9])
    assert len(calls) == 1 and [m.training for m in critic.modules()] == modes
    model.agent.model.Q = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("probe failed"))
    with pytest.raises(RuntimeError, match="probe failed"):
        critic_probe(model, critic, torch.ones(1, 2), "actor", {"log_std_min": -20}, noise, indices)
    assert [m.training for m in critic.modules()] == modes


def test_actor_snapshot_digest_depends_on_values_and_dtype():
    actor = torch.nn.Linear(2, 2)
    before = state_digest(actor)
    assert state_digest(deepcopy(actor)) == before
    assert state_digest(deepcopy(actor).double()) != before
    with torch.no_grad():
        actor.bias[0] += 1
    assert state_digest(actor) != before
