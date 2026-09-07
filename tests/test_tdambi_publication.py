"""TDAMBI curve identity and immutable native-prior reference ingestion."""

import copy
import hashlib
import json
import pytest

from utils import ambi_benchmark as storage
from utils import eval_series_data as data
from utils import eval_series as series


SOURCE = "rwgao_b-brown-university/ambi/xq3zva9u"
CHECKPOINT_SHA = "a" * 64


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _native_seed(seed):
    payload = f"tdmpc2-mppi-eval:12345:policy_prior_mean:{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**32


def _config():
    return {
        "inner_operator": "tdambi", "inner_rounds": 6, "inner_rollouts_per_round": 512,
        "inner_rollout_horizon": 3, "inner_updates_per_round": 3, "inner_batch_size": 512,
        "inner_replay_capacity": 12288, "inner_replay_sampling": "with_replacement",
        "inner_actor_lr": 3e-4, "inner_critic_lr": 3e-4, "inner_actor_adam_eps": 1e-5,
        "inner_adam_eps": 1e-8, "inner_actor_grad_clip_norm": 20.0,
        "inner_critic_grad_clip_norm": 20.0, "inner_critic_target_tau": .01,
        "inner_critic_target_update_interval": 1, "tdambi_entropy_coef": 1e-4,
        "tdambi_value_coef": .1, "tdambi_scale_tau": .01, "discount": .99,
        "episodic": False, "dropout": .01, "log_std_mapping": "tdmpc2_tanh",
        "log_std_min": -10.0, "log_std_max": 2.0, "inner_log_std_mapping": "tdmpc2_tanh",
        "inner_log_std_min": -10.0, "inner_log_std_max": 2.0,
        "inner_q_actor_reduction": "mean_pair", "inner_q_target_reduction": "min_pair",
        "q_pair_size": 2,
    }


def _resolved():
    return {"selector": "inner_budget/tdambi_3", "algorithm_config": {
        "alg": "TDAMBI/TDAMBI", "env": "DMControl-v0", "alg_params": {"obs": "state", **_config()}},
        "environment": {"id": "DMControl-v0", "params": {"task": "humanoid-walk", "obs": "state"}}}


@pytest.fixture
def native_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "scientific_identity", lambda algorithm, *args, **kwargs: {
        "algorithm": algorithm, "source_sha256": "science-v1"})
    monkeypatch.setattr(storage, "code_identity", lambda: {"commit": "b" * 40, "dirty": False})
    settings = {"iterations": 6, "num_samples": 512, "num_elites": 64, "num_pi_trajs": 24,
                "outer_planning_horizon": 3, "min_std": .05, "max_std": 2.0,
                "temperature": .5, "episodic": False, "obs": "state"}
    metadata = {"checkpoint": {"step": 300000}, "trial_run_params": {
        "alg": "TDMPC2/TDMPC2Baseline", "env": "DMControl-v0", "alg_params": settings,
        "resolved_runtime": {"observation": {"action_dim": 21, "mode": "state"}}},
        "experiment_params": {"env_params": _resolved()["environment"]["params"]}}
    meta_path = _dump(tmp_path / "native" / "checkpoint.metadata.json", metadata)
    _dump(meta_path.parent / "provenance.json", {"source_run": SOURCE,
          "checkpoint_sha256": CHECKPOINT_SHA, "code_sha": "c" * 40,
          "metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest()})
    episodes = []
    for seed in (101, 102):
        pair = {"environment_seed": seed}
        for controller in ("policy_prior_mean", "native_mppi"):
            value = float(seed - 100 + (controller == "native_mppi"))
            pair[controller] = {"controller": controller, "controller_seed": _native_seed(seed),
                                "return": value, "length": 2, "terminated": False,
                                "truncated": False, "capped": True, "seconds": .1,
                                "steps": [{"reward": value / 2}, {"reward": value / 2}]}
        episodes.append(pair)
    payload = {"schema_version": 1, "algorithm": "TDMPC2/TDMPC2Baseline",
               "checkpoint_metadata": {"step": 300000}, "checkpoint_sha256": CHECKPOINT_SHA,
               "environment": "DMControl-v0", "resolved_runtime": {
                   "observation": {"action_dim": 21}},
               "frozen_state": {"unchanged": True, "model_digest_before": "frozen",
                                "model_digest_after": "frozen", "num_updates_before": 20,
                                "num_updates_after": 20},
               "planner": {"configured_iterations": 6, "effective_iterations": 8,
                           "num_samples": 512, "num_elites": 64, "num_pi_trajs": 24,
                           "planning_horizon": 3, "model_transitions_per_action": 12336},
               "protocol": {"environment_seed_first": 101, "environment_seed_last": 102,
                            "controller_seed_base": 12345, "max_steps": 2,
                            "planner_rng": "independent fixed namespaced stream per environment seed"},
               "episodes": episodes}
    return _dump(meta_path.parent / "paired.json", payload), metadata, payload


def _identity(config=None):
    return data.planner_identity(config or _config(), {}, "TDAMBI/TDAMBI", "tanh_mean")


@pytest.mark.parametrize("key,value", [
    ("inner_updates_per_round", 6), ("inner_rounds", 8), ("inner_actor_lr", 1e-4),
    ("inner_critic_lr", 1e-4), ("tdambi_entropy_coef", .01), ("tdambi_value_coef", 1.0),
    ("tdambi_scale_tau", .1), ("inner_adam_eps", 1e-5), ("inner_actor_adam_eps", 1e-8),
    ("inner_critic_target_tau", .1), ("inner_actor_grad_clip_norm", 10),
    ("inner_replay_sampling", "without_replacement"), ("inner_log_std_min", -20),
])
def test_active_native_settings_identify_tdambi_curves(key, value):
    config = _config()
    baseline = _identity(config)
    config[key] = value
    assert _identity(config) != baseline


def test_scales_sac_temperature_and_unused_controls_do_not_split_tdambi():
    config = _config()
    expected = _identity(config)
    config.update(tdambi_scale=4000, tdambi_scale_before=12, inner_temperature=1e-8,
                  inner_temperature_mode="auto", inner_temperature_lr=123,
                  inner_mppi_iterations=999, wandb_run_name="cosmetic", inner_actor_lora_rank=64)
    assert _identity(config) == expected
    label = data.descriptive_label({"planner": expected, "backbone": SOURCE,
                                    "science": {"algorithm": "TDAMBI/TDAMBI"}})
    assert label == "TD-MPC2 · TDAMBI G3 J6 N512 H3 B512"
    assert "scale_initialization" in expected["semantics"]


def test_prior_planner_still_ignores_native_inner_settings():
    config = {**_config(), "inner_operator": "none"}
    assert _identity(config) == {"type": "prior", "action_rule": "tanh_mean"}


def test_tdambi_new_attempts_and_append_require_matching_behavior(tmp_path):
    identity = {"backbone": SOURCE, "planner": _identity(),
                "protocol": storage.protocol_for(_resolved(), 12345, 500),
                "science": {"algorithm": "TDAMBI/TDAMBI", "source_sha256": "pinned-science"}}
    template = {"identity": identity, "label": data.descriptive_label(identity)}
    first = series.create_run(tmp_path / "registry", template, "first", "eval", "entity", "oscar-owner")
    second = series.create_run(tmp_path / "registry", template, "repeat", "eval", "entity", "oscar-owner")
    assert first["run_id"] != second["run_id"]
    series.validate_identity(first, identity)
    changed = copy.deepcopy(identity)
    changed["planner"]["settings"]["tdambi_entropy_coef"] *= 2
    with pytest.raises(series.SeriesError, match="Incompatible append"):
        series.validate_identity(first, changed)
    changed = copy.deepcopy(identity)
    changed["protocol"]["controller_seed"] = 55
    with pytest.raises(series.SeriesError, match="Incompatible append"):
        series.validate_identity(first, changed)


def test_old_round_default_preserves_sac_identity_and_step_is_distinct():
    config = {"inner_operator": "sac"}
    legacy = data.planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    config["inner_update_timing"] = "round"
    assert data.planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean") == legacy
    config["inner_update_timing"] = "step"
    assert data.planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean") != legacy


def test_resolved_tdambi_preflight_matches_executed_configuration(native_reference, monkeypatch):
    import gymnasium as gym
    from RL.TDAMBI import TDAMBI

    env = gym.make("Pendulum-v1", max_episode_steps=2)
    params = {"device": "cpu", "model_size": None, "enc_dim": 16, "mlp_dim": 16,
              "latent_dim": 8, "num_enc_layers": 2, "simnorm_dim": 4, "num_q": 2,
              "num_bins": 11, "vmin": -5., "vmax": 5., "batch_size": 4,
              "train_unroll_horizon": 1, "outer_planning_horizon": 1,
              "buffer_size": 32, "seed_steps": 4, "pretrain_steps": 1,
              "wandb": False, "inner_rounds": 1, "inner_rollouts_per_round": 4,
              "inner_rollout_horizon": 1, "inner_updates_per_round": 1,
              "inner_batch_size": 4, "inner_replay_capacity": 8, "inner_diagnostics_every": 1}
    config = {"alg": "TDAMBI/TDAMBI", "env": "DMControl-v0", "seed": 12345,
              "device": "cpu", "total_steps": 10, "alg_params": params}
    resolved = {"selector": "inner_budget/tdambi_1", "algorithm_config": config,
                "environment": _resolved()["environment"]}
    try:
        model = TDAMBI("TDAMBI", env, params, config, {"frozen_checkpoint_evaluation": True})
        observation, _ = env.reset(seed=101)
        model.predict(observation)
        expected = data.planner_identity(vars(model.cfg), {}, "TDAMBI/TDAMBI", "tanh_mean")
        monkeypatch.setattr(TDAMBI, "__init__", lambda *args, **kwargs: pytest.fail("Preflight constructed a learner"))
        actual = data.identity_for_ambi_checkpoint(
            {"sha256": CHECKPOINT_SHA, "source_run": SOURCE, "source_run_verified": True,
             "metadata": native_reference[1]}, resolved, storage.protocol_for(resolved, 12345, 2),
            [101, 102], {"commit": "b" * 40}, path=native_reference[0], env=env)
        assert actual["planner"] == expected
        assert expected["settings"]["tdambi_entropy_coef"] == params.get("entropy_coef", 1e-4)
        assert "inner_temperature" not in expected["settings"]
    finally:
        env.close()


def _reference(path, protocol=None, **kwargs):
    return storage.reference_returns(path, CHECKPOINT_SHA,
                                     protocol or storage.protocol_for(_resolved(), 12345, 2),
                                     source_run=SOURCE, seeds=[101, 102], **kwargs)


def test_native_reference_keeps_distinct_rng_protocols_and_exact_files(native_reference, tmp_path):
    path, metadata, _ = native_reference
    reference = _reference(path)
    assert reference == {101: 1., 102: 2.}
    assert reference.provenance["source_protocol"]["seed_scheme"] != reference.provenance[
        "evaluation_protocol"]["seed_scheme"]
    bundle = storage.BenchmarkBundle(tmp_path / "bundle", checkpoint={"sha256": CHECKPOINT_SHA,
                                    "source_run": SOURCE, "source_run_verified": True, "metadata": metadata},
                                    protocol=storage.protocol_for(_resolved(), 12345, 2), reference=reference)
    assert (bundle.path / "reference/paired.json").read_bytes() == path.read_bytes()
    run = bundle.start_run(_resolved(), "episodes")
    for seed in (101, 102):
        events = [
            {"episode_id": f"seed-{seed}", "decision_index": 0, "event_index": 0,
             "phase": "critic_update", "round_index": 0, "critic_updates": 1,
             "actor_updates": 0, "temperature_updates": 0,
             "metrics": {"critic_loss": .5}},
            {"episode_id": f"seed-{seed}", "decision_index": 0, "event_index": 1,
             "phase": "actor_update", "round_index": 0, "critic_updates": 1,
             "actor_updates": 1, "temperature_updates": 0,
             "metrics": {"actor_q_scaled_mean": 2., "actor_native_scaled_entropy": 10.,
                         "actor_q_scale_before": 3., "actor_q_scale_after": 4.}},
        ]
        bundle.episode(run, {"seed": seed, "return": float(seed - 98), "length": 2,
                            "terminated": False, "truncated": False, "truncated_by_evaluator": True,
                            "control_seconds": .5, "model_metrics": {
                                "inner_actor_optimizer_steps": 18, "inner_critic_optimizer_steps": 18,
                                "inner_temperature_optimizer_steps": 0}}, events)
    bundle.finish_run(run, result={"resolved_config": _config(), "outer_state_unchanged": True,
                                  "outer_updates_before": 20, "outer_updates_after": 20,
                                  "environment_seeds": [101, 102]})
    bundle.finish()
    record, = data.normalize_bundle(bundle.path)
    assert record["identity"]["backbone"] == SOURCE
    assert record["metrics"]["eval/paired_gain_mean"] == 2
    assert record["metrics"]["eval/paired_gain_sample_std"] == 0
    assert record["metrics"]["work/temperature_updates"] == 0
    assert "reference/paired.json" in record["artifact_files"]
    from report_ambi_benchmark import load_bundles, render_html
    report = load_bundles([bundle.path])
    assert report["metric_catalog"]["actor_native_scaled_entropy"]["preferred_axis"] == "actor_updates"
    assert report["metric_catalog"]["critic_loss"]["preferred_axis"] == "critic_updates"
    assert "alpha" not in report["metric_catalog"]
    assert not any("soft_score" in name for name in report["metric_catalog"])
    assert "actor_q_scale_after" in render_html(report, title="TDAMBI native inner updates")
    saved_id = record["record_id"]
    assert data.normalize_bundle(bundle.path)[0]["record_id"] == saved_id
    manifest_path = bundle.path / "manifest.json"
    manifest = storage.read_json(manifest_path)
    manifest["runs"][0]["episodes"][0]["paired_return_delta"] += 1
    storage.atomic_json(manifest_path, manifest, overwrite=True)
    with pytest.raises(ValueError, match="Paired return delta"):
        data.normalize_bundle(bundle.path)


@pytest.mark.parametrize("key,value", [("max_steps", 500), ("controller_seed", 55),
                                      ("action_rule", "sample"), ("observation", "rgb"),
                                      ("prior_reference", "unknown")])
def test_native_reference_rejects_incompatible_evaluation_protocol(native_reference, key, value):
    protocol = storage.protocol_for(_resolved(), 12345, 2)
    protocol[key] = value
    with pytest.raises(ValueError, match="protocol"):
        _reference(native_reference[0], protocol)


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.update(checkpoint_sha256="f" * 64), "hash mismatch|checkpoint"),
    (lambda value: value["episodes"][0]["policy_prior_mean"].update(controller_seed=0), "RNG seed"),
    (lambda value: value["episodes"][0]["policy_prior_mean"].update(return_value=0, **{"return": 50}), "Return differs"),
    (lambda value: value["frozen_state"].update(unchanged=False), "Frozen"),
])
def test_invalid_native_prior_science_is_rejected(native_reference, mutation, match):
    path, _, payload = native_reference
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        _reference(_dump(path, payload))


def test_reference_changed_after_validation_is_not_copied(native_reference, tmp_path):
    path, metadata, payload = native_reference
    reference = _reference(path)
    payload["ignored_extra_field"] = "changed"
    _dump(path, payload)
    with pytest.raises(ValueError, match="changed after validation"):
        storage.BenchmarkBundle(tmp_path / "bundle", checkpoint={"sha256": CHECKPOINT_SHA,
                                "metadata": metadata}, protocol=storage.protocol_for(_resolved(), 12345, 2),
                                reference=reference)


def test_inventory_verified_reference_remains_portable_without_original_files(native_reference, tmp_path):
    path, metadata, _ = native_reference
    provenance_path = path.parent / "provenance.json"
    provenance = storage.read_json(provenance_path)
    provenance.pop("checkpoint_sha256")
    _dump(provenance_path, provenance)
    inventory = _dump(tmp_path / "inventory.json", {"source_run": SOURCE, "checkpoints": [
        {"step": 300000, "sha256": CHECKPOINT_SHA, "metadata_sha256": provenance["metadata_sha256"]}]})
    reference = _reference(path, checkpoint_inventory=inventory)
    checkpoint = {"sha256": CHECKPOINT_SHA, "source_run": SOURCE,
                  "source_run_verified": True, "metadata": metadata}
    bundle = storage.BenchmarkBundle(tmp_path / "bundle", checkpoint=checkpoint,
                                    protocol=storage.protocol_for(_resolved(), 12345, 2), reference=reference)
    path.parent.rename(tmp_path / "moved-original-reference")
    inventory.unlink()
    values, files = data._bundle_prior_reference(bundle.manifest, bundle.path / "manifest.json",
                                                {"sha256": CHECKPOINT_SHA, "step": 300000}, SOURCE)
    assert values == reference
    assert "reference/provenance/checkpoint-inventory-0.json" in files
