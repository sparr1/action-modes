"""The selected evaluation preserves saved controllers and rejects wrong evidence."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_prior_eval as campaign
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inventory(tmp_path):
    entries = []
    recipe = json.loads((ROOT / "configs/dmcontrol/algs/ambi_full_h3_j10_k3_reward_only.json").read_text())
    for index, (run_id, rounds, interval) in enumerate(campaign.SOURCES):
        weights = tmp_path / f"checkpoint-{index}"
        weights.write_bytes(f"weights {index}".encode())
        run = deepcopy(recipe)
        run["alg_params"].update(inner_rounds=rounds, inner_solve_interval=interval,
                                 inner_replay_capacity=rounds * 128 * 3, wandb_run_name=f"Source {index}")
        metadata = Path(str(weights) + ".metadata.json")
        metadata.write_text(json.dumps({
            "schema_version": 1, "checkpoint": {"step": 500000, "kind": "periodic", "episode": 1000,
                                                  "best_score": None, "best_window": 100},
            "trial_run_params": run,
            "experiment_params": {"env_params": {"task": "humanoid-walk", "obs": "state"}},
        }))
        entries.append({"source_run_id": run_id, "source_run_name": f"Source {index}",
                        "source_run": f"rwgao_b-brown-university/ambi/{run_id}",
                        "checkpoint": str(weights), "metadata": str(metadata), "step": 500000,
                        "sha256": campaign.file_hash(weights), "metadata_sha256": campaign.file_hash(metadata),
                        "J": rounds, "K": interval, "latest_at_selection": 500000})
    return {"schema_version": 1, "campaign_id": "test", "entries": entries}


@pytest.mark.parametrize("index", [0, 1, 2])
def test_matrix_preserves_source_solve_and_disables_prior_work(inventory, index):
    from tests.test_ambi_research_presets import _build_cfg

    entry = campaign.verify_entry(inventory, index)
    context = load_checkpoint_context(entry["checkpoint"], entry["metadata"])
    matrix = load_preset_matrix(campaign.MATRIX)
    sac = resolve_preset(campaign.MATRIX, "comparison/sac", matrix, checkpoint_context=context)
    assert sac["algorithm_config"]["alg_params"] == context.trial_run_params["alg_params"]
    cfg = _build_cfg(sac)
    assert (cfg.inner_rounds, cfg.inner_solve_interval) == (entry["J"], entry["K"])
    assert cfg.inner_temperature_mode == "auto"
    assert cfg.inner_temperature_initialization == "inherit_outer"
    prior = resolve_preset(campaign.MATRIX, "comparison/prior", matrix, checkpoint_context=context)
    prior_cfg = _build_cfg(prior)
    assert (prior_cfg.inner_operator, prior_cfg.inner_rounds, prior_cfg.inner_solve_interval) == ("none", 0, 1)
    assert prior_cfg.inner_critic_updates_per_action == prior_cfg.inner_actor_updates_per_action == 0


def test_inventory_rejects_hash_source_and_selection_changes(inventory):
    changed = deepcopy(inventory)
    changed["entries"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        campaign.verify_entry(changed, 0)
    changed = deepcopy(inventory)
    changed["entries"].reverse()
    with pytest.raises(ValueError, match="order"):
        campaign.verify_entry(changed, 0)
    changed = deepcopy(inventory)
    changed["entries"][0]["latest_at_selection"] = 475000
    with pytest.raises(ValueError, match="selection rule"):
        campaign.verify_entry(changed, 0)


def test_implicit_k1_default_is_accepted(inventory):
    entry = inventory["entries"][0]
    metadata = campaign.read_json(entry["metadata"])
    metadata["trial_run_params"]["alg_params"].pop("inner_solve_interval")
    Path(entry["metadata"]).write_text(json.dumps(metadata))
    entry["metadata_sha256"] = campaign.file_hash(entry["metadata"])
    assert campaign.verify_entry(inventory, 0)["K"] == 1


def result_payload(entry, *, smoke=False):
    seeds, steps = ([101], 6) if smoke else (campaign.SEEDS, 500)
    results = []
    for selector in campaign.SELECTORS:
        prior = selector.endswith("prior")
        results.append({"selector": selector, "outer_state_unchanged": True,
                        "outer_updates_before": 20, "outer_updates_after": 20,
                        "controller_seed": 55, "environment_seeds": seeds,
                        "episodes": [{"seed": seed, "length": steps, "return": 1.,
                                      "terminated": False, "truncated": True,
                                      "truncated_by_evaluator": smoke} for seed in seeds],
                        "model_metrics": {
                            key: {"count": len(seeds) * steps, "sum": 0, "min": 0, "max": 0}
                            for key in ("inner_model_steps", "inner_actor_optimizer_steps",
                                        "inner_critic_optimizer_steps", "inner_temperature_optimizer_steps")
                        },
                        "resolved_config": {"inner_operator": "none" if prior else "sac",
                                            "inner_rounds": 0 if prior else entry["J"],
                                            "inner_solve_interval": 1 if prior else entry["K"],
                                            "inner_rollout_horizon": 3, "inner_temperature_mode": "auto",
                                            "inner_temperature_initialization": "inherit_outer",
                                            "inner_hold_encoder": "solve_snapshot"}})
    return {"frozen_outer_learning": True, "deterministic_execution": True,
            "checkpoint_sha256": entry["sha256"], "results": results}


def test_validation_rejects_partial_coverage_mutation_and_cadence(inventory):
    entry = inventory["entries"][1]
    payload = result_payload(entry)
    assert campaign.validate_results(payload, entry) == {"episodes": 10, "decisions": 5000}
    for mutation, message in [
        (lambda p: p["results"][1]["episodes"].pop(), "coverage"),
        (lambda p: p["results"][1].update(outer_updates_after=21), "Outer state"),
        (lambda p: p["results"][1]["resolved_config"].update(inner_solve_interval=1), "controller changed"),
        (lambda p: p["results"][1].update(togo_return_probe={"rollouts": 32}), "to-go"),
    ]:
        changed = deepcopy(payload)
        mutation(changed)
        with pytest.raises(ValueError, match=message):
            campaign.validate_results(changed, entry)


@pytest.mark.parametrize("field,value", [("truncated_by_evaluator", True),
                                         ("terminated", True), ("truncated", False)])
def test_production_requires_native_episode_end(inventory, field, value):
    entry = inventory["entries"][0]
    payload = result_payload(entry)
    payload["results"][0]["episodes"][0][field] = value
    with pytest.raises(ValueError, match="native environment time limit"):
        campaign.validate_results(payload, entry)
    assert campaign.validate_results(result_payload(entry, smoke=True), entry, smoke=True)["decisions"] == 12


@pytest.mark.parametrize("key", ["inner_model_steps", "inner_actor_optimizer_steps",
                                 "inner_critic_optimizer_steps", "inner_temperature_optimizer_steps"])
def test_prior_requires_observed_zero_work(inventory, key):
    entry = inventory["entries"][0]
    payload = result_payload(entry)
    payload["results"][0]["model_metrics"][key]["max"] = 1
    with pytest.raises(ValueError, match="work counter"):
        campaign.validate_results(payload, entry)
    payload["results"][0]["model_metrics"].pop(key)
    with pytest.raises(ValueError, match="work counter"):
        campaign.validate_results(payload, entry)


def test_smoke_command_is_bounded_unpublished_and_no_clobber(inventory, tmp_path, monkeypatch):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    output = tmp_path / "smoke"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (output / "results.json").write_text(json.dumps(result_payload(inventory["entries"][0], smoke=True)))
        (output / "bundle").mkdir()
        (output / "bundle/manifest.json").write_text(json.dumps({"status": "complete"}))

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    monkeypatch.setattr(campaign, "runtime_info", lambda **kwargs: {"hostname": "test", "gpu_model": "test"})
    args = SimpleNamespace(mode="smoke", inventory=path, index=0, output=output,
                           expected_commit=None, eval_run_map=None, smoke_steps=6)
    receipt = campaign.run(args)
    assert receipt["episodes"] == 2 and receipt["decisions"] == 12
    assert commands[0][-4:] == ["--seeds", "101", "--max-steps", "6"]
    assert "--wandb" not in commands[0] and "--eval-run-map" not in commands[0]
    assert receipt["manifest_sha256"] == campaign.file_hash(output / "bundle/manifest.json")
    with pytest.raises(FileExistsError):
        campaign.run(args)


def test_changed_publication_identity_fails_before_compute(inventory, tmp_path, monkeypatch):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    mapping = {}
    for selector in campaign.SELECTORS:
        registry = tmp_path / selector.replace("/", "__")
        registry.mkdir()
        (registry / "run.json").write_text(json.dumps({"identity": {"expected": True}}))
        mapping[selector] = str(registry)
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(mapping))
    output = tmp_path / "production"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert "--eval-series-spec-dir" in command
        specs = Path(command[command.index("--eval-series-spec-dir") + 1])
        specs.mkdir()
        for selector in campaign.SELECTORS:
            (specs / (selector.replace("/", "__") + ".json")).write_text(
                json.dumps({"identity": {"expected": False}}))

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    args = SimpleNamespace(mode="run", inventory=path, index=0, output=output,
                           expected_commit=None, eval_run_map=map_path, smoke_steps=6)
    with pytest.raises(ValueError, match="identity changed"):
        campaign.run(args)
    assert len(commands) == 1
    assert not (output / "receipt.json").exists()
