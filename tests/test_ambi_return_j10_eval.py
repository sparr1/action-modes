"""J10 changes solve depth while preserving cadence, source state, and controls."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_j10_eval as campaign
from slurm import ambi_return_cadence_eval as cadence
from tests.test_ambi_return_prior_eval import inventory, result_payload
from tests.test_ambi_return_cadence_eval import reference_root, swap_result


def j10_result(entry, index, *, smoke=False):
    payload = swap_result(entry, 1 - index, smoke=smoke)
    result = payload["results"][0]
    result["selector"] = "rounds/j10"
    for key in ("alg_params", "resolved_config"):
        result[key].update(inner_rounds=10, inner_replay_capacity=3840)
    for key in ("inner_model_steps", "inner_critic_optimizer_steps",
                "inner_actor_optimizer_steps", "inner_temperature_optimizer_steps"):
        result["model_metrics"][key]["sum"] *= 10 / 8
        for episode in result["episodes"]:
            episode["model_metrics"][key] *= 10 / 8
    return payload


def third_result(entry, *, smoke=False):
    payload = result_payload(entry, smoke=smoke)
    sac = j10_result(entry, 1, smoke=smoke)["results"][0]
    sac["selector"] = "comparison/sac"
    payload["results"][1] = sac
    return payload


@pytest.mark.parametrize("index", [0, 1])
def test_j10_preserves_source_cadence_and_only_resizes_required_replay(inventory, index):
    from tests.test_ambi_research_presets import _build_cfg

    entry = cadence.verify_entry(inventory, index)
    context = cadence.load_checkpoint_context(entry["checkpoint"], entry["metadata"])
    resolved = cadence.resolve_preset(campaign.MATRIX, campaign.selection(index)[0],
                                      cadence.load_preset_matrix(campaign.MATRIX), checkpoint_context=context)
    expected = deepcopy(context.trial_run_params["alg_params"])
    expected.update(inner_rounds=10, inner_replay_capacity=3840)
    assert resolved["algorithm_config"]["alg_params"] == expected
    cfg = _build_cfg(resolved)
    assert cfg.inner_rounds == 10 and cfg.inner_solve_interval == (1, 3)[index]
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 3840
    assert cfg.inner_critic_updates_per_action == 160
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 40
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_temperature_initialization == "inherit_outer"
    assert campaign.validate_results(j10_result(entry, index), entry, index) == {
        "episodes": 5, "decisions": 2500, "solves": (2500, 835)[index]}
    assert campaign.validate_results(j10_result(entry, index, smoke=True), entry, index, smoke=True)["solves"] == (6, 2)[index]
    resolved["algorithm_config"]["alg_params"]["inner_replay_capacity"] = 3072
    with pytest.raises(ValueError, match="at least the cumulative"):
        _build_cfg(resolved)


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r["alg_params"].update(inner_rounds=8), "evaluation overrides"),
    (lambda r: r["alg_params"].update(inner_solve_interval=1), "evaluation overrides"),
    (lambda r: r["resolved_config"].update(inner_rounds=8), "Resolved controller"),
    (lambda r: r["episodes"][0]["model_metrics"].update(inner_model_steps=1026.048), "cadence/work"),
    (lambda r: r["model_metrics"]["inner_actor_optimizer_steps"].update(sum=26720), "cadence/work"),
    (lambda r: r["resolved_config"].update(inner_temperature_initialization="fixed"), "alpha"),
])
def test_j10_rejects_old_dose_or_changed_settings(inventory, mutation, message):
    entry = inventory["entries"][1]
    payload = j10_result(entry, 1)
    mutation(payload["results"][0])
    with pytest.raises(ValueError, match=message):
        campaign.validate_results(payload, entry, 1)


def test_third_pair_validates_prior_zero_work_and_j10_sac_dose(inventory):
    entry = inventory["entries"][2]
    payload = third_result(entry)
    assert campaign.validate_third_results(payload, entry) == {"episodes": 10, "decisions": 5000, "solves": 835}
    payload["results"][1]["model_metrics"]["inner_critic_optimizer_steps"]["sum"] = 106880
    with pytest.raises(ValueError, match="cadence/work"):
        campaign.validate_third_results(payload, entry)
    payload = third_result(entry)
    payload["results"][0]["model_metrics"]["inner_model_steps"]["max"] = 1
    with pytest.raises(ValueError, match="work counter"):
        campaign.validate_third_results(payload, entry)


def test_j10_accepts_original_k1_checkpoint_with_implicit_interval(inventory):
    entry = inventory["entries"][0]
    metadata = cadence.read_json(entry["metadata"])
    metadata["trial_run_params"]["alg_params"].pop("inner_solve_interval", None)
    Path(entry["metadata"]).write_text(json.dumps(metadata))
    entry["metadata_sha256"] = cadence.file_hash(entry["metadata"])
    cadence.verify_entry(inventory, 0)
    context = cadence.load_checkpoint_context(entry["checkpoint"], entry["metadata"])
    resolved = cadence.resolve_preset(campaign.MATRIX, "rounds/j10", cadence.load_preset_matrix(campaign.MATRIX),
                                      checkpoint_context=context)
    assert "inner_solve_interval" not in resolved["algorithm_config"]["alg_params"]
    payload = j10_result(entry, 0)
    payload["results"][0]["alg_params"] = resolved["algorithm_config"]["alg_params"]
    assert campaign.validate_results(payload, entry, 0)["solves"] == 2500


@pytest.mark.parametrize("index,mode", [(0, "prepare"), (0, "smoke"), (0, "run"),
                                       (1, "run"), (2, "prepare"), (2, "smoke"), (2, "run")])
def test_command_selection_reference_reuse_and_receipt_identity(inventory, tmp_path, monkeypatch, index, mode):
    inventory["campaign_id"] = campaign.CAMPAIGN_ID
    entry = inventory["entries"][index]
    source = tmp_path / "inventory.json"
    cadence.write_json(source, inventory)
    reference = reference_root(tmp_path, entry, index) if index < 2 and mode != "smoke" else tmp_path / "absent"
    output = tmp_path / mode
    selectors = ["rounds/j10"] if index < 2 else ["comparison/prior", "comparison/sac"]
    map_path = None
    if mode == "run":
        mapping = {}
        for selector in selectors:
            registry = tmp_path / selector.replace("/", "__")
            registry.mkdir()
            cadence.write_json(registry / "run.json", {"identity": {"selector": selector}})
            mapping[selector] = str(registry)
        map_path = tmp_path / "map.json"
        cadence.write_json(map_path, mapping)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--eval-series-spec-dir" in command:
            specs = Path(command[command.index("--eval-series-spec-dir") + 1])
            specs.mkdir()
            for selector in selectors:
                cadence.write_json(specs / (selector.replace("/", "__") + ".json"), {"identity": {"selector": selector}})
        else:
            result = (j10_result(entry, index, smoke=mode == "smoke") if index < 2 else
                      third_result(entry, smoke=mode == "smoke"))
            cadence.write_json(output / "results.json", result)
            (output / "bundle").mkdir()
            manifest = {"status": "complete", "runs": [{"selector": selector} for selector in selectors]}
            if mode == "run" and index < 2:
                manifest["reference"] = {"manifest_sha256": cadence.verify_reference(reference, index, entry)["manifest_sha256"]}
            cadence.write_json(output / "bundle/manifest.json", manifest)

    monkeypatch.setattr(cadence.subprocess, "run", fake_run)
    monkeypatch.setattr(cadence, "runtime_info", lambda **kwargs: {"hostname": "test"})
    monkeypatch.setattr(cadence.original, "runtime_info", lambda **kwargs: {"hostname": "test"})
    receipt = campaign.run(SimpleNamespace(mode=mode, inventory=source, index=index, output=output,
                                           reference_root=reference, expected_commit=None,
                                           eval_run_map=map_path, smoke_steps=6))
    assert receipt["campaign_id"] == campaign.CAMPAIGN_ID
    assert (receipt["J"], receipt["source_J"], receipt["eval_J"]) == (10, (8 if index < 2 else 10), 10)
    assert (receipt["source_K"], receipt["eval_K"]) == (entry["K"], entry["K"])
    assert (receipt["source_H"], receipt["eval_H"]) == (3, 3)
    for command in commands:
        assert "--eval-run-map" not in command and "--wandb" not in command
        if index < 2:
            assert command[command.index("--preset") + 1] == "rounds/j10"
        else:
            assert "--preset" not in command
            assert command[command.index("--matrix") + 1] == str(cadence.original.MATRIX)
    assert ("--reference-bundle" in commands[-1]) == (index < 2 and mode == "run")
    if mode == "smoke":
        assert commands[-1][-4:] == ["--seeds", "101", "--max-steps", "6"]
        assert receipt["decisions"] == (6 if index < 2 else 12)
    elif mode == "run":
        assert receipt["decisions"] == (2500 if index < 2 else 5000)
        assert receipt["solves"] == (2500 if index == 0 else 835)


def test_third_rejects_old_325k_before_compute(inventory, tmp_path, monkeypatch):
    inventory["campaign_id"] = campaign.CAMPAIGN_ID
    entry = inventory["entries"][2]
    metadata = cadence.read_json(entry["metadata"])
    metadata["checkpoint"]["step"] = 325000
    Path(entry["metadata"]).write_text(json.dumps(metadata))
    entry.update(step=325000, latest_at_selection=325000, metadata_sha256=cadence.file_hash(entry["metadata"]))
    source = tmp_path / "inventory.json"
    cadence.write_json(source, inventory)
    monkeypatch.setattr(cadence.original, "run", lambda _: pytest.fail("Must reject before evaluation"))
    with pytest.raises(ValueError, match="500k checkpoint"):
        campaign.run(SimpleNamespace(inventory=source, index=2))


def test_third_invalid_work_never_gets_validated_receipt(inventory, tmp_path, monkeypatch):
    inventory["campaign_id"] = campaign.CAMPAIGN_ID
    source, output = tmp_path / "inventory.json", tmp_path / "smoke"
    cadence.write_json(source, inventory)

    def fake_run(*args, **kwargs):
        payload = third_result(inventory["entries"][2], smoke=True)
        payload["results"][1]["model_metrics"]["inner_critic_optimizer_steps"]["sum"] = 256
        cadence.write_json(output / "results.json", payload)
        (output / "bundle").mkdir()
        cadence.write_json(output / "bundle/manifest.json", {"status": "complete"})

    monkeypatch.setattr(cadence.subprocess, "run", fake_run)
    monkeypatch.setattr(cadence.original, "runtime_info", lambda **kwargs: {})
    with pytest.raises(ValueError, match="cadence/work"):
        campaign.run(SimpleNamespace(mode="smoke", inventory=source, index=2, output=output,
                                     reference_root=None, expected_commit=None, eval_run_map=None, smoke_steps=6))
    assert not (output / "receipt.json").exists()
