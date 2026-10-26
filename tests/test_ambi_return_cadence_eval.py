"""Cadence swaps change one saved setting and reuse verified prior episodes."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_cadence_eval as campaign
from tests.test_ambi_return_prior_eval import inventory, result_payload


def swap_result(entry, index, *, smoke=False):
    selector, interval = campaign.selection(index)
    payload = result_payload(entry, smoke=smoke)
    result = payload["results"][1]
    payload["results"] = [result]
    result["selector"] = selector
    result["alg_params"] = campaign.read_json(entry["metadata"])["trial_run_params"]["alg_params"]
    result["alg_params"].update(inner_solve_interval=interval, device="cuda", wandb=False)
    result["resolved_config"]["inner_solve_interval"] = interval
    steps = 6 if smoke else 500
    solves = (steps + interval - 1) // interval
    doses = {"inner_model_steps": 3072, "inner_critic_optimizer_steps": 128,
             "inner_actor_optimizer_steps": 32, "inner_temperature_optimizer_steps": 32,
             "inner_solve_performed": 1}
    for episode in result["episodes"]:
        episode["solver_seed"] = campaign.solver_seed(55, "episode", episode["seed"])
        episode["model_metrics"] = {key: dose * solves / steps for key, dose in doses.items()}
    n = len(result["episodes"])
    result["model_metrics"] = {key: {"count": n * steps, "sum": dose * solves * n} for key, dose in doses.items()}
    return payload


@pytest.mark.parametrize("index", [0, 1])
def test_matrix_changes_only_requested_cadence(inventory, index):
    entry = campaign.verify_entry(inventory, index)
    context = campaign.load_checkpoint_context(entry["checkpoint"], entry["metadata"])
    selector, interval = campaign.selection(index)
    resolved = campaign.resolve_preset(campaign.MATRIX, selector, campaign.load_preset_matrix(campaign.MATRIX),
                                       checkpoint_context=context)
    expected = deepcopy(context.trial_run_params["alg_params"])
    expected["inner_solve_interval"] = interval
    assert resolved["algorithm_config"]["alg_params"] == expected
    assert campaign.validate_results(swap_result(entry, index), entry, index) == {
        "episodes": 5, "decisions": 2500, "solves": (835, 2500)[index]}
    assert campaign.validate_results(swap_result(entry, index, smoke=True), entry, index, smoke=True) == {
        "episodes": 1, "decisions": 6, "solves": (2, 6)[index]}


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r["episodes"][0].update(truncated_by_evaluator=True), "native environment"),
    (lambda r: r.update(outer_updates_after=99), "Outer state"),
    (lambda r: r["episodes"][0].update(solver_seed=55), "solver seed"),
    (lambda r: r["alg_params"].update(inner_rounds=10), "beyond the requested cadence"),
    (lambda r: r["resolved_config"].update(inner_temperature_mode="fixed"), "alpha"),
    (lambda r: r["episodes"][0]["model_metrics"].update(inner_model_steps=1), "cadence/work"),
    (lambda r: r["model_metrics"]["inner_solve_performed"].update(sum=834), "cadence/work"),
])
def test_rejects_changed_protocol_or_work(inventory, mutation, message):
    entry = inventory["entries"][0]
    payload = swap_result(entry, 0)
    mutation(payload["results"][0])
    with pytest.raises(ValueError, match=message):
        campaign.validate_results(payload, entry, 0)


def reference_root(tmp_path, entry, index):
    root = tmp_path / "reference"
    cell = root / "production" / f"cell-{index}"
    (cell / "bundle").mkdir(parents=True)
    payload = result_payload(entry)
    campaign.write_json(cell / "results.json", payload)
    resolved = campaign.resolve_preset(campaign.MATRIX, campaign.selection(index)[0],
                                       campaign.load_preset_matrix(campaign.MATRIX),
                                       checkpoint_context=campaign.load_checkpoint_context(entry["checkpoint"], entry["metadata"]))
    manifest = {"schema_version": 1, "status": "complete", "checkpoint": {"sha256": entry["sha256"]},
                "protocol": campaign.protocol_for(resolved, 55, 500), "runs": [{
                    "config": {"alg_params": {"inner_operator": "none"}}, "status": "complete",
                    "episodes": payload["results"][0]["episodes"]}]}
    campaign.write_json(cell / "bundle/manifest.json", manifest)
    campaign.write_json(cell / "receipt.json", {"validated": True, "mode": "run",
                                                "results_sha256": campaign.file_hash(cell / "results.json"),
                                                "manifest_sha256": campaign.file_hash(cell / "bundle/manifest.json")})
    return root


def test_reference_requires_hashes_and_same_protocol(inventory, tmp_path):
    entry = inventory["entries"][0]
    root = reference_root(tmp_path, entry, 0)
    assert campaign.verify_reference(root, 0, entry)["bundle"].endswith("cell-0/bundle")
    path = root / "production/cell-0/bundle/manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash changed"):
        campaign.verify_reference(root, 0, entry)


@pytest.mark.parametrize("mode", ["prepare", "smoke", "run"])
def test_commands_run_only_swap_and_never_stage(inventory, tmp_path, monkeypatch, mode):
    entry = inventory["entries"][0]
    source = tmp_path / "inventory.json"
    campaign.write_json(source, inventory)
    reference = reference_root(tmp_path, entry, 0) if mode != "smoke" else tmp_path / "does-not-exist"
    output = tmp_path / mode
    map_path = None
    if mode == "run":
        registry = tmp_path / "registry"
        registry.mkdir()
        campaign.write_json(registry / "run.json", {"identity": {"match": True}})
        map_path = tmp_path / "run-map.json"
        campaign.write_json(map_path, {"cadence/k3": str(registry)})
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--eval-series-spec-dir" in command:
            specs = Path(command[command.index("--eval-series-spec-dir") + 1])
            specs.mkdir()
            campaign.write_json(specs / "cadence__k3.json", {"identity": {"match": True}})
        else:
            campaign.write_json(output / "results.json", swap_result(entry, 0, smoke=mode == "smoke"))
            (output / "bundle").mkdir()
            manifest = {"status": "complete", "runs": [{"selector": "cadence/k3"}]}
            if mode == "run":
                manifest["reference"] = {"manifest_sha256": campaign.verify_reference(reference, 0, entry)["manifest_sha256"]}
            campaign.write_json(output / "bundle/manifest.json", manifest)

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    monkeypatch.setattr(campaign, "runtime_info", lambda **kwargs: {"hostname": "test", "gpu_model": "test"})
    args = SimpleNamespace(mode=mode, inventory=source, index=0, output=output, reference_root=reference,
                           expected_commit=None, eval_run_map=map_path, smoke_steps=6)
    receipt = campaign.run(args)
    assert (receipt["source_K"], receipt["eval_K"]) == (1, 3)
    for command in commands:
        assert command[command.index("--preset") + 1] == "cadence/k3"
        assert "--wandb" not in command and "--eval-run-map" not in command
    if mode == "run":
        assert len(commands) == 2 and "--reference-bundle" in commands[-1]
        assert receipt["decisions"] == 2500
    else:
        assert len(commands) == 1 and "--reference-bundle" not in commands[0]
    if mode == "smoke":
        assert commands[0][-4:] == ["--seeds", "101", "--max-steps", "6"]
        assert receipt["decisions"] == 6 and "reference" not in receipt
    with pytest.raises(FileExistsError):
        campaign.run(args)


def test_rejects_other_checkpoint_and_prior_rerun(inventory):
    with pytest.raises(ValueError, match="two selected"):
        campaign.selection(2)
    entry = inventory["entries"][0]
    payload = swap_result(entry, 0)
    payload["results"].append(payload["results"][0])
    with pytest.raises(ValueError, match="only the requested"):
        campaign.validate_results(payload, entry, 0)
