"""Only missing J10/K1 arms run, with paired priors and strict work accounting."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_j10_k1_eval as campaign
from slurm import ambi_return_cadence_eval as cadence
from tests.test_ambi_return_prior_eval import inventory
from tests.test_ambi_return_cadence_eval import reference_root
from tests.test_ambi_return_j10_eval import j10_result


def result_payload(entry, *, smoke=False):
    payload = j10_result(entry, 0, smoke=smoke)
    payload["results"][0]["selector"] = "rounds/j10_k1"
    return payload


def paired_reference(tmp_path, entry, index):
    # The reference protocol is independent of controller J/K; the paired
    # result still carries and validates this source checkpoint's saved J/K.
    root = reference_root(tmp_path, entry, 1)
    if index == 2:
        (root / "production/cell-1").rename(root / "production/cell-2")
    return root


@pytest.mark.parametrize("index", [1, 2])
def test_matrix_only_changes_requested_j_k_and_replay(inventory, index):
    from tests.test_ambi_research_presets import _build_cfg

    entry = cadence.verify_entry(inventory, index)
    context = cadence.load_checkpoint_context(entry["checkpoint"], entry["metadata"])
    resolved = cadence.resolve_preset(campaign.MATRIX, campaign.selection(index)[0],
                                      cadence.load_preset_matrix(campaign.MATRIX), checkpoint_context=context)
    expected = deepcopy(context.trial_run_params["alg_params"])
    expected.update(inner_rounds=10, inner_solve_interval=1, inner_replay_capacity=3840)
    assert resolved["algorithm_config"]["alg_params"] == expected
    cfg = _build_cfg(resolved)
    assert (cfg.inner_rounds, cfg.inner_solve_interval, cfg.inner_rollout_horizon) == (10, 1, 3)
    assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 3840
    assert cfg.inner_critic_updates_per_action == 160
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 40
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_temperature_initialization == "inherit_outer"
    assert campaign.validate_results(result_payload(entry), entry, index) == {
        "episodes": 5, "decisions": 2500, "solves": 2500}
    assert campaign.validate_results(result_payload(entry, smoke=True), entry, index, smoke=True) == {
        "episodes": 1, "decisions": 6, "solves": 6}


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r["alg_params"].update(inner_solve_interval=3), "evaluation overrides"),
    (lambda r: r["alg_params"].update(inner_replay_capacity=3072), "evaluation overrides"),
    (lambda r: r["resolved_config"].update(inner_rounds=8), "Resolved controller"),
    (lambda r: r["resolved_config"].update(inner_temperature_initialization="fixed"), "alpha"),
    (lambda r: r["episodes"][0]["model_metrics"].update(inner_model_steps=1280), "cadence/work"),
    (lambda r: r["model_metrics"]["inner_critic_optimizer_steps"].update(sum=133600), "cadence/work"),
    (lambda r: r["episodes"][0].update(truncated_by_evaluator=True), "native environment"),
    (lambda r: r.update(outer_updates_after=99), "Outer state"),
])
def test_rejects_wrong_cadence_dose_or_protocol(inventory, mutation, message):
    entry = inventory["entries"][2]
    payload = result_payload(entry)
    mutation(payload["results"][0])
    with pytest.raises(ValueError, match=message):
        campaign.validate_results(payload, entry, 2)


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize("mode", ["prepare", "smoke", "run"])
def test_worker_reuses_correct_pair_and_only_runs_missing_arm(inventory, tmp_path, monkeypatch, index, mode):
    inventory["campaign_id"] = campaign.CAMPAIGN_ID
    entry = inventory["entries"][index]
    source, output = tmp_path / "inventory.json", tmp_path / mode
    cadence.write_json(source, inventory)
    reference = paired_reference(tmp_path, entry, index) if mode != "smoke" else tmp_path / "absent"
    map_path = None
    if mode == "run":
        registry = tmp_path / "registry"
        registry.mkdir()
        cadence.write_json(registry / "run.json", {"identity": {"match": True}})
        map_path = tmp_path / "map.json"
        cadence.write_json(map_path, {"rounds/j10_k1": str(registry)})
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--eval-series-spec-dir" in command:
            specs = Path(command[command.index("--eval-series-spec-dir") + 1])
            specs.mkdir()
            cadence.write_json(specs / "rounds__j10_k1.json", {"identity": {"match": True}})
        else:
            cadence.write_json(output / "results.json", result_payload(entry, smoke=mode == "smoke"))
            (output / "bundle").mkdir()
            manifest = {"status": "complete", "runs": [{"selector": "rounds/j10_k1"}]}
            if mode == "run":
                manifest["reference"] = {"manifest_sha256": cadence.file_hash(reference / f"production/cell-{index}/bundle/manifest.json")}
            cadence.write_json(output / "bundle/manifest.json", manifest)

    monkeypatch.setattr(cadence.subprocess, "run", fake_run)
    monkeypatch.setattr(cadence, "runtime_info", lambda **kwargs: {"hostname": "test"})
    receipt = campaign.run(SimpleNamespace(mode=mode, inventory=source, index=index, output=output,
                                           reference_root=reference, expected_commit=None,
                                           eval_run_map=map_path, smoke_steps=6))
    assert receipt["campaign_id"] == campaign.CAMPAIGN_ID
    assert (receipt["J"], receipt["source_J"], receipt["eval_J"]) == (10, entry["J"], 10)
    assert (receipt["source_K"], receipt["eval_K"], receipt["source_H"], receipt["eval_H"]) == (3, 1, 3, 3)
    assert receipt["selector"] == "rounds/j10_k1"
    for command in commands:
        assert command[command.index("--preset") + 1] == "rounds/j10_k1"
        assert "--eval-run-map" not in command and "--wandb" not in command
    assert ("--reference-bundle" in commands[-1]) == (mode == "run")
    if mode == "run":
        assert commands[-1][commands[-1].index("--reference-bundle") + 1] == str(reference / f"production/cell-{index}/bundle")
        assert receipt["episodes"] == 5 and receipt["solves"] == receipt["decisions"] == 2500
    elif mode == "smoke":
        assert commands[-1][-4:] == ["--seeds", "101", "--max-steps", "6"]
        assert receipt["episodes"] == 1 and receipt["solves"] == receipt["decisions"] == 6


def test_existing_index0_and_other_inventory_cannot_run(inventory, tmp_path):
    with pytest.raises(ValueError, match="reuse index0"):
        campaign.run(SimpleNamespace(index=0))
    source = tmp_path / "inventory.json"
    cadence.write_json(source, inventory)
    with pytest.raises(ValueError, match="campaign inventory"):
        campaign.run(SimpleNamespace(index=1, inventory=source))


def test_default_shared_runner_still_rejects_j10_source(inventory, tmp_path):
    source = tmp_path / "inventory.json"
    cadence.write_json(source, inventory)
    with pytest.raises(ValueError, match="selected J8"):
        cadence.run(SimpleNamespace(index=2, inventory=source), select=campaign.selection)


def test_third_reference_requires_same_500k_checkpoint(inventory, tmp_path):
    entry = inventory["entries"][2]
    root = paired_reference(tmp_path, entry, 2)
    changed = {**entry, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        cadence.verify_reference(root, 2, changed, matrix=campaign.MATRIX, selector="rounds/j10_k1")
