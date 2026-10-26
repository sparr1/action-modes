"""H4 changes only the requested horizon/cadence and required replay capacity."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_h4_eval as campaign
from slurm import ambi_return_cadence_eval as cadence
from tests.test_ambi_return_prior_eval import inventory
from tests.test_ambi_return_cadence_eval import reference_root, swap_result


def h4_result(entry, *, smoke=False):
    payload = swap_result(entry, 1, smoke=smoke)
    result = payload["results"][0]
    result["selector"] = "horizon/h4_k1"
    for key in ("alg_params", "resolved_config"):
        result[key].update(inner_solve_interval=1, inner_rollout_horizon=4, inner_replay_capacity=4096)
    result["resolved_config"].pop("inner_hold_encoder", None)
    for episode in result["episodes"]:
        episode["model_metrics"]["inner_model_steps"] = 4096.
    metric = result["model_metrics"]["inner_model_steps"]
    metric["sum"] = metric["count"] * 4096
    return payload


@pytest.mark.parametrize("index", [0, 1])
def test_only_horizon_cadence_and_required_capacity_change(inventory, index):
    entry = cadence.verify_entry(inventory, index)
    context = cadence.load_checkpoint_context(entry["checkpoint"], entry["metadata"])
    resolved = cadence.resolve_preset(campaign.MATRIX, campaign.selection(index)[0],
                                      cadence.load_preset_matrix(campaign.MATRIX), checkpoint_context=context)
    expected = deepcopy(context.trial_run_params["alg_params"])
    expected.update(inner_solve_interval=1, inner_rollout_horizon=4, inner_replay_capacity=4096)
    assert resolved["algorithm_config"]["alg_params"] == expected
    assert campaign.validate_results(h4_result(entry), entry, index) == {
        "episodes": 5, "decisions": 2500, "solves": 2500}
    assert campaign.validate_results(h4_result(entry, smoke=True), entry, index, smoke=True) == {
        "episodes": 1, "decisions": 6, "solves": 6}


@pytest.mark.parametrize("index", [0, 1])
def test_resolved_h4_budget_requires_capacity_resize(inventory, index):
    from tests.test_ambi_research_presets import _build_cfg

    entry = inventory["entries"][index]
    resolved = cadence.resolve_preset(campaign.MATRIX, campaign.selection(index)[0],
                                      cadence.load_preset_matrix(campaign.MATRIX),
                                      checkpoint_context=cadence.load_checkpoint_context(entry["checkpoint"], entry["metadata"]))
    cfg = _build_cfg(resolved)
    assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 4096
    assert cfg.inner_critic_updates_per_action == 128
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 32
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_temperature_initialization == "inherit_outer"
    assert cfg.inner_solve_interval == 1 and not hasattr(cfg, "inner_hold_encoder")
    resolved["algorithm_config"]["alg_params"]["inner_replay_capacity"] = 3072
    with pytest.raises(ValueError, match="at least the cumulative"):
        _build_cfg(resolved)


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r["alg_params"].update(inner_replay_capacity=3072), "Controller settings"),
    (lambda r: r["resolved_config"].update(inner_replay_capacity=3072), "replay capacity"),
    (lambda r: r["resolved_config"].update(inner_rollout_horizon=3), "Resolved controller"),
    (lambda r: r["alg_params"].update(inner_rounds=10), "Controller settings"),
    (lambda r: r["episodes"][0]["model_metrics"].update(inner_model_steps=3072), "cadence/work"),
    (lambda r: r["model_metrics"]["inner_critic_optimizer_steps"].update(sum=319999), "cadence/work"),
    (lambda r: r["episodes"][0].update(truncated_by_evaluator=True), "native environment"),
    (lambda r: r.update(outer_updates_after=99), "Outer state"),
])
def test_rejects_wrong_horizon_work_or_frozen_protocol(inventory, mutation, message):
    entry = inventory["entries"][0]
    payload = h4_result(entry)
    mutation(payload["results"][0])
    with pytest.raises(ValueError, match=message):
        campaign.validate_results(payload, entry, 0)


@pytest.mark.parametrize("index,mode", [(0, "prepare"), (0, "smoke"), (1, "run")])
def test_new_commands_reuse_prior_without_mutating_old_campaign(inventory, tmp_path, monkeypatch, index, mode):
    entry = inventory["entries"][index]
    source = tmp_path / "inventory.json"
    cadence.write_json(source, inventory)
    reference = reference_root(tmp_path, entry, index) if mode != "smoke" else tmp_path / "absent"
    output = tmp_path / mode
    map_path = None
    if mode == "run":
        registry = tmp_path / "registry"
        registry.mkdir()
        cadence.write_json(registry / "run.json", {"identity": {"horizon": 4}})
        map_path = tmp_path / "map.json"
        cadence.write_json(map_path, {"horizon/h4_k1": str(registry)})
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--eval-series-spec-dir" in command:
            specs = Path(command[command.index("--eval-series-spec-dir") + 1])
            specs.mkdir()
            cadence.write_json(specs / "horizon__h4_k1.json", {"identity": {"horizon": 4}})
        else:
            cadence.write_json(output / "results.json", h4_result(entry, smoke=mode == "smoke"))
            (output / "bundle").mkdir()
            manifest = {"status": "complete", "runs": [{"selector": "horizon/h4_k1"}]}
            if mode == "run":
                manifest["reference"] = {"manifest_sha256": cadence.verify_reference(reference, index, entry)["manifest_sha256"]}
            cadence.write_json(output / "bundle/manifest.json", manifest)

    old_defaults = (cadence.MATRIX, cadence.CAMPAIGN_ID, cadence.selection(0), cadence.selection(1))
    monkeypatch.setattr(cadence.subprocess, "run", fake_run)
    monkeypatch.setattr(cadence, "runtime_info", lambda **kwargs: {"hostname": "test"})
    receipt = campaign.run(SimpleNamespace(mode=mode, inventory=source, index=index, output=output,
                                           reference_root=reference, expected_commit=None,
                                           eval_run_map=map_path, smoke_steps=6))
    assert (cadence.MATRIX, cadence.CAMPAIGN_ID, cadence.selection(0), cadence.selection(1)) == old_defaults
    assert receipt["campaign_id"] == campaign.CAMPAIGN_ID
    assert (receipt["source_H"], receipt["eval_H"], receipt["eval_K"]) == (3, 4, 1)
    assert receipt["source_K"] == entry["K"]
    for command in commands:
        assert command[command.index("--matrix") + 1] == str(campaign.MATRIX)
        assert command[command.index("--preset") + 1] == "horizon/h4_k1"
        assert "--eval-run-map" not in command and "--wandb" not in command
    assert ("--reference-bundle" in commands[-1]) == (mode == "run")
    if mode == "smoke":
        assert commands[-1][-4:] == ["--seeds", "101", "--max-steps", "6"]
        assert receipt["decisions"] == 6
    elif mode == "run":
        assert receipt["decisions"] == 2500 and len(commands) == 2
