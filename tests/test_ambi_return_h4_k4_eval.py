"""H4/K4 executes the requested held policy and never repeats its controls."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_h4_k4_eval as campaign
from slurm import ambi_return_cadence_eval as cadence
from tests.test_ambi_return_prior_eval import inventory
from tests.test_ambi_return_cadence_eval import reference_root
from tests.test_ambi_return_h4_eval import h4_result


def k4_result(entry, *, smoke=False, steps=6):
    payload = h4_result(entry, smoke=smoke)
    result = payload["results"][0]
    result["selector"] = "horizon/h4_k4"
    result["alg_params"]["inner_solve_interval"] = 4
    result["resolved_config"].update(inner_solve_interval=4, inner_hold_encoder="solve_snapshot")
    steps = steps if smoke else 500
    solves = (steps + 3) // 4
    doses = {"inner_model_steps": 4096, "inner_critic_optimizer_steps": 128,
             "inner_actor_optimizer_steps": 32, "inner_temperature_optimizer_steps": 32,
             "inner_solve_performed": 1}
    for episode in result["episodes"]:
        episode["length"] = steps
        episode["model_metrics"] = {key: dose * solves / steps for key, dose in doses.items()}
        episode["model_metrics"]["inner_policy_held"] = (steps - solves) / steps
    n = len(result["episodes"])
    result["model_metrics"] = {key: {"count": n * steps, "sum": dose * solves * n} for key, dose in doses.items()}
    result["model_metrics"]["inner_policy_held"] = {"count": n * steps, "sum": (steps - solves) * n}
    return payload


@pytest.mark.parametrize("index", [0, 1])
def test_matrix_and_config_preserve_saved_settings_except_horizon_cadence_capacity(inventory, index):
    from tests.test_ambi_research_presets import _build_cfg

    entry = cadence.verify_entry(inventory, index)
    context = cadence.load_checkpoint_context(entry["checkpoint"], entry["metadata"])
    resolved = cadence.resolve_preset(campaign.MATRIX, campaign.selection(index)[0],
                                      cadence.load_preset_matrix(campaign.MATRIX), checkpoint_context=context)
    expected = deepcopy(context.trial_run_params["alg_params"])
    expected.update(inner_solve_interval=4, inner_rollout_horizon=4, inner_replay_capacity=4096)
    assert resolved["algorithm_config"]["alg_params"] == expected
    cfg = _build_cfg(resolved)
    assert cfg.inner_solve_interval == 4 and cfg.inner_hold_encoder == "solve_snapshot"
    assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 4096
    assert cfg.inner_critic_updates_per_action == 128
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 32
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_temperature_initialization == "inherit_outer"
    assert campaign.validate_results(k4_result(entry), entry, index) == {
        "episodes": 5, "decisions": 2500, "solves": 625, "held_decisions": 1875}


@pytest.mark.parametrize("steps,solves,held", [(1, 1, 0), (3, 1, 2), (4, 1, 3), (5, 2, 3), (6, 2, 4)])
def test_partial_blocks_have_exact_solve_and_hold_counts(inventory, steps, solves, held):
    entry = inventory["entries"][0]
    assert campaign.validate_results(k4_result(entry, smoke=True, steps=steps), entry, 0,
                                     smoke=True, smoke_steps=steps) == {
        "episodes": 1, "decisions": steps, "solves": solves, "held_decisions": held}


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r["resolved_config"].update(inner_hold_encoder="live"), "Held encoder"),
    (lambda r: r["episodes"][0]["model_metrics"].update(inner_solve_performed=124/500), "cadence/work"),
    (lambda r: r["episodes"][0]["model_metrics"].update(inner_policy_held=374/500), "held-policy"),
    (lambda r: r["model_metrics"]["inner_policy_held"].update(sum=1874), "held-policy"),
    (lambda r: r["alg_params"].update(inner_rounds=10), "evaluation overrides"),
    (lambda r: r["episodes"][0].update(truncated_by_evaluator=True), "native environment"),
    (lambda r: r.update(outer_updates_after=99), "Outer state"),
])
def test_rejects_wrong_hold_state_or_episode_counts_even_when_total_is_correct(inventory, mutation, message):
    entry = inventory["entries"][0]
    payload = k4_result(entry)
    mutation(payload["results"][0])
    with pytest.raises(ValueError, match=message):
        campaign.validate_results(payload, entry, 0)


def test_engine_k4_uses_new_observations_and_resets_after_partial_block():
    import torch
    from tests.test_ambi_root_local_sac import _tiny_model

    model = _tiny_model(inner_rounds=1, inner_updates_per_round=2, inner_solve_interval=4,
                        inner_rollout_horizon=4, inner_replay_capacity=8, inner_critic_bernoulli_p=0.)
    try:
        agent, engine = model.agent, model.agent.inner_engine
        agent.act(torch.zeros(3), t0=True, eval_mode=True)
        observations = []
        hook = engine._held_encoder.register_forward_pre_hook(lambda _, inputs: observations.append(inputs[0].clone()))
        observed = [agent.last_inner_metrics["inner_solve_performed"]]
        for decision in range(1, 6):
            observation = torch.ones(3) * decision / 10
            agent.act(observation, eval_mode=True)
            torch.testing.assert_close(observations[-1].reshape(-1), observation)
            observed.append(agent.last_inner_metrics["inner_solve_performed"])
        hook.remove()
        assert observed == [1, 0, 0, 0, 1, 0]
        engine.reset_for_evaluation(55)
        agent.act(torch.ones(3), t0=True, eval_mode=True)
        assert agent.last_inner_metrics["inner_episode_decision_index"] == 0
        assert agent.last_inner_metrics["inner_solve_performed"] == 1
        assert agent.last_inner_metrics["inner_action_age"] == 0
    finally:
        model.close()
        model.env.close()


@pytest.mark.parametrize("index,mode", [(0, "prepare"), (0, "smoke"), (1, "run")])
def test_commands_evaluate_only_k4_and_reuse_original_prior(inventory, tmp_path, monkeypatch, index, mode):
    entry = inventory["entries"][index]
    source = tmp_path / "inventory.json"
    cadence.write_json(source, inventory)
    reference = reference_root(tmp_path, entry, index) if mode != "smoke" else tmp_path / "absent"
    output = tmp_path / mode
    map_path = None
    if mode == "run":
        registry = tmp_path / "registry"
        registry.mkdir()
        cadence.write_json(registry / "run.json", {"identity": {"horizon": 4, "cadence": 4}})
        map_path = tmp_path / "map.json"
        cadence.write_json(map_path, {"horizon/h4_k4": str(registry)})
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--eval-series-spec-dir" in command:
            specs = Path(command[command.index("--eval-series-spec-dir") + 1])
            specs.mkdir()
            cadence.write_json(specs / "horizon__h4_k4.json", {"identity": {"horizon": 4, "cadence": 4}})
        else:
            cadence.write_json(output / "results.json", k4_result(entry, smoke=mode == "smoke"))
            (output / "bundle").mkdir()
            manifest = {"status": "complete", "runs": [{"selector": "horizon/h4_k4"}]}
            if mode == "run":
                manifest["reference"] = {"manifest_sha256": cadence.verify_reference(reference, index, entry)["manifest_sha256"]}
            cadence.write_json(output / "bundle/manifest.json", manifest)

    monkeypatch.setattr(cadence.subprocess, "run", fake_run)
    monkeypatch.setattr(cadence, "runtime_info", lambda **kwargs: {"hostname": "test"})
    receipt = campaign.run(SimpleNamespace(mode=mode, inventory=source, index=index, output=output,
                                           reference_root=reference, expected_commit=None,
                                           eval_run_map=map_path, smoke_steps=6))
    assert receipt["campaign_id"] == campaign.CAMPAIGN_ID
    assert (receipt["source_H"], receipt["eval_H"], receipt["eval_K"]) == (3, 4, 4)
    assert receipt["source_K"] == entry["K"]
    for command in commands:
        assert command[command.index("--matrix") + 1] == str(campaign.MATRIX)
        assert command[command.index("--preset") + 1] == "horizon/h4_k4"
        assert "--eval-run-map" not in command and "--wandb" not in command
    assert ("--reference-bundle" in commands[-1]) == (mode == "run")
    if mode == "smoke":
        assert commands[-1][-4:] == ["--seeds", "101", "--max-steps", "6"]
        assert receipt["solves"] == 2 and receipt["held_decisions"] == 4
    elif mode == "run":
        assert receipt["solves"] == 625 and receipt["held_decisions"] == 1875 and len(commands) == 2
