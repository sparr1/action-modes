"""Scientific identity and arithmetic at the artifact/publication boundary."""

import copy
import ast
import json
from pathlib import Path
import statistics

import pytest

from utils import eval_series_data as data


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "scientific_identity", lambda *args: {"source_sha256": "science-v1"})
    config = {"inner_operator": "sac", "inner_rounds": 6, "inner_rollouts_per_round": 512,
              "inner_rollout_horizon": 3, "inner_actor_lr": 5e-5, "inner_critic_lr": 1e-4,
              "inner_temperature_mode": "auto", "inner_temperature_initialization": "inherit_outer",
              "inner_temperature": 0.125, "inner_temperature_lr": 3e-4,
              "inner_actor_updates_per_action": 18, "inner_critic_updates_per_action": 36,
              "inner_temperature_updates_per_action": 18, "inner_actor_adaptation": "clone",
              "inner_critic_adaptation": "clone", "inner_schedule_mode": "legacy",
              "inner_bootstrap_source": "inner_target", "episode_length": 500}
    episodes = [{"seed": seed, "solver_seed": seed + 1000, "return": 10.0 + index,
                 "length": 500, "terminated": False, "truncated": True, "control_seconds": 5.0,
                 "model_metrics": {"inner_actor_optimizer_steps": 18, "inner_model_steps": 9216},
                 "paired_return_delta": float(index), "status": "complete"}
                for index, seed in enumerate(range(101, 106))]
    value = {"schema_version": 1, "status": "complete", "evaluation_id": "legacy-attempt",
             "checkpoint": {"sha256": "a" * 64, "source_run": "entity/project/prior", "source_run_verified": True,
                            "metadata": {"checkpoint": {"step": 100000}}},
             "code": {"commit": "b" * 40, "dirty": False},
             "protocol": {"environment": {"id": "DMControl-v0", "params": {"task": "humanoid-walk"}},
                          "max_steps": 500, "controller_seed": 55, "action_rule": "tanh_mean",
                          "seed_scheme": "sha256-v1"},
             "runs": [{"selector": "dose/c6", "status": "complete", "kind": "episodes",
                       "config": {"alg": "AMBITDMPC2/AMBITDMPC2", "resolved_runtime": {"inner_operator": "none"}},
                       "resolved_config": config, "episodes": episodes, "trace_files": [],
                       "result": {"resolved_config": copy.deepcopy(config), "outer_state_unchanged": True,
                                  "outer_updates_before": 99999, "outer_updates_after": 99999,
                                  "environment_seeds": list(range(101, 106)), "return": {"mean": 999}}}]}
    path = dump(tmp_path / "bundle" / "manifest.json", value)
    return path, value


def test_actual_resolved_settings_and_sample_std(bundle):
    path, _ = bundle
    record, = data.load_records(path)
    assert record["identity"]["planner"]["type"] == "sac"
    assert record["identity"]["planner"]["settings"]["inner_critic_updates_per_action"] == 36
    assert record["metrics"]["eval/return_mean"] == 12
    assert record["metrics"]["eval/return_sample_std"] == statistics.stdev(range(10, 15))
    assert record["metrics"]["eval/paired_gain_sample_std"] == statistics.stdev(range(5))
    assert record["metrics"]["runtime/control_seconds"] == 25
    assert record["metrics"]["work/actor_updates"] == 45000
    assert record["metrics"]["work/model_steps"] == 23040000


def test_learned_values_and_cosmetics_do_not_split_planners(bundle):
    path, value = bundle
    first, = data.load_records(path)
    config = value["runs"][0]["resolved_config"]
    config.update({"device": "cuda:3", "wandb_run_name": "renamed", "inner_temperature": .0001,
                   "inner_reward_scale": 39, "inner_actor_lora_rank": 128,
                   "inner_explorer_actor_updates_per_action": 300, "inner_explorer_active": False,
                   "inner_mppi_num_samples": 900, "inner_diagnostics_every": 1})
    second, = data.load_records(dump(path, value))
    assert first["identity"] == second["identity"]


@pytest.mark.parametrize("key,value", [("inner_actor_lr", 1e-6), ("inner_critic_updates_per_action", 72),
                                      ("inner_bootstrap_source", "outer_target"),
                                      ("inner_temperature_initialization", "fixed")])
def test_active_settings_split_identity(bundle, key, value):
    path, payload = bundle
    first, = data.load_records(path)
    payload["runs"][0]["resolved_config"][key] = value
    second, = data.load_records(dump(path, payload))
    assert first["identity"]["planner"] != second["identity"]["planner"]


def test_prior_ignores_all_inactive_inner_settings(bundle):
    path, value = bundle
    value["runs"][0]["resolved_config"]["inner_operator"] = "none"
    first, = data.load_records(dump(path, value))
    value["runs"][0]["resolved_config"].update(inner_actor_lr=99, inner_critic_updates_per_action=999)
    second, = data.load_records(dump(path, value))
    assert first["identity"]["planner"] == second["identity"]["planner"] == {
        "type": "prior", "action_rule": "tanh_mean"}


def test_record_id_deduplicates_upload_retries_but_not_changed_measurements(bundle):
    path, value = bundle
    first, = data.load_records(path)
    value.update(elapsed_seconds=123, status="failed", evaluation_id="different-upload-copy")
    run = value["runs"][0]
    run.update(publication_seconds=999, status="failed", wandb_path="foreign/display/id")
    run["episodes"][0]["control_seconds"] = 100
    second, = data.load_records(dump(path, value))
    assert first["record_id"] == second["record_id"]
    assert second["provenance"]["scientific_status"] == "complete"
    run["episodes"][0]["return"] += 1
    third, = data.load_records(dump(path, value))
    assert first["record_id"] != third["record_id"]


@pytest.mark.parametrize("mutation,match", [
    (lambda v: v["runs"][0]["episodes"].pop(), "episode seeds"),
    (lambda v: v["runs"][0]["episodes"][0].update(status="failed"), "Incomplete episode"),
    (lambda v: v["runs"][0]["episodes"][0].update(return_value=None, **{"return": float("nan")}), "nonfinite"),
    (lambda v: v["runs"][0]["result"].update(outer_state_unchanged=False), "Frozen"),
    (lambda v: v["runs"][0]["result"].update(outer_updates_after=20), "Frozen"),
    (lambda v: v["runs"][0]["episodes"][0].update(truncated=False), "completion flag"),
])
def test_incomplete_or_false_scientific_results_are_rejected(bundle, mutation, match):
    path, value = bundle
    mutation(value)
    with pytest.raises(ValueError, match=match):
        data.load_records(dump(path, value))


def test_early_termination_is_a_complete_episode(bundle):
    path, value = bundle
    value["runs"][0]["episodes"][0].update(length=20, terminated=True, truncated=False)
    record, = data.load_records(dump(path, value))
    assert record["metrics"]["eval/episodes"] == 5
    assert record["metrics"]["work/environment_decisions"] == 2020


def test_inventory_repairs_missing_source_and_rejects_conflicts(bundle):
    path, value = bundle
    value["checkpoint"]["source_run"] = None
    value["checkpoint"]["source_run_verified"] = False
    dump(path, value)
    with pytest.raises(ValueError, match="Missing or conflicting"):
        data.load_records(path)
    inventory = dump(path.parent.parent / "checkpoint-manifest.json", {
        "source_run": "entity/project/prior", "checkpoints": [{"step": 100000, "sha256": "a" * 64}]})
    record, = data.load_records(path)
    assert record["identity"]["backbone"] == "entity/project/prior"
    assert record["provenance"]["checkpoint_source_verified"]
    with pytest.raises(ValueError, match="conflicting"):
        data.load_records(path, inventory_path=inventory, source_run="entity/project/other")


def test_declared_but_unverified_source_requires_inventory(bundle):
    path, value = bundle
    value["checkpoint"]["source_run_verified"] = False
    with pytest.raises(ValueError, match="unverified"):
        data.load_records(dump(path, value))


def test_transfer_inventory_validates_hash_and_named_training_directory(bundle):
    path, value = bundle
    cp = value["checkpoint"]
    cp["source_run_verified"] = False
    cp["metadata"]["trial_run_params"] = {"alg_params": {"wandb_run_name": "Prior training"}}
    inventory = dump(path.parent.parent / "transfer.json", {"runs": [
        {"name": "Prior training", "models_directory": "seed55/job1/models"}], "files": [
        {"kind": "weights", "path": "seed55/job1/models/checkpoint_100000", "step": 100000,
         "sha256": "a" * 64}]})
    record, = data.load_records(dump(path, value), inventory_path=inventory)
    assert record["provenance"]["checkpoint_source_verified"]
    cp["metadata"]["trial_run_params"]["alg_params"]["wandb_run_name"] = "Wrong training"
    with pytest.raises(ValueError, match="training run"):
        data.load_records(dump(path, value), inventory_path=inventory)


def test_observation_bank_identity_is_not_episode_protocol(bundle):
    path, value = bundle
    first, = data.load_records(path)
    value["protocol"].update(root_bank_id="checkpoint-dependent-bank", probe_rollouts=32)
    second, = data.load_records(dump(path, value))
    assert first["identity"] == second["identity"]


def test_only_audited_control_timing_is_excluded_from_scientific_ast():
    old = ast.parse('def run(model):\n action = model.predict(deterministic=True)\n return {"action": action}')
    timed = ast.parse('def run(model):\n control_seconds = 0.0\n prediction_started = time.perf_counter()\n action = model.predict(deterministic=True)\n control_seconds += time.perf_counter() - prediction_started\n return {"action": action, "control_seconds": control_seconds}')
    old_hash = ast.dump(data._WithoutControlTiming().visit(old))
    assert old_hash == ast.dump(data._WithoutControlTiming().visit(timed))
    changed = ast.parse('def run(model):\n action = model.predict(deterministic=False)\n return {"action": action}')
    assert old_hash != ast.dump(data._WithoutControlTiming().visit(changed))


def test_descriptive_label_identifies_backbone_and_planner(bundle):
    path, _ = bundle
    record, = data.load_records(path)
    assert "prior" in record["label"]
    assert "SAC C6/A3/T3 inner Q" in record["label"]
    assert "100000" not in record["label"]


def test_labels_distinguish_outer_terminal_from_every_transition_bootstrap(bundle):
    path, _ = bundle
    record, = data.load_records(path)
    identity = record["identity"]
    identity["planner"]["settings"]["inner_bootstrap_source"] = "outer_target"
    assert "outer Q throughout" in data.descriptive_label(identity)
    identity["planner"]["type"] = "xqc"
    identity["planner"]["settings"].pop("inner_bootstrap_source")
    identity["planner"]["settings"]["inner_terminal_bootstrap"] = "outer"
    assert "outer terminal Q" in data.descriptive_label(identity)


def test_mppi_labels_distinguish_terminal_value_backends():
    identity = {"backbone": "entity/project/prior", "science": {"algorithm": "AMBIXQC/AMBIXQC"},
                "planner": {"type": "mppi", "backend": "tdmpc2_mppi_over_frozen_xqc",
                            "settings": {"horizon": 3, "num_samples": 512, "num_elites": 64,
                                         "num_pi_trajs": 24, "effective_iterations": 8}}}
    assert "online XQC Q × frozen scale" in data.descriptive_label(identity)
    identity["planner"]["backend"] = "native_tdmpc2"
    identity["science"]["algorithm"] = "TDMPC2/TDMPC2Baseline"
    assert "online TD-MPC2 Q" in data.descriptive_label(identity)


def test_xqc_mppi_uses_evaluation_controller_not_stale_prior_config(bundle):
    path, value = bundle
    run = value["runs"][0]
    run["config"]["alg"] = "AMBIXQC/AMBIXQC"
    run["resolved_config"]["inner_operator"] = "none"
    run["result"].update(controller="mppi", action_rule="weighted_elite_gumbel_no_execution_noise",
                         evaluation_controller={"type": "mppi", "settings": {"iterations": 6,
                         "effective_iterations": 8, "num_samples": 512}, "protocol": {
                         "algorithm": "tdmpc2_mppi_over_frozen_xqc", "reward_scale": 50,
                         "terminal_value_source": "online_xqc_twin_mean"}})
    first, = data.load_records(dump(path, value))
    assert first["controller"] == "mppi"
    assert first["identity"]["planner"]["settings"]["effective_iterations"] == 8
    run["result"]["evaluation_controller"]["protocol"]["reward_scale"] = 100
    second, = data.load_records(dump(path, value))
    assert first["identity"] == second["identity"]


def test_xqc_legacy_inner_terminal_default_matches_explicit_resolver_default(bundle):
    path, value = bundle
    run = value["runs"][0]
    run["config"]["alg"] = "AMBIXQC/AMBIXQC"
    run["resolved_config"]["inner_operator"] = "xqc"
    old, = data.load_records(dump(path, value))
    run["resolved_config"]["inner_terminal_bootstrap"] = "inner"
    explicit, = data.load_records(dump(path, value))
    assert old["identity"] == explicit["identity"]
    run["resolved_config"]["inner_terminal_bootstrap"] = "outer"
    outer, = data.load_records(dump(path, value))
    assert old["identity"] != outer["identity"]


def test_missing_and_nonfinite_diagnostics_remain_distinct(bundle):
    path, value = bundle
    value["runs"][0]["episodes"][0]["model_metrics"]["nan_metric"] = float("nan")
    value["runs"][0]["episodes"][0]["model_metrics"]["missing_metric"] = None
    record, = data.load_records(dump(path, value))
    assert record["metrics"]["diagnostics/nan_metric_nonfinite"] == {"nonfinite": "nan"}
    assert "diagnostics/missing_metric_nonfinite" not in record["metrics"]
    json.dumps(record, allow_nan=False)


def test_missing_trace_files_are_explicit_and_path_escape_rejected(bundle):
    path, value = bundle
    value["runs"][0]["trace_files"] = ["controller/seed-101.jsonl.gz"]
    record, = data.load_records(dump(path, value))
    assert record["provenance"]["missing_artifact_files"] == ["controller/seed-101.jsonl.gz"]
    value["runs"][0]["trace_files"] = ["../escape.json"]
    with pytest.raises(ValueError, match="escapes"):
        data.load_records(dump(path, value))


def test_bank_only_is_not_an_episode_curve(bundle):
    path, value = bundle
    value["runs"][0].update(kind="bank", episodes=[], roots=[{"id": "root-1"}])
    assert data.load_records(dump(path, value)) == []


def test_local_tdmpc2_legacy_fixtures_are_compatible():
    root = Path(__file__).resolve().parents[2] / "benchmark-results/tdmpc2-prior-mppi-eval-20260906/results"
    paths = [root / f"step_{step}" / "paired.json" for step in (100000, 450000)]
    if not all(path.is_file() for path in paths):
        pytest.skip("Optional legacy artifact fixtures are not available")
    prior, mppi = data.load_records(paths[0])
    later_prior, later_mppi = data.load_records(paths[1])
    assert prior["controller"] == "policy_prior"
    assert mppi["controller"] == "native_mppi"
    assert prior["identity"] == later_prior["identity"]
    assert mppi["identity"] == later_mppi["identity"]
    assert mppi["metrics"]["eval/paired_gain_mean"] == pytest.approx(-146.66, abs=.01)
    assert "runtime/control_seconds" not in prior["metrics"]
    assert prior["metrics"]["runtime/evaluation_seconds"] > 0
    metadata = json.loads((paths[0].parent / "checkpoint.metadata.json").read_text())
    provenance = json.loads((paths[0].parent / "provenance.json").read_text())
    for record in (prior, mppi):
        actual = data.identity_for_tdmpc2_checkpoint(record["checkpoint"], metadata, record["controller"],
                                                     record["identity"]["protocol"], provenance, path=paths[0])
        assert actual == record["identity"]


def test_local_xqc_legacy_fixtures_recover_source_and_mppi():
    root = Path(__file__).resolve().parents[2] / "benchmark-results/ambixqc-mppi-eval-20260906/oscar/production"
    paths = sorted(root.glob("step_*/bundle/manifest.json"))
    if not paths:
        pytest.skip("Optional legacy artifact fixtures are not available")
    prior, mppi = data.load_records(paths[0])
    assert prior["identity"]["backbone"].endswith("/axqc-prior-92441d99-5959199")
    assert prior["identity"]["planner"]["type"] == "prior"
    assert mppi["identity"]["planner"]["backend"] == "tdmpc2_mppi_over_frozen_xqc"
    assert mppi["provenance"]["missing_artifact_files"] == []


def test_local_old_xqc_record_matches_preflight_explicit_inner_terminal_default(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "benchmark-results"
    paths = sorted((root / "ambixqc-inner-eval-j6-20260906").glob("**/bundle/manifest.json"))
    if not paths:
        pytest.skip("Optional legacy artifact fixtures are not available")
    path = next(path for path in paths if json.loads(path.read_text())["runs"][0].get("resolved_config", {}).get("inner_operator") == "xqc")
    payload = json.loads(path.read_text())
    run = payload["runs"][0]
    assert "inner_terminal_bootstrap" not in run["resolved_config"]
    inventory = root / "ambixqc-inner-eval-j6-20260906/checkpoint-manifest.json"
    record, = data.load_records(path, inventory_path=inventory)
    # Exercise preflight with the one resolver field introduced by the new
    # outer-terminal ablation; other settings remain the actual saved settings.
    resolved = {**run["resolved_config"], "inner_terminal_bootstrap": "inner"}
    monkeypatch.setattr(data, "resolved_checkpoint_config", lambda *args, **kwargs: resolved)
    actual = data.identity_for_ambi_checkpoint(
        payload["checkpoint"], {"algorithm_config": run["config"]}, payload["protocol"],
        run["result"]["environment_seeds"], payload["code"], path=path, inventory_path=inventory)
    assert actual == record["identity"]


def test_local_sac_preflight_matches_executed_settings_without_learner(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "benchmark-results"
    path = root / "inner-bench-analysis-20260905/episode_traces/f3z60w1i/manifest.json"
    inventory = root / "prior-checkpoint-transfers/hydra-20260905T215710Z/manifest.json"
    if not path.is_file() or not inventory.is_file():
        pytest.skip("Optional legacy artifact fixtures are not available")
    import torch
    import numpy as np
    from RL.AMBITDMPC2 import AMBITDMPC2
    monkeypatch.setattr(AMBITDMPC2, "__init__", lambda *args, **kwargs: pytest.fail("Constructed learner"))
    payload = json.loads(path.read_text())
    record, = data.load_records(path, inventory_path=inventory)
    run = payload["runs"][0]
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    actual = data.identity_for_ambi_checkpoint(
        payload["checkpoint"], {"algorithm_config": run["config"]}, payload["protocol"],
        run["result"]["environment_seeds"], payload["code"], path=path, inventory_path=inventory)
    assert actual == record["identity"]
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


def test_new_explicit_historical_defaults_preserve_legacy_planner_identity():
    legacy = {"inner_operator": "sac", "inner_adam_eps": 1e-8,
              "sac_actor_loss_scale_mode": "none"}
    current = {**legacy, "inner_actor_adam_eps": 1e-8,
               "inner_update_timing": "round", "inner_actor_entropy_mode": "squashed",
               "inner_critic_target_initialization": "online", "inner_critic_loss_coef": 1.0,
               "inner_actor_loss_scale_update": "per_action"}
    identity = lambda cfg: data.planner_identity(cfg, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert identity(current) == identity(legacy)
    for key, alternate in {
        "inner_actor_adam_eps": 1e-5,
        "inner_update_timing": "step",
        "inner_actor_entropy_mode": "tdmpc2_scaled",
        "inner_critic_target_initialization": "outer_target",
        "inner_critic_loss_coef": 0.1,
    }.items():
        assert identity({**current, key: alternate}) != identity(current), key
    scaled = {**current, "sac_actor_loss_scale_mode": "tdmpc2_percentile_range"}
    frozen = identity(scaled)
    assert frozen["settings"]["inner_actor_loss_scale_update"] == "per_action"
    assert frozen != identity({**scaled, "inner_actor_loss_scale_update": "per_update"})


@pytest.mark.parametrize("bank", ("reward_qscale", "entropy_qscale", "entropy_autotemp", "reward_autotemp"))
@pytest.mark.parametrize("selector", ("controller/prior", "controller/mppi", "inner/fixed", "inner/adaptive"))
def test_four_bank_prospective_identity_matches_executed_config_and_ignores_checkpoint_scalars(
    bank, selector, tmp_path, monkeypatch
):
    from tests.test_td_ambi_prior_bank_presets import BANKS, _cfg, _resolved
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from RL.tdmpc2_core.ambi_mppi import FrozenAMBIMPPIController

    monkeypatch.setattr(data, "scientific_identity", lambda *args: {"source_sha256": "current-science"})
    resolved = _resolved(bank, selector)
    observation = {"mode": "state", "shape": [67], "action_dim": 21, "episode_length": 500}
    checkpoint = {
        "sha256": "a" * 64,
        "source_run": f"rwgao_b-brown-university/ambi/{BANKS[bank]}",
        "source_run_verified": True,
        "metadata": {
            "checkpoint": {"step": 25000},
            "trial_run_params": {
                **copy.deepcopy(resolved["algorithm_config"]),
                "resolved_runtime": {"observation": observation},
            },
        },
    }
    protocol = {"environment": resolved["environment"], "observation": observation,
                "action_rule": "tanh_mean", "controller_seed": 55, "max_steps": 500,
                "seed_scheme": "sha256-v1"}
    seeds = [101, 102, 103, 104, 105]
    identity = data.identity_for_ambi_checkpoint(
        checkpoint, resolved, protocol, seeds, {"commit": "b" * 40}, path=tmp_path / "manifest.json"
    )
    cfg = _cfg(resolved)
    result = {}
    action_rule = protocol["action_rule"]
    if selector == "controller/mppi":
        # Compare the metadata-only constructor against a real initialized
        # controller. Narrow network widths do not change MPPI's protocol.
        agent = AMBITDMPC2Agent(_cfg(resolved, small=True))
        agent.model.eval()
        controller = FrozenAMBIMPPIController(agent, resolved["evaluation_controller"]["params"])
        result["evaluation_controller"] = {
            "type": "mppi", "settings": controller.settings, "protocol": controller.protocol,
        }
        action_rule = controller.protocol["action_rule"]
    executed = data.planner_identity(vars(cfg), result, resolved["algorithm_config"]["alg"], action_rule)
    assert identity["planner"] == executed
    assert identity["backbone"].endswith(BANKS[bank])

    # Checkpoint weights, saved S and learned alpha are data points on the same
    # curve. Their initialization/adaptation policies remain in planner identity.
    later = copy.deepcopy(checkpoint)
    later["sha256"] = "c" * 64
    later["metadata"]["checkpoint"]["step"] = 2000000
    later["actor_loss_scale_state"] = {"value": 21.0}
    later["log_ent_coef"] = -16.2
    later_resolved = copy.deepcopy(resolved)
    later_resolved["algorithm_config"]["alg_params"]["inner_temperature"] = 9e-8
    later_identity = data.identity_for_ambi_checkpoint(
        later, later_resolved, protocol, seeds, {"commit": "b" * 40}, path=tmp_path / "later.json"
    )
    assert later_identity == identity
    if selector.startswith("inner/"):
        counterpart = _resolved(bank, "inner/adaptive" if selector == "inner/fixed" else "inner/fixed")
        other = data.identity_for_ambi_checkpoint(
            checkpoint, counterpart, protocol, seeds, {"commit": "b" * 40}, path=tmp_path / "other.json"
        )
        assert identity["planner"] != other["planner"]
        settings = identity["planner"]["settings"]
        assert settings["inner_actor_entropy_mode"] == "tdmpc2_scaled"
        assert settings["inner_critic_target_initialization"] == "outer_target"
        if bank.endswith("qscale"):
            assert settings["inner_actor_loss_scale_update"] == (
                "per_action" if selector == "inner/fixed" else "per_update"
            )
        else:
            assert settings["inner_temperature_mode"] == (
                "inherit_outer" if selector == "inner/fixed" else "auto"
            )
            assert settings["inner_target_entropy"] == -441
