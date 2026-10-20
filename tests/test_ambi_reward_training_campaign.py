"""The reward-only GPU gate binds one unchanged recipe to durable resume."""

from argparse import Namespace
from copy import deepcopy
import subprocess
from types import SimpleNamespace

import pytest

from slurm import ambi_reward_training as campaign


@pytest.mark.parametrize("path", list(campaign.CAMPAIGNS))
def test_actual_campaigns_validate_and_smoke_preserves_production_shapes(path):
    _, operator, total_steps, target_entropy = campaign.CAMPAIGNS[path]
    name, manifest, original = campaign.validate_campaign(path)
    selected, small_manifest, run, overrides = campaign.smoke_recipe(12, 120, path)
    expected = deepcopy(original)
    expected["alg_params"].update(wandb=False, seed_steps=500, pretrain_steps=2, buffer_size=4096)
    expected.update(total_steps=633, checkpoint_every=633, save_strat="all")
    assert selected == name and run == expected
    assert overrides == {"wandb": False, "seed_steps": 500, "pretrain_steps": 2, "buffer_size": 4096}
    expected_manifest = deepcopy(manifest)
    expected_manifest.update(checkpoint_every=633, save_strat="all", save_trials="none", logs="none")
    assert small_manifest == expected_manifest
    assert original["alg_params"]["inner_operator"] == operator
    assert original["alg_params"]["inner_solve_interval"] == (3 if operator == "sac" else 1)
    assert original["checkpoint_every"] == 25000 and original["save_strat"] == "all"
    assert original["total_steps"] == total_steps
    assert run["alg_params"]["target_entropy"] == original["alg_params"]["target_entropy"] == target_entropy


@pytest.mark.parametrize("path", list(campaign.CAMPAIGNS))
def test_campaign_binds_config_name_budget_and_entropy_to_approved_manifest(monkeypatch, path):
    name, manifest, original = campaign.validate_campaign(path)
    changed = deepcopy(original)
    monkeypatch.setattr(campaign, "recipe", lambda index, selected: (name, manifest, changed))
    for key, value in (("total_steps", original["total_steps"] + 500000),
                       ("target_entropy", -21. if original["alg_params"]["target_entropy"] == -10.5 else -10.5)):
        changed = deepcopy(original)
        (changed["alg_params"] if key == "target_entropy" else changed)[key] = value
        with pytest.raises(AssertionError):
            campaign.validate_campaign(path)
    monkeypatch.setattr(campaign, "recipe", lambda index, selected: (name + "-renamed", manifest, original))
    with pytest.raises(AssertionError, match="identity"):
        campaign.validate_campaign(path)


def test_campaign_rejects_unapproved_manifest():
    with pytest.raises(AssertionError, match="Unapproved"):
        campaign.validate_campaign(campaign.ROOT / "configs/dmcontrol/experiments/ambi_held_training.json")


def test_projection_uses_selected_training_budget():
    short = campaign.production_projection(.02, 500000)
    long = campaign.production_projection(.02, 1000000)
    assert short["production_total_steps"] == 500000 and long["production_total_steps"] == 1000000
    assert long["projected_production_seconds"] == 20000
    assert long["projected_production_hours"] == 2 * short["projected_production_hours"]


def test_no_inner_gate_uses_longer_window_and_benchmarks_four_cores():
    assert campaign.smoke_defaults("none") == (24, 400)
    assert campaign.cpu_counts("none") == (1, 2, 4)
    _, _, run, _ = campaign.smoke_recipe(24, 400, campaign.MANIFESTS["none"])
    assert run["total_steps"] == run["checkpoint_every"] == 925
    for operator in ("sac", "mppi"):
        assert campaign.smoke_defaults(operator) == (12, 120)
        assert campaign.cpu_counts(operator) == (1, 2)


def test_no_inner_gate_requires_zero_realized_work_and_counts_prior_actions():
    metrics = dict.fromkeys(("inner_active", "inner_model_steps_budget", "inner_steps", "inner_model_steps",
                            "inner_total_model_steps", "inner_realized_model_steps", "inner_nominal_model_steps",
                            "inner_rounds", "inner_iterations", "inner_mppi_iterations", "inner_rollouts",
                            "inner_updates", "inner_update_slots", "inner_target_updates", "inner_q_evaluations",
                            "inner_replay_draws", "inner_buffer_size", "inner_critic_optimizer_steps",
                            "inner_actor_optimizer_steps", "inner_temperature_optimizer_steps", "inner_compile_fallback"), 0.)
    metrics["inner_policy_evaluations"] = 1.
    assert not campaign.validate_action(metrics, operator="none", decision=0, interval=1)
    for key in metrics:
        with pytest.raises(AssertionError):
            campaign.validate_action({**metrics, key: metrics[key] + 1}, operator="none", decision=1, interval=1)
    rows = [{"solve": False, "action_seconds": seconds, "outer_update_enqueue_seconds": .03}
            for seconds in (.01, .02)]
    summary = campaign.training_phases(rows, "none")
    assert summary["solve_decisions"] == summary["held_decisions"] == 0
    assert summary["prior_decisions"] == 2
    assert summary["prior_action_seconds_mean"] == pytest.approx(.015)
    assert summary["held_action_seconds_mean"] is summary["solve_action_seconds_mean"] is None
    assert summary["outer_update_enqueue_seconds_mean"] == .03
    assert campaign.training_phases(rows, "sac")["held_decisions"] == 2


def test_mppi_gate_rejects_accidental_inner_learning_or_wrong_model_dose():
    metrics = {"inner_critic_optimizer_steps": 0, "inner_actor_optimizer_steps": 0,
               "inner_temperature_optimizer_steps": 0, "inner_mppi_iterations": 8,
               "inner_model_steps": 12336, "inner_compile_fallback": 0}
    assert campaign.validate_action(metrics, operator="mppi", decision=0, interval=1)
    for key, value in (("inner_actor_optimizer_steps", 1), ("inner_model_steps", 12335),
                       ("inner_compile_fallback", 1)):
        with pytest.raises(AssertionError):
            campaign.validate_action({**metrics, key: value}, operator="mppi", decision=1, interval=1)


def test_sac_gate_counts_solves_and_rejects_held_work():
    metrics = {"inner_solve_performed": 0., "inner_policy_held": 1., "inner_action_age": 1.,
               "inner_episode_decision_index": 1., "inner_compile_fallback": 0., "inner_policy_evaluations": 1.}
    metrics.update(dict.fromkeys(("inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                   "inner_temperature_optimizer_steps", "inner_model_steps", "inner_updates", "inner_steps",
                   "inner_rollouts", "inner_replay_draws", "inner_realized_model_steps", "inner_total_model_steps",
                   "inner_q_evaluations"), 0.))
    assert not campaign.validate_action(metrics, operator="sac", decision=1, interval=3)
    with pytest.raises(AssertionError):
        campaign.validate_action({**metrics, "inner_critic_optimizer_steps": 1.}, operator="sac", decision=1, interval=3)


def test_optimizer_step_check_rejects_incomplete_actor_or_temperature_updates():
    def optimizer(counts):
        return SimpleNamespace(state={index: {"step": SimpleNamespace(item=lambda n=n: n)}
                                      for index, n in enumerate(counts)})
    assert campaign._optimizer_steps(optimizer([134, 134])) == 134
    for counts in ([], [133, 134]):
        with pytest.raises(AssertionError):
            campaign._optimizer_steps(optimizer(counts))


def receipt_fixture(operator, manifest=None):
    name, _, run = campaign.validate_campaign(manifest or campaign.MANIFESTS[operator])
    interval = run["alg_params"]["inner_solve_interval"]
    runtime = {"python": "3.10", "torch": "2.3.1", "cuda": "12.1", "executable": "/locked/python", "gpu": "L40S"}
    binding = {"commit": "pinned", "manifest_sha256": "manifest", "config_sha256": {name: "config"}}
    warmup, decisions = campaign.smoke_defaults(operator)
    solves = 0 if operator == "none" else decisions // interval
    held = 0 if operator == "none" else decisions - solves
    cases = [{"passed": True, "binding": binding, "runtime": runtime, "config": name,
              "operator": operator, "inner_solve_interval": interval, "physical_cores": count,
              "compile_fallback": False, "replay_storage_device": "cpu",
              "compile_graph_counts": {"before": 9, "after": 9}, "auxiliary_critic_present": False,
              "reward_only_targets": True, "primary_critic_sources": True, "outer_actor_entropy_auto": True,
              "checkpoint_roundtrip": True, "checkpoint_sidecars": True, "boundary_training_state_roundtrip": True,
              "outer_target_entropy": run["alg_params"]["target_entropy"], "warmup_active_decisions": warmup,
              "outer_actor_optimizer_steps": 2 + warmup + decisions, "outer_temperature_optimizer_steps": 2 + warmup + decisions,
              "mppi_primary_online_mean_pair_no_entropy_verified": operator == "mppi",
              "sampled_prior_execution_verified": operator == "none", "zero_inner_work_verified": operator == "none",
              "measured": {"decisions": decisions, "outer_updates": decisions, "seconds_per_decision": .3,
                           "solve_decisions": solves, "held_decisions": held, "prior_decisions": decisions if operator == "none" else 0,
                           **campaign.production_projection(.3, run["total_steps"])}}
             for count in campaign.cpu_counts(operator)]
    receipt = {"passed": True, "binding": binding, "runtime": runtime, "cases": cases, "artifacts": {}}
    return receipt, binding, runtime, dict(name=name, operator=operator, interval=interval)


@pytest.mark.parametrize("manifest", list(campaign.CAMPAIGNS))
def test_receipt_requires_exact_semantics_complete_training_and_measured_resources(manifest):
    operator = campaign.CAMPAIGNS[manifest][1]
    receipt, binding, runtime, identity = receipt_fixture(operator, manifest)
    for count in campaign.cpu_counts(operator):
        campaign.validate_receipt(receipt, binding, runtime, count, **identity)
    with pytest.raises(AssertionError, match="configuration"):
        campaign.validate_receipt(receipt, {**binding, "commit": "other"}, runtime, 1, **identity)
    with pytest.raises(AssertionError, match="runtime"):
        campaign.validate_receipt(receipt, binding, {**runtime, "gpu": "A100"}, 1, **identity)
    with pytest.raises(AssertionError, match="not measured"):
        campaign.validate_receipt(receipt, binding, runtime, 3, **identity)
    mutations = [
        lambda r: r["cases"].pop(),
        lambda r: r["cases"][0].update(auxiliary_critic_present=True),
        lambda r: r["cases"][0].update(replay_storage_device="cuda:0"),
        lambda r: r["cases"][0].update(reward_only_targets=False),
        lambda r: r["cases"][0].update(outer_temperature_optimizer_steps=133),
        lambda r: r["cases"][0]["measured"].update(outer_updates=119),
        lambda r: r["cases"][0]["measured"].update(held_decisions=1),
        lambda r: r["cases"][0]["compile_graph_counts"].update(after=10),
        lambda r: r["cases"][0].update(outer_target_entropy=-9.),
        lambda r: r["cases"][0]["measured"].update(production_total_steps=123),
        lambda r: r["cases"][0]["measured"].update(projected_production_seconds=1.),
    ]
    if operator == "mppi":
        mutations.append(lambda r: r["cases"][0].update(mppi_primary_online_mean_pair_no_entropy_verified=False))
    if operator == "none":
        mutations.extend([
            lambda r: r["cases"][0].update(sampled_prior_execution_verified=False),
            lambda r: r["cases"][0].update(zero_inner_work_verified=False),
            lambda r: r["cases"][0]["measured"].update(prior_decisions=399),
            lambda r: r["cases"][0]["measured"].update(solve_decisions=400),
        ])
    for mutate in mutations:
        bad = deepcopy(receipt)
        mutate(bad)
        with pytest.raises(AssertionError):
            campaign.validate_receipt(bad, binding, runtime, 1, **identity)


@pytest.mark.parametrize("manifest", list(campaign.CAMPAIGNS))
@pytest.mark.parametrize("restart", [0, 1])
@pytest.mark.parametrize("storage_root", list(campaign.STORAGE_ROOTS))
def test_launch_delegates_exact_recipe_and_lineage_to_canonical_resume(monkeypatch, tmp_path, manifest, restart, storage_root):
    operator = campaign.CAMPAIGNS[manifest][1]
    receipt, binding, runtime, identity = receipt_fixture(operator, manifest)
    gate = tmp_path / "receipt.json"
    campaign.write_new(gate, receipt)
    monkeypatch.setattr(campaign, "binding", lambda path: binding)
    monkeypatch.setattr(campaign, "runtime", lambda: runtime)
    for key, value in {"SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_ID": "123",
                       "SLURM_RESTART_COUNT": str(restart), "AMBI_DRAIN_AFTER_SECONDS": "3600",
                       "AMBI_DURABLE_ROOT": storage_root,
                       "AMBI_DURABLE_QUOTA_LABEL": "rgao48",
                       "AMBI_DURABLE_QUOTA_PATH": campaign.STORAGE_ROOTS[storage_root],
                       "AMBI_LINEAGE_DIR": storage_root + "/return-only-test"}.items():
        monkeypatch.setenv(key, value)
    for key in ("AMBI_RUN_CONFIG", "AMBI_ALG_DIR", "AMBI_PYTHON", "SLURM_SUBMIT_DIR"):
        monkeypatch.setenv(key, "test-placeholder")
    executions = []
    monkeypatch.setattr(campaign.os, "execv", lambda *args: executions.append(args))
    args = Namespace(manifest=manifest, gate=gate, output=tmp_path / "campaign")
    campaign.launch(args)
    record = campaign.read(args.output / f"launch-123.{restart}.json")
    assert record["run_params"] == campaign.recipe(0, args.manifest)[2]
    assert record["resume_mode"] == ("required" if restart else "new")
    assert "wandb_run_id" not in record
    assert record["storage"]["AMBI_DURABLE_ROOT"] == storage_root
    assert record["storage"]["AMBI_DURABLE_QUOTA_PATH"] == campaign.STORAGE_ROOTS[storage_root]
    assert executions == [("/bin/bash", ["/bin/bash", str(campaign.ROOT / "run_ambi_oscar.sh")])]
    assert campaign.os.environ["AMBI_RUN_CONFIG"] == str(args.manifest)


@pytest.mark.parametrize("root,quota_path", campaign.STORAGE_ROOTS.items())
def test_storage_selection_requires_approved_pair_and_contained_lineage(root, quota_path):
    campaign.validate_storage_selection(root, root + "/new-lineage", "rgao48", quota_path)
    for values in (
        (root, root + "/lineage", "data+rbalestr", quota_path),
        (root, root + "/lineage", "rgao48", "/oscar/other"),
        (root, root + "/lineage", "rgao48", "/oscar/scratch" if quota_path == "/oscar/home" else "/oscar/home"),
        (root, root, "rgao48", quota_path),
        (root, root + "/../outside", "rgao48", quota_path),
        (root, str(campaign.ROOT / "lineage"), "rgao48", quota_path),
        (str(campaign.ROOT), str(campaign.ROOT / "lineage"), "rgao48", quota_path),
        ("/tmp/unapproved", "/tmp/unapproved/lineage", "rgao48", quota_path),
    ):
        with pytest.raises((AssertionError, ValueError)):
            campaign.validate_storage_selection(*values)


def test_wrapper_preserves_explicit_resources_and_online_lineage_identity():
    path = campaign.ROOT / "slurm/run_ambi_reward_training_oscar.sbatch"
    subprocess.run(["bash", "-n", str(path)], check=True)
    source = path.read_text()
    directives = [line for line in source.splitlines() if line.startswith("#SBATCH")]
    assert not any(option in line for line in directives
                   for option in ("--time", "--cpus-per-task", "--gres", "--mem", "--array"))
    assert "#SBATCH --hint=nomultithread" in directives and "#SBATCH --requeue" in directives
    assert '"${AMBI_RETURN_MANIFEST:?' in source and '"${AMBI_DRAIN_AFTER_SECONDS:?' in source
    assert 'unset WANDB_RUN_ID WANDB_RESUME' in source
    assert 'WANDB_MODE=online' in source and 'WANDB_MODE=disabled' in source
    assert 'slurm/ambi_reward_training.py train' in source
    assert '${AMBI_DURABLE_ROOT:-/oscar/home/rgao48/ambi-durable}' in source
    for root, quota_path in campaign.STORAGE_ROOTS.items():
        assert f'{root}) expected_quota_path={quota_path}' in source
