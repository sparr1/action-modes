"""The reward-only GPU gate binds one unchanged recipe to durable resume."""

from argparse import Namespace
from copy import deepcopy
import subprocess
from types import SimpleNamespace

import pytest

from slurm import ambi_reward_training as campaign


@pytest.mark.parametrize("operator", ["sac", "mppi"])
def test_actual_campaigns_validate_and_smoke_preserves_production_shapes(operator):
    path = campaign.MANIFESTS[operator]
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


def receipt_fixture(operator):
    name, _, run = campaign.validate_campaign(campaign.MANIFESTS[operator])
    interval = run["alg_params"]["inner_solve_interval"]
    runtime = {"python": "3.10", "torch": "2.3.1", "cuda": "12.1", "executable": "/locked/python", "gpu": "L40S"}
    binding = {"commit": "pinned", "manifest_sha256": "manifest", "config_sha256": {name: "config"}}
    cases = [{"passed": True, "binding": binding, "runtime": runtime, "config": name,
              "operator": operator, "inner_solve_interval": interval, "physical_cores": count,
              "compile_fallback": False, "replay_storage_device": "cpu",
              "compile_graph_counts": {"before": 9, "after": 9}, "auxiliary_critic_present": False,
              "reward_only_targets": True, "primary_critic_sources": True, "outer_actor_entropy_auto": True,
              "checkpoint_roundtrip": True, "checkpoint_sidecars": True, "boundary_training_state_roundtrip": True,
              "outer_target_entropy": -10.5, "warmup_active_decisions": 12,
              "outer_actor_optimizer_steps": 134, "outer_temperature_optimizer_steps": 134,
              "mppi_primary_online_mean_pair_no_entropy_verified": operator == "mppi",
              "measured": {"decisions": 120, "outer_updates": 120, "seconds_per_decision": .3,
                           "solve_decisions": 120 // interval, "held_decisions": 120 - 120 // interval}}
             for count in campaign.CPU_COUNTS]
    receipt = {"passed": True, "binding": binding, "runtime": runtime, "cases": cases, "artifacts": {}}
    return receipt, binding, runtime, dict(name=name, operator=operator, interval=interval)


@pytest.mark.parametrize("operator", ["sac", "mppi"])
def test_receipt_requires_exact_semantics_complete_training_and_measured_resources(operator):
    receipt, binding, runtime, identity = receipt_fixture(operator)
    for count in campaign.CPU_COUNTS:
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
    ]
    if operator == "mppi":
        mutations.append(lambda r: r["cases"][0].update(mppi_primary_online_mean_pair_no_entropy_verified=False))
    for mutate in mutations:
        bad = deepcopy(receipt)
        mutate(bad)
        with pytest.raises(AssertionError):
            campaign.validate_receipt(bad, binding, runtime, 1, **identity)


@pytest.mark.parametrize("operator", ["sac", "mppi"])
@pytest.mark.parametrize("restart", [0, 1])
def test_launch_delegates_exact_recipe_and_lineage_to_canonical_resume(monkeypatch, tmp_path, operator, restart):
    receipt, binding, runtime, identity = receipt_fixture(operator)
    gate = tmp_path / "receipt.json"
    campaign.write_new(gate, receipt)
    monkeypatch.setattr(campaign, "binding", lambda path: binding)
    monkeypatch.setattr(campaign, "runtime", lambda: runtime)
    for key, value in {"SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_ID": "123",
                       "SLURM_RESTART_COUNT": str(restart), "AMBI_DRAIN_AFTER_SECONDS": "3600",
                       "AMBI_DURABLE_ROOT": "/oscar/home/rgao48/ambi-durable",
                       "AMBI_LINEAGE_DIR": "/oscar/home/rgao48/ambi-durable/return-only-test"}.items():
        monkeypatch.setenv(key, value)
    for key in ("AMBI_RUN_CONFIG", "AMBI_ALG_DIR", "AMBI_PYTHON", "SLURM_SUBMIT_DIR"):
        monkeypatch.setenv(key, "test-placeholder")
    executions = []
    monkeypatch.setattr(campaign.os, "execv", lambda *args: executions.append(args))
    args = Namespace(manifest=campaign.MANIFESTS[operator], gate=gate, output=tmp_path / "campaign")
    campaign.launch(args)
    record = campaign.read(args.output / f"launch-123.{restart}.json")
    assert record["run_params"] == campaign.recipe(0, args.manifest)[2]
    assert record["resume_mode"] == ("required" if restart else "new")
    assert "wandb_run_id" not in record
    assert executions == [("/bin/bash", ["/bin/bash", str(campaign.ROOT / "run_ambi_oscar.sh")])]
    assert campaign.os.environ["AMBI_RUN_CONFIG"] == str(args.manifest)


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
