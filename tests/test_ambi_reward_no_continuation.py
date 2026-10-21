"""Ordinary replay launches cannot reuse continuation gates or restart outputs."""

from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from slurm import ambi_reward_training as campaign


def gpu_receipt():
    manifest = campaign.ENTROPY_MANIFESTS[-10.5]
    name, _, run = campaign.validate_campaign(manifest)
    binding = {"commit": "tested", "manifest_sha256": "manifest"}
    runtime = {"gpu": "L40S", "executable": "/locked/python"}
    policy = campaign.execution_policy("off")
    cases = []
    for cores in (1, 2, 4):
        cases.append({
            "passed": True, "binding": binding, "runtime": runtime, "config": name,
            "execution_policy": policy, "operator": "none", "inner_solve_interval": 1,
            "physical_cores": cores, "compile_fallback": False, "replay_storage_device": "cuda:0",
            "compile_graph_counts": {"before": 4, "after": 4}, "auxiliary_critic_present": False,
            "reward_only_targets": True, "primary_critic_sources": True, "outer_actor_entropy_auto": True,
            "checkpoint_roundtrip": True, "checkpoint_sidecars": True, "boundary_training_state_roundtrip": True,
            "outer_target_entropy": -10.5, "warmup_active_decisions": 24,
            "outer_actor_optimizer_steps": 426, "outer_temperature_optimizer_steps": 426,
            "sampled_prior_execution_verified": True, "zero_inner_work_verified": True,
            "measured": {"decisions": 400, "outer_updates": 400, "seconds_per_decision": .02,
                         "solve_decisions": 0, "held_decisions": 0, "prior_decisions": 400,
                         **campaign.production_projection(.02, run["total_steps"])},
        })
    receipt = {"passed": True, "binding": binding, "runtime": runtime, "execution_policy": policy,
               "cases": cases, "artifacts": {}}
    identity = dict(name=name, operator="none", interval=1)
    return manifest, receipt, binding, runtime, identity


def test_gpu_gate_is_bound_to_continuation_and_replay_policy():
    _, receipt, binding, runtime, identity = gpu_receipt()
    campaign.validate_receipt(receipt, binding, runtime, 1, continuation="off", **identity)
    with pytest.raises(AssertionError, match="policy mismatch"):
        campaign.validate_receipt(receipt, binding, runtime, 1, **identity)
    mutations = (
        lambda value: value.pop("execution_policy"),
        lambda value: value["cases"][0].pop("execution_policy"),
        lambda value: value["cases"][0].update(replay_storage_device="cpu"),
        lambda value: value.update(execution_policy=campaign.execution_policy()),
        lambda value: value["cases"][0].update(execution_policy=campaign.execution_policy()),
    )
    for mutate in mutations:
        bad = deepcopy(receipt)
        mutate(bad)
        with pytest.raises(AssertionError):
            campaign.validate_receipt(bad, binding, runtime, 1, continuation="off", **identity)


@pytest.fixture
def launch_context(monkeypatch, tmp_path):
    manifest, receipt, binding, runtime, identity = gpu_receipt()
    gate = tmp_path / "receipt.json"
    campaign.write_new(gate, receipt)
    args = Namespace(manifest=manifest, gate=gate, output=tmp_path / "campaign", continuation="off")
    monkeypatch.setattr(campaign, "binding", lambda path: binding)
    monkeypatch.setattr(campaign, "runtime", lambda: runtime)
    for key, value in {"SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "0",
                       "WANDB_RUN_ID": "old-run", "WANDB_RESUME": "must"}.items():
        monkeypatch.setenv(key, value)
    for key in ("AMBI_LINEAGE_DIR", "AMBI_DRAIN_AFTER_SECONDS", "AMBI_DURABLE_ROOT",
                "AMBI_DURABLE_QUOTA_LABEL", "AMBI_DURABLE_QUOTA_PATH"):
        monkeypatch.delenv(key, raising=False)
    requests, executions = [], []

    def show_job(command, **kwargs):
        requests.append(command)
        return "JobId=123 Requeue=0 Restarts=0\n"

    monkeypatch.setattr(campaign.subprocess, "check_output", show_job)
    monkeypatch.setattr(campaign.os, "execv", lambda *values: executions.append(values))
    return args, requests, executions


def test_launch_keeps_recipe_and_model_snapshots_without_lineage(monkeypatch, launch_context):
    args, requests, executions = launch_context
    campaign.launch(args)
    assert requests == [["scontrol", "show", "job", "123", "--oneliner"]]
    executable, command = executions[0]
    assert executable == sys.executable
    record = campaign.read(args.output / "training/launch.json")
    assert command == [sys.executable, "main.py", "--run", record["generated_manifest"],
                       "--alg-dir", "configs/dmcontrol/algs", "--log-dir", str(args.output / "training"),
                       "--num-runs", "1"]
    expected = campaign.recipe(0, args.manifest)[2]
    document, published = campaign.noncontinuation_manifest(args.manifest, expected)
    assert record["run_params"] == published
    assert campaign.read(record["generated_manifest"]) == document
    assert record["generated_manifest_sha256"] == campaign.digest(record["generated_manifest"])
    assert expected["checkpoint_every"] == 25000 and expected["save_strat"] == "all"
    assert expected["total_steps"] == 1000000
    assert record["execution_policy"] == campaign.execution_policy("off")
    assert record["wandb_run_id"] == campaign.os.environ["WANDB_RUN_ID"] != "old-run"
    assert "WANDB_RESUME" not in campaign.os.environ
    assert "lineage_dir" not in record
    with pytest.raises(FileExistsError):
        campaign.launch(args)
    assert len(executions) == 1


@pytest.mark.parametrize("manifest", list(campaign.CAMPAIGNS))
def test_noncontinuation_publication_preserves_all_scientific_settings(manifest):
    name, original_document, original_run = campaign.validate_campaign(manifest)
    document, published = campaign.noncontinuation_manifest(manifest, original_run)
    assert "durable exact-resume generations" not in document["study_note"]
    assert "exact-resume" not in published["alg_params"]["wandb_tags"]
    assert "continuation-off" in published["alg_params"]["wandb_tags"]
    # Reproduce main.py's shallow override, including complete alg_params.
    resolved = {"name": name, **campaign.read(campaign.ROOT / "configs/dmcontrol/algs" / (name + ".json")),
                **document["overrides_alg"]}
    assert resolved == published
    resolved.pop("execution_policy")
    resolved["alg_params"]["wandb_tags"] = original_run["alg_params"]["wandb_tags"]
    assert resolved == original_run
    document["study_note"] = original_document["study_note"]
    document["overrides_alg"] = original_document["overrides_alg"]
    assert document == original_document


@pytest.mark.parametrize("scheduler_reply", ["JobId=123 Requeue=1", "JobId=123", "JobId=123 Requeue=0 Requeue=1"])
def test_launch_requires_scheduler_no_requeue_before_output_creation(monkeypatch, launch_context, scheduler_reply):
    args, _, executions = launch_context
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: scheduler_reply)
    with pytest.raises(AssertionError, match="no-requeue"):
        campaign.launch(args)
    assert not args.output.exists() and not executions


def test_launch_rejects_restart_before_output_creation(monkeypatch, launch_context):
    args, requests, executions = launch_context
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    with pytest.raises(AssertionError, match="cannot restart"):
        campaign.launch(args)
    assert not args.output.exists() and not requests and not executions


@pytest.mark.parametrize("output", [Path("relative-output"), campaign.ROOT, campaign.ROOT / "logs"])
def test_launch_rejects_nonexternal_output(launch_context, output):
    args, _, executions = launch_context
    args.output = output
    with pytest.raises(AssertionError, match="outside the source"):
        campaign.launch(args)
    assert not executions
