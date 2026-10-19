"""The single K3 campaign stays matched to the deployed aux-return recipe."""

from argparse import Namespace
from copy import deepcopy
import subprocess

import pytest

from slurm import ambi_held_training as campaign
from tests.test_ambi_root_local_sac import _build_cfg


def test_only_cadence_and_truthful_publication_identity_change():
    base = campaign.read(campaign.ROOT / "configs/dmcontrol/algs/ambi_full_h3_j8_fresh.json")
    name, manifest, run = campaign.validate_campaign()
    assert name == campaign.NAME
    expected = deepcopy(base)
    expected["alg_params"].update(inner_solve_interval=3, aux_return_joint_vmap=False)
    for field in ("wandb_tags", "wandb_group", "wandb_run_name"):
        expected["alg_params"][field] = run["alg_params"][field]
    expected["name"] = name
    assert run == expected
    p = run["alg_params"]
    cfg = _build_cfg(**p)
    assert cfg.inner_hold_encoder == "solve_snapshot"
    assert cfg.inner_model_step_budget == 3072
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
    assert cfg.inner_actor_scope == cfg.inner_critic_scope == cfg.inner_replay_scope == "action"
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_temperature_initialization == "inherit_outer"
    assert cfg.inner_entropy_enabled and cfg.inner_finite_horizon
    assert cfg.aux_return_detach_representation is False
    assert manifest["configs"] == [name] and manifest["trials"] == 1
    assert manifest["checkpoint_every"] == run["checkpoint_every"] == 25000
    assert manifest["save_strat"] == run["save_strat"] == "all"
    assert p["wandb_project"] == "ambi" and p["wandb_group"] == "ambi-held-training-20261018"
    assert p["wandb_run_name"] == "H3 | Fresh | J8 | K3 | aux-return | seed55"
    assert {"K3", "held-feedback-policy", "solve-snapshot-encoder", "return-return", "exact-resume"} <= set(p["wandb_tags"])
    assert "alg_params" not in manifest["overrides_alg"]


def receipt_fixture():
    runtime = {"python": "3.10", "torch": "2.3.1", "cuda": "12.1", "executable": "/locked/python", "gpu": "L40S"}
    binding = {"commit": "pinned", "config_sha256": {campaign.NAME: "config"}, "manifest_sha256": "manifest"}
    cases = [{"passed": True, "binding": binding, "runtime": runtime,
              "config": campaign.NAME, "physical_cores": cpu,
              "compile_fallback": False, "replay_storage_device": "cpu",
              "compile_graph_counts": {"before": 9, "after": 9},
              "measured": {"decisions": 120, "outer_updates": 120, "seconds_per_decision": .3}}
             for cpu in campaign.CPU_COUNTS]
    cuda = {"passed": True, "binding": binding, "runtime": runtime, "config": campaign.NAME,
            "interval": 3, "bernoulli_probability": 0., "compile_strict": True,
            "fallbacks": 0, "checkpoint_roundtrip": True, "boundary_training_state_roundtrip": True}
    return {"passed": True, "binding": binding, "runtime": runtime,
            "cuda_gate": cuda, "cases": cases, "artifacts": {}}, binding, runtime


def test_receipt_requires_matching_source_gpu_actual_cpu_replay_and_complete_work():
    receipt, binding, runtime = receipt_fixture()
    for cpu in campaign.CPU_COUNTS:
        campaign.validate_receipt(receipt, binding, runtime, cpu)
    for altered in ({**binding, "commit": "other"}, {**binding, "manifest_sha256": "other"}):
        with pytest.raises(AssertionError, match="source/configuration"):
            campaign.validate_receipt(receipt, altered, runtime, 1)
    with pytest.raises(AssertionError, match="another gpu"):
        campaign.validate_receipt(receipt, binding, {**runtime, "gpu": "A100"}, 1)
    with pytest.raises(AssertionError, match="not measured"):
        campaign.validate_receipt(receipt, binding, runtime, 4)
    mutations = (
        lambda x: x["cases"].pop(),
        lambda x: x["cases"][0].update(replay_storage_device="cuda:0"),
        lambda x: x["cases"][0]["compile_graph_counts"].update(after=10),
        lambda x: x["cases"][0]["measured"].update(outer_updates=119),
        lambda x: x["cuda_gate"].update(interval=1),
        lambda x: x["cuda_gate"].update(checkpoint_roundtrip=False),
        lambda x: x["cuda_gate"].update(boundary_training_state_roundtrip=False),
    )
    for mutate in mutations:
        bad = deepcopy(receipt)
        mutate(bad)
        with pytest.raises(AssertionError):
            campaign.validate_receipt(bad, binding, runtime, 1)


@pytest.mark.parametrize("restart", [0, 1])
def test_launch_delegates_durable_resume_without_new_run_id(monkeypatch, tmp_path, restart):
    receipt, binding, runtime = receipt_fixture()
    gate = tmp_path / "receipt.json"
    campaign.write_new(gate, receipt)
    monkeypatch.setattr(campaign, "binding", lambda path: binding)
    monkeypatch.setattr(campaign, "runtime", lambda: runtime)
    for key, value in {"SLURM_CPUS_PER_TASK": "2", "SLURM_JOB_ID": "123",
                       "SLURM_RESTART_COUNT": str(restart), "AMBI_DRAIN_AFTER_SECONDS": "3600",
                       "AMBI_DURABLE_ROOT": "/oscar/home/rgao48/ambi-durable",
                       "AMBI_LINEAGE_DIR": "/oscar/home/rgao48/ambi-durable/held-test"}.items():
        monkeypatch.setenv(key, value)
    for key in ("AMBI_RUN_CONFIG", "AMBI_ALG_DIR", "AMBI_PYTHON", "SLURM_SUBMIT_DIR"):
        monkeypatch.setenv(key, "test-placeholder")
    executions = []
    monkeypatch.setattr(campaign.os, "execv", lambda *args: executions.append(args))
    args = Namespace(manifest=campaign.MANIFEST, gate=gate, output=tmp_path / "campaign")
    campaign.launch(args)
    record = campaign.read(args.output / f"launch-123.{restart}.json")
    assert record["run_params"] == campaign.recipe(0, campaign.MANIFEST)[2]
    assert record["resume_mode"] == ("required" if restart else "new")
    assert "wandb_run_id" not in record
    assert executions == [("/bin/bash", ["/bin/bash", str(campaign.ROOT / "run_ambi_oscar.sh")])]
    assert campaign.os.environ["AMBI_RUN_CONFIG"] == str(campaign.MANIFEST)
    with pytest.raises(FileExistsError):
        campaign.launch(args)


def test_oscar_wrapper_has_explicit_resources_one_cell_and_canonical_resume():
    path = campaign.ROOT / "slurm/run_ambi_held_training_oscar.sbatch"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    directives = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert not any(option in line for line in directives
                   for option in ("--time", "--cpus-per-task", "--gres", "--mem", "--array"))
    assert "#SBATCH --requeue" in directives and "#SBATCH --hint=nomultithread" in directives
    assert 'AMBI_DRAIN_AFTER_SECONDS:?' in text and 'AMBI_LINEAGE_DIR:?' in text
    assert 'git rev-parse HEAD' in text and 'git status --porcelain --untracked-files=all' in text
    assert '-z "${SLURM_ARRAY_TASK_ID:-}"' in text
    assert 'unset WANDB_RUN_ID WANDB_RESUME' in text
    assert 'WANDB__REQUIRE_CORE=true' in text and 'WANDB_DIR="$wandb_root"' in text
    assert 'slurm/ambi_held_training.py train' in text
