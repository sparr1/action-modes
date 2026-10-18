"""Verify the one-cell soft-SAC launch gate and explicit resource contract."""

from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import pytest

from slurm import ambi_soft_training as campaign
from tests.test_ambi_root_local_sac import _build_cfg


def test_one_cell_gate_preserves_full_inner_shapes_and_checks_primary_source():
    assert campaign.campaign_name() == "ambi_full_h3_j8_soft"
    name, manifest, run = campaign.recipe(0, campaign.MANIFEST)
    cfg = _build_cfg(**run["alg_params"])
    critic = object()
    agent = SimpleNamespace(aux_return=None,
                            model=SimpleNamespace(_Qs=critic, state_dict=lambda: {}),
                            inner_engine=SimpleNamespace(_critic_base=critic, _horizon_critic=critic))
    campaign.validate_recipe(cfg, agent)
    cfg.inner_sac_critic_target = "reward_only"
    with pytest.raises(AssertionError):
        campaign.validate_recipe(cfg, agent)


def test_gpu_gate_only_shortens_collection_and_disables_external_logging():
    name, original_manifest, original = campaign.recipe(0, campaign.MANIFEST)
    selected, manifest, run, overrides = campaign.smoke_recipe(20, 128)
    expected = deepcopy(original)
    expected["alg_params"].update(wandb=False, seed_steps=500, pretrain_steps=2, buffer_size=4096)
    expected.update(total_steps=649, checkpoint_every=649, save_strat="all")
    assert selected == name and run == expected
    assert overrides == {"wandb": False, "seed_steps": 500, "pretrain_steps": 2, "buffer_size": 4096}
    expected_manifest = deepcopy(original_manifest)
    expected_manifest.update(checkpoint_every=649, save_strat="all", save_trials="none", logs="none")
    assert manifest == expected_manifest


def test_fresh_inner_metrics_accept_absent_bernoulli_diagnostic_but_reject_transfer():
    metrics = {"inner_critic_optimizer_steps": 128, "inner_actor_optimizer_steps": 32,
               "inner_compile_fallback": 0}
    campaign.validate_inner_metrics(metrics)
    campaign.validate_inner_metrics({**metrics, "inner_critic_bernoulli_applied": 0.})
    with pytest.raises(AssertionError):
        campaign.validate_inner_metrics({**metrics, "inner_critic_bernoulli_applied": 1.})


def receipt_fixture():
    runtime = {"python": "3.10", "torch": "2.3.1", "cuda": "12.1", "executable": "/locked/python", "gpu": "L40S"}
    binding = {"commit": "pinned", "manifest_sha256": "manifest", "config_sha256": {"soft": "a"}}
    receipt = {"passed": True, "runtime": runtime, "binding": binding,
               "cases": [{"physical_cores": cores, "passed": True} for cores in (1, 2, 4)]}
    return receipt, binding, runtime


def test_gpu_receipt_rejects_stale_config_unmeasured_cpu_and_other_gpu():
    receipt, binding, runtime = receipt_fixture()
    for cores in (1, 2, 4):
        campaign.validate_receipt(receipt, binding, runtime, cores)
    with pytest.raises(AssertionError, match="configuration"):
        campaign.validate_receipt(receipt, {**binding, "commit": "another"}, runtime, 1)
    with pytest.raises(AssertionError, match="another gpu"):
        campaign.validate_receipt(receipt, binding, {**runtime, "gpu": "A100"}, 1)
    with pytest.raises(AssertionError, match="not measured"):
        campaign.validate_receipt(receipt, binding, runtime, 8)
    broken = deepcopy(receipt)
    broken["cases"][-1]["passed"] = False
    with pytest.raises(AssertionError):
        campaign.validate_receipt(broken, binding, runtime, 1)
    broken = deepcopy(receipt)
    broken["cases"] = broken["cases"][:-1]
    with pytest.raises(AssertionError):
        campaign.validate_receipt(broken, binding, runtime, 1)


def test_production_launch_delegates_unchanged_recipe(monkeypatch, tmp_path):
    receipt, binding, runtime = receipt_fixture()
    gate = tmp_path / "gate.json"
    campaign.write_new(gate, receipt)
    monkeypatch.setattr(campaign, "binding", lambda path: binding)
    monkeypatch.setattr(campaign, "runtime", lambda: runtime)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("WANDB_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("WANDB_RUN_ID", "test-run")
    executions = []
    monkeypatch.setattr(campaign.os, "execv", lambda *args: executions.append(args))
    output = tmp_path / "production"
    args = Namespace(manifest=campaign.MANIFEST, gate=gate, output=output)
    campaign.launch(args)
    launched = campaign.read(output / "launch.json")
    assert launched["run_params"] == campaign.recipe(0, campaign.MANIFEST)[2]
    assert launched["config"] == "ambi_full_h3_j8_soft"
    assert executions == [(campaign.sys.executable,
        [campaign.sys.executable, "main.py", "--run", str(campaign.MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
         "--log-dir", str(output), "--alg-index", "0", "--trial-index", "0", "--num-runs", "1"])]
    with pytest.raises(FileExistsError):
        campaign.launch(args)


def test_oscar_launcher_requires_resources_exact_source_and_persistent_core_journal():
    text = (campaign.ROOT / "slurm/run_ambi_soft_training_oscar.sbatch").read_text()
    directives = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert not any(option in line for line in directives
                   for option in ("--time", "--cpus-per-task", "--gres", "--mem"))
    assert '#SBATCH --hint=nomultithread' in directives
    assert 'SLURM_CPUS_PER_TASK:?' in text
    assert 'git rev-parse HEAD' in text and 'git status --porcelain --untracked-files=all' in text
    assert '-z "${SLURM_ARRAY_TASK_ID:-}"' in text
    assert 'slurm/ambi_soft_training.py train' in text
    assert 'WANDB__REQUIRE_CORE=true' in text
    assert 'wandb_root="$AMBI_SOFT_OUTPUT/wandb-job-${SLURM_JOB_ID}"' in text
    assert 'WANDB_DIR="$wandb_root"' in text
