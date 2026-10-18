"""Keep the MPPI gate bound to both production recipes and measured resources."""

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest

from slurm import ambi_mppi_training as campaign
from tests.test_ambi_root_local_sac import _build_cfg


def test_two_cell_campaign_and_production_planner_shapes():
    names = campaign.campaign_names()
    assert names == ["ambi_full_h3_mppi_aux_return", "ambi_full_h3_mppi_soft"]
    for index, name in enumerate(names):
        selected, manifest, run = campaign.recipe(index, campaign.MANIFEST)
        assert selected == name and "alg_params" not in manifest["overrides_alg"]
        assert run["total_steps"] == 500000 and run["seed"] == 55
        assert run["checkpoint_every"] == manifest["checkpoint_every"] == 25000
        assert run["save_strat"] == manifest["save_strat"] == "all"
        cfg = _build_cfg(**run["alg_params"])
        assert (cfg.inner_operator, cfg.inner_rollout_horizon, cfg.inner_mppi_iterations) == ("mppi", 3, 8)
        assert (cfg.inner_mppi_num_samples, cfg.inner_mppi_num_elites, cfg.inner_mppi_num_pi_trajs) == (512, 64, 24)
        assert cfg.inner_critic_updates_per_action == cfg.inner_actor_updates_per_action == 0
        assert cfg.inner_mppi_eval_action == "weighted_elite" and cfg.inner_mppi_warm_start_scope == "episode"
        assert cfg.aux_return_mode == ("sac" if index == 0 else "off")
        assert cfg.inner_horizon_critic_source == ("aux_return" if index == 0 else "sac")


@pytest.mark.parametrize("index", [0, 1])
def test_gpu_gate_only_shortens_collection_and_disables_external_logging(index):
    name, original_manifest, original = campaign.recipe(index, campaign.MANIFEST)
    selected, manifest, run, overrides = campaign.smoke_recipe(index, 20, 128)
    expected = deepcopy(original)
    expected["alg_params"].update(wandb=False, seed_steps=500, pretrain_steps=2, buffer_size=4096)
    expected.update(total_steps=649, checkpoint_every=649, save_strat="all")
    assert selected == name and run == expected
    assert overrides == {"wandb": False, "seed_steps": 500, "pretrain_steps": 2, "buffer_size": 4096}
    expected_manifest = deepcopy(original_manifest)
    expected_manifest.update(checkpoint_every=649, save_strat="all", save_trials="none", logs="none")
    assert manifest == expected_manifest


def receipt_fixture():
    runtime = {"python": "3.10", "torch": "2.3.1", "cuda": "12.1", "executable": "/locked/python", "gpu": "H100"}
    binding = {"commit": "pinned", "manifest_sha256": "manifest", "config_sha256": {"aux": "a", "soft": "b"}}
    receipt = {"passed": True, "runtime": runtime, "binding": binding,
               "cases": [{"index": index, "physical_cores": cores, "passed": True}
                         for index in (0, 1) for cores in (1, 2)]}
    return receipt, binding, runtime


def test_gpu_receipt_rejects_stale_config_unmeasured_cpu_and_other_gpu():
    receipt, binding, runtime = receipt_fixture()
    campaign.validate_receipt(receipt, binding, runtime, 0, 1)
    campaign.validate_receipt(receipt, binding, runtime, 1, 2)
    with pytest.raises(AssertionError, match="configuration"):
        campaign.validate_receipt(receipt, {**binding, "commit": "another"}, runtime, 0, 1)
    with pytest.raises(AssertionError, match="another gpu"):
        campaign.validate_receipt(receipt, binding, {**runtime, "gpu": "L40S"}, 0, 1)
    with pytest.raises(AssertionError, match="not measured"):
        campaign.validate_receipt(receipt, binding, runtime, 0, 8)
    broken = deepcopy(receipt)
    broken["cases"][-1]["passed"] = False
    with pytest.raises(AssertionError):
        campaign.validate_receipt(broken, binding, runtime, 0, 1)
    broken = deepcopy(receipt)
    broken["cases"] = broken["cases"][:-1]
    with pytest.raises(AssertionError):
        campaign.validate_receipt(broken, binding, runtime, 0, 1)


def test_production_launch_delegates_unchanged_selected_recipe(monkeypatch, tmp_path):
    receipt, binding, runtime = receipt_fixture()
    gate = tmp_path / "gate.json"
    campaign.write_new(gate, receipt)
    monkeypatch.setattr(campaign, "binding", lambda path: binding)
    monkeypatch.setattr(campaign, "runtime", lambda: runtime)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("WANDB_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("WANDB_RUN_ID", "test-run")
    executions = []
    monkeypatch.setattr(campaign.os, "execv", lambda *args: executions.append(args))
    output = tmp_path / "production"
    args = Namespace(manifest=campaign.MANIFEST, gate=gate, index=1, output=output)
    campaign.launch(args)
    launched = campaign.read(output / "launch.json")
    assert launched["run_params"] == campaign.recipe(1, campaign.MANIFEST)[2]
    assert launched["config"] == "ambi_full_h3_mppi_soft"
    assert executions == [(campaign.sys.executable,
        [campaign.sys.executable, "main.py", "--run", str(campaign.MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
         "--log-dir", str(output), "--alg-index", "1", "--trial-index", "0", "--num-runs", "1"])]
    with pytest.raises(FileExistsError):
        campaign.launch(args)


def test_oscar_launcher_requires_resource_choices_and_exact_source():
    text = (Path(campaign.ROOT) / "slurm/run_ambi_mppi_training_oscar.sbatch").read_text()
    directives = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert not any(option in line for line in directives
                   for option in ("--time", "--cpus-per-task", "--gres", "--mem"))
    assert '#SBATCH --hint=nomultithread' in directives
    assert 'SLURM_CPUS_PER_TASK:?' in text
    assert 'git rev-parse HEAD' in text and 'git status --porcelain --untracked-files=all' in text
    assert '"$SLURM_ARRAY_TASK_ID" =~ ^[0-1]$' in text
    assert 'slurm/ambi_mppi_training.py train' in text
