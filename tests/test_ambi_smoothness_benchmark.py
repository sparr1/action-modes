"""Protect production-work preservation and the smoothness GPU gate."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from slurm import benchmark_ambi_smoothness as bench


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    algorithms = root / "algs"
    algorithms.mkdir()
    params = dict(inner_rounds=10, inner_rollout_horizon=3, inner_solve_interval=3,
                  inner_batch_size=256, outer_pave_batch_size=768,
                  outer_pave_mpr_coef=.004, outer_pave_action_smoothness_coef=0.,
                  outer_pave_obs_noise_std=.01, outer_pave_action_noise_std=.02,
                  critic_ensemble_backend="vmap", compile=True, compile_strict=True,
                  wandb=True, seed_steps=2500, pretrain_steps=2500, buffer_size=1000000)
    config = dict(env="DMControl-v0", seed=55, total_steps=1000000,
                  checkpoint_every=25000, save_strat="all", alg_params=params)
    manifest = dict(configs=["mpr"], overrides_alg={"seed": 56},
                    env_params={"domain_name": "humanoid", "task_name": "walk"},
                    checkpoint_every=25000, save_strat="all", save_trials="all", logs="timestamp")
    (algorithms / "mpr.json").write_text(json.dumps(config))
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(bench, "ROOT", root)
    return root, path, algorithms, config, manifest


def test_smoke_preserves_work_and_uses_shallow_manifest_overrides(campaign):
    _, path, algorithms, config, original_manifest = campaign
    name, manifest, run, overrides = bench.recipe(path, algorithms, 0, 12, 30)
    expected = {"name": "mpr", **deepcopy(config), "seed": 56}
    expected["alg_params"].update(wandb=False, seed_steps=500, pretrain_steps=2, buffer_size=4096)
    expected.update(total_steps=543, checkpoint_every=None, save_strat="none")
    assert run == expected and name == "mpr"
    assert overrides == dict(wandb=False, seed_steps=500, pretrain_steps=2, buffer_size=4096)
    assert manifest == {**original_manifest, "checkpoint_every": None,
                        "save_strat": "none", "save_trials": "none", "logs": "none"}
    assert json.loads(path.read_text()) == original_manifest
    assert json.loads((algorithms / "mpr.json").read_text()) == config
    replacement = {"outer_pave_batch_size": 768, "outer_pave_mpr_coef": .02}
    path.write_text(json.dumps({**original_manifest, "overrides_alg": {"alg_params": replacement}}))
    _, _, run, overrides = bench.recipe(path, algorithms, 0, 12, 30)
    assert run["alg_params"] == {**replacement, **overrides}


def test_rejects_external_recipe_and_invalid_index(campaign, tmp_path):
    _, path, algorithms, _, _ = campaign
    with pytest.raises(AssertionError, match="alg-index"):
        bench.recipe(path, algorithms, -1, 12, 30)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(ValueError):
        bench.source_path(outside)


def test_penalties_require_full_samples_finite_and_nonzero():
    selected = bench.active_terms(dict(outer_pave_batch_size=768, outer_pave_mpr_coef=.004,
                                      aux_return_pave_action_smoothness_coef=.004))
    row = {}
    for prefix, term in selected:
        row.update({f"{prefix}_sample_count": 768, f"{prefix}_{term}_loss": .5,
                    f"{prefix}_{term}_weighted_loss": .002})
    assert len(bench.validate_penalties([row], selected)) == 2
    for key, value in (("outer_pave_sample_count", 64), ("outer_pave_mpr_loss", float("nan")),
                       ("outer_pave_mpr_loss", 0.), ("aux_return_pave_action_smoothness_weighted_loss", 0.)):
        with pytest.raises(AssertionError):
            bench.validate_penalties([{**row, key: value}], selected)
    with pytest.raises(AssertionError, match="768"):
        bench.active_terms(dict(outer_pave_batch_size=64, outer_pave_mpr_coef=.004))
    assert bench.active_terms(dict(outer_pave_batch_size=None, outer_pave_mpr_coef=.004)) == [("outer_pave", "mpr")]


def test_all_campaign_recipes_preserve_full_pave_and_k3():
    manifest = bench.ROOT / "configs/dmcontrol/experiments/ambi_critic_smoothness_training.json"
    if not manifest.is_file():
        pytest.skip("Smoothness campaign configs not present")
    selected = []
    for index in range(4):
        _, _, run, _ = bench.recipe(manifest, bench.ROOT / "configs/dmcontrol/algs", index, 12, 30)
        params = run["alg_params"]
        assert (params["inner_rounds"], params["inner_rollout_horizon"], params["inner_solve_interval"]) == (10, 3, 3)
        assert params["batch_size"] * params["train_unroll_horizon"] == 768
        assert params["outer_pave_batch_size"] is None
        selected.append(bench.active_terms(params))
    assert selected == [[("outer_pave", "mpr")], [("outer_pave", "action_smoothness")],
                        [("aux_return_pave", "mpr")], [("aux_return_pave", "action_smoothness")]]


def test_cli_defaults_and_rejects_incomplete_k3_windows():
    args = ["--manifest", "manifest.json", "--alg-dir", "algs", "--alg-index", "0", "--output", "/tmp/out"]
    parsed = bench.parse_args(args)
    assert (parsed.cores, parsed.warmup, parsed.decisions) == ([1, 2], 12, 30)
    for extra in (["--decisions", "31"], ["--warmup", "10"], ["--cores", "1,1"],
                  ["--cores", "0"], ["--cores", "abc"], ["--decisions", "501"]):
        with pytest.raises(SystemExit):
            bench.parse_args(args + extra)


def test_driver_uses_isolated_workers_with_thread_limits(campaign, tmp_path, monkeypatch):
    root, path, algorithms, _, _ = campaign
    output = tmp_path / "output"
    args = bench.parse_args(["--manifest", str(path), "--alg-dir", str(algorithms),
                            "--alg-index", "0", "--output", str(output)])
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setattr(bench.os, "sched_getaffinity", lambda _: {3, 7}, raising=False)
    monkeypatch.setattr(bench, "physical_core_cpus", lambda _: [3, 7])
    monkeypatch.setattr(bench, "binding", lambda _: {"commit": "fixed"})
    calls = []

    def fake_run(command, **kwargs):
        worker_cpus = command[command.index("--worker-cpus") + 1]
        count = len(worker_cpus.split(","))
        environment = kwargs["env"]
        assert environment["WANDB_MODE"] == "disabled"
        assert environment["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        assert all(environment[key] == str(count) for key in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
        assert kwargs["cwd"] == root and kwargs["check"] is True
        report = Path(command[command.index("--output") + 1])
        report.write_text(json.dumps({"passed": True, "binding": {"commit": "fixed"},
                          "physical_cores": count, "measured": {"seconds_per_decision": 1. if count == 1 else .98}}))
        calls.append((worker_cpus, environment["TORCHINDUCTOR_CACHE_DIR"]))

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    bench.driver(args)
    assert [row[0] for row in calls] == ["3", "3,7"]
    assert calls[0][1] == calls[1][1]
    result = json.loads((output / "results.json").read_text())
    assert result["passed"] and result["recommended_cpus_within_5pct_of_best"] == 1
    with pytest.raises(FileExistsError):
        bench.driver(args)


def test_cpu_replay_survives_learn_budget_rebuild_without_checkpoint_files(tmp_path):
    from tests.test_ambi_alpha_carry_training_state import _training_model

    learner = _training_model("run")
    try:
        bench.configure_cpu_replay_without_checkpoints(learner, tmp_path / "measurement.json")
        before = learner.buffer
        learner.learn(15)
        assert learner.buffer is not before
        assert learner.buffer.resumable_storage
        assert learner.buffer._storage_device.type == "cpu"
        assert list(tmp_path.iterdir()) == []
    finally:
        learner.close()
        learner.env.close()
