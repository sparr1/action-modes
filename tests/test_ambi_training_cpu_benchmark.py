"""Protect the CPU comparison's physical affinity and complete-decision timer."""

from copy import deepcopy

import pytest

from slurm import benchmark_ambi_training_cpu as bench


def test_physical_core_selection_excludes_smt_and_distinguishes_sockets(tmp_path):
    topology = {0: (0, 0), 1: (0, 1), 4: (0, 0), 5: (0, 1),
                8: (1, 0), 9: (1, 1), 12: (1, 0), 13: (1, 1)}
    for cpu, (socket, core) in topology.items():
        path = tmp_path / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        (path / "physical_package_id").write_text(str(socket))
        (path / "core_id").write_text(str(core))
    assert bench.physical_core_cpus(reversed(list(topology)), tmp_path) == [0, 1, 8, 9]
    assert bench.physical_core_cpus([4, 5, 12, 13], tmp_path) == [4, 5, 12, 13]


def test_benchmark_preserves_production_work_and_only_shortens_collection():
    name, original_manifest, original = bench.recipe(5, bench.MANIFEST)
    selected_name, manifest, run, overrides = bench.benchmark_recipe(20, 128)
    expected = deepcopy(original)
    expected["alg_params"].update(wandb=False, seed_steps=500, pretrain_steps=2, buffer_size=4096)
    expected.update(total_steps=649, checkpoint_every=None, save_strat="none")
    assert name == selected_name == "ambi_full_h3_j8_bernoulli50"
    assert run == expected
    assert overrides == {"wandb": False, "seed_steps": 500, "pretrain_steps": 2, "buffer_size": 4096}
    expected_manifest = deepcopy(original_manifest)
    expected_manifest.update(checkpoint_every=None, save_strat="none", save_trials="none", logs="none")
    assert manifest == expected_manifest


def test_held_benchmark_selects_exact_production_recipe_without_changing_solve_dose():
    path = bench.ROOT / "configs/dmcontrol/experiments/ambi_held_training.json"
    name, _, original = bench.recipe(0, path)
    selected, _, run, overrides = bench.benchmark_recipe(12, 120, manifest_path=path, cell_index=0)
    expected = deepcopy(original)
    expected["alg_params"].update(overrides)
    expected.update(total_steps=633, checkpoint_every=None, save_strat="none")
    assert name == selected == "ambi_full_h3_j8_k3_fresh"
    assert run == expected
    params = run["alg_params"]
    assert (params["inner_solve_interval"], params["inner_rounds"], params["inner_rollout_horizon"]) == (3, 8, 3)
    assert params["inner_critic_bernoulli_p"] == 0
    assert params["aux_return_joint_vmap"] is False
    assert params["compile"] and params["compile_strict"]


def test_selectable_cpu_counts_do_not_require_eight_core_allocation():
    assert bench.cpu_counts("1,2,3") == (1, 2, 3)
    for invalid in ("", "0,1", "1,1", "-1,2", "1,two"):
        with pytest.raises(ValueError):
            bench.cpu_counts(invalid)


def test_phase_summary_preserves_controller_and_outer_timing_scope():
    rows = [{"solve": solve, "action_seconds": seconds, "outer_update_enqueue_seconds": .2}
            for solve, seconds in ((True, 6.), (False, .1), (False, .3))]
    result = bench.phase_summary(rows)
    assert result["solve_decisions"] == 1 and result["held_decisions"] == 2
    assert result["solve_action_seconds_mean"] == 6.
    assert result["held_action_seconds_mean"] == pytest.approx(.2)
    assert result["outer_update_enqueue_seconds_mean"] == pytest.approx(.2)


@pytest.mark.parametrize("probability", [0., .5])
def test_benchmark_work_validator_accepts_actual_held_metrics_and_rejects_work(probability):
    import torch
    from tests.test_ambi_root_local_sac import _tiny_model
    model = _tiny_model(inner_solve_interval=3, inner_critic_bernoulli_p=probability,
                        inner_finite_horizon=True, inner_execution_action="policy_sample")
    try:
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        model.agent.act(torch.ones(3), collect_diagnostics=False)
        metrics = dict(model.agent.last_inner_metrics)
        assert not bench.action_work(metrics, decision=1, interval=3, probability=probability)
        metrics["inner_replay_draws"] = 1
        with pytest.raises(AssertionError):
            bench.action_work(metrics, decision=1, interval=3, probability=probability)
    finally:
        model.close()
        model.env.close()


def test_timer_excludes_warmup_but_includes_outer_updates(monkeypatch):
    clock = {"wall": 100., "cpu": 50.}
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock["wall"])
    monkeypatch.setattr(bench.time, "process_time", lambda: clock["cpu"])
    syncs = []
    timer = bench.TrainingWindow(12, 3, lambda: syncs.append(clock["wall"]))
    timer.before_action(11, 9)
    timer.after_action(11, 100.)
    timer.after_decision(11, 10)
    assert timer.result is None and not timer.inner_seconds and not syncs
    for index, step in enumerate((12, 13, 14)):
        timer.before_action(step, 10 + index)
        clock["wall"] += 1.  # controller
        timer.after_action(step, 1.)
        clock["wall"] += 2.  # environment, replay and outer update
        clock["cpu"] += 2.
        timer.after_decision(step, 11 + index)
    result = timer.result
    assert syncs == [100., 109.]
    assert result["wall_seconds"] == 9.
    assert result["process_cpu_seconds"] == 6.
    assert result["average_cpu_cores_used"] == pytest.approx(2 / 3)
    assert result["decisions_per_second"] == pytest.approx(1 / 3)
    assert result["outer_updates"] == 3
    assert result["inner_seconds_per_action"] == 1.
    assert result["end_to_end_seconds"] == [3., 3., 3.]


def test_timer_rejects_missing_outer_work(monkeypatch):
    now = iter([0., 1.])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(now))
    timer = bench.TrainingWindow(12, 1, lambda: None)
    timer.before_action(12, 10)
    timer.after_action(12, .5)
    with pytest.raises(AssertionError):
        timer.after_decision(12, 10)
