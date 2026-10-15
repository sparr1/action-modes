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
