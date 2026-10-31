"""Ensure K1 timing includes exactly one real decision and outer update."""

import json

import pytest

from tests import benchmark_k1_training_graph as bench
from tests.benchmark_whole_inner_graph import make_agent


def test_k1_recipe_preserves_work_and_disables_publication():
    original = json.loads(bench.RECIPE.read_text())
    run, manifest, overrides = bench.recipe(12, 64)
    expected = dict(original["alg_params"])
    expected.update(overrides)
    assert run["alg_params"] == expected
    assert run["total_steps"] == 577
    assert run["alg_params"]["inner_solve_interval"] == 1
    assert run["alg_params"]["inner_rounds"] == 8
    assert run["alg_params"]["aux_return_mode"] == "off"
    assert not run["alg_params"]["wandb"]
    assert run["save_strat"] == manifest["save_trials"] == "none"


def test_k1_primary_critic_configuration_resolves():
    agent = make_agent(device="cpu", tiny=True, compile_enabled=False, config=bench.RECIPE)
    assert agent.aux_return is None
    assert agent.cfg.outer_critic_target == agent.cfg.inner_sac_critic_target == "reward_only"
    assert agent.inner_engine._critic_base is agent.inner_engine._horizon_critic is agent.model._Qs


def test_timer_includes_outer_work_and_excludes_warmup(monkeypatch):
    clock = {"wall": 0., "cpu": 0.}
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock["wall"])
    monkeypatch.setattr(bench.time, "process_time", lambda: clock["cpu"])
    syncs = []
    timer = bench.DecisionTimer(12, 2, lambda: syncs.append(clock["wall"]))
    timer.before(11, 7)
    timer.after_action(11, 100.)
    timer.after(11, 8)
    assert not timer.seconds
    for step in (12, 13):
        timer.before(step, step - 4)
        clock["wall"] += 1.
        timer.after_action(step, 1.)
        clock["wall"] += 2.
        clock["cpu"] += 2.
        timer.after(step, step - 3)
    assert syncs == [0., 6.]
    assert timer.elapsed == 6. and timer.cpu == 4.
    assert timer.inner_seconds == [1., 1.] and timer.seconds == [3., 3.]
    assert timer.updates == [1, 2]


def test_timer_rejects_missing_outer_update(monkeypatch):
    monkeypatch.setattr(bench.time, "perf_counter", lambda: 0.)
    timer = bench.DecisionTimer(12, 1, lambda: None)
    timer.before(12, 8)
    with pytest.raises(AssertionError, match="outer update"):
        timer.after(12, 8)
