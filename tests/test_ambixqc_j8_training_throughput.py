"""The J8 timing must execute the requested budget and include outer learning."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parent / "benchmarks/ambixqc_j8_training_throughput.py"
SPEC = importlib.util.spec_from_file_location("j8_timing", PATH)
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


def test_recipe_resolves_requested_j8_counts():
    from RL.AMBIXQC import AMBIXQC
    import gymnasium as gym

    run, manifest = timing.recipe("j8", True, 20, 96)
    wrapper = object.__new__(AMBIXQC)
    wrapper.env = gym.make("Pendulum-v1")
    wrapper.run_params = {**run, "device": "cpu"}
    wrapper.experiment_params = manifest
    wrapper.custom_params = run["alg_params"]
    try:
        cfg = wrapper._build_cfg({**run["alg_params"], "device": "cpu"})
    finally:
        wrapper.env.close()
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (128, 32, 32)
    assert cfg.inner_model_step_budget == 3072
    assert cfg.inner_replay_capacity == 3072
    assert cfg.utd == cfg.xqc_utd == 1
    assert cfg.aux_return_mode == "off"
    assert not cfg.wandb and run["checkpoint_every"] is None


def test_window_times_through_final_outer_update():
    clock = iter([10., 12., 15., 19.])
    cpu = iter([1., 7.])
    syncs = []
    window = timing.TrainingWindow(502, 3, lambda: syncs.append(True),
                                  wall=lambda: next(clock), cpu=lambda: next(cpu))
    window.before_action(502, 2)
    window.after_decision(502, 3, .2)
    window.after_decision(503, 4, .3)
    window.after_decision(504, 5, .4)
    assert window.result["wall_seconds"] == 9.
    assert window.result["outer_updates"] == 3
    assert window.result["decisions_per_second"] == pytest.approx(1 / 3)
    assert len(syncs) == 2


def test_window_rejects_missing_outer_update():
    clock = iter([0., 1.])
    window = timing.TrainingWindow(1, 1, lambda: None, wall=lambda: next(clock), cpu=lambda: 0)
    window.before_action(1, 0)
    with pytest.raises(RuntimeError, match="outer update"):
        window.after_decision(1, 0, .1)
