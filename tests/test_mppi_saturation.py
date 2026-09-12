"""Observer fidelity and the distinction between mixtures and mean actions."""
import math
import random

import numpy as np
import pytest
import torch

import evaluate_mppi_saturation as diagnostic
from RL.tdmpc2_core import mppi


def _callbacks():
    return mppi.MPPIModelCallbacks(action_dim=2,
        dynamics=lambda z, action: z + .01 * action,
        reward=lambda z, action: action[:, :1],
        policy=lambda z, generator: torch.randn(z.shape[0], 2, generator=generator, device=z.device).tanh(),
        terminal_q=lambda z, action, reduction, generator: (z + action).sum(-1, keepdim=True))


def _settings(device="cpu"):
    return dict(callbacks=_callbacks(), horizon=1, iterations=4, num_samples=128,
                num_elites=16, num_pi_trajs=6, temperature=.5, min_std=.05, max_std=2.,
                discount=.99, q_reduction="mean_pair", eval_mode=True,
                generator=torch.Generator(device=device).manual_seed(72))


def test_weighted_elite_saturation_is_not_saturation_of_mean():
    actions = torch.tensor([[[1.], [-1.]]])
    values = torch.tensor([[math.log(3.)], [0.]])
    stats, raw, mean = diagnostic.population_statistics(actions, values,
        num_elites=2, num_pi_trajs=1, temperature=1., min_std=.05, max_std=2.)
    assert stats["weighted_elite_selection"]["exact_boundary_fraction"] == 1.
    assert stats["optimized_mean"]["exact_boundary_fraction"] == 0.
    assert stats["weighted_elite_selection"]["mean_absolute_action"] == 1.
    assert stats["optimized_mean"]["mean_absolute_action"] == pytest.approx(.5)
    torch.testing.assert_close(mean, torch.tensor([[.5]]))
    assert np.asarray(raw["elite_weights"]) == pytest.approx(np.asarray([[.75], [.25]]))


def test_exact_and_near_bounds_are_component_fractions():
    actions = torch.tensor([[1., -1., .995, -.995, .5, 0.]])
    stats = diagnostic.boundary_statistics(actions)
    assert stats["exact_boundary_fraction"] == pytest.approx(2 / 6)
    assert stats["near_boundary_fraction"] == pytest.approx(4 / 6)
    assert stats["candidate_count"] == 1 and stats["component_count"] == 6


@pytest.mark.parametrize("values", [torch.tensor([[1.01]]), torch.tensor([[float("nan")]]), torch.empty(0, 1)])
def test_invalid_action_values_are_not_silently_clipped(values):
    with pytest.raises(ValueError):
        diagnostic.boundary_statistics(values)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_observer_preserves_actual_action_fit_model_work_and_rng(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")
    root = torch.zeros(1, 2, device=device)
    baseline_settings = _settings(device)
    baseline = mppi.mppi_plan(root, **baseline_settings)
    baseline_rng = baseline_settings["generator"].get_state()
    global_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    observed_settings = _settings(device)
    observed, records = diagnostic.observe_mppi(root, **observed_settings)
    assert torch.equal(observed.action, baseline.action)
    assert torch.equal(observed.next_mean, baseline.next_mean)
    assert observed.model_steps == baseline.model_steps == 512
    assert torch.equal(observed_settings["generator"].get_state(), baseline_rng)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    if cuda_rng is not None:
        assert all(torch.equal(before, after) for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert random.getstate() == python_rng
    assert all(np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state()))
    assert [r["iteration"] for r in records] == [1, 2, 3, 4]
    assert len(records[-1]["population"]["actions"][0]) == 128
    for key in baseline.metrics:
        assert observed.metrics[key] == baseline.metrics[key]


def test_observer_restores_original_estimator_on_error(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected estimator error")
    monkeypatch.setattr(mppi, "_estimate_value", fail)
    with pytest.raises(RuntimeError, match="injected estimator"):
        diagnostic.observe_mppi(torch.zeros(1, 2), **_settings())
    assert mppi._estimate_value is fail


def test_runtime_context_restores_mixed_modes_and_every_rng_on_error():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.1))
    model.train()
    model[0].eval()
    modes = [m.training for m in model.modules()]
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="injected"):
        with diagnostic.preserve_runtime(model):
            assert not any(m.training for m in model.modules())
            random.random()
            np.random.randn()
            torch.randn(8)
            raise RuntimeError("injected")
    assert [m.training for m in model.modules()] == modes
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
    assert random.getstate() == python_rng
    assert all(np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state()))
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


def test_actor_measurements_use_exact_provided_noise_and_zero_mean_noise():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []
        def pi(self, z, *, policy, noise, **bounds):
            self.seen.append(noise.clone())
            return noise.tanh(), {}
    model = Policy().train()
    noise = torch.tensor([[30., -30.], [0., 0.]])
    stats = diagnostic.actor_statistics(model, torch.zeros(1, 2), None, {}, noise)
    assert torch.equal(model.seen[0], noise)
    assert torch.equal(model.seen[1], torch.zeros(1, 2))
    assert stats["policy_samples"]["exact_boundary_fraction"] == .5
    assert stats["policy_mean"]["exact_boundary_fraction"] == 0.
    assert model.training


def test_aggregation_uses_equal_episodes_and_nested_root_solver_means():
    rows = []
    for episode, root, repeat, value in [("a", "a0", 0, 0.), ("a", "a0", 1, 0.),
                                        ("a", "a1", 0, 1.), ("b", "b0", 0, 1.)]:
        rows.append(dict(episode_id=episode, root_id=root, solver_repeat=repeat,
                         operator="sac", iteration=4, distribution="policy_samples",
                         **{metric: value for metric in diagnostic.METRICS}))
    result = diagnostic.aggregate_rows(rows)
    metric = result[0]["metrics"]["exact_boundary_fraction"]
    assert result[0]["episodes"] == 2
    assert metric["episode_values"] == [.5, 1.]
    assert metric["mean"] == .75
    assert metric["ci95"] == [.5, 1.]


def test_smoke_summary_does_not_claim_population_interval(tmp_path):
    row = dict(episode_id="a", root_id="a0", solver_repeat=0, operator="mppi", iteration=4,
               distribution="weighted_elite_selection", **{metric: .5 for metric in diagnostic.METRICS})
    record = {"summary": diagnostic.aggregate_rows([row])}
    assert record["summary"][0]["metrics"]["exact_boundary_fraction"]["ci95"] is None
    diagnostic.write_html(tmp_path / "report.html", record)
    report = (tmp_path / "report.html").read_text()
    assert "expected saturation" in report and "different action rules" in report


def test_existing_outputs_are_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        diagnostic.run(tmp_path / "checkpoint", tmp_path / "roots", output)
