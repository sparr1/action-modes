from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from RL.tdmpc2_core.mppi import MPPIModelCallbacks, mppi_plan


def _model_cfg(**overrides):
    values = {
        "multitask": False,
        "obs_shape": {"state": (3,)},
        "obs": "state",
        "task_dim": 0,
        "num_enc_layers": 2,
        "enc_dim": 16,
        "latent_dim": 8,
        "simnorm_dim": 4,
        "action_dim": 2,
        "mlp_dim": 16,
        "num_bins": 7,
        "bin_size": 10.0 / 6.0,
        "vmin": -5.0,
        "vmax": 5.0,
        "episodic": False,
        "dropout": 0.0,
        "log_std_min": -20.0,
        "log_std_max": 2.0,
        "tau": 0.005,
        "q_representation": "scalar",
        "num_q": 2,
        "q_pair_size": 2,
        "q_num_bins": 5,
        "q_vmin": -2.0,
        "q_vmax": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("representation", "num_q"),
    [("scalar", 2), ("distributional", 4)],
)
def test_mppi_supports_both_q_backends_without_mutating_model_or_global_rng(
    representation,
    num_q,
):
    model = SoftWorldModel(
        _model_cfg(q_representation=representation, num_q=num_q)
    ).eval()
    root_z = torch.zeros(1, model.cfg.latent_dim)
    model_before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    training_before = model.training
    global_rng_before = torch.random.get_rng_state().clone()

    result = mppi_plan(
        root_z,
        model,
        horizon=2,
        iterations=2,
        num_samples=6,
        num_elites=2,
        num_pi_trajs=1,
        temperature=0.5,
        min_std=0.05,
        max_std=1.0,
        discount=0.99,
        q_reduction="mean_all",
        termination_threshold=0.5,
        generator=torch.Generator().manual_seed(17),
        t0=True,
        eval_mode=True,
        materialize_metrics=False,
    )

    assert result.action.shape == (model.cfg.action_dim,)
    assert result.next_mean.shape == (2, model.cfg.action_dim)
    torch.testing.assert_close(result.action, result.next_mean[0].clamp(-1.0, 1.0))
    assert result.model_steps == 1 * (2 - 1) + 2 * 6 * 2
    assert result.metrics["planner_model_steps"] == result.model_steps
    assert torch.is_tensor(result.metrics["planner_value_mean"])
    assert result.metrics["planner_value_mean"].device == root_z.device
    assert all(
        torch.isfinite(torch.as_tensor(value))
        for value in result.metrics.values()
    )
    assert model.training is training_before
    torch.testing.assert_close(
        torch.random.get_rng_state(),
        global_rng_before,
        rtol=0,
        atol=0,
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)


def test_mppi_callback_mode_counts_every_transition_and_masks_terminated_value():
    transition_counter = {"model_steps": 0}
    seen_reductions = []

    def dynamics(z, action):
        transition_counter["model_steps"] += z.shape[0]
        return z

    def reward(z, action):
        return torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)

    def policy(z, *, generator):
        return torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)

    def terminal_q(z, action, *, reduction, generator):
        seen_reductions.append(reduction)
        return torch.full((z.shape[0], 1), 100.0, device=z.device, dtype=z.dtype)

    callbacks = MPPIModelCallbacks(
        action_dim=1,
        dynamics=dynamics,
        reward=reward,
        policy=policy,
        terminal_q=terminal_q,
        termination=lambda z: torch.ones(z.shape[0], 1, device=z.device),
    )
    result = mppi_plan(
        torch.zeros(1, 2),
        callbacks=callbacks,
        horizon=3,
        iterations=2,
        num_samples=5,
        num_elites=2,
        num_pi_trajs=2,
        temperature=0.5,
        min_std=0.05,
        max_std=1.0,
        discount=0.9,
        q_reduction="min_all",
        termination_threshold=0.5,
        generator=torch.Generator().manual_seed(2),
        eval_mode=True,
    )

    expected_steps = 2 * (3 - 1) + 2 * 5 * 3
    assert transition_counter["model_steps"] == expected_steps
    assert result.model_steps == expected_steps
    assert result.metrics["planner_policy_model_steps"] == 4
    assert result.metrics["planner_candidate_model_steps"] == 30
    assert result.metrics["planner_value_mean"] == pytest.approx(1.0)
    assert seen_reductions == ["min_all", "min_all"]


def test_mppi_warm_start_is_shifted_without_aliasing_or_mutating_input():
    callbacks = MPPIModelCallbacks(
        action_dim=1,
        dynamics=lambda z, action: z,
        reward=lambda z, action: action,
        policy=lambda z, *, generator: torch.zeros(
            z.shape[0], 1, device=z.device, dtype=z.dtype
        ),
        terminal_q=lambda z, action, *, reduction, generator: torch.zeros(
            z.shape[0], 1, device=z.device, dtype=z.dtype
        ),
    )
    previous_mean = torch.full((3, 1), 0.8)
    previous_before = previous_mean.clone()
    common = {
        "callbacks": callbacks,
        "horizon": 3,
        "iterations": 1,
        "num_samples": 32,
        "num_elites": 8,
        "num_pi_trajs": 0,
        "temperature": 1.0,
        "min_std": 1e-4,
        "max_std": 1e-4,
        "discount": 0.99,
        "q_reduction": "mean_all",
        "termination_threshold": 0.5,
        "previous_mean": previous_mean,
        "eval_mode": True,
    }

    warm = mppi_plan(
        torch.zeros(1, 2),
        generator=torch.Generator().manual_seed(11),
        t0=False,
        **common,
    )
    cold = mppi_plan(
        torch.zeros(1, 2),
        generator=torch.Generator().manual_seed(11),
        t0=True,
        **common,
    )

    torch.testing.assert_close(previous_mean, previous_before, rtol=0, atol=0)
    assert warm.next_mean.data_ptr() != previous_mean.data_ptr()
    torch.testing.assert_close(warm.action, warm.next_mean[0].clamp(-1.0, 1.0))
    torch.testing.assert_close(cold.action, cold.next_mean[0].clamp(-1.0, 1.0))
    assert warm.action.item() > 0.7
    assert abs(cold.action.item()) < 0.01


def test_mppi_rejects_ambiguous_models_and_invalid_population_controls():
    callbacks = MPPIModelCallbacks(
        action_dim=1,
        dynamics=lambda z, action: z,
        reward=lambda z, action: torch.zeros(z.shape[0], 1),
        policy=lambda z, *, generator: torch.zeros(z.shape[0], 1),
        terminal_q=lambda z, action, *, reduction, generator: torch.zeros(
            z.shape[0], 1
        ),
    )
    kwargs = {
        "horizon": 2,
        "iterations": 1,
        "num_samples": 4,
        "num_elites": 2,
        "num_pi_trajs": 0,
        "temperature": 0.5,
        "min_std": 0.05,
        "max_std": 1.0,
        "discount": 0.99,
        "q_reduction": "mean_all",
        "generator": torch.Generator().manual_seed(1),
    }

    with pytest.raises(ValueError, match="exactly one"):
        mppi_plan(torch.zeros(1, 2), **kwargs)
    with pytest.raises(ValueError, match="num_elites"):
        mppi_plan(
            torch.zeros(1, 2),
            callbacks=callbacks,
            **{**kwargs, "num_elites": 5},
        )
    with pytest.raises(ValueError, match="num_pi_trajs"):
        mppi_plan(
            torch.zeros(1, 2),
            callbacks=callbacks,
            **{**kwargs, "num_pi_trajs": 5},
        )
    with pytest.raises(ValueError, match="previous_mean"):
        mppi_plan(
            torch.zeros(1, 2),
            callbacks=callbacks,
            previous_mean=torch.zeros(3, 1),
            **kwargs,
        )


def test_mppi_rejects_train_mode_models_and_nonfinite_controls():
    model = SoftWorldModel(_model_cfg(dropout=0.2)).train()
    kwargs = {
        "horizon": 2,
        "iterations": 1,
        "num_samples": 4,
        "num_elites": 2,
        "num_pi_trajs": 0,
        "temperature": 0.5,
        "min_std": 0.05,
        "max_std": 1.0,
        "discount": 0.99,
        "q_reduction": "mean_all",
        "generator": torch.Generator().manual_seed(1),
    }
    with pytest.raises(ValueError, match="model.eval"):
        mppi_plan(torch.zeros(1, model.cfg.latent_dim), model, **kwargs)

    callbacks = MPPIModelCallbacks(
        action_dim=1,
        dynamics=lambda z, action: z,
        reward=lambda z, action: torch.zeros(z.shape[0], 1),
        policy=lambda z, *, generator: torch.zeros(z.shape[0], 1),
        terminal_q=lambda z, action, *, reduction, generator: torch.zeros(
            z.shape[0], 1
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        mppi_plan(
            torch.zeros(1, 2),
            callbacks=callbacks,
            **{**kwargs, "temperature": float("nan")},
        )
