import json
import math
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from evaluate_ambi_checkpoint import _critic_architecture_key
from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core import MODEL_SIZE
from RL.tdmpc2_core.common.q_representation import QRepresentation
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from utils.ambi_research import load_preset_matrix, resolve_preset


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_inner_decoupling.json"


def _build_cfg(**params):
    """Resolve AMBI configuration without allocating the model."""
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def _tiny_params(**overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "simnorm_dim": 4,
        "num_bins": 5,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "horizon": 2,
        "buffer_size": 32,
        "seed_steps": 4,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "wandb": False,
        "dropout": 0.0,
        "q_representation": "scalar",
        "num_q": 2,
        "q_num_bins": 5,
        "q_vmin": -5,
        "q_vmax": 5,
        "inner_operator": "sac",
        "inner_rounds": 3,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 2,
        "inner_updates_per_round": "auto",
        "inner_batch_size": 4,
        "inner_replay_capacity": 12,
        "inner_replay_sampling": "with_replacement",
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "auto",
        "inner_temperature_initialization": "inherit_outer",
        "inner_target_entropy": "inherit_outer",
        "inner_critic_target_tau": 0.005,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 0,
        "inner_mppi_num_elites": 2,
        "inner_mppi_num_pi_trajs": 0,
    }
    params.update(overrides)
    return params


def _model_from_params(params):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return AMBITDMPC2(
        "AMBITDMPC2",
        env,
        params,
        {"seed": 13, "device": "cpu", "env": "test", "total_steps": 10},
        {},
    )


def _tiny_model(**overrides):
    return _model_from_params(_tiny_params(**overrides))


def _tiny_legacy_model(**overrides):
    params = _tiny_params()
    params.pop("inner_rollouts_per_round")
    params.pop("inner_updates_per_round")
    params.update(overrides)
    return _model_from_params(params)


def _tiny_default_q_model(**overrides):
    params = _tiny_params()
    params.pop("q_representation")
    params.pop("num_q")
    params.update(overrides)
    return _model_from_params(params)


def _assert_finite_metrics(metrics):
    for value in metrics.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
        elif isinstance(value, (int, float)):
            assert math.isfinite(float(value))


def test_reference_defaults_resolve_tdmpc2_size_five_distributional_critic():
    cfg = _build_cfg(model_size=5)

    assert cfg.q_representation == "distributional"
    assert cfg.num_q == 5
    assert cfg.q_pair_size == 2
    assert cfg.q_num_bins == 101
    assert cfg.q_vmin == pytest.approx(-10.0)
    assert cfg.q_vmax == pytest.approx(10.0)
    assert cfg.dropout == pytest.approx(0.01)
    assert cfg.inner_q_target_reduction == "min_pair"
    assert cfg.inner_q_actor_reduction == "min_pair"
    assert cfg.mppi_terminal_q_reduction == "mean_pair"

    assert cfg.inner_schedule_mode == "canonical"
    assert cfg.inner_rounds == 4
    assert cfg.inner_rollouts_per_round == 64
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_updates_per_round == "auto"
    assert cfg.inner_model_step_budget == 768
    assert cfg.inner_replay_capacity == 768
    assert cfg.inner_batch_size == 128
    assert cfg.inner_actor_adaptation == "clone"
    assert cfg.inner_critic_adaptation == "clone"
    assert cfg.inner_temperature_mode == "auto"
    assert cfg.inner_temperature_initialization == "inherit_outer"
    assert cfg.inner_target_entropy == "inherit_outer"
    assert cfg.inner_actor_lr == pytest.approx(5e-5)
    assert cfg.inner_critic_lr == pytest.approx(5e-5)
    assert cfg.inner_temperature_lr == pytest.approx(5e-5)
    assert cfg.inner_critic_target_tau == pytest.approx(0.005)

    debug = _build_cfg(model_size=1)
    assert debug.q_representation == "distributional"
    assert debug.num_q == 2


def test_default_size_five_backend_outputs_and_trains_all_five_heads():
    cfg = _build_cfg(model_size=5)
    # Exercise the resolved critic contract without allocating the full 5M model.
    cfg.enc_dim = 16
    cfg.mlp_dim = 16
    cfg.latent_dim = 8
    cfg.num_enc_layers = 2
    cfg.simnorm_dim = 4
    model = SoftWorldModel(cfg)
    z = torch.randn(3, cfg.latent_dim)
    action = torch.randn(3, cfg.action_dim).tanh()

    predictions = model.q_predictions(z, action)
    assert predictions.shape == (5, 3, 101)
    loss = model.critic_loss(predictions, torch.randn(3, 1))
    loss.backward()

    for head in model._Qs:
        gradients = [
            parameter.grad
            for parameter in head.parameters()
            if parameter.requires_grad
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert any(torch.count_nonzero(gradient).item() for gradient in gradients)


def test_one_random_q_pair_is_shared_by_the_whole_minibatch(monkeypatch):
    backend = QRepresentation(
        "distributional",
        num_q=5,
        pair_size=2,
        num_bins=5,
        vmin=-2,
        vmax=2,
    )
    values = torch.tensor(
        [
            [1.0, 8.0, 2.0, 7.0],
            [5.0, 3.0, 9.0, 1.0],
            [4.0, 6.0, 0.0, 5.0],
            [2.0, 7.0, 4.0, 3.0],
            [6.0, 1.0, 8.0, 2.0],
        ]
    ).unsqueeze(-1)
    calls = []

    def pair_for_batch(device, **kwargs):
        calls.append((device, kwargs))
        return torch.tensor([1, 4], device=device)

    monkeypatch.setattr(backend, "_pair_indices", pair_for_batch)
    minimum = backend.reduce(values, "min_pair")
    mean = backend.reduce(values, "mean_pair")

    assert len(calls) == 2  # Exactly once for each whole-batch Q call.
    torch.testing.assert_close(minimum, torch.minimum(values[1], values[4]))
    torch.testing.assert_close(mean, (values[1] + values[4]) / 2)


def test_five_head_full_clone_and_target_start_as_exact_independent_copies():
    model = _tiny_model(
        q_representation="distributional",
        num_q=5,
        inner_updates_per_round=0,
    )
    agent = model.agent
    engine = agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)

    outer = agent.model._Qs
    local = engine.state.critic
    target = engine.state.critic_target
    assert len(outer) == len(local) == len(target) == 5
    assert local is not outer
    assert target is not local
    for outer_head, local_head, target_head in zip(outer, local, target):
        assert local_head is not outer_head
        assert target_head is not local_head
        for outer_parameter, local_parameter, target_parameter in zip(
            outer_head.parameters(),
            local_head.parameters(),
            target_head.parameters(),
        ):
            torch.testing.assert_close(local_parameter, outer_parameter, rtol=0, atol=0)
            torch.testing.assert_close(
                target_parameter, local_parameter, rtol=0, atol=0
            )
            assert local_parameter.data_ptr() != outer_parameter.data_ptr()
            assert target_parameter.data_ptr() != local_parameter.data_ptr()
            assert local_parameter.requires_grad
            assert not target_parameter.requires_grad


def test_every_tdmpc2_model_size_declares_the_expected_critic_ensemble():
    assert {size: spec["num_q"] for size, spec in MODEL_SIZE.items()} == {
        1: 2,
        5: 5,
        19: 5,
        48: 5,
        317: 8,
    }


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, ("distributional", 5, 101, -10.0, 10.0)),
        ({"model_size": None}, ("distributional", 5, 101, -10.0, 10.0)),
        ({"model_size": "1"}, ("distributional", 2, 101, -10.0, 10.0)),
        ({"model_size": 19}, ("distributional", 5, 101, -10.0, 10.0)),
        ({"model_size": 48}, ("distributional", 5, 101, -10.0, 10.0)),
        ({"model_size": 317}, ("distributional", 8, 101, -10.0, 10.0)),
        (
            {"q_representation": "scalar", "model_size": 317},
            ("scalar", 2, 1, None, None),
        ),
        (
            {
                "model_size": "5",
                "q_num_bins": "51",
                "q_vmin": "-8",
                "q_vmax": "12",
            },
            ("distributional", 5, 51, -8.0, 12.0),
        ),
    ],
)
def test_evaluator_resolves_the_same_critic_architecture_as_ambi(params, expected):
    resolved = {"algorithm_config": {"alg_params": params}}
    assert _critic_architecture_key(resolved) == expected


def test_new_schedule_rejects_mixed_total_budget_or_component_totals():
    with pytest.raises(ValueError, match="Cannot mix canonical J/N/H/G"):
        _build_cfg(
            inner_rollouts_per_round=4,
            inner_model_step_budget=24,
        )
    with pytest.raises(ValueError, match="Cannot mix canonical J/N/H/G"):
        _build_cfg(
            inner_updates_per_round=2,
            inner_actor_updates_per_action=4,
        )


def test_v1_schedule_aliases_map_to_j_n_h_g():
    with pytest.warns(DeprecationWarning):
        cfg = _build_cfg(
            inner_iterations=2,
            inner_rollouts=4,
            inner_horizon=3,
            inner_updates_per_iteration=5,
        )

    assert cfg.inner_schedule_mode == "canonical"
    assert cfg.inner_rounds == 2
    assert cfg.inner_rollouts_per_round == 4
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_updates_per_round == 5
    assert cfg.inner_model_step_budget == 24


def test_legacy_total_schedule_is_preserved_exactly_for_one_release():
    with pytest.warns(DeprecationWarning):
        cfg = _build_cfg(
            inner_model_step_budget=24,
            inner_rounds=2,
            inner_rollout_horizon=3,
            inner_critic_updates_per_action=4,
            inner_actor_updates_per_action=5,
            inner_temperature_mode="auto",
            inner_temperature_updates_per_action=3,
        )

    assert cfg.inner_schedule_mode == "legacy"
    assert cfg.inner_rollouts_per_round == 4
    assert cfg.inner_critic_updates_per_action == 4
    assert cfg.inner_actor_updates_per_action == 5
    assert cfg.inner_temperature_updates_per_action == 3
    assert cfg.inner_expected_update_slots == 5
    assert cfg.inner_nominal_critic_utd == pytest.approx(4 / 24)


def test_legacy_unequal_totals_keep_their_exact_front_loaded_allocations(
    monkeypatch,
):
    with pytest.warns(DeprecationWarning):
        model = _tiny_legacy_model(
            inner_rounds=2,
            inner_rollout_horizon=2,
            inner_model_step_budget=8,
            inner_critic_updates_per_action=3,
            inner_actor_updates_per_action=1,
            inner_temperature_mode="inherit_outer",
            inner_temperature_updates_per_action=0,
        )
    engine = model.agent.inner_engine
    allocations = []
    original_updates = engine._run_update_counts

    def record_updates(**counts):
        allocations.append(
            (
                counts["critic_count"],
                counts["actor_count"],
                counts["temperature_count"],
            )
        )
        return original_updates(**counts)

    monkeypatch.setattr(engine, "_run_update_counts", record_updates)
    model.agent.act(torch.zeros(3), collect_diagnostics=False)

    assert allocations == [(2, 1, 0), (1, 0, 0)]
    assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 3
    assert model.agent.last_inner_metrics["inner_actor_optimizer_steps"] == 1
    assert model.agent.last_inner_metrics["inner_update_slots"] == 3


def test_canonical_auto_schedule_collects_then_updates_at_every_round(monkeypatch):
    model = _tiny_model()
    agent = model.agent
    engine = agent.inner_engine
    outer_alpha = agent.alpha.detach().clone()
    events = []
    original_collect = engine._collect_round
    original_updates = engine._run_update_counts

    def record_collect(root_z):
        events.append(("collect", engine.state.replay.size, engine.state.actor_steps))
        result = original_collect(root_z)
        events.append(
            ("collected", engine.state.replay.size, int(result["lengths"].sum()))
        )
        return result

    def record_updates(**counts):
        events.append(
            (
                "update",
                engine.state.replay.size,
                counts["critic_count"],
                counts["actor_count"],
                counts["temperature_count"],
            )
        )
        return original_updates(**counts)

    monkeypatch.setattr(engine, "_collect_round", record_collect)
    monkeypatch.setattr(engine, "_run_update_counts", record_updates)
    agent.act(torch.zeros(3), collect_diagnostics=False)

    assert events == [
        ("collect", 0, 0),
        ("collected", 4, 4),
        ("update", 4, 4, 4, 4),
        ("collect", 4, 4),
        ("collected", 8, 4),
        ("update", 8, 4, 4, 4),
        ("collect", 8, 8),
        ("collected", 12, 4),
        ("update", 12, 4, 4, 4),
    ]
    metrics = agent.last_inner_metrics
    assert metrics["inner_model_steps"] == 12
    assert metrics["inner_update_slots"] == 12
    assert metrics["inner_critic_optimizer_steps"] == 12
    assert metrics["inner_actor_optimizer_steps"] == 12
    assert metrics["inner_temperature_optimizer_steps"] == 12
    assert metrics["inner_buffer_size"] == 12
    assert metrics["inner_nominal_model_steps"] == 12
    assert metrics["inner_realized_model_steps"] == 12
    assert metrics["inner_requested_update_slots"] == 12
    assert metrics["inner_updates_per_round_realized"] == 4
    assert metrics["inner_critic_utd"] == pytest.approx(1.0)
    assert metrics["inner_actor_utd"] == pytest.approx(1.0)
    assert metrics["inner_temperature_utd"] == pytest.approx(1.0)
    assert metrics["inner_alpha_initial"] == pytest.approx(outer_alpha.item())
    assert metrics["inner_alpha_delta"] == pytest.approx(
        float(metrics["inner_alpha_final"] - metrics["inner_alpha_initial"])
    )
    assert abs(float(metrics["inner_alpha_delta"])) > 0.0
    torch.testing.assert_close(agent.alpha, outer_alpha, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("updates_per_round", "expected_updates"),
    [("auto", 6), (2, 4)],
)
def test_auto_uses_realized_round_transitions_but_explicit_g_is_fixed(
    updates_per_round, expected_updates
):
    model = _tiny_model(
        episodic=True,
        horizon=3,
        inner_rounds=2,
        inner_rollouts_per_round=3,
        inner_rollout_horizon=3,
        inner_updates_per_round=updates_per_round,
        inner_replay_capacity=18,
    )

    def terminate_immediately(z, task=None, unnormalized=False):
        del task
        value = torch.ones(z.shape[0], 1, device=z.device)
        return value if not unnormalized else torch.full_like(value, 20.0)

    model.agent.model.termination = terminate_immediately
    model.agent.act(torch.zeros(3), collect_diagnostics=False)
    metrics = model.agent.last_inner_metrics

    assert metrics["inner_model_steps_budget"] == 18
    assert metrics["inner_model_steps"] == 6
    assert metrics["inner_nominal_model_steps"] == 18
    assert metrics["inner_realized_model_steps"] == 6
    assert metrics["inner_update_slots"] == expected_updates
    assert metrics["inner_requested_update_slots"] == expected_updates
    assert metrics["inner_updates_per_round_realized"] == pytest.approx(
        expected_updates / 2
    )
    assert metrics["inner_critic_utd"] == pytest.approx(expected_updates / 6)
    assert metrics["inner_actor_utd"] == pytest.approx(expected_updates / 6)
    assert metrics["inner_temperature_utd"] == pytest.approx(expected_updates / 6)
    assert metrics["inner_critic_optimizer_steps"] == expected_updates
    assert metrics["inner_actor_optimizer_steps"] == expected_updates
    assert metrics["inner_temperature_optimizer_steps"] == expected_updates
    assert model.agent.last_inner_rollout_lengths == [1] * 6


def test_canonical_without_replacement_fails_on_realized_round_underfill():
    model = _tiny_model(
        episodic=True,
        horizon=3,
        inner_rounds=2,
        inner_rollouts_per_round=3,
        inner_rollout_horizon=3,
        inner_updates_per_round="auto",
        inner_batch_size=4,
        inner_replay_capacity=18,
        inner_replay_sampling="without_replacement",
    )

    def terminate_immediately(z, task=None, unnormalized=False):
        del task
        value = torch.ones(z.shape[0], 1, device=z.device)
        return value if not unnormalized else torch.full_like(value, 20.0)

    model.agent.model.termination = terminate_immediately
    with pytest.raises(ValueError, match="without replacement"):
        model.agent.act(torch.zeros(3), collect_diagnostics=False)


def test_action_local_alpha_inherits_outer_without_aliasing_and_resets():
    model = _tiny_model(inner_updates_per_round=0)
    agent = model.agent
    engine = agent.inner_engine
    assert agent.log_ent_coef is not None

    with torch.no_grad():
        agent.log_ent_coef.fill_(math.log(0.37))
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)

    assert engine.state.log_alpha is not agent.log_ent_coef
    assert engine.state.log_alpha.data_ptr() != agent.log_ent_coef.data_ptr()
    assert engine.alpha.item() == pytest.approx(0.37)
    assert engine._resolved_inner_target_entropy() == pytest.approx(
        agent.target_entropy
    )
    outer_before = agent.alpha.detach().clone()
    with torch.no_grad():
        engine.state.log_alpha.add_(0.4)
    torch.testing.assert_close(agent.alpha, outer_before, rtol=0, atol=0)

    engine._clear_expired(t0=False, include_action=True)
    with torch.no_grad():
        agent.log_ent_coef.fill_(math.log(0.61))
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=False)

    assert engine.alpha.item() == pytest.approx(0.61)
    assert engine.state.temperature_optim is None


def test_canonical_five_head_full_clone_runs_finite_end_to_end():
    model = _tiny_model(
        q_representation="distributional",
        num_q=5,
        inner_rounds=2,
        inner_rollouts_per_round=2,
        inner_rollout_horizon=2,
        inner_updates_per_round=1,
        inner_replay_capacity=8,
    )

    model.learn(total_timesteps=6)

    assert model._global_step == 6
    assert model.agent.num_updates > 0
    assert model.agent.model.critic_signature == {
        "q_representation": "distributional",
        "num_q": 5,
        "q_num_bins": 5,
        "q_vmin": -5.0,
        "q_vmax": 5.0,
    }
    assert model.agent.last_inner_metrics["inner_model_steps"] == 8
    assert model.agent.last_inner_metrics["inner_update_slots"] == 2
    _assert_finite_metrics(model.agent.last_inner_metrics)
    _assert_finite_metrics(model._last_train_metrics)


def test_default_distributional_checkpoint_requires_explicit_scalar_opt_in(tmp_path):
    default = _tiny_default_q_model()
    assert default.cfg.q_representation == "distributional"
    assert default.cfg.num_q == 5
    checkpoint = tmp_path / "default-distributional.pt"
    default.agent.save(checkpoint)

    scalar = _tiny_model(q_representation="scalar", num_q=2)
    with pytest.raises(ValueError, match="critic specification"):
        scalar.agent.load(checkpoint)

    scalar_checkpoint = tmp_path / "explicit-scalar.pt"
    scalar.agent.save(scalar_checkpoint)
    restored_scalar = _tiny_model(q_representation="scalar", num_q=2)
    restored_scalar.agent.load(scalar_checkpoint)
    assert restored_scalar.agent.model.critic_signature["q_representation"] == "scalar"


def test_automatic_temperature_default_is_sac_only():
    sac = _build_cfg(inner_operator="sac")
    assert sac.inner_temperature_mode == "auto"
    assert sac.inner_temperature_initialization == "inherit_outer"
    assert sac.inner_updates_per_round == "auto"

    for operator in ("td3", "mppi", "none"):
        cfg = _build_cfg(inner_operator=operator)
        assert cfg.inner_temperature_mode != "auto"
        assert cfg.inner_temperature_updates_per_action == 0


def test_canonical_auto_temperature_inherits_outer_unless_explicitly_overridden():
    cfg = _build_cfg(
        inner_rounds=2,
        inner_rollouts_per_round=4,
        inner_rollout_horizon=3,
        inner_updates_per_round=2,
        inner_temperature_mode="auto",
    )
    assert cfg.inner_temperature_initialization == "inherit_outer"

    frozen_critic = _build_cfg(inner_critic_adaptation="frozen")
    assert frozen_critic.inner_critic_updates_per_action == 0
    assert frozen_critic.inner_nominal_critic_utd == 0.0
    assert frozen_critic.inner_expected_update_slots == 768


def test_research_operator_presets_remove_inapplicable_sac_schedule_fields():
    matrix = load_preset_matrix(MATRIX)
    none = resolve_preset(MATRIX, "inner_operator/none", matrix=matrix)
    mppi = resolve_preset(MATRIX, "inner_operator/mppi", matrix=matrix)

    for resolved in (none, mppi):
        params = resolved["algorithm_config"]["alg_params"]
        assert "inner_rounds" not in params
        assert "inner_rollouts_per_round" not in params
        assert "inner_updates_per_round" not in params

    none_cfg = _build_cfg(**none["algorithm_config"]["alg_params"])
    assert none_cfg.inner_rounds == 0
    assert none_cfg.inner_mppi_iterations == 0

    mppi_cfg = _build_cfg(**mppi["algorithm_config"]["alg_params"])
    assert mppi_cfg.inner_rounds == 0
    assert mppi_cfg.inner_mppi_iterations == 2
    assert mppi_cfg.inner_mppi_num_samples == 128


def test_reference_presets_resolve_expected_q_and_schedule_contract():
    main = json.loads((ROOT / "configs/algs/AntAMBITDMPC2.json").read_text())[
        "alg_params"
    ]
    full_copy = json.loads(
        (ROOT / "configs/algs/AntAMBITDMPC2FullCopy.json").read_text()
    )["alg_params"]
    debug = json.loads((ROOT / "configs/algs/AntAMBITDMPC2Debug.json").read_text())[
        "alg_params"
    ]

    for params in (main, full_copy):
        cfg = _build_cfg(**params)
        assert cfg.model_size == 5
        assert cfg.q_representation == "distributional"
        assert cfg.num_q == 5
        assert cfg.inner_actor_adaptation == "clone"
        assert cfg.inner_critic_adaptation == "clone"
        assert cfg.inner_rounds == 4
        assert cfg.inner_rollouts_per_round == 64
        assert cfg.inner_rollout_horizon == 3
        assert cfg.inner_updates_per_round == "auto"

    debug_cfg = _build_cfg(**debug)
    assert debug_cfg.model_size == 1
    assert debug_cfg.q_representation == "distributional"
    assert debug_cfg.num_q == 2
    assert debug_cfg.inner_rounds == 2
    assert debug_cfg.inner_rollouts_per_round == 4
    assert debug_cfg.inner_rollout_horizon == 3
    assert debug_cfg.inner_updates_per_round == "auto"
