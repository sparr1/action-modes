"""Configuration and collection contracts for selectable prior distillation."""

import pytest
import torch

from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_inner_decoupling import _model
from utils.resume_identity import scientific_trial_parameters


def test_default_prior_objective_preserves_existing_recipe():
    cfg = _build_cfg()
    assert cfg.outer_actor_objective == "q_maximization"
    assert cfg.reanalyze_interval == 10
    assert cfg.reanalyze_batch_size == 20
    assert cfg.store_teacher_policy is False
    assert cfg.store_behavior_policy is False


def test_distillation_preserves_entropy_and_q_settings():
    common = dict(
        outer_actor_entropy_mode="tdmpc2_scaled", ent_coef=0.004,
        outer_critic_target="reward_only", inner_sac_critic_target="entropy_augmented",
        inner_target_entropy="auto", outer_q_target_reduction="mean_all",
        inner_q_actor_reduction="mean_pair",
    )
    baseline = _build_cfg(**common)
    cfg = _build_cfg(outer_actor_objective="DISTILLATION", **common)
    assert cfg.outer_actor_objective == "distillation"
    assert cfg.store_teacher_policy is True
    assert cfg.store_behavior_policy is False
    for key in common:
        assert getattr(cfg, key) == getattr(baseline, key)
    cached = _build_cfg(outer_actor_objective="distillation", reanalyze_interval=0)
    assert cached.reanalyze_interval == 0


@pytest.mark.parametrize("key,value", [
    ("outer_actor_objective", "both"), ("outer_actor_objective", None),
    ("reanalyze_interval", -1), ("reanalyze_interval", True),
    ("reanalyze_interval", 1.5), ("reanalyze_batch_size", 0),
    ("reanalyze_batch_size", False), ("reanalyze_batch_size", 1.5),
])
def test_invalid_distillation_controls_are_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: value})


@pytest.mark.parametrize("overrides,match", [
    ({"inner_operator": "td3"}, "inner_operator"),
    ({"inner_actor_adaptation": "frozen"}, "inner_actor_adaptation"),
    ({"inner_actor_initialization": "random"}, "inner_actor_initialization"),
    ({"inner_explorer_mode": "frozen_random"}, "inner_explorer_mode"),
    ({"inner_critic_bernoulli_p": 0.5}, "inner_critic_bernoulli_p"),
    ({"inner_actor_scope": "episode"}, "inner_actor_scope"),
    ({"inner_replay_scope": "run"}, "inner_replay_scope"),
    ({"inner_actor_writeback_coef": 0.1}, "inner_actor_writeback_coef"),
    ({"inner_critic_writeback_coef": 0.1}, "inner_critic_writeback_coef"),
    ({"outer_behavior_policy_kl_schedule": "smooth"}, "outer_behavior_policy_kl_schedule"),
    ({"inner_updates_per_round": 0}, "positive actor updates"),
])
def test_distillation_rejects_incompatible_inner_recipes(overrides, match):
    with pytest.raises(ValueError, match=match):
        _build_cfg(outer_actor_objective="distillation", **overrides)


def test_disabled_objective_does_not_restrict_existing_inner_variants():
    cfg = _build_cfg(inner_actor_initialization="random")
    assert cfg.outer_actor_objective == "q_maximization"
    assert cfg.inner_actor_initialization == "random"


def test_distillation_requires_nonnegative_temporal_weights():
    with pytest.raises(ValueError, match="rho >= 0"):
        _build_cfg(outer_actor_objective="distillation", rho=-1.0)
    cfg = _build_cfg(outer_actor_objective="distillation", rho=0.0)
    assert cfg.rho == 0.0
    # Existing Q-maximization configurations keep their previous validation.
    baseline = _build_cfg(rho=-1.0)
    assert baseline.rho == -1.0


def test_distillation_retains_real_replay_mixing_and_horizon_conditioning():
    cfg = _build_cfg(
        outer_actor_objective="distillation", inner_outer_replay_fraction=0.5,
    )
    assert cfg.inner_outer_replay_fraction == 0.5
    conditioned = _build_cfg(
        outer_actor_objective="distillation", inner_finite_horizon=True,
        inner_horizon_conditioning="one_hot", inner_diagnostic_rollouts=0,
    )
    assert conditioned.inner_horizon_conditioning == "one_hot"


def _identity(params):
    return scientific_trial_parameters({
        "alg": "AMBITDMPC2/AMBITDMPC2", "seed": 7, "alg_params": params,
    })


def test_distillation_identity_preserves_disabled_fingerprints():
    assert _identity({}) == _identity({
        "outer_actor_objective": "q_maximization", "reanalyze_interval": 10,
        "reanalyze_batch_size": 20, "store_teacher_policy": False,
    })
    enabled = _identity({"outer_actor_objective": "distillation"})
    assert enabled == _identity({
        "outer_actor_objective": "DISTILLATION", "reanalyze_interval": 10,
        "reanalyze_batch_size": 20, "store_teacher_policy": True,
    })
    assert enabled != _identity({})
    assert enabled != _identity({"outer_actor_objective": "distillation", "reanalyze_interval": 0})
    assert enabled != _identity({"outer_actor_objective": "distillation", "reanalyze_batch_size": 4})


def test_collection_stages_teacher_on_successor_and_preserves_growth():
    model = _model(outer_actor_objective="distillation")
    try:
        obs = torch.zeros(3)
        assert model._start_episode_staging(obs) == 1
        assert not model._episode_staging["teacher_valid"][0]
        assert model._episode_staging["teacher_outer_version"][0] == -1
        assert "behavior_policy_valid" not in model._episode_staging.keys()
        model._pending_teacher_policy = {
            "pre_tanh_mean": torch.tensor([0.75]),
            "log_std": torch.tensor([-1.25]), "outer_version": 8,
        }
        model._stage_transition(1, obs, torch.tensor([0.2]), 1.0, False)
        assert model._pending_teacher_policy is None
        assert model._episode_staging["teacher_valid"][1]
        assert model._episode_staging["teacher_outer_version"][1] == 8
        model._ensure_episode_staging_capacity(len(model._episode_staging) + 1)
        torch.testing.assert_close(model._episode_staging["teacher_pre_tanh_mean"][1], torch.tensor([0.75]))
        model._random_action_norm()
        model._stage_transition(2, obs, torch.tensor([0.1]), 0.0, False)
        assert not model._episode_staging["teacher_valid"][2]
        assert model._episode_staging["teacher_outer_version"][2] == -1
    finally:
        model.env.close()


def test_wrapper_captures_teacher_and_direct_prior_remains_invalid(monkeypatch):
    model = _model(outer_actor_objective="distillation")
    try:
        obs = torch.zeros(3)
        model._start_episode_staging(obs)
        calls = []

        def act(*args, **kwargs):
            calls.append(kwargs)
            model.agent.last_teacher_policy = {
                "pre_tanh_mean": torch.tensor([0.5]), "log_std": torch.tensor([-0.7]),
            }
            return torch.tensor([0.1])

        monkeypatch.setattr(model.agent, "act", act)
        model.agent.outer_version = 9
        model._act_agent(obs, t0=True, eval_mode=False)
        assert calls[0]["capture_teacher_policy"] is True
        assert model._pending_teacher_policy["outer_version"] == 9
        with pytest.raises(RuntimeError, match="Teacher-policy metadata"):
            model._act_agent(obs, t0=False, eval_mode=False)
        model._stage_transition(1, obs, torch.tensor([0.1]), 0.0, False)
        model._outer_policy_episode_selected = True
        model._outer_policy_action_generator = torch.Generator().manual_seed(1)
        monkeypatch.setattr(model.agent, "act_outer_policy", lambda *a, **kw: torch.tensor([0.2]))
        model._act_agent(obs, t0=False, eval_mode=False)
        model._stage_transition(2, obs, torch.tensor([0.2]), 0.0, False)
        assert not model._episode_staging["teacher_valid"][2]
    finally:
        model.env.close()


def test_distillation_metrics_flow_through_existing_update_logging():
    model = _model()
    try:
        model._accumulate_train_metrics({
            "distillation_kl": torch.tensor(0.3),
            "distillation_reanalyze_roots": 12,
        })
        metrics = model._wandb_update_window.pop_floats()
        assert metrics["train/distillation_kl"] == pytest.approx(0.3)
        assert metrics["train/distillation_reanalyze_roots"] == 12
    finally:
        model.env.close()
