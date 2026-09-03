import copy
import warnings

import gymnasium as gym
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.common.search_config import resolve_inner_search_semantics


def _base_params(**overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "simnorm_dim": 4,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 4,
        "outer_planning_horizon": 2,
        "buffer_size": 32,
        "seed_steps": 2,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "wandb": False,
        "dropout": 0.0,
        "q_representation": "scalar",
        "num_q": 2,
        "q_num_bins": 11,
        "q_vmin": -5,
        "q_vmax": 5,
        "inner_operator": "sac",
        "inner_q_objective": "finite_horizon",
        "inner_critic_horizon_mode": "shared",
        "inner_return_estimator": "td0",
        "inner_return_steps": None,
        "inner_return_lambda": None,
        "inner_leaf_q_source": "outer_target",
        "inner_leaf_value_samples": 1,
        "inner_search_replay_retention": "action",
        "inner_offpolicy_mode": "none",
        "inner_search_bootstrap_critic": "target",
        "inner_target_update_event": "optimizer_step",
        "inner_depth_update_order": "mixed",
        "inner_rounds": 2,
        "inner_rollouts_per_round": 3,
        "inner_rollout_horizon": 4,
        "inner_critic_updates_per_round": 1,
        "inner_actor_updates_per_round": 1,
        "inner_batch_size": 2,
        "inner_replay_capacity": 24,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "inherit_outer",
        "inner_critic_target_tau": 0.25,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 0,
        "inner_behavior_action": "policy_sample",
        "inner_behavior_std_scale": 1.0,
        "inner_behavior_noise_std": 0.0,
        "inner_mppi_num_elites": 2,
        "inner_mppi_num_pi_trajs": 0,
    }
    params.update(overrides)
    return params


def _finite_params(layout, estimator, *, retention, offpolicy, **overrides):
    params = _base_params(
        inner_critic_horizon_mode=layout,
        inner_return_estimator=estimator,
        inner_search_replay_retention=retention,
        inner_offpolicy_mode=offpolicy,
    )
    if estimator == "n_step":
        params["inner_return_steps"] = 2
    elif estimator in {"lambda_return", "retrace"}:
        params["inner_return_lambda"] = 0.5
    elif estimator == "full_suffix":
        params.update(
            inner_search_bootstrap_critic="none",
            inner_target_update_event="none",
        )
    params.update(overrides)
    return params


def _vtrace_params(**overrides):
    params = _base_params(
        inner_operator="vtrace",
        inner_return_lambda=0.5,
        inner_offpolicy_mode="per_decision_is",
        outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only",
        inner_vtrace_distill_updates=1,
        inner_vtrace_distill_action_samples=1,
    )
    params.update(overrides)
    return params


def _build_cfg(params, env):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = env
    algorithm.run_params = {
        "alg_params": params,
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.custom_params = params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return algorithm._build_cfg({"device": "cpu", **params})


def _model(params):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return AMBITDMPC2(
        "AMBITDMPC2",
        env,
        params,
        {"seed": 73, "device": "cpu", "env": "test", "total_steps": 10},
        {},
    )


_RETURN_REUSE_CASES = {
    "td0": (("round", "none"), ("action", "none")),
    "n_step": (
        ("round", "none"),
        ("action", "uncorrected"),
        ("action", "per_decision_is"),
        ("action", "resimulate"),
    ),
    "lambda_return": (
        ("round", "none"),
        ("action", "uncorrected"),
        ("action", "per_decision_is"),
        ("action", "resimulate"),
    ),
    "full_suffix": (
        ("round", "none"),
        ("action", "uncorrected"),
        ("action", "per_decision_is"),
        ("action", "resimulate"),
    ),
    "retrace": (("action", "per_decision_is"),),
}


def test_complete_valid_search_semantic_cross_product_resolves():
    """Cover every Q layout/return/reuse/leaf/ordinary target combination."""

    target_strategies = (
        ("target", "optimizer_step"),
        ("target", "round_end"),
        ("frozen_target", "none"),
        ("online", "none"),
    )
    resolved = 0
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    try:
        for layout in ("shared", "depth_conditioned", "stage_heads"):
            for estimator, reuse_cases in _RETURN_REUSE_CASES.items():
                for retention, offpolicy in reuse_cases:
                    strategies = (
                        (("none", "none"),)
                        if estimator == "full_suffix"
                        else target_strategies
                    )
                    for bootstrap, event in strategies:
                        for leaf_source in ("outer_target", "outer_online"):
                            cfg = _build_cfg(
                                _finite_params(
                                    layout,
                                    estimator,
                                    retention=retention,
                                    offpolicy=offpolicy,
                                    inner_leaf_q_source=leaf_source,
                                    inner_search_bootstrap_critic=bootstrap,
                                    inner_target_update_event=event,
                                ),
                                env,
                            )
                            semantics = resolve_inner_search_semantics(cfg)
                            assert semantics.is_finite_q
                            assert semantics.critic_horizon_mode == layout
                            assert semantics.return_estimator == estimator
                            assert semantics.replay_retention == retention
                            assert semantics.offpolicy_mode == offpolicy
                            assert semantics.leaf_q_source == leaf_source
                            assert semantics.bootstrap_critic == bootstrap
                            assert semantics.target_update_event == event
                            assert cfg.inner_bootstrap_source is None
                            resolved += 1

        # Hard leaf-to-root propagation is the additional valid target family.
        for layout in ("depth_conditioned", "stage_heads"):
            for retention in ("round", "action"):
                for leaf_source in ("outer_target", "outer_online"):
                    cfg = _build_cfg(
                        _finite_params(
                            layout,
                            "td0",
                            retention=retention,
                            offpolicy="none",
                            inner_leaf_q_source=leaf_source,
                            inner_target_update_event="depth_stage",
                            inner_depth_update_order="backward",
                            inner_critic_target_tau=1.0,
                        ),
                        env,
                    )
                    assert cfg.inner_effective_target_update_event == "depth_stage"
                    resolved += 1

        # V-trace has its own complete layout/retention/leaf/target surface.
        for layout in ("shared", "depth_conditioned", "stage_heads"):
            for retention in ("round", "action"):
                for leaf_source in ("outer_target", "outer_online"):
                    for bootstrap, event in (
                        ("target", "optimizer_step"),
                        ("target", "round_end"),
                        ("frozen_target", "none"),
                    ):
                        cfg = _build_cfg(
                            _vtrace_params(
                                inner_critic_horizon_mode=layout,
                                inner_search_replay_retention=retention,
                                inner_leaf_q_source=leaf_source,
                                inner_search_bootstrap_critic=bootstrap,
                                inner_target_update_event=event,
                            ),
                            env,
                        )
                        semantics = resolve_inner_search_semantics(cfg)
                        assert semantics.is_vtrace
                        assert semantics.target_update_event == event
                        resolved += 1
    finally:
        env.close()

    # This count makes accidental loss of a coherent cell visible.
    assert resolved == 332


@pytest.mark.parametrize(
    ("estimator", "retention", "offpolicy", "message"),
    [
        (estimator, retention, offpolicy, message)
        for estimator in _RETURN_REUSE_CASES
        for retention in ("round", "action")
        for offpolicy in ("none", "uncorrected", "per_decision_is", "resimulate")
        if (retention, offpolicy) not in _RETURN_REUSE_CASES[estimator]
        for message in (
            (
                "TD\\(0\\).*needs no trajectory importance correction"
                if estimator == "td0"
                else "Retrace requires action-retained replay"
                if estimator == "retrace"
                else "Fresh-round multistep trajectories"
                if retention == "round"
                else "Action-retained multistep replay"
            ),
        )
    ],
)
def test_invalid_return_replay_correction_cartesian_complement_is_rejected(
    estimator, retention, offpolicy, message
):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    try:
        with pytest.raises(ValueError, match=message):
            _build_cfg(
                _finite_params(
                    "shared",
                    estimator,
                    retention=retention,
                    offpolicy=offpolicy,
                ),
                env,
            )
    finally:
        env.close()


@pytest.fixture(scope="module")
def finite_exact_model():
    model = _model(
        _finite_params(
            "shared", "td0", retention="action", offpolicy="none"
        )
    )
    yield model
    model.env.close()


@pytest.fixture(scope="module")
def vtrace_exact_model():
    model = _model(_vtrace_params())
    yield model
    model.env.close()


@pytest.mark.parametrize(
    ("config_field", "spec_field", "replacement"),
    [
        ("inner_operator", "operator", "vtrace"),
        ("inner_q_objective", "q_objective", "legacy_continuing"),
        (
            "inner_critic_horizon_mode",
            "critic_horizon_mode",
            "stage_heads",
        ),
        ("inner_return_estimator", "return_estimator", "n_step"),
        ("inner_return_steps", "return_steps", 2),
        ("inner_return_lambda", "return_lambda", 0.5),
        ("inner_leaf_q_source", "leaf_q_source", "outer_online"),
        ("inner_leaf_value_samples", "leaf_value_samples", 2),
        (
            "inner_search_replay_retention",
            "replay_retention",
            "round",
        ),
        ("inner_offpolicy_mode", "offpolicy_mode", "per_decision_is"),
        (
            "inner_search_bootstrap_critic",
            "bootstrap_critic",
            "frozen_target",
        ),
        ("inner_target_update_event", "target_update_event", "round_end"),
        ("inner_depth_update_order", "depth_update_order", "backward"),
    ],
)
def test_exact_resume_rejects_each_finite_search_semantic_field(
    finite_exact_model, config_field, spec_field, replacement
):
    assert config_field.startswith("inner_")
    state = copy.deepcopy(finite_exact_model.agent.training_state_dict())
    state["outer"]["critic_target_spec"]["inner_search"][spec_field] = (
        replacement
    )
    with pytest.raises(ValueError, match="critic-target specification"):
        finite_exact_model.agent.load_training_state_dict(state)


@pytest.mark.parametrize(
    ("config_field", "spec_field", "replacement"),
    [
        ("inner_vtrace_rho_clip", "rho_clip", 0.75),
        ("inner_vtrace_c_clip", "c_clip", 0.75),
        ("inner_vtrace_pg_rho_clip", "pg_rho_clip", 0.75),
        ("inner_vtrace_distill_updates", "distill_updates", 2),
        (
            "inner_vtrace_distill_action_samples",
            "distill_action_samples",
            2,
        ),
    ],
)
def test_exact_resume_rejects_each_vtrace_semantic_field(
    vtrace_exact_model, config_field, spec_field, replacement
):
    assert config_field.startswith("inner_vtrace_")
    state = copy.deepcopy(vtrace_exact_model.agent.training_state_dict())
    state["outer"]["critic_target_spec"]["inner_search"]["vtrace"][
        spec_field
    ] = replacement
    with pytest.raises(ValueError, match="critic-target specification"):
        vtrace_exact_model.agent.load_training_state_dict(state)


def _assert_outer_model_equal(left, right):
    assert tuple(left.state_dict()) == tuple(right.state_dict())
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


def test_portable_outer_checkpoint_transfers_across_every_search_layout_family():
    source = _model(
        _finite_params(
            "shared", "td0", retention="action", offpolicy="none"
        )
    )
    legacy = _model(_base_params(inner_q_objective="legacy_continuing"))
    receivers = [
        _model(
            _finite_params(
                "depth_conditioned",
                "n_step",
                retention="round",
                offpolicy="none",
            )
        ),
        _model(
            _finite_params(
                "stage_heads",
                "full_suffix",
                retention="action",
                offpolicy="resimulate",
            )
        ),
        _model(
            _vtrace_params(
                inner_critic_horizon_mode="stage_heads",
                inner_search_replay_retention="round",
            )
        ),
    ]
    all_models = [source, legacy, *receivers]
    try:
        checkpoint = source.agent.checkpoint_state()
        top_level_keys = tuple(checkpoint)
        model_keys = tuple(checkpoint["model"])
        assert tuple(legacy.agent.checkpoint_state()) == top_level_keys
        assert tuple(legacy.agent.model.state_dict()) == model_keys
        for receiver in receivers:
            assert tuple(receiver.agent.checkpoint_state()) == top_level_keys
            assert tuple(receiver.agent.model.state_dict()) == model_keys
            receiver.agent.load(checkpoint)
            _assert_outer_model_equal(receiver.agent.model, source.agent.model)
    finally:
        for model in all_models:
            model.env.close()
