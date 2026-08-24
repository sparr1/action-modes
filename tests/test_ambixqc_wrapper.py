import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest

import main as training_main
from RL.AMBIXQC import AMBIXQC
from RL.TDMPC2 import TDMPC2Baseline
from RL.xqc_core import OFFICIAL_XQC_COMMIT
from utils.resume_identity import SUPPORTED_RESUME_ALGORITHMS
from utils.wandb_utils import WandbAccumulator


ROOT = Path(__file__).resolve().parents[1]


def _build_cfg(**params):
    algorithm = object.__new__(AMBIXQC)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.experiment_params = {}
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def test_default_config_is_xqc_over_told_without_ambi_auxiliary_modes():
    cfg = _build_cfg()

    assert cfg.mpc is False
    assert cfg.utd == 1
    assert cfg.discount_min == pytest.approx(0.99)
    assert cfg.discount_max == pytest.approx(0.99)
    assert cfg.inner_operator == "xqc"
    assert cfg.inner_actor_adaptation == "clone"
    assert cfg.inner_critic_adaptation == "clone"
    assert cfg.inner_actor_scope == "action"
    assert cfg.inner_critic_scope == "action"
    assert cfg.inner_temperature_scope == "action"
    assert cfg.inner_replay_scope == "action"

    assert cfg.xqc_actor_net_arch == (256, 256, 256, 256)
    assert cfg.xqc_critic_net_arch == (512, 512, 512, 512)
    assert cfg.xqc_num_atoms == 101
    assert cfg.xqc_vmin == pytest.approx(-5.0)
    assert cfg.xqc_vmax == pytest.approx(5.0)
    assert cfg.xqc_tau == pytest.approx(0.005)
    assert cfg.xqc_policy_delay == 3
    assert cfg.xqc_init_temperature == pytest.approx(0.01)
    assert cfg.xqc_resolved_target_entropy == pytest.approx(-0.5)
    assert cfg.xqc_lr_transition_steps == 12
    assert cfg.xqc_official_commit == OFFICIAL_XQC_COMMIT
    assert cfg.xqc_reward_normalization is True

    # TOLD's reward classifier and XQC's critic distribution are deliberately
    # separate supports.
    assert cfg.num_bins == 101
    assert cfg.vmin == pytest.approx(-10.0)
    assert cfg.vmax == pytest.approx(10.0)
    assert cfg.xqc_vmin == pytest.approx(-5.0)
    assert cfg.xqc_vmax == pytest.approx(5.0)


def test_default_inner_budget_resolves_action_local_xqc_delay_counts():
    cfg = _build_cfg()

    assert cfg.inner_rounds == 2
    assert cfg.inner_rollouts_per_round == 32
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_updates_per_round == 4
    assert cfg.inner_model_step_budget == 192
    assert cfg.inner_replay_capacity == 192
    assert cfg.inner_expected_update_slots == 8
    assert cfg.inner_critic_updates_per_action == 8
    assert cfg.inner_actor_updates_per_action == 3
    assert cfg.inner_temperature_updates_per_action == 3
    assert cfg.inner_temperature_lr == pytest.approx(cfg.inner_actor_lr)


@pytest.mark.parametrize(
    "params",
    [
        {"mpc": True},
        {"utd": 2},
        {"inner_operator": "sac"},
        {"inner_actor_adaptation": "lora"},
        {"inner_actor_scope": "episode"},
        {"inner_mppi_iterations": 2},
        {"outer_q_target_reduction": "min_all"},
        {"q_representation": "distributional"},
        {"actor_lr": 3e-4},
        {"num_atoms": 201},
        {"temp_lr": 3e-4},
        {"normalize_last_layer": True},
        {"xqc_reward_normalization": False},
        {"reward_normalization": False},
        {"gamma": 0.5},
        {"learning_rate": 0.9},
        {"discount_min": 0.2},
        {"outer_planning_horizon": 5},
        {"inner_model_step_budget": 10},
        {"inner_adam_eps": 1e-8},
        {"xqc_gamma": 0.99},
        {"compile": True},
        {"compile_strict": True},
    ],
)
def test_incompatible_or_inert_options_fail_early(params):
    with pytest.raises(ValueError):
        _build_cfg(**params)


def test_first_port_rejects_pixels_at_the_wrapper_boundary():
    with pytest.raises(NotImplementedError, match="state observations only"):
        _build_cfg(obs="rgb")


def test_inner_replay_must_hold_the_fresh_action_tree():
    with pytest.raises(ValueError, match="hold all imagined transitions"):
        _build_cfg(
            inner_rounds=2,
            inner_rollouts_per_round=4,
            inner_rollout_horizon=3,
            inner_replay_capacity=23,
        )

    with pytest.raises(ValueError, match="with_replacement"):
        _build_cfg(inner_replay_sampling="without_replacement")


def test_long_inner_horizon_preserves_the_model_bias_warning():
    with pytest.warns(UserWarning, match="model-bias"):
        cfg = _build_cfg(
            train_unroll_horizon=2,
            inner_rollout_horizon=3,
        )

    assert cfg.inner_horizon_ratio == pytest.approx(1.5)


@pytest.mark.parametrize("discount", [True, -0.1, 0.0, 1.01, float("nan")])
def test_bellman_and_reward_normalizer_discount_is_validated_early(discount):
    with pytest.raises(ValueError, match="discount"):
        _build_cfg(discount=discount)


@pytest.mark.parametrize("value_coef", [True, 0.0, -1.0, float("nan")])
def test_xqc_value_loss_coefficient_is_validated_before_training(value_coef):
    with pytest.raises(ValueError, match="value_coef"):
        _build_cfg(value_coef=value_coef)


def test_real_transition_hook_forwards_boundaries_to_agent():
    wrapper = object.__new__(AMBIXQC)
    calls = []
    wrapper.agent = SimpleNamespace(
        observe_reward=lambda reward, terminated, truncated: calls.append(
            (reward, terminated, truncated)
        )
    )

    wrapper._observe_transition(2.5, False, True)

    assert calls == [(2.5, False, True)]


def test_native_tdmpc_transition_hook_remains_a_noop():
    wrapper = object.__new__(TDMPC2Baseline)
    assert wrapper._observe_transition(1.0, False, True) is None


def test_training_loop_observes_raw_reward_once_before_replay_staging():
    events = []

    class OneStepEnv:
        def step(self, action):
            events.append(("env", action))
            return "next", 3.25, False, True, {}

    class EmptyBuffer:
        num_eps = 0

        def add(self, rows):
            events.append(("buffer", rows))

    wrapper = object.__new__(TDMPC2Baseline)
    wrapper.env = OneStepEnv()
    wrapper.buffer = EmptyBuffer()
    wrapper.cfg = SimpleNamespace(seed_steps=10, episodic=False)
    wrapper._global_step = 0
    wrapper._episode_return = 0.0
    wrapper._episode_len = 0
    wrapper._eval_freq = None
    wrapper._checkpointing = None
    wrapper._last_reward = 0.0
    wrapper._last_terminated = False
    wrapper._last_truncated = False
    wrapper._last_info = {}
    wrapper._reuse_observation_tensor = lambda obs: obs
    wrapper._start_episode_staging = lambda obs: 1
    wrapper._random_action_norm = lambda: "raw-action"
    wrapper._record_action_metrics = lambda **kwargs: None
    wrapper._unscale_action = lambda action: action
    wrapper._observe_transition = lambda reward, terminated, truncated: events.append(
        ("observe", reward, terminated, truncated)
    )
    wrapper._stage_transition = (
        lambda row, obs, action, reward, terminated: events.append(
            ("stage", reward, terminated)
        )
    )
    wrapper._episode_staging = ["raw-replay"]
    wrapper._accumulate_reward_metrics = lambda *args: None
    wrapper._log_step = lambda *args: None
    wrapper._log_wandb_step = lambda *args, **kwargs: None
    wrapper._maybe_checkpoint = lambda: None

    completed, pending = wrapper._run_training_episode(
        "obs", 1, eval_pending=False
    )

    assert completed is True
    assert pending is False
    assert events == [
        ("env", "raw-action"),
        ("observe", 3.25, False, True),
        ("stage", 3.25, False),
        ("buffer", ["raw-replay"]),
    ]


def test_algorithm_registration_and_resume_identity_are_explicit():
    assert training_main._learn_resets_env_with_seed("AMBIXQC/AMBIXQC")
    assert "AMBIXQC/AMBIXQC" not in SUPPORTED_RESUME_ALGORITHMS


def test_runtime_metadata_records_xqc_semantics():
    cfg = _build_cfg()
    model = SimpleNamespace(
        cfg=cfg,
        agent=SimpleNamespace(
            critic_signature={
                "q_representation": "xqc_c51",
                "num_q": 2,
                "num_atoms": 101,
                "vmin": -5.0,
                "vmax": 5.0,
            }
        ),
        env=SimpleNamespace(),
    )

    metadata = training_main._resolved_runtime_metadata(
        model,
        trial_run_params={"alg": "AMBIXQC/AMBIXQC", "seed": 3},
    )

    assert metadata["algorithm"] == "AMBIXQC/AMBIXQC"
    assert metadata["critic"]["q_representation"] == "xqc_c51"
    assert metadata["xqc"]["xqc_official_commit"] == OFFICIAL_XQC_COMMIT
    assert metadata["xqc"]["xqc_policy_delay"] == 3
    assert metadata["xqc"]["xqc_adam_eps"] == pytest.approx(1e-8)
    assert metadata["xqc"]["xqc_optimizer_backend"] == "auto"
    assert metadata["xqc"]["xqc_reward_normalization"] is True
    assert metadata["xqc"]["discount"] == pytest.approx(0.99)
    assert metadata["inner_budget"]["inner_operator"] == "xqc"
    assert metadata["inner_budget"]["transitions_per_action"] == 192


def test_xqc_specific_inner_metrics_are_routed_to_the_shared_wandb_window():
    wrapper = object.__new__(AMBIXQC)
    wrapper.agent = SimpleNamespace(
        last_inner_rollout_lengths=[3],
        last_inner_metrics={
            "inner_active": 1.0,
            "inner_rollouts": 1.0,
            "inner_steps": 3.0,
            "inner_updates": 1.0,
            "inner_critic_optimizer_steps": 1.0,
            "inner_actor_optimizer_steps": 1.0,
            "inner_temperature_optimizer_steps": 1.0,
            "inner_q1_mean": 1.0,
            "inner_q2_mean": 0.5,
            "inner_q_disagreement_mean": 0.5,
            "inner_actor_update_accepted": 1.0,
            "inner_policy_entropy": 0.75,
            "inner_reward_scale": 2.5,
            "inner_actor_learning_rate": 5e-5,
            "inner_critic_learning_rate": 5e-5,
            "inner_temperature_learning_rate": 5e-5,
            "inner_algorithm_xqc": 1.0,
            "inner_final_outer_policy_kl": 0.25,
        },
    )
    wrapper._wandb_train_window = WandbAccumulator()
    wrapper._inner_steps_total = 0
    wrapper._inner_updates_total = 0
    wrapper._wandb_inner_seconds = 0.0
    wrapper._wandb_inner_actions = 0
    wrapper._wandb_inner_steps = 0

    wrapper._record_action_metrics(planned=True, action_seconds=0.1)
    payload = wrapper._wandb_train_window.pop()

    assert payload["train/inner_q1_mean"] == pytest.approx(1.0)
    assert payload["train/inner_q2_mean"] == pytest.approx(0.5)
    assert payload["train/inner_q_disagreement_mean"] == pytest.approx(0.5)
    assert payload["train/inner_actor_update_accepted"] == pytest.approx(1.0)
    assert payload["train/inner_policy_entropy"] == pytest.approx(0.75)
    assert payload["train/inner_reward_scale"] == pytest.approx(2.5)
    assert payload["train/inner_actor_learning_rate"] == pytest.approx(5e-5)
    assert payload["train/inner_algorithm_xqc"] == pytest.approx(1.0)
    assert payload["train/inner_final_outer_policy_kl"] == pytest.approx(0.25)
    assert payload["train/inner_final_outer_policy_kl_count"] == pytest.approx(1.0)
    assert payload["train/inner_final_outer_policy_kl_min"] == pytest.approx(0.25)
    assert payload["train/inner_final_outer_policy_kl_max"] == pytest.approx(0.25)


def test_exact_trainer_resume_is_explicitly_unsupported_in_v1():
    wrapper = object.__new__(AMBIXQC)
    with pytest.raises(NotImplementedError, match="not supported in v1"):
        wrapper.enable_training_resume(total_timesteps=12)


def test_training_budget_cannot_diverge_from_xqc_schedule():
    wrapper = object.__new__(AMBIXQC)
    wrapper.cfg = SimpleNamespace(steps=12)
    with pytest.raises(ValueError, match="construction-time step budget"):
        wrapper.learn(total_timesteps=13)


def test_real_wrapper_learn_smoke_exercises_outer_and_one_inner_xqc_action():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    params = {
        "device": "cpu",
        "model_size": 1,
        "xqc_actor_net_arch": [16],
        "xqc_critic_net_arch": [16],
        "xqc_num_atoms": 11,
        "buffer_size": 64,
        "batch_size": 2,
        "seed_steps": 5,
        "pretrain_steps": 1,
        "train_unroll_horizon": 1,
        "inner_rollout_horizon": 1,
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_updates_per_round": 1,
        "inner_batch_size": 2,
        "inner_replay_capacity": 2,
        "inner_diagnostics_every": 1,
        "eval_freq": None,
        "compile": False,
        "wandb": False,
    }
    run_params = {
        "seed": 7,
        "device": "cpu",
        "env": "Pendulum-v1",
        "total_steps": 7,
    }
    model = AMBIXQC("AMBIXQC", env, params, run_params, {})
    try:
        model.learn(total_timesteps=7)

        assert model._global_step == 7
        assert model.agent.num_updates == 2
        assert model.agent.reward_normalizer.count == pytest.approx(7.0)
        assert model.agent.inner_engine.action_index == 1
    finally:
        model.env.close()


def test_humanoid_smoke_config_is_small_and_unambiguously_ambixqc():
    algorithm = json.loads(
        (
            ROOT
            / "configs/dmcontrol/algs/ambixqc_humanoid_walk_state_smoke.json"
        ).read_text()
    )
    experiment = json.loads(
        (
            ROOT
            / "configs/dmcontrol/experiments/ambixqc_humanoid_walk_state_smoke.json"
        ).read_text()
    )

    assert algorithm["alg"] == "AMBIXQC/AMBIXQC"
    assert algorithm["alg_params"]["mpc"] is False
    assert algorithm["alg_params"]["utd"] == 1
    assert algorithm["alg_params"]["inner_rounds"] == 1
    assert algorithm["alg_params"]["inner_rollouts_per_round"] == 4
    assert algorithm["alg_params"]["inner_updates_per_round"] == 1
    assert algorithm["alg_params"]["seed_steps"] == 500
    assert algorithm["alg_params"]["pretrain_steps"] == 1
    assert algorithm["alg_params"]["buffer_size"] == 2000
    assert algorithm["alg_params"]["batch_size"] == 16
    assert algorithm["total_steps"] == algorithm["alg_params"]["seed_steps"] + 2
    assert experiment["configs"] == ["ambixqc_humanoid_walk_state_smoke"]
    assert experiment["study_type"] == "functional_smoke_test"
    assert experiment["env_params"]["task"] == "humanoid-walk"
    assert experiment["env_params"]["obs"] == "state"
