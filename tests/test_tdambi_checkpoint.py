"""Native-checkpoint transfer and evaluation-only TDAMBI configuration."""

import copy
import json
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from RL.TDAMBI import TDAMBI, native_evaluation_params
from RL.TDMPC2 import TDMPC2Baseline
from utils.ambi_research import PresetMatrixError, load_preset_matrix, normalize_selectors, resolve_preset
from utils.checkpoint_context import load_checkpoint_context


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/tdambi_humanoid_inner_benchmark.json"


def tiny_native_params(**overrides):
    params = dict(device="cpu", model_size=None, enc_dim=16, mlp_dim=16,
                  latent_dim=8, num_enc_layers=2, num_q=3, simnorm_dim=8,
                  num_bins=11, vmin=-5, vmax=5, batch_size=4, buffer_size=100,
                  episode_length=5, seed_steps=4, train_unroll_horizon=2,
                  outer_planning_horizon=2, dropout=0.2, mpc=False,
                  entropy_coef=0.0003, lr=0.0002, tau=0.03, value_coef=0.17)
    params.update(overrides)
    return params


def make_tdambi(**overrides):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    params = tiny_native_params(**{**dict(inner_rounds=1, inner_rollouts_per_round=4,
                                          inner_rollout_horizon=2, inner_updates_per_round=0,
                                          inner_batch_size=4, inner_replay_capacity=16),
                                    **overrides})
    return TDAMBI("TDAMBI", env, params,
                  {"seed": 3, "device": "cpu", "env": "Pendulum-v1", "total_steps": 10},
                  {"frozen_checkpoint_evaluation": True})


@pytest.fixture
def pair():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    native = TDMPC2Baseline("TDMPC2", env, tiny_native_params(),
                            {"seed": 3, "device": "cpu", "env": "Pendulum-v1", "total_steps": 10}, {})
    adapted = make_tdambi()
    # Different online/target weights exercise the saved-target contract.
    with torch.no_grad():
        for head in native.agent.model._Qs:
            head[-1].bias.copy_(torch.linspace(-0.4, 0.6, 11))
        for head in native.agent.model._target_Qs:
            head[-1].bias.copy_(torch.linspace(0.7, -0.2, 11))
    native.agent.model.eval()
    adapted.agent.load(copy.deepcopy(native.agent.checkpoint_state()))
    yield native, adapted
    native.flush_checkpoints()
    adapted.close()
    env.close()
    adapted.env.close()


def test_native_checkpoint_preserves_model_predictions_and_saved_target(pair):
    native, adapted = pair
    source, target = native.agent.model, adapted.agent.model
    obs = torch.tensor([[0.2, -0.3, 0.7], [-0.1, 0.0, 0.4]])
    z = source.encode(obs, None)
    torch.testing.assert_close(target.encode(obs), z, rtol=0, atol=0)
    torch.manual_seed(71)
    action, info = source.pi(z, None)
    torch.manual_seed(71)
    copied_action, copied_info = target.pi_tdmpc2(z)
    torch.testing.assert_close(copied_action, action, rtol=0, atol=0)
    for key in ("mean", "log_std", "entropy", "scaled_entropy"):
        torch.testing.assert_close(copied_info[key], info[key], rtol=0, atol=0)
    torch.testing.assert_close(target.next(z, action), source.next(z, action, None), rtol=0, atol=0)
    torch.testing.assert_close(target.reward(z, action), source.reward(z, action, None), rtol=0, atol=0)
    for target_q in (False, True):
        for native_reduction, shared_reduction in (("min", "min_pair"), ("avg", "mean_pair")):
            torch.manual_seed(98)
            actual = target.Q(z, action, return_type=shared_reduction, target=target_q)
            torch.manual_seed(98)
            expected = source.Q(z, action, None, return_type=native_reduction, target=target_q)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(target._Qs[0][-1].bias, target._target_Qs[0][-1].bias)
    assert all(not param.requires_grad for param in target.parameters())


def test_no_update_prediction_matches_native_central_action(pair):
    native, adapted = pair
    observation = [0.2, -0.3, 0.7]
    expected, _ = native.predict(observation, deterministic=True, episode_start=True)
    actual, _ = adapted.predict(observation, deterministic=True, episode_start=True)
    assert (actual == expected).all()
    assert adapted.agent.last_inner_metrics["inner_actor_optimizer_steps"] == 0
    assert adapted.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 0


@pytest.mark.parametrize("mutation,match", [
    ("target", "saved target"), ("bounds", "policy bounds"),
    ("shape", "architecture"), ("nan", "non-finite"), ("sac", "native TD-MPC2"),
])
def test_load_preflight_rejects_incompatible_state_without_mutation(pair, mutation, match):
    native, adapted = pair
    state = copy.deepcopy(native.agent.checkpoint_state())
    if mutation == "target":
        state["model"] = {key: value for key, value in state["model"].items()
                          if not key.startswith("_target_Qs.")}
    elif mutation == "bounds":
        state["model"]["log_std_min"] = state["model"]["log_std_min"] + 1
    elif mutation == "shape":
        state["model"]["_pi.2.weight"] = torch.zeros(4, 16)
    elif mutation == "nan":
        state["model"]["_pi.2.weight"][0, 0] = float("nan")
    else:
        state["entropy_spec"] = {}
    before = copy.deepcopy(adapted.agent.model.state_dict())
    with pytest.raises(ValueError, match=match):
        adapted.agent.load(state)
    for key, expected in before.items():
        torch.testing.assert_close(adapted.agent.model.state_dict()[key], expected, rtol=0, atol=0)


def test_tdambi_rejects_training_and_stochastic_execution(pair):
    _, adapted = pair
    with pytest.raises(RuntimeError, match="evaluation-only"):
        adapted.learn(total_timesteps=1)
    with pytest.raises(RuntimeError, match="evaluation-only"):
        adapted.agent.update(None)
    with pytest.raises(ValueError, match="central action"):
        adapted.predict([0.0, 0.0, 0.0], deterministic=False)
    with pytest.raises(ValueError, match="evaluation-only"):
        TDAMBI("TDAMBI", adapted.env, tiny_native_params(), {}, {})


@pytest.mark.parametrize("override", [
    {"inner_finite_horizon": True}, {"inner_explorer_mode": "shared_mixture"},
    {"inner_critic_dropout_enabled": False}, {"inner_actor_writeback_coef": 0.1},
    {"inner_actor_scope": "episode"}, {"inner_temperature_mode": "auto"},
    {"inner_actor_adaptation": "lora"}, {"inner_bootstrap_source": "outer_online"},
    {"inner_actor_updates_per_round": 3, "inner_critic_updates_per_round": 3},
    {"inner_log_std_min": -12}, {"q_num_bins": 21}, {"compile": True},
])
def test_tdambi_rejects_non_native_or_unsupported_controls(override):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    instance = object.__new__(TDAMBI)
    instance.env = env
    instance.run_params = {"device": "cpu"}
    try:
        with pytest.raises(ValueError, match="TDAMBI"):
            instance._build_cfg({**tiny_native_params(), **override})
    finally:
        env.close()


def _native_context(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"metadata-only")
    payload = {"schema_version": 1, "checkpoint": {"kind": "step", "step": 100000,
                                                  "episode": 100, "best_score": None,
                                                  "best_window": 100},
               "trial_run_params": {"alg": "TDMPC2/TDMPC2Baseline", "env": "Pendulum-v1",
                                    "seed": 55, "alg_params": tiny_native_params(
                                        train_unroll_horizon=3, outer_planning_horizon=3)},
               "experiment_params": {"env_params": {"max_episode_steps": 5}}}
    Path(f"{checkpoint}.metadata.json").write_text(json.dumps(payload))
    return load_checkpoint_context(checkpoint)


def test_checkpoint_matrix_inherits_native_settings_and_expands_three_budgets(tmp_path):
    context = _native_context(tmp_path)
    before = copy.deepcopy(context)
    matrix = load_preset_matrix(MATRIX)
    assert normalize_selectors(matrix) == ["inner_budget/tdambi_3"]
    for budget in (3, 6, 12):
        resolved = resolve_preset(MATRIX, f"inner_budget/tdambi_{budget}", checkpoint_context=context)
        assert resolved["algorithm_config"]["alg"] == "TDAMBI/TDAMBI"
        assert resolved["source_algorithm"] == "TDMPC2/TDMPC2Baseline"
        params = resolved["algorithm_config"]["alg_params"]
        instance = object.__new__(TDAMBI)
        instance.env = gym.make("Pendulum-v1", max_episode_steps=5)
        instance.run_params = {"device": "cpu"}
        try:
            cfg = instance._build_cfg(params)
        finally:
            instance.env.close()
        assert cfg.inner_operator == "tdambi"
        assert cfg.inner_rounds == 6
        assert cfg.inner_rollouts_per_round == 512
        assert cfg.inner_rollout_horizon == 3
        assert cfg.inner_batch_size == 512
        assert cfg.inner_replay_capacity == 12288
        assert cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action == 6 * budget
        assert cfg.inner_temperature_updates_per_action == 0
        assert cfg.inner_actor_lr == cfg.inner_critic_lr == 0.0002
        assert cfg.tdambi_entropy_coef == 0.0003
        assert cfg.tdambi_value_coef == 0.17
        assert cfg.tdambi_scale_tau == cfg.inner_critic_target_tau == 0.03
    assert context == before


def test_tdambi_matrix_rejects_ambi_backbone_and_architecture_override(tmp_path):
    context = _native_context(tmp_path)
    context.trial_run_params["alg"] = "AMBITDMPC2/AMBITDMPC2"
    with pytest.raises(PresetMatrixError, match="native TDMPC2"):
        resolve_preset(MATRIX, "inner_budget/tdambi_3", checkpoint_context=context)
    matrix = load_preset_matrix(MATRIX)
    matrix["shared_alg_params"]["num_q"] = 4
    with pytest.raises(PresetMatrixError, match="incompatible overrides"):
        resolve_preset(MATRIX, "inner_budget/tdambi_3", matrix, checkpoint_context=context)


def test_tdambi_evaluator_saves_frozen_bundle_and_native_provenance(tmp_path, monkeypatch):
    import gzip
    from evaluate_ambi_checkpoint import evaluate_matrix
    # Host Git subprocesses are independent of evaluator science and can fail
    # when macOS forks after PyTorch has initialized its thread pool.
    monkeypatch.setattr("utils.ambi_benchmark.code_identity",
                        lambda: {"commit": "test-implementation", "dirty": False})
    monkeypatch.setattr("evaluate_ambi_checkpoint._make_env", lambda resolved: gym.make(
        resolved["environment"]["id"], **resolved["environment"]["params"]
    ))

    env = gym.make("Pendulum-v1", max_episode_steps=3)
    native = TDMPC2Baseline("TDMPC2", env, tiny_native_params(),
                            {"seed": 55, "device": "cpu", "env": "Pendulum-v1", "total_steps": 10}, {})
    checkpoint = tmp_path / "native.pt"
    torch.save(native.agent.checkpoint_state(), checkpoint)
    native.flush_checkpoints()
    env.close()
    metadata = {
        "schema_version": 1,
        "checkpoint": {"kind": "step", "step": 100000, "episode": 5,
                       "best_score": None, "best_window": 100},
        "trial_run_params": {"alg": "TDMPC2/TDMPC2Baseline", "env": "Pendulum-v1",
                             "seed": 55, "alg_params": tiny_native_params()},
        "experiment_params": {"env_params": {"max_episode_steps": 3}},
    }
    Path(f"{checkpoint}.metadata.json").write_text(json.dumps(metadata))
    matrix = load_preset_matrix(MATRIX)
    matrix["shared_alg_params"].update(inner_rounds=1, inner_rollouts_per_round=4,
                                       inner_rollout_horizon=2, inner_batch_size=4,
                                       inner_replay_capacity=16)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    bundle_path = tmp_path / "bundle"
    payload = evaluate_matrix(matrix_path, checkpoint, seeds=[101], max_steps=3,
                               bundle_dir=bundle_path, source_run="entity/project/native-source")
    result = payload["results"][0]
    assert result["outer_state_unchanged"] is True
    assert result["episodes"][0]["length"] == 3
    assert result["episodes"][0]["truncated"] is True
    assert result["episodes"][0]["model_metrics"]["inner_actor_optimizer_steps"] == 3
    assert result["episodes"][0]["model_metrics"]["inner_critic_optimizer_steps"] == 3
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["checkpoint"]["source_run"] == "entity/project/native-source"
    assert manifest["checkpoint"]["metadata"]["trial_run_params"]["alg"] == "TDMPC2/TDMPC2Baseline"
    assert manifest["runs"][0]["config"]["alg"] == "TDAMBI/TDAMBI"
    traces = list(bundle_path.rglob("*.jsonl.gz"))
    assert len(traces) == 1
    with gzip.open(traces[0], "rt") as handle:
        events = [json.loads(line) for line in handle]
    assert sum(event["phase"] == "decision" for event in events) == 3
    assert sum(event["phase"] == "update" for event in events) == 9


def test_tdambi_rejects_bank_probes_before_environment_creation(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    context = _native_context(tmp_path)
    monkeypatch.setattr(evaluator, "_make_env", lambda resolved: pytest.fail("environment was created"))
    with pytest.raises(ValueError, match="episode evaluation only"):
        evaluator.evaluate_matrix(MATRIX, tmp_path / "checkpoint.pt",
                                  metadata_path=context.source,
                                  bundle_dir=tmp_path / "bundle", root_bank_path=tmp_path / "bank.json")


def test_native_model_size_retains_its_actual_ensemble_over_inactive_num_q():
    from RL.tdmpc2_core import MODEL_SIZE
    params = native_evaluation_params({"model_size": 5, "num_q": 2})
    assert params["num_q"] == MODEL_SIZE[5]["num_q"]
    assert native_evaluation_params({"model_size": None, "num_q": 2})["num_q"] == 2
