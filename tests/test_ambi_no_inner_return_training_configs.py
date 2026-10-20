"""No-inner control isolates collection while preserving reward-only outer learning."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_finite
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params
from tests.test_pave_integration import _batch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "ambi_full_no_inner_reward_only"
GROUP = "ambi-return-only-training-20261019"


def recipe():
    return json.loads((ROOT / f"configs/dmcontrol/algs/{CONFIG}.json").read_text())


def test_recipe_only_replaces_planning_with_sampled_prior_collection():
    base = json.loads((ROOT / "configs/dmcontrol/algs/ambi_full_h3_mppi_reward_only.json").read_text())
    current = recipe()
    expected = deepcopy(base)
    params = expected["alg_params"]
    for key in list(params):
        if key.startswith("inner_mppi_") or key == "mppi_terminal_q_reduction":
            params.pop(key)
    params.update(inner_operator="none", inner_model_step_budget=0,
                  inner_execution_action="policy_sample")
    for key in ("wandb_tags", "wandb_run_name"):
        params[key] = current["alg_params"][key]
    assert current == expected
    assert not any(key.startswith("inner_mppi_") for key in current["alg_params"])

    cfg = _build_cfg(**current["alg_params"])
    assert cfg.inner_operator == "none" and cfg.mpc is False
    assert cfg.inner_solve_interval == 1 and not hasattr(cfg, "inner_hold_encoder")
    for key in ("inner_model_step_budget", "inner_rounds", "inner_rollouts_per_round",
                "inner_updates_per_round", "inner_mppi_iterations",
                "inner_critic_updates_per_action", "inner_actor_updates_per_action",
                "inner_temperature_updates_per_action", "inner_diagnostic_rollouts"):
        assert getattr(cfg, key) == 0, key
    assert cfg.inner_execution_action == "policy_sample"
    assert cfg.inner_execution_std_scale == 1. and cfg.inner_execution_noise_std == 0.
    assert cfg.inner_critic_bernoulli_p == 0.
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0.
    assert cfg.train_unroll_horizon == cfg.outer_planning_horizon == 3
    assert cfg.model_size == 5 and (cfg.enc_dim, cfg.mlp_dim, cfg.latent_dim) == (256, 512, 512)
    assert cfg.q_representation == "distributional" and cfg.critic_value_mode == "single"
    assert (cfg.num_q, cfg.q_num_bins, cfg.q_vmin, cfg.q_vmax, cfg.q_pair_size) == (5, 101, -10, 10, 2)
    assert cfg.outer_q_target_reduction == "min_pair" and cfg.outer_q_actor_reduction == "mean_pair"
    assert cfg.outer_critic_target == "reward_only" and cfg.aux_return_mode == "off"
    assert cfg.inner_actor_source == cfg.inner_critic_source == "sac"
    assert cfg.ent_coef == "auto_1.0" and cfg.target_entropy == -10.5
    assert cfg.outer_actor_entropy_mode == "squashed" and cfg.sac_actor_loss_scale_mode == "none"
    assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
    assert cfg.aux_return_joint_vmap is False
    assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.utd == 1
    assert cfg.batch_size == 256 and cfg.buffer_size == 1000000


def test_one_cell_manifest_preserves_training_and_publication_contract():
    manifest = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_return_only_no_inner_training.json").read_text())
    run = recipe()
    assert manifest["configs"] == [CONFIG] and manifest["trials"] == 1
    assert "alg_params" not in manifest["overrides_alg"]
    assert manifest["overrides_alg"]["seed"] == run["seed"] == 55
    assert manifest["overrides_alg"]["total_steps"] == run["total_steps"] == 500000
    assert manifest["checkpoint_every"] == run["checkpoint_every"] == 25000
    assert manifest["save_strat"] == run["save_strat"] == "all"
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    params = run["alg_params"]
    assert params["wandb_project"] == "ambi" and params["wandb_group"] == GROUP
    assert params["wandb_run_name"] == "No inner | reward-only Q | auto entropy | seed55"
    assert {"no-inner-loop", "zero-imagined-work", "sampled-prior-actions", "outer-unroll-H3",
            "no-auxiliary-critic", "reward-only-primary-critic", "automatic-actor-entropy",
            "target-entropy-10p5", GROUP} <= set(params["wandb_tags"])
    assert not {"mppi", "held-policy", "soft-Q-bootstrap", "episode-plan-warm-start"} & set(params["wandb_tags"])
    assert params["eval_freq"] is None and not params["eval_inner_comparison"]


@pytest.fixture
def learner():
    params = recipe()["alg_params"]
    params.update(device="cpu", compile=False, compile_strict=False, wandb=False,
                  model_size=None, enc_dim=32, mlp_dim=32, latent_dim=16,
                  num_enc_layers=2, simnorm_dim=8, batch_size=2,
                  seed_steps=4, pretrain_steps=1, buffer_size=32)
    result = _model_from_params(params)
    yield result
    result.close()
    result.env.close()


def test_training_action_samples_current_prior_without_imagination(learner, monkeypatch):
    agent, engine = learner.agent, learner.agent.inner_engine
    original = agent.model.pi_action
    samples = []

    def sample(z, **kwargs):
        assert kwargs["policy"] is agent.model._pi
        assert not kwargs.get("deterministic", False)
        assert kwargs["generator"] is not None
        action = original(z, **kwargs)
        mean = original(z, policy=agent.model._pi, deterministic=True)
        samples.append((action.detach().clone(), mean.detach().clone()))
        return action

    def forbidden(*args, **kwargs):
        pytest.fail("Prior-only collection attempted planning, Q evaluation or inner learning")

    before = {key: value.clone() for key, value in agent.model.state_dict().items()}
    monkeypatch.setattr(agent.model, "pi_action", sample)
    for method in ("next", "next_from_joint", "reward", "reward_from_joint", "Q"):
        monkeypatch.setattr(agent.model, method, forbidden)
    for method in ("_prepare_workspace", "_act_rl", "_act_mppi"):
        monkeypatch.setattr(engine, method, forbidden)
    for index in range(2):
        action = learner._act_agent(torch.zeros(3), t0=index == 0, eval_mode=False)
        torch.testing.assert_close(action, samples[index][0][0])
        assert not torch.equal(action, samples[index][1][0])
        metrics = agent.last_inner_metrics
        assert metrics["inner_policy_evaluations"] == 1
        for key in ("inner_active", "inner_model_steps", "inner_rollouts", "inner_updates",
                    "inner_mppi_iterations", "inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                    "inner_temperature_optimizer_steps", "inner_target_updates", "inner_q_evaluations",
                    "inner_buffer_size", "inner_replay_draws"):
            assert metrics[key] == 0, key
    assert len(samples) == 2 and not torch.equal(samples[0][0], samples[1][0])
    assert engine.state.actor is None and engine.state.critic is None and engine.state.replay is None
    for key, value in agent.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_reward_only_target_and_outer_actor_temperature_learning(learner, monkeypatch):
    agent = learner.agent
    assert agent.aux_return is None
    assert not any("aux_return" in key for key in agent.model.state_dict())
    assert not hasattr(agent.model, "_aux_return_Qs")
    assert not hasattr(agent.model, "_target_aux_return_Qs")
    assert agent.ent_coef_optim is not None and agent.log_ent_coef.requires_grad
    assert agent.alpha.item() == pytest.approx(1.) and agent.target_entropy == -10.5
    q_calls = []

    def policy(z, **kwargs):
        return z.new_zeros(*z.shape[:-1], 1), {
            "log_prob": z.new_full((*z.shape[:-1], 1), -999.),
            "entropy": z.new_full((*z.shape[:-1], 1), 999.),
        }

    def q(z, action, **kwargs):
        q_calls.append(kwargs)
        return z.new_full((*z.shape[:-1], 1), 7.)

    with monkeypatch.context() as patch:
        patch.setattr(agent.model, "pi", policy)
        patch.setattr(agent.model, "Q", q)
        target = agent._soft_td_target(torch.zeros(2, agent.cfg.latent_dim),
                                      torch.full((2, 1), 2.), torch.tensor([[0.], [1.]]))
    torch.testing.assert_close(target, torch.tensor([[2. + agent.discount * 7.], [2.]]))
    assert q_calls == [{"target": True, "reduction": "min_pair"}]

    before_actor = [parameter.detach().clone() for parameter in agent.model._pi.parameters()]
    before_q = [parameter.detach().clone() for parameter in agent.model._Qs.parameters()]
    before_alpha = agent.log_ent_coef.detach().clone()
    info = agent._update(*_batch(agent))
    _assert_finite(info)
    assert not any(key.startswith("aux_return_") for key in info)
    assert any(not torch.equal(before, after) for before, after in zip(before_actor, agent.model._pi.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(before_q, agent.model._Qs.parameters()))
    assert not torch.equal(agent.log_ent_coef, before_alpha)
