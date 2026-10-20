"""Two fresh one-million-decision controls vary only the actor entropy target."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params


ROOT = Path(__file__).resolve().parents[1]
CASES = [("10p5", -10.5), ("21", -21.)]
GROUP = "ambi-return-only-training-20261019"


def recipe(suffix):
    return json.loads((ROOT / f"configs/dmcontrol/algs/ambi_full_no_inner_reward_only_1m_entropy{suffix}.json").read_text())


@pytest.mark.parametrize("suffix,target", CASES)
def test_only_budget_entropy_target_and_publication_change(suffix, target):
    old = json.loads((ROOT / "configs/dmcontrol/algs/ambi_full_no_inner_reward_only.json").read_text())
    assert old["total_steps"] == 500000 and old["alg_params"]["target_entropy"] == -10.5
    expected = deepcopy(old)
    expected["total_steps"] = 1000000
    expected["alg_params"]["target_entropy"] = target
    current = recipe(suffix)
    for key in ("wandb_tags", "wandb_run_name"):
        expected["alg_params"][key] = current["alg_params"][key]
    assert current == expected
    cfg = _build_cfg(**current["alg_params"])
    assert cfg.inner_operator == "none" and not cfg.mpc
    assert cfg.inner_execution_action == "policy_sample" and cfg.inner_solve_interval == 1
    for key in ("inner_model_step_budget", "inner_rounds", "inner_mppi_iterations",
                "inner_actor_updates_per_action", "inner_critic_updates_per_action",
                "inner_temperature_updates_per_action"):
        assert getattr(cfg, key) == 0
    assert cfg.train_unroll_horizon == 3 and cfg.model_size == 5
    assert cfg.num_q == 5 and cfg.q_representation == "distributional"
    assert cfg.aux_return_mode == "off" and cfg.outer_critic_target == "reward_only"
    assert cfg.ent_coef == "auto_1.0" and cfg.target_entropy == target
    assert cfg.outer_actor_entropy_mode == "squashed" and cfg.sac_actor_loss_scale_mode == "none"
    assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.utd == 1
    assert cfg.buffer_size == 1000000 and cfg.batch_size == 256
    assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"


@pytest.mark.parametrize("suffix,target", CASES)
def test_single_cell_manifest_has_matching_budget_and_distinct_identity(suffix, target):
    current = recipe(suffix)
    manifest = json.loads((ROOT / f"configs/dmcontrol/experiments/ambi_return_only_no_inner_1m_entropy{suffix}.json").read_text())
    old_manifest = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_return_only_no_inner_training.json").read_text())
    expected = deepcopy(old_manifest)
    expected["configs"] = [f"ambi_full_no_inner_reward_only_1m_entropy{suffix}"]
    expected["overrides_alg"]["total_steps"] = 1000000
    expected["study_note"] = manifest["study_note"]
    assert manifest == expected
    assert manifest["trials"] == 1
    assert manifest["overrides_alg"]["total_steps"] == current["total_steps"] == 1000000
    assert manifest["overrides_alg"]["seed"] == current["seed"] == 55
    assert manifest["checkpoint_every"] == current["checkpoint_every"] == 25000
    assert current["total_steps"] // current["checkpoint_every"] == 40
    assert manifest["save_strat"] == current["save_strat"] == "all"
    assert "alg_params" not in manifest["overrides_alg"]
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    params = current["alg_params"]
    assert params["wandb_run_name"] == f"No inner | reward-only Q | auto entropy {target:g} | 1M decisions | seed55"
    assert params["wandb_project"] == "ambi" and params["wandb_group"] == GROUP
    tags = set(params["wandb_tags"])
    assert {"1m-decisions", "2m-environment-steps", "target-entropy-" + suffix,
            "no-inner-loop", "no-auxiliary-critic", "sampled-prior-actions",
            "reward-only-primary-critic", "automatic-actor-entropy", "exact-resume", GROUP} <= tags
    assert not {"500k-decisions", "1m-environment-steps"} & tags
    assert {tag for tag in tags if tag.startswith("target-entropy-")} == {"target-entropy-" + suffix}
    assert params["eval_freq"] is None and not params["eval_inner_comparison"]


@pytest.mark.parametrize("suffix,target", CASES)
def test_actual_learner_uses_requested_automatic_temperature_target(suffix, target):
    params = recipe(suffix)["alg_params"]
    params.update(device="cpu", compile=False, compile_strict=False, wandb=False,
                  model_size=None, enc_dim=32, mlp_dim=32, latent_dim=16,
                  num_enc_layers=2, simnorm_dim=8, batch_size=2,
                  seed_steps=4, pretrain_steps=1, buffer_size=32)
    learner = _model_from_params(params)
    try:
        agent = learner.agent
        assert agent.target_entropy == target
        assert agent.ent_coef_optim is not None and agent.log_ent_coef.requires_grad
        assert agent.alpha.item() == pytest.approx(1.)
        assert agent.aux_return is None
        assert not any("aux_return" in key for key in agent.model.state_dict())
        assert agent.inner_engine._actor_base is agent.model._pi
        assert learner.cfg.outer_critic_target == "reward_only"
    finally:
        learner.close()
        learner.env.close()
