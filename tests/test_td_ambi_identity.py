"""Inner objective choices remain scientific state without changing old defaults."""

from copy import deepcopy

import pytest

from tests.test_ambi_inner_decoupling import _fixed_reward_model
from utils.resume_identity import scientific_trial_parameters


def test_historical_inner_defaults_keep_their_scientific_identity():
    old = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {}}
    explicit = deepcopy(old)
    explicit["alg_params"].update(
        inner_critic_loss_coef=1.0,
        inner_actor_loss_scale_update="per_action",
        inner_critic_target_initialization="online",
        inner_actor_adam_eps=1e-8,
    )
    assert scientific_trial_parameters(old) == scientific_trial_parameters(explicit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("inner_critic_loss_coef", 0.1),
        ("inner_actor_loss_scale_update", "per_update"),
        ("inner_critic_target_initialization", "outer_target"),
        ("inner_actor_adam_eps", 1e-5),
    ],
)
def test_inner_tdmpc2_choices_change_scientific_identity(field, value):
    old = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {}}
    changed = deepcopy(old)
    changed["alg_params"][field] = value
    assert scientific_trial_parameters(old) != scientific_trial_parameters(changed)


def test_checkpoint_records_inner_tdmpc2_update_semantics():
    holder = _fixed_reward_model(
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        inner_critic_loss_coef=0.1,
        inner_actor_loss_scale_update="per_update",
        inner_critic_target_initialization="outer_target",
        inner_actor_adam_eps=1e-5,
    )
    try:
        spec = holder.agent._critic_target_spec()["inner_solve"]
        assert spec["critic_loss_coef"] == 0.1
        assert spec["actor_loss_scale_update"] == "per_update"
        assert spec["critic_target_initialization"] == "outer_target"
        assert spec["actor_adam_eps"] == 1e-5
    finally:
        holder.env.close()
