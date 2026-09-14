"""Post-launch, opt-in CUDA smoke for all six reward-only prior backbones."""

import os

import pytest
import torch

from tests.test_ambi_prior_sac_cuda_gate import _run_production_shape_cuda_gate


CASES = (
    ("ambi_prior_reward_clip_target21", "direct_clamp", -21.0, 1.0),
    ("ambi_prior_reward_clip_target10p5", "direct_clamp", -10.5, 1.0),
    ("ambi_prior_reward_smooth_target21", "tdmpc2_tanh", -21.0, 1.0),
    ("ambi_prior_reward_smooth_target10p5", "tdmpc2_tanh", -10.5, 1.0),
    ("ambi_prior_reward_clip_fixed0p0001", "direct_clamp", -21.0, 0.0001),
    ("ambi_prior_reward_smooth_fixed0p0001", "tdmpc2_tanh", -21.0, 0.0001),
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production-shape gate requires CUDA")
@pytest.mark.skipif(os.environ.get("AMBI_RUN_REAL_DMCONTROL_TESTS") != "1",
                    reason="set AMBI_RUN_REAL_DMCONTROL_TESTS=1 on an allocated GPU")
@pytest.mark.parametrize("name,mapping,target,alpha", CASES)
def test_reward_prior_production_shape_cuda_gate(tmp_path, name, mapping, target, alpha):
    _run_production_shape_cuda_gate(
        tmp_path, name, mapping, target,
        manifest_name="ambi_prior_reward_critic_study",
        critic_target="reward_only", expected_alpha=alpha,
    )
