from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core import boom_diagnostics as diagnostics
from tests.test_ambi_boom import _agent, _batch, _labels
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree


def _pair(device="cpu", compiled=False):
    base = _agent(outer_boom_alignment="replay_likelihood", q_representation="distributional", num_q=5, q_pair_size=2,
                  outer_q_actor_reduction="mean_pair", dropout=.1,
                  outer_boom_diagnostics_early_every=1)
    cfg = deepcopy(base.cfg)
    cfg.device, cfg.compile, cfg.compile_strict = device, compiled, compiled
    cfg.outer_boom_compile_mode = "reduce-overhead" if compiled else "default"
    base = AMBITDMPC2Agent(cfg)
    other_cfg = deepcopy(cfg)
    other_cfg.outer_boom_diagnostics = True
    observed = AMBITDMPC2Agent(other_cfg)
    observed.load_training_state_dict(_clone_tree(base.training_state_dict()))
    base.actor_loss_scale.fill_(4.)
    observed.actor_loss_scale.fill_(4.)
    return base, observed


@pytest.mark.parametrize("all_invalid", [False, True])
def test_probes_preserve_entire_training_update_and_rng(all_invalid):
    base, observed = _pair()
    for _ in range(2):
        data = _batch(base)
        mean, std, valid = _labels(base, valid=not all_invalid)
        labels = dict(behavior_pre_tanh_mean=mean, behavior_log_std=std,
                      behavior_policy_valid=valid, behavior_pre_tanh_action=data[1].atanh())
        rng = torch.get_rng_state()
        expected = base._update(*data, **labels)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = observed._update(*data, **labels)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        _assert_tree_equal(observed.training_state_dict(), base.training_state_dict())
        _assert_tree_equal({k:actual[k] for k in expected}, expected)
        assert all(torch.isfinite(v) for k,v in actual.items() if k.startswith("boom_diag_"))
        assert actual["boom_diag_adam_step_norm"] > 0
        assert actual["boom_diag_policy_kl_before_after"] >= -1e-6
        if all_invalid:
            assert actual["boom_diag_alignment_unit_grad_norm"] == 0
            assert actual["boom_diag_coefficient_estimate_valid"] == 0


def test_actual_step_norm_and_gaussian_kl_match_independent_oracles():
    agent = _agent(outer_boom_alignment="replay_likelihood", outer_boom_diagnostics=True)
    zs = torch.randn(2, 4, agent.cfg.latent_dim)
    mean, std, valid = _labels(agent)
    metrics, saved = diagnostics.before_step(agent, zs, mean.tanh(), mean, valid)
    with torch.no_grad():
        for p in agent.model._pi.parameters():
            p.add_(.0003)
    actual = diagnostics.after_step(agent, saved)
    expected = .0003 * sum(p.numel() for p in agent.model._pi.parameters()) ** .5
    assert actual["boom_diag_adam_step_norm"].item() == pytest.approx(expected, rel=1e-4)
    new = agent.model.policy_stats(saved["states"])
    old = saved["stats"]
    kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(old["pre_tanh_mean"], old["log_std"].exp()),
        torch.distributions.Normal(new["pre_tanh_mean"], new["log_std"].exp()),
    ).sum(-1).mean()
    torch.testing.assert_close(actual["boom_diag_policy_kl_before_after"], kl, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(metrics["boom_diag_prospective_ratio_lambda_0p042"],
                               metrics["boom_diag_prospective_ratio_lambda_0p0042"] * 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_diagnostics_preserve_compiled_adam_rng_and_graph_storage():
    base, observed = _pair("cuda", compiled=True)
    # Restore each model's initial state after compiler warmup, whose internal
    # graph setup can differ on the first call. Compare actual logical updates.
    snapshots = [_clone_tree(a.training_state_dict()) for a in (base, observed)]
    zs = torch.randn(3, base.cfg.batch_size, base.cfg.latent_dim, device="cuda")
    mean, std, valid = (x.cuda() for x in _labels(base))
    kwargs = dict(behavior_pre_tanh_mean=mean, behavior_log_std=std, behavior_policy_valid=valid,
                  replay_action=mean.tanh(), behavior_pre_tanh_action=mean)
    for agent in (base, observed):
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            agent._update_actor(zs, **kwargs)
    for agent, state in zip((base, observed), snapshots):
        agent.load_training_state_dict(state)
    for _ in range(3):
        cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.compiler.cudagraph_mark_step_begin()
        expected = base._update_actor(zs, **kwargs)
        expected_rng = torch.cuda.get_rng_state()
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(gpu)
        torch.compiler.cudagraph_mark_step_begin()
        actual = observed._update_actor(zs, **kwargs)
        torch.cuda.synchronize()
        _assert_tree_equal(observed.training_state_dict(), base.training_state_dict())
        torch.testing.assert_close(torch.cuda.get_rng_state(), expected_rng, rtol=0, atol=0)
        for k in expected:
            torch.testing.assert_close(torch.as_tensor(actual[k]), torch.as_tensor(expected[k]), rtol=0, atol=0)
        assert all(torch.isfinite(v) for k,v in actual.items() if k.startswith("boom_diag_"))
