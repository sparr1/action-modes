"""Actor-entropy-only intervention for the pinned released BOOM reference.

The released action sampler, alignment Gaussian helper, Q weighting, and gate
retain their equations. Only the actor's entropy log probability is replaced.
"""
import json
import math

import torch
import torch.nn.functional as F


def corrected_log_probability(eps, log_std, pre_tanh):
    """Joint Gaussian log density minus the exact stable tanh Jacobian."""
    gaussian = (-0.5 * eps.square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
    jacobian = (2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(-1, keepdim=True)
    return gaussian - jacobian


def corrected_actor_pi(model, z, task):
    """Same single-task sample and RNG draw as upstream WorldModel.pi."""
    from boom.common import math as upstream_math

    assert task is None and not model.cfg.multitask
    mean, log_std = model._pi(z).chunk(2, dim=-1)
    log_std = upstream_math.log_std(log_std, model.log_std_min, model.log_std_dif)
    eps = torch.randn_like(mean)
    pre_tanh = mean + eps * log_std.exp()
    action = pre_tanh.tanh()
    log_prob = corrected_log_probability(eps, log_std, pre_tanh)
    # Detached comparisons never replace the native helper used by alignment.
    with torch.no_grad():
        jac = torch.log(torch.relu(1 - action.square()) + 1e-6).sum(-1, keepdim=True)
        native_log_prob = upstream_math.gaussian_logprob(eps, log_std) - jac
        gaussian = (-0.5 * eps.square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        diagnostics = {
            "post_tanh_entropy_floored": -(gaussian - jac).mean(),
            "post_tanh_entropy_exact": -log_prob.detach().mean(),
            "upstream_entropy_statistic": -native_log_prob.mean(),
        }
    return (mean.tanh(), action, log_prob, log_std), diagnostics


def validate_entropy_ablation(agent, output):
    """Verify identical actions, alignment gradients, and RNG for fixed weights."""
    from boom.common import math as upstream_math
    from upstream_reference_acceleration import rng_state, restore_rng

    assert agent.cfg.reference_entropy == "ambi-posttanh"
    z = torch.randn(3, agent.cfg.batch_size, agent.cfg.latent_dim, device=agent.device)
    target = torch.rand(3, agent.cfg.batch_size, agent.cfg.action_dim, device=agent.device) * 2 - 1
    weights = torch.softmax(torch.randn(3, agent.cfg.batch_size, device=agent.device), dim=-1)
    params = tuple(agent.model._pi.parameters())
    state = rng_state()
    observed = agent.observe
    agent.observe = False
    results = []
    try:
        for active in (False, True):
            restore_rng(state)
            agent.in_actor = active
            mean, action, log_prob, log_std = agent.model.pi(z, None)
            std = log_std.exp().detach().clamp_min(agent.cfg.min_std)
            score = upstream_math.gaussian_logprob((action - target) / std, std.log()).squeeze(-1)
            alignment = -(weights * score / 3).sum()
            gradients = torch.autograd.grad(alignment, params)
            results.append((tuple(x.detach().clone() for x in (mean, action, log_std)),
                            tuple(g.detach().clone() for g in gradients), rng_state(),
                            float(log_prob.detach().mean())))
    finally:
        agent.in_actor = False
        agent.observe = observed
    for a, b in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(results[0][2][2], results[1][2][2])
    assert all(torch.equal(a, b) for a, b in zip(results[0][2][3], results[1][2][3]))
    receipt = {
        "actions_exact": True, "alignment_gradients_exact": True, "torch_rng_exact": True,
        "entropy_coefficient": float(agent.cfg.entropy_coef),
        "released_log_prob_mean": results[0][3], "corrected_log_prob_mean": results[1][3],
        "alignment_gaussian": "unchanged released helper, including extra action-dimension factor",
    }
    (output / "entropy_ablation_parity.json").write_text(json.dumps(receipt, indent=2))
    print("ENTROPY_ABLATION_PARITY_PASSED", json.dumps(receipt), flush=True)
