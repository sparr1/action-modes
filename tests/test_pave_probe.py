"""Frozen PAVE calibration uses actual weighted TD gradients and joint clipping."""

from copy import deepcopy
import math
import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core import pave_probe
from RL.tdmpc2_core.pave import critic_pave_loss
from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture
def agents():
    wrappers = []

    def create(**overrides):
        options = dict(inner_rounds=1, inner_updates_per_round=1, ent_coef=.1,
                       aux_return_ent_coef=.1, compile=False, dropout=0.)
        options.update(overrides)
        wrapper = _tiny_model(**options)
        wrappers.append(wrapper)
        with torch.no_grad():
            for name in ("_Qs", "_aux_return_Qs"):
                if hasattr(wrapper.agent.model, name):
                    for head in getattr(wrapper.agent.model, name):
                        head[-1].weight.normal_(std=.1)
        return wrapper.agent

    yield create
    for wrapper in wrappers:
        wrapper.env.close()


def batch(agent):
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    generator = torch.Generator().manual_seed(714)
    return (
        torch.randn(h + 1, b, 3, generator=generator),
        torch.randn(h, b, 1, generator=generator).tanh(),
        torch.randn(h, b, 1, generator=generator),
        torch.zeros(h, b, 1),
    )


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)
    elif isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def snapshot(agent):
    return {
        "training": deepcopy(agent.training_state_dict()),
        "modes": {name: module.training for name, module in agent.named_modules()},
        "gradients": {name: None if parameter.grad is None else parameter.grad.clone()
                      for name, parameter in agent.named_parameters()},
        "requires_grad": {name: parameter.requires_grad
                          for name, parameter in agent.named_parameters()},
        "torch": torch.random.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def flat_gradient(loss, parameters, *, retain_graph=False):
    parameters = tuple(parameters)
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                    allow_unused=True)
    return torch.cat([torch.zeros_like(parameter).flatten() if gradient is None
                      else gradient.detach().flatten()
                      for parameter, gradient in zip(parameters, gradients)])


def fixed_targets(agent, inputs, monkeypatch):
    obs, action, reward, _ = inputs
    primary = reward + torch.arange(action.shape[0]).reshape(-1, 1, 1) * .7
    auxiliary = reward - torch.arange(action.shape[0]).reshape(-1, 1, 1) * .3 + .4
    monkeypatch.setattr(agent, "_soft_td_target", lambda *args, **kwargs: primary)
    if agent.aux_return is not None:
        monkeypatch.setattr(agent.aux_return, "td_target", lambda *args, **kwargs: auxiliary)
    return primary, auxiliary


def manual_td(agent, inputs, targets):
    """Independent per-time reduction; do not reuse the learner loss helper."""
    obs, action, _, _ = inputs
    z = agent.model.encode(obs[0])
    states = []
    for recorded_action in action:
        states.append(z)
        z = agent.model.next(z, recorded_action)
    losses = {}
    for name, attribute, coefficient, target in (
        ("primary", "_Qs", agent.cfg.critic_coef, targets[0]),
        ("auxiliary", "_aux_return_Qs", agent.cfg.aux_return_critic_coef, targets[1]),
    ):
        if not hasattr(agent.model, attribute):
            continue
        terms = []
        for time, (latent, recorded_action) in enumerate(zip(states, action)):
            predictions = getattr(agent.model, attribute)._forward_eager(
                torch.cat((latent, recorded_action), dim=-1)
            )
            mean_loss = agent.model.q_backend.loss(
                predictions, target[time], reduction="none",
            ).mean()
            terms.append(mean_loss * agent.cfg.rho ** time)
        losses[name] = coefficient * sum(terms) / len(terms)
    return losses


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("critics", [("primary",), ("auxiliary",), ("primary", "auxiliary")])
@pytest.mark.parametrize("horizon", [1, 3])
def test_probe_td_norms_match_independent_weighted_recurrent_training_loss(
    agents, monkeypatch, representation, critics, horizon,
):
    agent = agents(q_representation=representation, aux_return_mode="sac",
                   train_unroll_horizon=horizon, inner_rollout_horizon=1, rho=.6,
                   critic_coef=.7, aux_return_critic_coef=.17)
    inputs = batch(agent)
    targets = fixed_targets(agent, inputs, monkeypatch)
    agent.model.train()
    td_losses = manual_td(agent, inputs, targets)
    expected = {
        name: flat_gradient(loss, getattr(agent.model, "_Qs" if name == "primary"
                                         else "_aux_return_Qs").parameters(), retain_graph=True)
        for name, loss in td_losses.items()
    }
    report = pave_probe.probe_batch(
        agent, *inputs, seed=901, update_number=11, critics=critics,
        batch_size=2, scales=(.25,),
    )
    assert set(report["critics"]) == set(critics)
    assert report["sample_count"] == 2
    for name in critics:
        actual = report["critics"][name]
        expected_norm = expected[name].double().norm().item()
        assert actual["td_gradient_norm"] == pytest.approx(expected_norm, rel=3e-5, abs=1e-8)
        assert actual["gradient_gram"][0][0] == pytest.approx(expected_norm ** 2, rel=6e-5, abs=1e-10)
        assert actual["td_coefficient"] == (.7 if name == "primary" else .17)
        assert actual["td_norm_reliable"]
        assert all(value == 1. for value in actual["finite_gradient_fractions"].values())


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_primary_only_backbone_needs_no_auxiliary_network(agents, representation):
    agent = agents(q_representation=representation)
    report = pave_probe.probe_batch(agent, *batch(agent), seed=901, update_number=11,
                                   critics=("primary",), scales=(1.,))
    assert set(report["critics"]) == {"primary"}
    assert all(candidate["scope"] == "primary" for candidate in report["candidates"])
    assert report["baseline"]["finite_gradient_fraction"] == 1.


@pytest.mark.parametrize("failure", [False, True])
def test_probe_preserves_full_training_state_gradients_modes_and_all_rng_even_on_failure(
    agents, monkeypatch, failure,
):
    agent = agents(aux_return_mode="return_actor", dropout=.25)
    inputs = batch(agent)
    agent._update(*inputs)  # Populate Adam, counters, and independent auxiliary state.
    agent.model.train()
    agent.model._Qs[0].eval()
    for index, parameter in enumerate(agent.parameters()):
        if parameter.requires_grad and index % 3 == 0:
            parameter.grad = torch.full_like(parameter, .0123)
    next(agent.model._pi.parameters()).requires_grad_(False)
    before = snapshot(agent)

    def forbidden(*args, **kwargs):
        raise RuntimeError("intentional geometry failure")

    if failure:
        monkeypatch.setattr(pave_probe, "critic_pave_loss", forbidden)
        with pytest.raises(RuntimeError, match="intentional geometry failure"):
            pave_probe.probe_batch(agent, *inputs, seed=901, update_number=11,
                                  critics=("primary", "auxiliary"), scales=(1.,))
    else:
        report = pave_probe.probe_batch(agent, *inputs, seed=901, update_number=11,
                                       critics=("primary", "auxiliary"), scales=(1.,))
        assert report["schema"] == "ambi-pave-gradient-probe-batch"
        assert report["version"] == 1
    assert_tree_equal(snapshot(agent), before)


def test_probe_uses_ordinary_compile_wrappers_but_keeps_geometry_eager(agents, monkeypatch):
    agent = agents(aux_return_mode="sac")
    calls = {"ordinary": 0, "ensemble": 0, "geometry": 0}
    in_geometry = False
    ensembles = [agent.model._Qs, agent.model._target_Qs,
                 agent.model._aux_return_Qs, agent.model._target_aux_return_Qs]

    def ensemble_wrapper(ensemble):
        def wrapped(*args, **kwargs):
            assert not in_geometry
            calls["ensemble"] += 1
            return ensemble._forward_eager(*args, **kwargs)
        return wrapped

    wrappers = [ensemble_wrapper(ensemble) for ensemble in ensembles]
    for ensemble, wrapped in zip(ensembles, wrappers):
        ensemble._compile_enabled = True
        ensemble._compiled_forward = wrapped
        ensemble._compiled_detached_forward = wrapped

    def ordinary(*args, **kwargs):
        calls["ordinary"] += 1
        return agent._outer_update_region.eager(*args, **kwargs)

    def geometry(*args, **kwargs):
        nonlocal in_geometry
        in_geometry = True
        calls["geometry"] += 1
        try:
            return critic_pave_loss(*args, **kwargs)
        finally:
            in_geometry = False

    monkeypatch.setattr(pave_probe, "critic_pave_loss", geometry)
    agent._outer_update_region.enabled = True
    agent._outer_update_region._compiled = ordinary
    pave_probe.probe_batch(agent, *batch(agent), seed=901, update_number=11,
                          critics=("primary", "auxiliary"), scales=(1.,))
    assert calls["ordinary"] == 1
    assert calls["ensemble"] >= 4
    assert calls["geometry"] == 2
    for ensemble, wrapped in zip(ensembles, wrappers):
        assert ensemble._compile_enabled
        assert ensemble._compiled_forward is wrapped
        assert ensemble._compiled_detached_forward is wrapped
        assert not ensemble.compile_failed
    assert agent._outer_update_region.enabled
    assert agent._outer_update_region._compiled is ordinary
    assert not agent._outer_update_region.failed


def test_candidate_coefficients_and_full_clipping_match_direct_combined_autograd(agents, monkeypatch):
    agent = agents(aux_return_mode="sac", critic_coef=.8, aux_return_critic_coef=.19,
                   grad_clip_norm=.05, train_unroll_horizon=3, rho=.7)
    inputs = batch(agent)
    targets = fixed_targets(agent, inputs, monkeypatch)
    geometry = {}

    def capture(critic, decode, z, action, **kwargs):
        name = "primary" if critic is agent.model._Qs else "auxiliary"
        geometry[name] = (critic, decode, z, action,
                          {key: value for key, value in kwargs.items()
                           if key in {"noisy_z", "next_z", "probe", "curvature_margin"}})
        return critic_pave_loss(critic, decode, z, action, **kwargs)

    monkeypatch.setattr(pave_probe, "critic_pave_loss", capture)
    report = pave_probe.probe_batch(agent, *inputs, seed=901, update_number=11,
                                   critics=("primary", "auxiliary"), scales=(.5,))
    assert set(geometry) == {"primary", "auxiliary"}
    obs, action, reward, terminated = inputs
    with torch.no_grad():
        next_z = agent.model.encode(obs[1:])
    agent.model.train()
    ordinary = agent._outer_update_kernel(obs[0], action, reward, terminated, next_z, targets[0])
    aux_loss, _ = agent.aux_return.critic_loss(ordinary[0], action, targets[1])
    baseline = ordinary[-1] + agent.cfg.aux_return_critic_coef * aux_loss
    parameters = tuple(agent._world_critic_params)
    baseline_gradient = flat_gradient(baseline, parameters, retain_graph=True)
    baseline_norm = baseline_gradient.double().norm().item()
    assert report["baseline"]["full_gradient_norm"] == pytest.approx(baseline_norm, rel=3e-5)
    baseline_clip = min(1., agent.cfg.grad_clip_norm / (baseline_norm + 1e-6))
    assert report["baseline"]["clip_multiplier"] == pytest.approx(baseline_clip, rel=3e-5)

    for candidate in report["candidates"]:
        if candidate["scope"] != "combined":
            continue
        coefficients = candidate["weights"]
        assert coefficients == {"mpr": .05, "vfc": .05, "curvature": (
            .005 if candidate["variant"] == "reference" else .05
        )}
        loss = baseline
        for name, args in geometry.items():
            critic, decode, z, selected_action, kwargs = args
            pave_loss, _ = critic_pave_loss(
                critic, decode, z, selected_action, **kwargs,
                **{f"{key}_coef": value for key, value in coefficients.items()},
            )
            td_gradient = flat_gradient(baseline, critic.parameters(), retain_graph=True).double()
            pave_gradient = flat_gradient(pave_loss, critic.parameters(), retain_graph=True).double()
            td_norm, pave_norm = td_gradient.norm().item(), pave_gradient.norm().item()
            local = candidate["critics"][name]
            assert local["pave_gradient_norm"] == pytest.approx(pave_norm, rel=3e-5)
            assert local["ratio_to_td"] == pytest.approx(pave_norm / td_norm, rel=3e-5)
            assert local["cosine_with_td"] == pytest.approx(
                torch.dot(td_gradient, pave_gradient).item() / (td_norm * pave_norm),
                rel=3e-5, abs=1e-7,
            )
            assert local["combined_critic_gradient_norm"] == pytest.approx(
                (td_gradient + pave_gradient).norm().item(), rel=3e-5,
            )
            loss = loss + pave_loss
        actual_gradient = flat_gradient(loss, parameters, retain_graph=True)
        actual_norm = actual_gradient.double().norm().item()
        actual_clip = min(1., agent.cfg.grad_clip_norm / (actual_norm + 1e-6))
        assert candidate["full_gradient_norm"] == pytest.approx(actual_norm, rel=3e-5)
        assert candidate["clip_multiplier"] == pytest.approx(actual_clip, rel=3e-5)
        assert candidate["clip_multiplier_relative_to_baseline"] == pytest.approx(
            actual_clip / baseline_clip, rel=3e-5,
        )


def test_near_zero_td_norm_is_reported_as_unreliable_without_infinite_ratios(agents):
    agent = agents(aux_return_mode="sac")
    report = pave_probe.probe_batch(agent, *batch(agent), seed=901, update_number=11,
                                   critics=("primary", "auxiliary"), scales=(1.,),
                                   td_norm_floor=1e30)
    assert all(not critic["td_norm_reliable"] for critic in report["critics"].values())
    for candidate in report["candidates"]:
        for critic in candidate["critics"].values():
            assert critic["ratio_to_td"] is None
            assert critic["cosine_with_td"] is None


def test_exact_zero_td_gradient_does_not_produce_infinite_pave_ratio(agents, monkeypatch):
    agent = agents(q_representation="scalar")
    with torch.no_grad():
        for head in agent.model._Qs:
            head[-1].weight.zero_()
            head[-1].bias.zero_()
    inputs = batch(agent)
    monkeypatch.setattr(agent, "_soft_td_target", lambda *args, **kwargs: torch.zeros_like(inputs[2]))
    report = pave_probe.probe_batch(agent, *inputs, seed=901, update_number=11,
                                   critics=("primary",), scales=(1.,))
    assert report["critics"]["primary"]["td_gradient_norm"] == 0.
    assert not report["critics"]["primary"]["td_norm_reliable"]
    for candidate in report["candidates"]:
        assert candidate["critics"]["primary"]["ratio_to_td"] is None
        assert candidate["critics"]["primary"]["cosine_with_td"] is None


def test_candidate_norm_algebra_resolves_cancellation_without_negative_roundoff():
    vectors = [torch.tensor([3., 4.], dtype=torch.float64),
               torch.tensor([1., 2.], dtype=torch.float64),
               torch.tensor([-1., -2.], dtype=torch.float64),
               torch.zeros(2, dtype=torch.float64)]
    gram = pave_probe._gram(vectors)
    result = pave_probe._candidate_statistics(
        gram, {"mpr": 1., "vfc": 1., "curvature": 1.}, 1e-10,
    )
    assert result["pave_gradient_norm"] == pytest.approx(0., abs=1e-12)
    assert result["ratio_to_td"] == pytest.approx(0., abs=1e-12)
    assert result["cosine_with_td"] is None
    assert result["combined_critic_gradient_norm"] == pytest.approx(5.)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_two_probe_batches_with_real_aot_autograd_and_eager_geometry(agents, monkeypatch, representation):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    actual_compile = torch.compile
    compiled_regions = []

    def aot_compile(function, **options):
        compiled_regions.append(getattr(function, "__name__", str(function)))
        return actual_compile(function, backend="aot_eager", **options)

    monkeypatch.setattr(torch, "compile", aot_compile)
    torch._dynamo.reset()
    try:
        options = dict(aux_return_mode="sac", q_representation=representation,
                       num_q=5 if representation == "distributional" else 2)
        eager = agents(**options)
        compiled = agents(**options, compile=True, compile_strict=True)
        compiled.load(deepcopy(eager.checkpoint_state()))
        before = snapshot(compiled)
        for index in range(2):
            obs, action, reward, terminated = batch(compiled)
            inputs = (obs + .03 * index, action, reward, terminated)
            expected = pave_probe.probe_batch(eager, *inputs, seed=901 + index,
                                              update_number=11 + index, scales=(1.,))
            actual = pave_probe.probe_batch(compiled, *inputs, seed=901 + index,
                                            update_number=11 + index, scales=(1.,))
            assert actual["baseline"]["finite_gradient_fraction"] == 1.
            assert actual["baseline"]["full_gradient_norm"] == pytest.approx(
                expected["baseline"]["full_gradient_norm"], rel=3e-5,
            )
            for name in ("primary", "auxiliary"):
                np.testing.assert_allclose(actual["critics"][name]["gradient_gram"],
                                           expected["critics"][name]["gradient_gram"],
                                           rtol=5e-5, atol=1e-8)
            assert not compiled._outer_update_region.failed
            assert all(not getattr(compiled.model, name).compile_failed for name in (
                "_Qs", "_target_Qs", "_aux_return_Qs", "_target_aux_return_Qs",
            ))
            assert_tree_equal(snapshot(compiled), before)
            if index == 0:
                first_compile_count = len(compiled_regions)
                cached_outer = compiled._outer_update_region._compiled
                assert cached_outer is not None
            else:
                assert len(compiled_regions) == first_compile_count
                assert compiled._outer_update_region._compiled is cached_outer
        assert "_outer_update_kernel" in compiled_regions
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("auxiliary", [False, True])
def test_default_critic_selection_uses_only_available_networks(agents, auxiliary):
    agent = agents(aux_return_mode="sac" if auxiliary else "off")
    report = pave_probe.probe_batch(agent, *batch(agent), seed=901,
                                   update_number=11, scales=(1.,))
    assert set(report["critics"]) == ({"primary", "auxiliary"} if auxiliary else {"primary"})


@pytest.mark.parametrize("critics", [("auxiliary",), ("primary", "auxiliary")])
def test_explicit_unavailable_auxiliary_is_rejected_before_state_mutation(agents, critics):
    agent = agents()
    inputs = batch(agent)
    before = snapshot(agent)
    with pytest.raises(ValueError, match="unavailable"):
        pave_probe.probe_batch(agent, *inputs, seed=901, update_number=11, critics=critics)
    assert_tree_equal(snapshot(agent), before)


def test_wrong_observation_dimension_is_rejected_before_state_mutation(agents):
    agent = agents()
    obs, action, reward, terminated = batch(agent)
    before = snapshot(agent)
    with pytest.raises(ValueError, match="Observation dimension"):
        pave_probe.probe_batch(agent, obs[..., :2], action, reward, terminated,
                              seed=901, update_number=11)
    assert_tree_equal(snapshot(agent), before)
