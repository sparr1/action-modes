"""Geometry telemetry must be comparable, sparse, and observational."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from RL.TDMPC2 import _DeviceMeanAccumulator
from RL.tdmpc2_core.critic_geometry import (
    critic_geometry_diagnostics, geometry_diagnostics_due, gradient_consistency_metrics,
)
from RL.tdmpc2_core.pave import critic_pave_loss, prepare_pave_inputs
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_pave_identity import identity
from tests.test_pave_integration import _batch, _nonconstant_critics
from utils.pave_identity import CRITIC_GEOMETRY_DEFAULTS, resolve_critic_geometry_controls


@pytest.fixture
def wrappers():
    created = []

    def create(**options):
        result = _tiny_model(inner_operator="none", inner_rounds=0,
                             inner_rollouts_per_round=0, inner_updates_per_round=0,
                             ent_coef=.1, **options)
        created.append(result)
        return result

    yield create
    for wrapper in created:
        wrapper.env.close()


def test_defaults_enable_both_diagnostics_without_adding_training_losses():
    cfg = _build_cfg()
    assert cfg.critic_geometry_diagnostics is True
    assert cfg.critic_geometry_every == 1000
    assert cfg.critic_geometry_batch_size == 64
    assert cfg.critic_geometry_obs_noise_std == .01
    assert cfg.critic_geometry_action_noise_std == .02
    assert cfg.outer_pave_mpr_coef == cfg.outer_pave_action_smoothness_coef == 0
    assert not geometry_diagnostics_due(cfg, 0)
    assert not geometry_diagnostics_due(cfg, 999)
    assert geometry_diagnostics_due(cfg, 1000)
    assert geometry_diagnostics_due(cfg, 2000)


@pytest.mark.parametrize("unsupported", [{"obs": "rgb"}, {"critic_value_mode": "return_entropy"}])
def test_unsupported_models_auto_disable_but_reject_explicit_enable(unsupported):
    assert resolve_critic_geometry_controls(unsupported)["critic_geometry_diagnostics"] is False
    with pytest.raises(ValueError, match="requires state observations"):
        resolve_critic_geometry_controls({**unsupported, "critic_geometry_diagnostics": True})


@pytest.mark.parametrize("key,bad", [
    ("critic_geometry_diagnostics", 1), ("critic_geometry_diagnostics", "true"),
    ("critic_geometry_every", 0), ("critic_geometry_every", True),
    ("critic_geometry_batch_size", 0), ("critic_geometry_batch_size", 2.5),
    ("critic_geometry_seed", -1), ("critic_geometry_seed", False),
    ("critic_geometry_obs_noise_std", float("nan")),
    ("critic_geometry_action_noise_std", float("inf")),
    ("critic_geometry_obs_noise_std", -.1), ("critic_geometry_action_noise_std", True),
])
def test_invalid_controls_are_rejected_even_when_disabled(key, bad):
    with pytest.raises(ValueError, match=key):
        resolve_critic_geometry_controls({"critic_geometry_diagnostics": False, key: bad})


def test_numpy_integer_cap_and_full_batch_are_supported():
    assert _build_cfg(critic_geometry_batch_size=np.int64(3)).critic_geometry_batch_size == 3
    assert _build_cfg(critic_geometry_batch_size=None).critic_geometry_batch_size is None


@pytest.mark.parametrize("base", [{}, {"outer_pave_mpr_coef": .004},
                                  {"outer_pave_action_smoothness_coef": .004}])
def test_logging_controls_do_not_change_scientific_identity(base):
    assert identity(**base) == identity(**base, **CRITIC_GEOMETRY_DEFAULTS)
    assert identity(**base) == identity(**base, **{
        **CRITIC_GEOMETRY_DEFAULTS, "critic_geometry_diagnostics": True,
        "critic_geometry_every": 1, "critic_geometry_batch_size": None,
        "critic_geometry_obs_noise_std": .2, "critic_geometry_seed": 7,
    })


def test_gradient_metrics_preserve_head_sum_and_measure_direction_and_scale():
    original = torch.tensor([[[1., 0.]], [[-1., 0.]]])
    perturbed = torch.tensor([[[0., 2.]], [[0., -2.]]])
    values = gradient_consistency_metrics(original, perturbed, "mpr")
    assert values["mpr_loss"] == 5  # Opposing heads must not cancel.
    assert values["mpr_gradient_change_rms"] == pytest.approx(2.5 ** .5)
    assert values["mpr_relative_gradient_change"] == pytest.approx(5 ** .5)
    assert values["mpr_cosine_similarity"] == 0
    assert values["mpr_cosine_valid_fraction"] == 1
    scaled = gradient_consistency_metrics(3 * original, 3 * perturbed, "mpr")
    assert scaled["mpr_loss"] == 9 * values["mpr_loss"]
    torch.testing.assert_close(scaled["mpr_relative_gradient_change"], values["mpr_relative_gradient_change"])
    assert gradient_consistency_metrics(original, -original, "mpr")["mpr_cosine_similarity"] == -1


def test_flat_critics_are_not_reported_as_perfect_directional_agreement():
    zeros = torch.zeros(2, 3, 4)
    values = gradient_consistency_metrics(zeros, zeros, "action_smoothness")
    assert values["action_smoothness_loss"] == 0
    assert values["action_smoothness_cosine_valid_fraction"] == 0
    assert torch.isnan(values["action_smoothness_cosine_similarity"])
    assert torch.isnan(values["action_smoothness_relative_gradient_change"])
    accumulator = _DeviceMeanAccumulator()
    accumulator.update(values, prefix="train/")
    published = accumulator.floats()
    assert published["train/action_smoothness_loss"] == 0
    assert "train/action_smoothness_cosine_similarity" not in published


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("backend", ["loop", "vmap"])
def test_diagnostics_match_training_losses_without_higher_order_graphs(wrappers, monkeypatch,
                                                                     representation, backend):
    agent = wrappers(aux_return_mode="sac", q_representation=representation,
                     critic_ensemble_backend=backend, dropout=.3,
                     critic_geometry_every=1, critic_geometry_batch_size=3).agent
    _nonconstant_critics(agent)
    obs, action, _, _ = _batch(agent)
    agent.model.train()
    agent.model._Qs[0].eval()
    modes = [(module, module.training) for module in agent.model.modules()]
    for parameter in agent.model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    grads = [None if p.grad is None else p.grad.clone() for p in agent.model.parameters()]
    state = deepcopy(agent.checkpoint_state())
    rng = torch.random.get_rng_state().clone()
    autograd_grad = torch.autograd.grad
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs.get("create_graph", False))
        return autograd_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record)
    actual = critic_geometry_diagnostics(agent.model, obs, action, agent.cfg, update_number=9)
    assert calls and not any(calls)
    assert all(not value.requires_grad for value in actual.values())
    assert all(module.training == mode for module, mode in modes)
    _assert_tree_equal(agent.checkpoint_state(), state)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for parameter, expected in zip(agent.model.parameters(), grads):
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    inputs = prepare_pave_inputs(
        agent.model, obs, action, seed=agent.cfg.critic_geometry_seed, update_number=9,
        batch_size=3, needs_noise=True, needs_action_noise=True,
        needs_next=False, needs_probe=False,
    )
    for name, attr in (("primary", "_Qs"), ("aux_return", "_aux_return_Qs")):
        _, expected = critic_pave_loss(
            getattr(agent.model, attr), agent.model.q_backend.decode,
            inputs["z"], inputs["action"], noisy_z=inputs["noisy_z"],
            noisy_action=inputs["noisy_action"], mpr_coef=1., action_smoothness_coef=1.,
        )
        for term in ("mpr", "action_smoothness"):
            torch.testing.assert_close(actual[f"critic_geometry_{name}_{term}_loss"], expected[f"{term}_loss"])
    assert actual["critic_geometry_sample_count"] == 3


def test_probe_is_independent_of_training_regularizer_selection(wrappers):
    agent = wrappers(critic_geometry_every=1).agent
    _nonconstant_critics(agent)
    obs, action, _, _ = _batch(agent)
    expected = critic_geometry_diagnostics(agent.model, obs, action, agent.cfg, 2)
    agent.cfg.outer_pave_mpr_coef = 1.
    agent.cfg.outer_pave_action_smoothness_coef = 0.
    agent.cfg.outer_pave_obs_noise_std = .5
    agent.cfg.outer_pave_action_noise_std = .8
    agent.cfg.outer_pave_batch_size = 1
    torch.rand(17)
    actual = critic_geometry_diagnostics(agent.model, obs, action, agent.cfg, 2)
    _assert_tree_equal(actual, expected)
    later = critic_geometry_diagnostics(agent.model, obs, action, agent.cfg, 3)
    assert not torch.equal(later["critic_geometry_action_perturbation_rms"], expected["critic_geometry_action_perturbation_rms"])


@pytest.mark.parametrize("enabled,update", [(False, 1000), (True, 999)])
def test_disabled_and_off_cadence_do_no_model_work(wrappers, monkeypatch, enabled, update):
    agent = wrappers(critic_geometry_diagnostics=enabled).agent
    obs, action, _, _ = _batch(agent)

    def forbidden(*args, **kwargs):
        raise AssertionError("No diagnostic work was due")

    monkeypatch.setattr(agent.model, "encode", forbidden)
    rng = torch.random.get_rng_state().clone()
    assert critic_geometry_diagnostics(agent.model, obs, action, agent.cfg, update) == {}
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_modes_and_rng_are_restored_after_failed_probe(wrappers, monkeypatch):
    agent = wrappers(critic_geometry_every=1).agent
    obs, action, _, _ = _batch(agent)
    agent.model._Qs[0].eval()
    modes = [(module, module.training) for module in agent.model.modules()]
    rng = torch.random.get_rng_state().clone()

    def fail(*args):
        torch.rand(7)
        raise RuntimeError("intentional diagnostic failure")

    monkeypatch.setattr(agent.model, "encode", fail)
    with pytest.raises(RuntimeError, match="intentional diagnostic failure"):
        critic_geometry_diagnostics(agent.model, obs, action, agent.cfg, 1)
    assert all(module.training == mode for module, mode in modes)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("auxiliary,regularizer", [(False, "none"), (False, "mpr"),
                                                 (True, "none"), (True, "action_smoothness")])
def test_logging_preserves_complete_training_updates_and_emits_both_metrics(wrappers, auxiliary, regularizer):
    options = dict(aux_return_mode="sac" if auxiliary else "off", dropout=.2,
                   critic_geometry_every=1, critic_geometry_batch_size=None)
    if regularizer != "none":
        options[f"outer_pave_{regularizer}_coef"] = .004
    off = wrappers(critic_geometry_diagnostics=False, **options)
    on = wrappers(critic_geometry_diagnostics=True, **options)
    _nonconstant_critics(off.agent)
    on.agent.load(deepcopy(off.agent.checkpoint_state()))
    batch = _batch(off.agent)
    for _ in range(2):
        rng = torch.random.get_rng_state().clone()
        expected = off.agent._update(*batch)
        next_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng)
        actual = on.agent._update(*batch)
        _assert_tree_equal({k: v for k, v in actual.items() if not k.startswith("critic_geometry_")}, expected)
        _assert_tree_equal(on.agent.checkpoint_state(), off.agent.checkpoint_state())
        torch.testing.assert_close(torch.random.get_rng_state(), next_rng, rtol=0, atol=0)
        for critic in ("primary", "aux_return") if auxiliary else ("primary",):
            for metric in ("mpr_loss", "action_smoothness_loss", "action_gradient_rms"):
                assert f"critic_geometry_{critic}_{metric}" in actual
        on._accumulate_train_metrics(actual)
    on._accumulate_train_metrics(expected)  # An unsampled update must not insert zeros.
    payload = on._wandb_update_window.floats(include_stats=True)
    key = "train/critic_geometry_primary_action_smoothness_loss"
    assert payload[key] >= 0
    assert payload[key + "_count"] == 2
    assert payload["train/critic_loss_count"] == 3
