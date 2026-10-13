"""PAVE gradient ownership, training accounting, resume, and AOT boundaries."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core import pave
from RL.tdmpc2_core.pave import outer_pave_loss
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture
def agents():
    wrappers = []

    def create(**overrides):
        options = dict(inner_rounds=1, inner_updates_per_round=1, ent_coef=.1)
        options.update(overrides)
        wrapper = _tiny_model(**options)
        wrappers.append(wrapper)
        return wrapper.agent

    yield create
    for wrapper in wrappers:
        wrapper.env.close()


def _weights(*prefixes):
    return {
        f"{prefix}_{term}_coef": value
        for prefix in prefixes
        for term, value in (("mpr", .1), ("vfc", .1), ("curvature", .01))
    }


def _batch(agent):
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    generator = torch.Generator().manual_seed(991)
    return (
        torch.randn(h + 1, b, 3, generator=generator),
        torch.randn(h, b, 1, generator=generator).tanh(),
        torch.randn(h, b, 1, generator=generator),
        torch.zeros(h, b, 1),
    )


def _nonconstant_critics(agent):
    with torch.no_grad():
        for attr in ("_Qs", "_aux_return_Qs"):
            if hasattr(agent.model, attr):
                for head in getattr(agent.model, attr):
                    head[-1].weight.normal_(std=.1)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("prefixes", [
    ("outer_pave",), ("aux_return_pave",), ("outer_pave", "aux_return_pave"),
])
def test_pave_backward_reaches_only_selected_online_critics(agents, representation, prefixes):
    agent = agents(q_representation=representation, aux_return_mode="return_actor",
                   dropout=.3, **_weights(*prefixes))
    _nonconstant_critics(agent)
    obs, action, _, _ = _batch(agent)
    agent.model.train()
    # Preserve heterogeneous child modes, not just the ensemble's root flag.
    agent.model._Qs[0].eval()
    modes = [(module, module.training) for module in agent.model.modules()]
    before_rng = torch.random.get_rng_state().clone()
    loss, metrics = outer_pave_loss(agent.model, obs, action, agent.cfg, update_number=7)
    torch.testing.assert_close(torch.random.get_rng_state(), before_rng, rtol=0, atol=0)
    assert all(module.training == mode for module, mode in modes)
    loss.backward()

    for attr, prefix in (("_Qs", "outer_pave"), ("_aux_return_Qs", "aux_return_pave")):
        parameters = list(getattr(agent.model, attr).parameters())
        if prefix in prefixes:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
            assert metrics[f"{prefix}_sample_count"] == 4
            assert 0 <= metrics[f"{prefix}_curvature_violation_fraction"] <= 1
        else:
            assert all(p.grad is None for p in parameters)
            assert not any(key.startswith(prefix) for key in metrics)
    for attr in ("_encoder", "_dynamics", "_reward", "_pi", "_return_pi",
                 "_target_Qs", "_target_aux_return_Qs"):
        assert all(p.grad is None for p in getattr(agent.model, attr).parameters())
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())


def test_disabled_helper_does_no_forward_and_consumes_no_rng(agents, monkeypatch):
    agent = agents()
    obs, action, _, _ = _batch(agent)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Disabled PAVE must not call the model")

    monkeypatch.setattr(agent.model, "encode", forbidden)
    monkeypatch.setattr(agent.model._Qs, "_forward_eager", forbidden)
    before = torch.random.get_rng_state().clone()
    loss, metrics = outer_pave_loss(agent.model, obs, action, agent.cfg, update_number=1)
    assert loss.item() == 0
    assert metrics == {}
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_disabled_full_update_is_bitwise_equivalent_to_legacy_configuration(agents):
    baseline = agents(dropout=.2)
    explicit = agents(dropout=.2, outer_pave_batch_size=None,
                      outer_pave_obs_noise_std=.9, outer_pave_curvature_margin=3.,
                      **{key: 0. for key in _weights("outer_pave", "aux_return_pave")})
    # Missing controls exercise the pre-feature configuration shape.
    for key in list(vars(baseline.cfg)):
        if key.startswith(("outer_pave_", "aux_return_pave_")):
            delattr(baseline.cfg, key)
    explicit.load(deepcopy(baseline.checkpoint_state()))
    before = torch.random.get_rng_state().clone()
    expected = baseline._update(*_batch(baseline))
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(before)
    actual = explicit._update(*_batch(explicit))
    _assert_tree_equal(actual, expected)
    _assert_tree_equal(explicit.checkpoint_state(), baseline.checkpoint_state())
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert not any("pave" in key for key in actual)
    assert "outer_pave" not in explicit._critic_target_spec()


@pytest.mark.parametrize("term,view_count", [("mpr", 2), ("vfc", 2), ("curvature", 1)])
def test_only_enabled_terms_encode_observations_and_draw_noise(agents, monkeypatch, term, view_count):
    agent = agents(**{f"outer_pave_{term}_coef": .1})
    obs, action, _, _ = _batch(agent)
    encode = agent.model.encode
    views = []

    def record_encode(value):
        views.append(value)
        return encode(value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Inactive PAVE term consumed random draws")

    monkeypatch.setattr(agent.model, "encode", record_encode)
    if term != "mpr":
        monkeypatch.setattr(torch, "randn", forbidden)
    if term != "curvature":
        monkeypatch.setattr(torch, "randint", forbidden)
    _, metrics = outer_pave_loss(agent.model, obs, action, agent.cfg, update_number=2)
    assert len(views) == view_count
    for candidate in ("mpr", "vfc", "curvature"):
        assert (f"outer_pave_{candidate}_loss" in metrics) is (candidate == term)


def test_active_sampling_depends_on_update_number_not_global_rng(agents):
    agent = agents(outer_pave_batch_size=2, **_weights("outer_pave"))
    _nonconstant_critics(agent)
    obs, action, _, _ = _batch(agent)
    first, first_metrics = outer_pave_loss(agent.model, obs, action, agent.cfg, update_number=3)
    torch.rand(27)
    repeated, repeated_metrics = outer_pave_loss(agent.model, obs, action, agent.cfg, update_number=3)
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    _assert_tree_equal(repeated_metrics, first_metrics)
    changed, _ = outer_pave_loss(agent.model, obs, action, agent.cfg, update_number=4)
    assert not torch.equal(changed, first)


@pytest.mark.parametrize("cap,expected_rows", [(None, 4), (3, 3), (99, 4)])
def test_observation_pairing_simnorm_and_shared_random_inputs(agents, monkeypatch, cap, expected_rows):
    agent = agents(aux_return_mode="sac", outer_pave_batch_size=cap,
                   **_weights("outer_pave", "aux_return_pave"))
    obs, action, _, _ = _batch(agent)
    encode = agent.model.encode
    encoded_views = []
    geometry_inputs = []
    critic_loss = pave.critic_pave_loss

    def record_encode(observations):
        assert not torch.is_grad_enabled()
        latents = encode(observations)
        encoded_views.append((observations.clone(), latents))
        return latents

    def record_geometry(critic, decode, z, actions, **kwargs):
        geometry_inputs.append((z, actions, kwargs))
        return critic_loss(critic, decode, z, actions, **kwargs)

    monkeypatch.setattr(agent.model, "encode", record_encode)
    monkeypatch.setattr(pave, "critic_pave_loss", record_geometry)
    _, metrics = outer_pave_loss(agent.model, obs, action, agent.cfg, update_number=2)
    assert len(encoded_views) == 3
    current, noisy, successor = (pair[0] for pair in encoded_views)
    assert current.shape[0] == expected_rows
    assert not torch.equal(current, noisy)
    original = obs[:-1].flatten(0, 1)
    matches = (current[:, None] == original[None]).all(dim=-1)
    assert torch.equal(matches.sum(dim=1), torch.ones(expected_rows, dtype=torch.long))
    rows = matches.to(torch.long).argmax(dim=1)
    assert rows.unique().numel() == expected_rows
    torch.testing.assert_close(successor, obs[1:].flatten(0, 1)[rows], rtol=0, atol=0)
    torch.testing.assert_close(geometry_inputs[0][1], action.flatten(0, 1)[rows], rtol=0, atol=0)
    for _, latents in encoded_views:
        assert not latents.requires_grad
        groups = latents.reshape(expected_rows, -1, agent.cfg.simnorm_dim)
        assert (groups >= 0).all() and (groups <= 1).all()
        torch.testing.assert_close(groups.sum(-1), torch.ones_like(groups[..., 0]))
    primary, auxiliary = geometry_inputs
    assert primary[0] is auxiliary[0]
    assert primary[1] is auxiliary[1]
    for key in ("noisy_z", "next_z", "probe"):
        assert primary[2][key] is auxiliary[2][key]
    assert set(primary[2]["probe"].flatten().tolist()) <= {-1., 1.}
    assert metrics["outer_pave_sample_count"] == metrics["aux_return_pave_sample_count"] == expected_rows


def test_update_accounts_for_independent_pave_weights_in_single_optimizer_step(agents):
    agent = agents(aux_return_mode="sac", critic_coef=.37,
                   aux_return_critic_coef=.23, outer_pave_batch_size=3,
                   **_weights("outer_pave", "aux_return_pave"))
    _nonconstant_critics(agent)
    info = agent._update(*_batch(agent))
    expected = (agent.cfg.consistency_coef * info["consistency_loss"]
                + agent.cfg.reward_coef * info["reward_loss"]
                + agent.cfg.termination_coef * info["termination_loss"]
                + agent.cfg.critic_coef * info["critic_loss"]
                + agent.cfg.aux_return_critic_coef * info["aux_return_critic_loss"])
    for prefix in ("outer_pave", "aux_return_pave"):
        weighted = sum(getattr(agent.cfg, f"{prefix}_{term}_coef")
                       * info[f"{prefix}_{term}_loss"]
                       for term in ("mpr", "vfc", "curvature"))
        torch.testing.assert_close(info[f"{prefix}_weighted_loss"], weighted)
        for term in ("mpr", "vfc", "curvature"):
            torch.testing.assert_close(info[f"{prefix}_{term}_weighted_loss"],
                                       getattr(agent.cfg, f"{prefix}_{term}_coef")
                                       * info[f"{prefix}_{term}_loss"])
        assert info[f"{prefix}_sample_count"] == 3
        expected = expected + weighted
    torch.testing.assert_close(info["total_loss"], expected)
    assert agent.num_updates == agent.aux_return.num_updates == 1
    for attr in ("_Qs", "_aux_return_Qs"):
        for parameter in getattr(agent.model, attr).parameters():
            assert agent.optim.state[parameter]["step"].item() == 1


@pytest.mark.parametrize("aux_mode", ["off", "sac"])
@pytest.mark.parametrize("source_active", [False, True])
def test_portable_transfer_can_toggle_pave_without_architecture_change(agents, aux_mode, source_active):
    prefixes = ("outer_pave",) if aux_mode == "off" else ("outer_pave", "aux_return_pave")
    source = agents(aux_return_mode=aux_mode, **(_weights(*prefixes) if source_active else {}))
    source._update(*_batch(source))
    target = agents(aux_return_mode=aux_mode, **({} if source_active else _weights(*prefixes)))
    target.load(deepcopy(source.checkpoint_state()))
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    assert ("outer_pave" in target._critic_target_spec()) is (not source_active)


@pytest.mark.parametrize("change", [
    {"outer_pave_mpr_coef": .2}, {"outer_pave_vfc_coef": 0.},
    {"outer_pave_curvature_coef": .02}, {"aux_return_pave_mpr_coef": .2},
    {"aux_return_pave_vfc_coef": 0.}, {"aux_return_pave_curvature_coef": .02},
    {"outer_pave_batch_size": None}, {"outer_pave_obs_noise_std": .02},
    {"outer_pave_curvature_margin": 0.},
])
def test_exact_resume_rejects_changed_pave_settings_before_mutation(agents, change):
    options = dict(aux_return_mode="sac", **_weights("outer_pave", "aux_return_pave"))
    source = agents(**options)
    source.prepare_training_resume_boundary()
    target = agents(**(options | change))
    before = deepcopy(target.training_state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="critic-target specification"):
        target.load_training_state_dict(deepcopy(source.training_state_dict()))
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_exact_resume_reproduces_next_update_and_pave_sampling(agents):
    options = dict(aux_return_mode="return_actor", outer_pave_batch_size=2,
                   dropout=.2, **_weights("outer_pave", "aux_return_pave"))
    source = agents(**options)
    source._update(*_batch(source))
    source.prepare_training_resume_boundary()
    saved = deepcopy(source.training_state_dict())
    restored = agents(**options)
    restored.load_training_state_dict(saved)
    _assert_tree_equal(restored.training_state_dict(), saved)
    rng = torch.random.get_rng_state().clone()
    expected = source._update(*_batch(source))
    torch.random.set_rng_state(rng)
    actual = restored._update(*_batch(restored))
    _assert_tree_equal(actual, expected)
    source.prepare_training_resume_boundary()
    restored.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())


@pytest.mark.parametrize("change", ["enable", "disable", "seed"])
def test_exact_resume_rejects_changed_pave_activation_or_seed(agents, change):
    source = agents(**({} if change == "enable" else _weights("outer_pave")))
    target = agents(**({} if change == "disable" else _weights("outer_pave")))
    if change == "seed":
        target.cfg.seed += 1
    source.prepare_training_resume_boundary()
    before = deepcopy(target.training_state_dict())
    with pytest.raises(ValueError, match="critic-target specification"):
        target.load_training_state_dict(deepcopy(source.training_state_dict()))
    _assert_tree_equal(target.training_state_dict(), before)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_two_updates_with_real_aot_autograd_and_eager_pave(agents, monkeypatch, representation):
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
        agent = agents(compile=True, compile_strict=True, aux_return_mode="sac",
                       q_representation=representation,
                       num_q=5 if representation == "distributional" else 2,
                       **_weights("outer_pave", "aux_return_pave"))
        _nonconstant_critics(agent)
        for _ in range(2):
            info = agent._update(*_batch(agent))
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in info.values())
            assert not agent._outer_update_region.failed
        assert agent.num_updates == 2
        assert "_outer_update_kernel" in compiled_regions
        assert agent._outer_update_region._compiled is not None
    finally:
        torch._dynamo.reset()
