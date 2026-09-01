import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from RL.tdmpc2_core.common import layers
from RL.tdmpc2_core.common.parameter_noise import (
    adapt_parameter_noise_stddev,
    actor_mean_raw,
    classify_parameter_noise_actor,
    deterministic_actor_actions,
    deterministic_population_actions,
    make_perturbed_actor_parameters,
    parameter_noise_action_rms,
    population_actor_mean_raw,
    post_tanh_action_rms,
    sample_parameter_deltas,
)


ACTION_DIM = 3


def _actor():
    actor = layers.mlp(5, [11, 7], 2 * ACTION_DIM)
    actor.register_buffer("test_sentinel", torch.tensor([2.5]))
    return actor


def _state_snapshot(actor):
    return {
        name: value.detach().clone()
        for name, value in actor.state_dict().items()
    }


def _assert_state_unchanged(actor, snapshot):
    assert set(actor.state_dict()) == set(snapshot)
    for name, expected in snapshot.items():
        torch.testing.assert_close(
            actor.state_dict()[name], expected, rtol=0, atol=0
        )


def test_actor_classification_matches_current_clone_layout_and_rejects_unknowns():
    actor = _actor()
    spec = classify_parameter_noise_actor(actor, ACTION_DIM)

    assert spec.hidden_linear_parameter_names == (
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
    )
    assert spec.layer_norm_parameter_names == (
        "0.ln.weight",
        "0.ln.bias",
        "1.ln.weight",
        "1.ln.bias",
    )
    assert spec.perturbable_names == (
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
        "2.weight",
        "2.bias",
    )
    assert spec.final_weight_name == "2.weight"
    assert spec.final_bias_name == "2.bias"
    assert spec.buffer_names == ("test_sentinel",)

    wrong_head = layers.mlp(5, [7], 2 * ACTION_DIM + 1)
    with pytest.raises(ValueError, match="mean/log-std"):
        classify_parameter_noise_actor(wrong_head, ACTION_DIM)

    unexpected = _actor()
    unexpected.register_parameter("extra", nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="unclassified"):
        classify_parameter_noise_actor(unexpected, ACTION_DIM)

    with pytest.raises(TypeError, match="nn.Sequential"):
        classify_parameter_noise_actor(nn.Linear(5, 2 * ACTION_DIM), ACTION_DIM)

    plain_hidden = nn.Sequential(
        nn.Linear(5, 7),
        nn.Linear(7, 2 * ACTION_DIM),
    )
    with pytest.raises(ValueError, match="exactly NormedLinear"):
        classify_parameter_noise_actor(plain_hidden, ACTION_DIM)

    normed_final = nn.Sequential(
        layers.NormedLinear(5, 7),
        layers.NormedLinear(7, 2 * ACTION_DIM),
    )
    with pytest.raises(ValueError, match="exactly nn.Linear"):
        classify_parameter_noise_actor(normed_final, ACTION_DIM)


def test_private_generator_samples_exact_iid_deltas_without_global_rng_use():
    actor = _actor()
    spec = classify_parameter_noise_actor(actor, ACTION_DIM)
    global_state = torch.random.get_rng_state().clone()
    generator = torch.Generator().manual_seed(7301)
    initial_private_state = generator.get_state().clone()

    actual = sample_parameter_deltas(actor, spec, 4, generator=generator)

    torch.testing.assert_close(
        torch.random.get_rng_state(), global_state, rtol=0, atol=0
    )
    assert not torch.equal(generator.get_state(), initial_private_state)
    replay = torch.Generator().manual_seed(7301)
    parameters = dict(actor.named_parameters())
    for name in spec.perturbable_names:
        parameter = parameters[name]
        if name in {spec.final_weight_name, spec.final_bias_name}:
            expected = torch.zeros((4, *parameter.shape))
            expected[:, :ACTION_DIM] = torch.randn(
                (4, ACTION_DIM, *parameter.shape[1:]),
                generator=replay,
            )
            assert torch.count_nonzero(actual[name][:, ACTION_DIM:]) == 0
        else:
            expected = torch.randn((4, *parameter.shape), generator=replay)
        torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
        assert not actual[name].requires_grad


def test_perturbed_parameters_mask_layernorm_buffers_and_final_logstd_rows():
    actor = _actor()
    spec = classify_parameter_noise_actor(actor, ACTION_DIM)
    snapshot = _state_snapshot(actor)
    deltas = sample_parameter_deltas(
        actor,
        spec,
        5,
        generator=torch.Generator().manual_seed(19),
    )
    perturbed = make_perturbed_actor_parameters(actor, spec, deltas, 0.125)
    base = dict(actor.named_parameters())

    assert set(perturbed) == set(base)
    for name in spec.hidden_linear_parameter_names:
        expected = base[name].detach().unsqueeze(0) + 0.125 * deltas[name]
        torch.testing.assert_close(perturbed[name], expected, rtol=0, atol=0)
    for name in spec.layer_norm_parameter_names:
        expected = base[name].detach().unsqueeze(0).expand_as(perturbed[name])
        torch.testing.assert_close(perturbed[name], expected, rtol=0, atol=0)

    for name in (spec.final_weight_name, spec.final_bias_name):
        torch.testing.assert_close(
            perturbed[name][:, :ACTION_DIM],
            base[name][:ACTION_DIM].detach().unsqueeze(0)
            + 0.125 * deltas[name][:, :ACTION_DIM],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            perturbed[name][:, ACTION_DIM:],
            base[name][ACTION_DIM:].detach().unsqueeze(0).expand_as(
                perturbed[name][:, ACTION_DIM:]
            ),
            rtol=0,
            atol=0,
        )
    assert "test_sentinel" not in perturbed
    _assert_state_unchanged(actor, snapshot)

    for name, value in perturbed.items():
        assert value.untyped_storage().data_ptr() != (
            base[name].untyped_storage().data_ptr()
        )

    fixed_population = {
        name: value.detach().clone() for name, value in perturbed.items()
    }
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(1.0)
        perturbed[spec.layer_norm_parameter_names[0]][0, 0] = -123.0
    for name, expected in fixed_population.items():
        if name == spec.layer_norm_parameter_names[0]:
            expected = expected.clone()
            expected[0, 0] = -123.0
        torch.testing.assert_close(perturbed[name], expected, rtol=0, atol=0)
    assert base[spec.layer_norm_parameter_names[0]][0] != -123.0

    invalid = dict(deltas)
    invalid[spec.final_bias_name] = invalid[spec.final_bias_name].clone()
    invalid[spec.final_bias_name][:, ACTION_DIM:] = 1.0
    with pytest.raises(ValueError, match="log-std rows"):
        make_perturbed_actor_parameters(actor, spec, invalid, 0.1)


def _explicit_population_mean(actor, parameters, latents):
    buffers = {
        name: value.detach().clone() for name, value in actor.named_buffers()
    }
    outputs = []
    for index in range(latents.shape[0]):
        one_actor_parameters = {
            name: value[index] for name, value in parameters.items()
        }
        output = functional_call(
            actor,
            (one_actor_parameters, buffers),
            (latents[index],),
            strict=True,
        )
        outputs.append(output[..., :ACTION_DIM])
    return torch.stack(outputs)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4, 20])
def test_vmap_population_matches_explicit_loop_oracle_and_chunking(chunk_size):
    actor = _actor()
    actor.eval()
    spec = classify_parameter_noise_actor(actor, ACTION_DIM)
    snapshot = _state_snapshot(actor)
    training_flags = tuple(module.training for module in actor.modules())
    deltas = sample_parameter_deltas(
        actor,
        spec,
        7,
        generator=torch.Generator().manual_seed(41),
    )
    parameters = make_perturbed_actor_parameters(actor, spec, deltas, 0.07)
    latents = torch.linspace(-1.5, 1.5, 7 * 4 * 5).reshape(7, 4, 5)
    global_state = torch.random.get_rng_state().clone()

    expected_mean = _explicit_population_mean(actor, parameters, latents)
    actual_mean = population_actor_mean_raw(
        actor,
        spec,
        parameters,
        latents,
        chunk_size=chunk_size,
    )
    actual_actions = deterministic_population_actions(
        actor,
        spec,
        parameters,
        latents,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_actions, torch.tanh(expected_mean))
    assert actual_mean.shape == (7, 4, ACTION_DIM)
    torch.testing.assert_close(
        torch.random.get_rng_state(), global_state, rtol=0, atol=0
    )
    assert tuple(module.training for module in actor.modules()) == training_flags
    _assert_state_unchanged(actor, snapshot)


def test_base_mean_and_actions_are_stateless_and_use_only_mean_head():
    actor = _actor()
    actor.eval()
    snapshot = _state_snapshot(actor)
    latent = torch.randn(2, 5)
    direct = actor(latent)

    mean_raw = actor_mean_raw(actor, latent, action_dim=ACTION_DIM)
    action = deterministic_actor_actions(actor, latent, action_dim=ACTION_DIM)

    torch.testing.assert_close(mean_raw, direct[:, :ACTION_DIM])
    torch.testing.assert_close(action, torch.tanh(direct[:, :ACTION_DIM]))
    _assert_state_unchanged(actor, snapshot)


def test_post_tanh_rms_helpers_match_manual_population_displacement():
    actor = _actor()
    actor.eval()
    spec = classify_parameter_noise_actor(actor, ACTION_DIM)
    deltas = sample_parameter_deltas(
        actor,
        spec,
        6,
        generator=torch.Generator().manual_seed(83),
    )
    parameters = make_perturbed_actor_parameters(actor, spec, deltas, 0.04)
    latents = torch.randn(6, 3, 5)
    perturbed = deterministic_population_actions(actor, spec, parameters, latents)
    reference = deterministic_actor_actions(actor, latents, action_dim=ACTION_DIM)
    expected = (perturbed - reference).square().mean().sqrt()

    torch.testing.assert_close(post_tanh_action_rms(reference, perturbed), expected)
    torch.testing.assert_close(
        parameter_noise_action_rms(
            actor,
            spec,
            parameters,
            latents,
            chunk_size=4,
        ),
        expected,
    )

    zero_parameters = make_perturbed_actor_parameters(actor, spec, deltas, 0.0)
    # vmap lowers the per-policy linear algebra differently from the shared-
    # weight base call, so identical parameters can differ at float32 roundoff.
    assert parameter_noise_action_rms(
        actor, spec, zero_parameters, latents
    ) < 1e-6


def test_stddev_adaptation_is_bounded_log_proportional_and_validated():
    assert adapt_parameter_noise_stddev(0.1, 0.1, 0.4) == pytest.approx(0.2)
    assert adapt_parameter_noise_stddev(0.1, 1.0, 0.01) == pytest.approx(0.05)
    assert adapt_parameter_noise_stddev(0.1, 0.0, 0.2) == pytest.approx(0.2)
    assert adapt_parameter_noise_stddev(
        0.1, 0.3, 0.0, min_stddev=1e-5
    ) == pytest.approx(1e-5)
    assert adapt_parameter_noise_stddev(
        0.9,
        0.1,
        1.0,
        max_stddev=1.0,
    ) == pytest.approx(1.0)
    assert adapt_parameter_noise_stddev(
        0.1,
        2.0,
        5e-324,
    ) == pytest.approx(0.05)

    with pytest.raises(ValueError, match="measured_action_rms"):
        adapt_parameter_noise_stddev(0.1, float("nan"), 0.2)
    with pytest.raises(ValueError, match="adaptation_rate"):
        adapt_parameter_noise_stddev(0.1, 0.1, 0.2, adaptation_rate=0.0)
    with pytest.raises(ValueError, match="max_update_ratio"):
        adapt_parameter_noise_stddev(0.1, 0.1, 0.2, max_update_ratio=0.9)


def test_stale_specs_and_population_shape_errors_fail_closed():
    actor = _actor()
    spec = classify_parameter_noise_actor(actor, ACTION_DIM)
    actor.register_buffer("late_buffer", torch.ones(1))
    with pytest.raises(ValueError, match="buffer layout"):
        sample_parameter_deltas(
            actor,
            spec,
            2,
            generator=torch.Generator().manual_seed(1),
        )

    actor = _actor()
    spec = classify_parameter_noise_actor(actor, ACTION_DIM)
    deltas = sample_parameter_deltas(
        actor,
        spec,
        2,
        generator=torch.Generator().manual_seed(2),
    )
    parameters = make_perturbed_actor_parameters(actor, spec, deltas, 0.1)
    with pytest.raises(ValueError, match=r"shape \[K, R, D\]"):
        population_actor_mean_raw(actor, spec, parameters, torch.randn(2, 5))
    with pytest.raises(ValueError, match="same positive K"):
        population_actor_mean_raw(actor, spec, parameters, torch.randn(3, 1, 5))
    with pytest.raises(ValueError, match="input width"):
        population_actor_mean_raw(actor, spec, parameters, torch.randn(2, 1, 6))
    with pytest.raises(ValueError, match="chunk_size"):
        population_actor_mean_raw(
            actor,
            spec,
            parameters,
            torch.randn(2, 1, 5),
            chunk_size=0,
        )

    nonfinite = dict(deltas)
    nonfinite[spec.hidden_linear_parameter_names[0]] = nonfinite[
        spec.hidden_linear_parameter_names[0]
    ].clone()
    nonfinite[spec.hidden_linear_parameter_names[0]][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        make_perturbed_actor_parameters(actor, spec, nonfinite, 0.1)
