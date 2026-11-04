"""Same-draw reference checks for root-batched SAC execution optimizations."""

import pytest
import torch

from RL.tdmpc2_core.batched_teacher import BatchedTeacherCore, _clipped_root_gradients
from tests.test_ambi_batched_teacher import _agent, Tape


ALL = frozenset({"selective_q", "actor_cache", "fused_updates"})


class ReplayTape:
    def __init__(self, entries):
        self.entries, self.index = entries, 0

    def __call__(self, kind, shape, upper=None):
        expected_kind, expected_upper, value = self.entries[self.index]
        self.index += 1
        assert (kind, upper, tuple(shape)) == (expected_kind, expected_upper, tuple(value.shape))
        return value


def assert_core_close(actual, expected, *, rtol=2e-4, atol=3e-6):
    for name in ("actor", "critic", "target", "prior_actor", "prior_critic"):
        for key, value in getattr(actual, name).items():
            reference = getattr(expected, name)[key]
            torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)
            if name in {"actor", "critic"}:
                assert (value.grad is None) == (reference.grad is None)
                if value.grad is not None:
                    torch.testing.assert_close(value.grad, reference.grad, rtol=rtol, atol=atol)
    for name in ("scale", "log_alpha", "alpha_fixed"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=rtol, atol=atol)
    torch.testing.assert_close(actual.replay[:, :actual.size], expected.replay[:, :expected.size], rtol=rtol, atol=atol)
    assert (actual.position, actual.size, actual.critic_steps, actual.counts) == (
        expected.position, expected.size, expected.critic_steps, expected.counts,
    )
    for optimizer, reference in zip(actual.optimizers, expected.optimizers):
        for state, target in zip(optimizer.state.values(), reference.state.values()):
            for name in state:
                torch.testing.assert_close(state[name], target[name], rtol=rtol, atol=atol)


@pytest.mark.parametrize("features", [frozenset({name}) for name in sorted(ALL)] + [ALL])
@pytest.mark.parametrize("options", [
    {"q_representation": "scalar", "inner_temperature_mode": "auto"},
    {"q_representation": "distributional", "inner_finite_horizon": True,
     "inner_q_actor_reduction": "mean_pair", "mppi_terminal_q_reduction": "min_pair",
     "dropout": .2, "inner_critic_dropout_enabled": True},
    {"q_representation": "distributional",
     "inner_actor_entropy_mode": "tdmpc2_scaled", "inner_log_std_mapping": "tdmpc2_tanh",
     "sac_actor_loss_scale_mode": "tdmpc2_percentile_range", "inner_actor_loss_scale_update": "per_update"},
    {"q_representation": "distributional", "inner_bootstrap_source": "outer_online",
     "inner_q_target_reduction": "mean_all", "inner_q_actor_reduction": "min_all",
     "inner_replay_sampling": "without_replacement"},
])
def test_optimized_full_solve_preserves_reference_draws_losses_and_optimizer_state(features, options):
    torch.manual_seed(463)
    agent = _agent(inner_rounds=3, inner_critic_updates_per_round=4,
                   inner_actor_updates_per_round=2, inner_temperature_updates_per_round=2,
                   inner_batch_size=4, inner_replay_capacity=24, **options)
    reference = BatchedTeacherCore(agent, 3, optimization_features=())
    optimized = BatchedTeacherCore(agent, 3, optimization_features=features)
    # Run twice after changing the shared priors. Cached statistics must never
    # survive the last actor block, round boundaries, or the next fresh solve.
    for iteration in range(2):
        if iteration:
            with torch.no_grad():
                for parameter in agent.model.parameters():
                    parameter.add_(.002)
        roots = torch.randn_like(reference.root)
        reference.root.copy_(roots)
        optimized.root.copy_(roots)
        tape = Tape(reference)
        expected = reference.run(tape)
        reader = ReplayTape(tape.entries)
        actual = optimized.run(reader)
        assert reader.index == len(tape.entries)
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, rtol=2e-4, atol=3e-6)
        assert_core_close(optimized, reference)


@pytest.mark.parametrize("reduction", ["min_pair", "mean_pair"])
@pytest.mark.parametrize("pair_size", [1, 2])
def test_selected_heads_match_full_ensemble_action_gradients_with_explicit_dropout(reduction, pair_size):
    torch.manual_seed(731)
    agent = _agent(q_representation="distributional", q_pair_size=pair_size,
                   inner_q_actor_reduction=reduction, dropout=.4, inner_critic_dropout_enabled=True)
    reference = BatchedTeacherCore(agent, 2, optimization_features=())
    optimized = BatchedTeacherCore(agent, 2, optimization_features={"selective_q"})
    reference.reset()
    optimized.reset()
    z = torch.randn(2, 7, agent.cfg.latent_dim)
    actions = torch.randn(2, 7, agent.cfg.action_dim)
    expected_actions = actions.clone().requires_grad_()
    actual_actions = actions.clone().requires_grad_()
    # Different ordering and nonadjacent head selection in the two roots.
    pairs = torch.tensor([[2, 0], [1, 2]])[:, :pair_size]
    masks = reference._masks(z.shape[1], reference._draw)
    expected = reference._reduced_values(
        {key: value.detach() for key, value in reference.critic.items()},
        z, expected_actions, reduction, pairs, masks, training=True,
    )
    actual = optimized._reduced_values(
        {key: value.detach() for key, value in optimized.critic.items()},
        z, actual_actions, reduction, pairs, masks, training=True,
    )
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual_actions.grad, expected_actions.grad, rtol=2e-4, atol=3e-6)
    assert all(p.grad is None for p in optimized.critic.values())
    assert agent.model.q_backend.num_q == 3


def test_cache_visits_resident_next_states_in_minibatch_tiles_per_critic_block(monkeypatch):
    agent = _agent(inner_rounds=3, inner_critic_updates_per_round=4,
                   inner_actor_updates_per_round=1, inner_replay_capacity=24)
    # Exercise the low-level ring contract even though production action-local
    # validation reserves enough capacity for the entire configured solve.
    agent.cfg.inner_replay_capacity = 11
    core = BatchedTeacherCore(agent, 2, optimization_features={"actor_cache"})
    rows = []
    cache = core.regions["actor_cache"].eager
    def observe(next_z):
        rows.append(next_z.shape[1])
        return cache(next_z)
    monkeypatch.setattr(core.regions["actor_cache"], "eager", observe)
    core.root.normal_()
    core.run()
    # The 8/11/11 resident rows need 2/3/3 tiles; the last row of the
    # nondivisible rounds is padded without adding a replay label.
    assert rows == [4] * 8
    rows.clear()
    core.run()
    assert rows == [4] * 8


@pytest.mark.parametrize("size", [1, 4, 5, 11])
def test_cache_tiles_preserve_minibatch_layout_padding_indices_and_rng(monkeypatch, size):
    torch.manual_seed(613)
    agent = _agent(inner_critic_updates_per_round=4, inner_actor_updates_per_round=1,
                   inner_batch_size=4, inner_finite_horizon=True)
    core = BatchedTeacherCore(agent, 2, optimization_features={"actor_cache"})
    core.reset()
    core.replay.normal_()
    core.size = size
    offset = sum(core.field_widths[:3])
    indices = torch.tensor([[size - 1, 0, size // 2, size - 1],
                            [0, size - 1, 0, size // 2]])
    batch = core.replay.gather(1, indices[..., None].expand(-1, -1, core.replay.shape[-1]))
    reference_next_z = batch[..., offset:offset + agent.cfg.latent_dim]
    observed = []
    cache = core.regions["actor_cache"].eager

    def observe(next_z):
        assert next_z.shape == reference_next_z.shape
        assert next_z.stride() == reference_next_z.stride()
        assert next_z.storage_offset() == reference_next_z.storage_offset()
        observed.append(next_z.clone())
        return cache(next_z)

    monkeypatch.setattr(core.regions["actor_cache"], "eager", observe)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        raw = core._cache_replay_actor_outputs()
        expected = cache(reference_next_z)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert raw.shape == (2, size, 2 * agent.cfg.action_dim)
    assert len(observed) == (size + 3) // 4
    for tile, value in enumerate(observed):
        start, valid = tile * 4, min(4, size - tile * 4)
        torch.testing.assert_close(value[:, :valid],
                                   core.replay[:, start:start + valid, offset:offset + agent.cfg.latent_dim],
                                   rtol=0, atol=0)
        assert torch.count_nonzero(value[:, valid:]).item() == 0
    actual = raw.gather(1, indices[..., None].expand(-1, -1, raw.shape[-1]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cache_padded_cost_allows_equal_work_and_skips_more_expensive_round(monkeypatch):
    agent = _agent(inner_critic_updates_per_round=2, inner_actor_updates_per_round=1,
                   inner_batch_size=5)
    core = BatchedTeacherCore(agent, 2, optimization_features={"actor_cache"})
    sizes, rows = [], []
    cache_replay = core._cache_replay_actor_outputs
    cache_tile = core.regions["actor_cache"].eager

    def observe_replay():
        sizes.append(core.size)
        return cache_replay()

    def observe_tile(next_z):
        rows.append(next_z.shape[1])
        return cache_tile(next_z)

    monkeypatch.setattr(core, "_cache_replay_actor_outputs", observe_replay)
    monkeypatch.setattr(core.regions["actor_cache"], "eager", observe_tile)
    core.run()
    # First round: ceil(8/5)*5 == 2 critic updates * 5 rows, so cache.
    # Second round: ceil(16/5)*5 > 10, so ordinary minibatch forwards win.
    assert sizes == [8]
    assert rows == [5, 5]


def test_cache_is_not_used_for_interleaved_actor_updates(monkeypatch):
    agent = _agent()
    core = BatchedTeacherCore(agent, 2, optimization_features={"actor_cache"})
    assert not core.helper._uses_component_update_schedule
    monkeypatch.setattr(core.regions["actor_cache"], "eager",
                        lambda *args: pytest.fail("Actor statistics cannot remain cached across actor updates"))
    core.run()


def test_cache_ring_overwrite_full_solve_matches_reference():
    torch.manual_seed(151)
    agent = _agent(q_representation="distributional", inner_rounds=3,
                   inner_critic_updates_per_round=4, inner_actor_updates_per_round=2,
                   inner_replay_capacity=24, inner_batch_size=4,
                   dropout=.2, inner_critic_dropout_enabled=True, inner_finite_horizon=True)
    # Deliberately force ring overwrite inside the low-level core. Public
    # action-local recipes reserve a full solve and do not need this override.
    agent.cfg.inner_replay_capacity = 11
    reference = BatchedTeacherCore(agent, 3, optimization_features=())
    optimized = BatchedTeacherCore(agent, 3, optimization_features=ALL)
    reference.root.normal_()
    optimized.root.copy_(reference.root)
    tape = Tape(reference)
    expected = reference.run(tape)
    reader = ReplayTape(tape.entries)
    actual = optimized.run(reader)
    assert reader.index == len(tape.entries)
    assert optimized.size == 11 and optimized.position == 24 % 11
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=2e-4, atol=3e-6)
    assert_core_close(optimized, reference)


def test_cache_is_skipped_when_it_would_evaluate_more_rows(monkeypatch):
    agent = _agent(inner_critic_updates_per_round=2, inner_actor_updates_per_round=1,
                   inner_batch_size=2)
    core = BatchedTeacherCore(agent, 2, optimization_features={"actor_cache"})
    monkeypatch.setattr(core.regions["actor_cache"], "eager",
                        lambda *args: pytest.fail("Caching should not increase actor forward rows"))
    core.run()


def test_cached_statistics_preserve_fresh_sampling_and_entropy():
    agent = _agent(inner_actor_entropy_mode="tdmpc2_scaled", inner_log_std_mapping="tdmpc2_tanh")
    core = BatchedTeacherCore(agent, 2)
    core.reset()
    z = torch.randn(2, 7, agent.cfg.latent_dim)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        raw = core._cache_actor_outputs(z)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    first = None
    for noise in (torch.zeros(2, 7, agent.cfg.action_dim), torch.ones(2, 7, agent.cfg.action_dim)):
        actual, info = core._pi(core.actor, z, noise, raw_output=raw)
        expected, expected_info = core._pi(core.actor, z, noise)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key in info:
            torch.testing.assert_close(info[key], expected_info[key], rtol=0, atol=0)
        if first is None:
            first = actual
        else:
            assert not torch.equal(first, actual)


def test_fused_clipping_matches_independent_root_clipping_and_preserves_target_storage():
    torch.manual_seed(92)
    gradients = (torch.randn(3, 2, 4), torch.randn(3, 4))
    gradients = tuple(g * torch.tensor([1e-3, 1., 1e3]).reshape(3, *([1] * (g.ndim - 1))) for g in gradients)
    clipped = _clipped_root_gradients(gradients, .5)
    for root in range(3):
        parameters = [torch.nn.Parameter(torch.zeros_like(g[root])) for g in gradients]
        for p, g in zip(parameters, gradients):
            p.grad = g[root].clone()
        torch.nn.utils.clip_grad_norm_(parameters, .5)
        for actual, expected in zip(clipped, parameters):
            torch.testing.assert_close(actual[root], expected.grad, rtol=1e-6, atol=1e-7)
    agent = _agent()
    core = BatchedTeacherCore(agent, 2, optimization_features={"fused_updates"})
    core.reset()
    pointers = {key: value.data_ptr() for key, value in core.target.items()}
    with torch.no_grad():
        for value in core.critic.values():
            value.add_(.1)
    expected = {key: value.lerp(core.critic[key], agent.cfg.inner_critic_target_tau) for key, value in core.target.items()}
    core._update_targets()
    for key, value in core.target.items():
        assert value.data_ptr() == pointers[key]
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_optimized_regions_compile_fullgraph_with_identical_explicit_random_draws():
    torch._dynamo.reset()
    agent = _agent(q_representation="distributional", dropout=.2,
                   inner_critic_dropout_enabled=True, inner_finite_horizon=True,
                   inner_critic_updates_per_round=3, inner_actor_updates_per_round=1,
                   inner_actor_entropy_mode="tdmpc2_scaled")
    core = BatchedTeacherCore(agent, 2, optimization_features=ALL)
    core.root.normal_()
    tape = Tape(core)
    expected = core.run(tape)
    for region in core.regions.values():
        region.enabled = True
        region._compiled = torch.compile(region.eager, backend="aot_eager", fullgraph=True, dynamic=False)
    reader = ReplayTape(tape.entries)
    actual = core.run(reader)
    assert reader.index == len(tape.entries)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=1e-4, atol=3e-6)
    torch._dynamo.reset()


def test_disabled_optimization_default_and_explicit_reference_have_identical_arithmetic():
    agent = _agent(reanalyze_batched_optimizations=False)
    configured = BatchedTeacherCore(agent, 2)
    reference = BatchedTeacherCore(agent, 2, optimization_features=())
    assert configured.optimization_features == frozenset()
    assert set(configured.regions) == {"rollout", "actor", "critic"}
    roots = torch.randn_like(reference.root)
    configured.root.copy_(roots)
    reference.root.copy_(roots)
    tape = Tape(reference)
    expected = reference.run(tape)
    reader = ReplayTape(tape.entries)
    actual = configured.run(reader)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    assert_core_close(configured, reference, rtol=0, atol=0)
