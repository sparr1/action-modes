"""Independent-root SAC arithmetic, state isolation, and CUDA capture gates."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.batched_teacher import (
    BatchedTeacherCore, BatchedTeacherSolver, clip_root_grad_norm_,
)
from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.reanalyze import LazyReanalyzer
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params
from tests.test_ambi_reanalyze import (
    _packet, _ReplayProbe, _globals, _assert_globals, models,  # noqa: F401
)
from tests.test_ambi_inner_decoupling import _clone_tree, _assert_tree_equal


def _agent(*, device="cpu", action_dim=3, **overrides):
    params = _tiny_params(
        inner_rounds=2, inner_rollouts_per_round=4, inner_rollout_horizon=2,
        inner_updates_per_round=2, inner_replay_capacity=16, num_q=2,
        q_pair_size=2, reanalyze_backend="batched", reanalyze_parallel_roots=3,
        inner_outer_policy_kl_coef=0.2,
    )
    params.update(overrides)
    if "inner_critic_updates_per_round" in params:
        params.pop("inner_updates_per_round")
    params.setdefault("target_entropy", -1.)
    params["inner_target_entropy"] = -1.
    if params.get("sac_actor_loss_scale_mode", "none") != "none":
        params.update(ent_coef=.01, inner_temperature_mode="inherit_outer")
    if params["q_representation"] == "distributional":
        params["num_q"] = 3
    cfg = _build_cfg(**params)
    cfg.device = device
    cfg.action_dim = action_dim
    cfg.action_dims = [action_dim]
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    return agent


class Tape:
    def __init__(self, core, entries=None, root=None):
        self.core, self.root = core, root
        self.entries = [] if entries is None else entries
        self.index = 0

    def __call__(self, kind, shape, upper=None):
        if self.root is None:
            value = self.core._draw(kind, shape, upper)
            self.entries.append((kind, upper, value.clone()))
        else:
            expected_kind, expected_upper, original = self.entries[self.index]
            assert (kind, upper) == (expected_kind, expected_upper)
            value = original[self.root:self.root + 1]
            assert value.shape == shape
        self.index += 1
        return value


def _assert_root_state(batch, single, root, *, rtol=1e-4, atol=2e-6):
    for name in ("actor", "critic", "target", "prior_actor", "prior_critic"):
        for key, actual in getattr(batch, name).items():
            torch.testing.assert_close(actual[root:root + 1], getattr(single, name)[key], rtol=rtol, atol=atol)
    for name in ("scale", "log_alpha"):
        torch.testing.assert_close(getattr(batch, name)[root:root + 1], getattr(single, name), rtol=rtol, atol=atol)
    torch.testing.assert_close(batch.replay[root:root + 1, :batch.size], single.replay[:, :single.size], rtol=rtol, atol=atol)
    assert (batch.position, batch.size, batch.critic_steps) == (single.position, single.size, single.critic_steps)
    for batch_optimizer, single_optimizer in zip(batch.optimizers, single.optimizers):
        for actual, expected in zip(batch_optimizer.state.values(), single_optimizer.state.values()):
            for key in actual:
                value = actual[key] if key == "step" else actual[key][root:root + 1]
                torch.testing.assert_close(value, expected[key], rtol=rtol, atol=atol)


@pytest.mark.parametrize("options", [
    {},
    {"q_representation": "distributional", "inner_finite_horizon": True,
     "dropout": 0.2, "inner_critic_dropout_enabled": True},
    {"outer_actor_entropy_mode": "tdmpc2_scaled", "inner_actor_entropy_mode": "tdmpc2_scaled",
     "sac_actor_loss_scale_mode": "tdmpc2_percentile_range", "inner_actor_loss_scale_update": "per_update",
     "inner_critic_target_initialization": "outer_target"},
    {"inner_temperature_mode": "fixed", "inner_temperature": .01, "inner_behavior_action": "mean"},
    {"inner_temperature_mode": "inherit_outer", "ent_coef": .03, "inner_bootstrap_source": "outer_online"},
    {"inner_replay_sampling": "without_replacement", "inner_bootstrap_source": "outer_target",
     "inner_behavior_action": "mean_plus_gaussian"},
    {"inner_critic_updates_per_round": 3, "inner_actor_updates_per_round": 1,
     "inner_temperature_updates_per_round": 1},
])
def test_batched_solve_matches_independent_parameters_replay_and_adam(options):
    torch.manual_seed(81)
    agent = _agent(**options)
    if agent.actor_loss_scale is not None:
        agent.actor_loss_scale.fill_(2.)
    core = BatchedTeacherCore(agent, 3)
    core.root.copy_(torch.randn_like(core.root))
    tape = Tape(core)
    actual = core.run(tape)
    for root in range(3):
        single = BatchedTeacherCore(agent, 1)
        single.root.copy_(core.root[root:root + 1])
        reader = Tape(single, tape.entries, root)
        expected = single.run(reader)
        assert reader.index == len(tape.entries)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a[root:root + 1], e, rtol=1e-4, atol=2e-6)
        _assert_root_state(core, single, root)
    assert all(parameter.grad is None for parameter in agent.model.parameters())


@pytest.mark.parametrize("representation,scaled", [("scalar", False), ("distributional", True)])
def test_batched_losses_and_gradients_match_native_sac(representation, scaled):
    agent = _agent(q_representation=representation, dropout=0.,
                   inner_finite_horizon=True, mppi_terminal_q_reduction="mean_all",
                   sac_actor_loss_scale_mode="tdmpc2_percentile_range" if scaled else "none",
                   inner_actor_loss_scale_update="per_update" if scaled else "per_action")
    core = BatchedTeacherCore(agent, 2)
    core.reset()
    noise = torch.randn(2, agent.cfg.inner_rollout_horizon, agent.cfg.inner_rollouts_per_round, agent.cfg.action_dim)
    core.root.normal_()
    with torch.no_grad():
        batch = core._rollout(noise)[:, :4]
    policy_noise = torch.randn(2, 4, agent.cfg.action_dim)
    prior_noise = torch.randn_like(policy_noise)
    pairs = torch.tensor([[0, core.heads - 1], [1, 0]])
    alpha = core.root.new_full((2, 1, 1), .03)
    critic_loss = core._critic_loss(batch, alpha, policy_noise, pairs, None, prior_noise, None)
    critic_loss.sum().backward()
    actor_loss, entropy, scale = core._actor_loss(batch[..., :agent.cfg.latent_dim], alpha, policy_noise, pairs, None, True)
    actor_loss.sum().backward()
    for root in range(2):
        engine = InnerImprovementEngine(agent)
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        fields = batch[root].split(core.field_widths, dim=-1)
        outputs = engine._sac_critic_kernel(*fields[:5], alpha[root].reshape(()), policy_noise[root], pairs[root],
                                            fields[5], prior_noise[root], actor_loss_scale=core.scale[root] if scaled else None)
        outputs[0].backward()
        torch.testing.assert_close(critic_loss[root], outputs[0], rtol=2e-5, atol=2e-6)
        for head, module in enumerate(engine.state.critic):
            for key, parameter in module.named_parameters():
                torch.testing.assert_close(core.critic[key].grad[root, head], parameter.grad, rtol=1e-4, atol=2e-6)
        outputs = engine._sac_actor_kernel(fields[0], alpha[root].reshape(()), policy_noise[root], pairs[root], True,
                                           q_scale=core.scale[root].reshape(1) if scaled else None)
        outputs[2].backward()
        torch.testing.assert_close(actor_loss[root], outputs[2], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(entropy[root], outputs[1], rtol=2e-5, atol=2e-6)
        for key, parameter in engine.state.actor.named_parameters():
            torch.testing.assert_close(core.actor[key].grad[root], parameter.grad, rtol=1e-4, atol=2e-6)
        if scaled:
            torch.testing.assert_close(scale[root].reshape(1), outputs[-1])
        with torch.no_grad():
            actual = core._rollout(noise)[root]
            expected = engine._dense_rollout_kernel(core.root[root:root + 1], noise[root], None)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA."))])
def test_root_clipping_and_adam_do_not_share_norm_or_moments(device):
    parameters = [torch.nn.Parameter(torch.randn(3, 2, 4, device=device)), torch.nn.Parameter(torch.randn(3, 4, device=device))]
    reference = [[torch.nn.Parameter(p[r].detach().clone()) for p in parameters] for r in range(3)]
    optimizer = torch.optim.Adam(parameters, lr=.1, eps=1e-5, fused=device == "cuda")
    separate = [torch.optim.Adam(p, lr=.1, eps=1e-5) for p in reference]
    for step in range(3):
        for p in parameters:
            p.grad = torch.randn_like(p) * p.new_tensor([1e-3, 1., 1e3]).reshape(3, *([1] * (p.ndim - 1)))
        for r, params in enumerate(reference):
            for p, original in zip(params, parameters):
                p.grad = original.grad[r].clone()
            torch.nn.utils.clip_grad_norm_(params, .5)
            separate[r].step()
        clip_root_grad_norm_(parameters, .5)
        optimizer.step()
        for r, params in enumerate(reference):
            for p, original in zip(params, parameters):
                torch.testing.assert_close(original[r], p)
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[original][key][r], separate[r].state[p][key])


def test_ring_wrap_and_oversized_appends_match_native_replay():
    agent = _agent()
    core = BatchedTeacherCore(agent, 2)
    core.reset()
    references = [LatentReplayBuffer(core.capacity, agent.cfg.latent_dim, agent.cfg.action_dim, "cpu") for _ in range(2)]
    for count in (5, 9, 17, 3):
        rows = torch.randn(2, count, sum(core.field_widths))
        core.append(rows)
        for root, replay in enumerate(references):
            replay.add_packed(rows[root])
            assert (replay.pos, replay.size) == (core.position, core.size)
            torch.testing.assert_close(core.replay[root, :core.size], replay._storage[:replay.size], rtol=0, atol=0)


def test_solver_isolates_rng_outer_state_and_refreshes_current_priors():
    agent = _agent()
    solver = BatchedTeacherSolver(agent)
    roots = torch.randn(5, agent.cfg.latent_dim)
    seeds = [31, 32, 33, 34, 35]
    agent.model.train()
    agent.model._pi[0].eval()
    modes = [(module, module.training) for module in agent.modules()]
    for parameter in agent.model.parameters():
        parameter.grad = torch.randn_like(parameter)
    gradients = [p.grad.clone() for p in agent.model.parameters()]
    before = deepcopy(agent.model.state_dict())
    live = deepcopy(agent.inner_engine.training_state_dict())
    rng = _globals()
    first = solver.solve(roots, seeds)
    second = solver.solve(roots, seeds)
    for a, b in zip(first[:3], second[:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    _assert_globals(rng)
    _assert_tree_equal(agent.model.state_dict(), before)
    _assert_tree_equal(agent.inner_engine.training_state_dict(), live)
    assert all(module.training == mode for module, mode in modes)
    for p, grad in zip(agent.model.parameters(), gradients):
        torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
    assert first[3]["distillation_reanalyze_padding_roots"] == 1
    assert first[2][0] == 5 * 2 * 4 * 2
    with torch.no_grad():
        agent.model._pi[-1].bias.add_(.1)
    third = solver.solve(roots, seeds)
    assert not torch.equal(first[0], third[0])


def test_batched_refresh_resume_and_protocol_validation(models):
    model = models(outer_actor_objective="distillation", reanalyze_backend="batched", reanalyze_parallel_roots=2)
    original = LazyReanalyzer(model.agent)
    obs, teacher = _packet()
    original.refresh(_ReplayProbe(), obs, teacher, 10)
    saved = _clone_tree(original.state_dict())
    expected = original.refresh(_ReplayProbe(), obs, teacher, 20)[0]
    restored = LazyReanalyzer(model.agent)
    restored.load_state_dict(saved)
    actual = restored.refresh(_ReplayProbe(), obs, teacher, 20)[0]
    _assert_tree_equal(actual, expected)
    assert saved["version"] == 3 and actual["valid"].sum() == 4
    for key, invalid in (("reanalyze_parallel_roots", 3), ("reanalyze_backend", "native"), ("reanalyze_rng_protocol", "other"), ("reanalyze_optimizer", "other")):
        state = _clone_tree(saved)
        state[key] = invalid
        before = _clone_tree(restored.state_dict())
        with pytest.raises(ValueError):
            restored.load_state_dict(state)
        _assert_tree_equal(restored.state_dict(), before)


@pytest.mark.parametrize("backend", ["batched", "batched_cuda_graph"])
def test_backend_selection_and_restrictions_are_opt_in(backend):
    assert _build_cfg(reanalyze_backend=backend, inner_outer_replay_fraction=.5).outer_actor_objective == "q_maximization"
    with pytest.raises(ValueError, match="inner_outer_replay_fraction"):
        _build_cfg(outer_actor_objective="distillation", reanalyze_backend=backend, inner_outer_replay_fraction=.5)


def test_tensor_regions_compile_without_breaks_and_preserve_explicit_draws():
    torch._dynamo.reset()
    agent = _agent(q_representation="distributional", inner_finite_horizon=True)
    core = BatchedTeacherCore(agent, 2)
    core.root.normal_()
    tape = Tape(core)
    expected = core.run(tape)
    reader_index = 0

    def replay(kind, shape, upper=None):
        nonlocal reader_index
        expected_kind, expected_upper, value = tape.entries[reader_index]
        reader_index += 1
        assert (kind, upper, shape) == (expected_kind, expected_upper, value.shape)
        return value

    for region in core.regions.values():
        region.enabled = True
        region._compiled = torch.compile(region.eager, backend="aot_eager", fullgraph=True, dynamic=False)
    actual = core.run(replay)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=2e-6)
    torch._dynamo.reset()


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a scheduled CUDA GPU.")


@CUDA
@pytest.mark.parametrize("compiled,scaled", [(False, False), (True, False), (True, True)])
def test_cuda_graph_matches_uncaptured_batch_and_is_exactly_repeatable(compiled, scaled):
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    agent = _agent(device="cuda", compile=compiled, compile_strict=compiled,
                   q_representation="distributional", dropout=.1, inner_critic_dropout_enabled=True,
                   inner_finite_horizon=True,
                   sac_actor_loss_scale_mode="tdmpc2_percentile_range" if scaled else "none",
                   inner_actor_loss_scale_update="per_update" if scaled else "per_action",
                   inner_actor_entropy_mode="tdmpc2_scaled" if scaled else "squashed")
    eager = BatchedTeacherSolver(agent)
    agent.cfg.reanalyze_backend = "batched_cuda_graph"
    graph = BatchedTeacherSolver(agent)
    roots = torch.randn(5, agent.cfg.latent_dim, device="cuda")
    seeds = [31, 32, 33, 34, 35]
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    for iteration in range(2):
        with torch.no_grad():
            for parameter in agent.model.parameters():
                parameter.add_(.001)
            if agent.log_ent_coef is not None:
                agent.log_ent_coef.add_(.1)
            else:
                agent.fixed_ent_coef.mul_(1.1)
            if scaled:
                agent.actor_loss_scale.fill_(2. + iteration)
        expected = eager.solve(roots, seeds)
        actual = graph.solve(roots, seeds)
        for a, e in zip(actual[:3], expected[:3]):
            torch.testing.assert_close(a, e, rtol=5e-4, atol=1e-5)
        for root in range(3):
            # Both workspaces contain the same final padded chunk.
            for key in graph.core.actor:
                torch.testing.assert_close(graph.core.actor[key][root], eager.core.actor[key][root], rtol=5e-4, atol=1e-5)
        repeated = graph.solve(roots, seeds)
        for a, e in zip(actual[:3], repeated[:3]):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    assert all(p.grad is None for p in agent.model.parameters())
