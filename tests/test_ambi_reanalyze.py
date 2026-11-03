"""Fresh teacher extraction, replay-root selection, and learner isolation."""

import math
import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.reanalyze import LazyReanalyzer
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_outer_replay import _buffer, _episode
from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture
def models():
    opened = []

    def create(**overrides):
        options = {
            "inner_rounds": 1, "inner_updates_per_round": 1,
            "inner_replay_capacity": 8, "inner_temperature_mode": "inherit_outer",
            "inner_execution_std_scale": 0.4,
        }
        options.update(overrides)
        model = _tiny_model(**options)
        model.cfg.reanalyze_batch_size = 2
        opened.append(model)
        return model

    yield create
    for model in opened:
        model.env.close()


def _packet(action_dim=1):
    index = torch.tensor([[1, 2, 50], [2, 3, 51]])
    episode = torch.tensor([[0, 0, 1], [0, 0, 1]])
    obs = torch.zeros(3, 3, 3)
    obs[:-1] = index[..., None].expand(2, 3, 3).float() / 50
    teacher = {
        "mean": torch.zeros(2, 3, action_dim),
        "log_std": torch.zeros(2, 3, action_dim),
        "valid": torch.zeros(2, 3, 1, dtype=torch.bool),
        "version": torch.full((2, 3, 1), -1, dtype=torch.long),
        "index": index, "episode": episode,
    }
    return obs, teacher


class _ReplayProbe:
    def __init__(self, *, mix=False):
        self.writes = []
        self.real = _buffer() if mix else None
        self.draws = []
        if mix:
            self.real.add(_episode())

    @property
    def num_sampleable_transitions(self):
        return 0 if self.real is None else self.real.num_sampleable_transitions

    def sample_transitions(self, count, *, generator):
        self.draws.append(generator)
        return self.real.sample_transitions(count, generator=generator)

    def write_teacher_targets(self, teacher, positions, mean, log_std, outer_version):
        self.writes.append((positions.clone(), mean.clone(), log_std.clone(), outer_version))
        result = _clone_tree(teacher)
        for position, mu, std in zip(positions.tolist(), mean, log_std):
            step, column = position
            mask = (
                (teacher["index"] == teacher["index"][step, column])
                & (teacher["episode"] == teacher["episode"][step, column])
            )
            result["mean"][mask] = mu
            result["log_std"][mask] = std
            result["valid"][mask] = True
            result["version"][mask] = outer_version
        return result


def _globals():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_globals(expected):
    assert random.getstate() == expected[0]
    current = np.random.get_state()
    assert current[0] == expected[1][0]
    np.testing.assert_array_equal(current[1], expected[1][1])
    assert current[2:] == expected[1][2:]
    torch.testing.assert_close(torch.random.get_rng_state(), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("conditioned", [False, True])
def test_capture_is_unscaled_rng_free_and_preserves_action_contract(models, conditioned):
    model = models(**({"inner_horizon_conditioning": "one_hot", "inner_finite_horizon": True}
                      if conditioned else {}))
    agent = model.agent
    agent.model.eval()
    root = agent.model.encode(torch.zeros(1, 3)).detach()
    engine = InnerImprovementEngine(agent)
    engine.reset_for_evaluation(79, reuse_action_pool=True)
    plain = engine.act(root, t0=True, collect_diagnostics=False, return_behavior_policy=True)
    expected_rng = _clone_tree(engine.rng.training_state_dict())
    assert engine.last_teacher_policy is None
    engine.reset_for_evaluation(79, reuse_action_pool=True)
    global_rng = _globals()
    captured = engine.act(
        root, t0=True, collect_diagnostics=False,
        return_behavior_policy=True, capture_teacher_policy=True,
    )
    assert len(plain) == len(captured) == 4
    assert not plain[0].requires_grad and not captured[0].requires_grad
    torch.testing.assert_close(captured[0], plain[0], rtol=0, atol=0)
    _assert_tree_equal(engine.rng.training_state_dict(), expected_rng)
    _assert_globals(global_rng)
    target = engine.last_teacher_policy
    assert set(target) == {"pre_tanh_mean", "log_std"}
    assert all(not value.requires_grad for value in target.values())
    torch.testing.assert_close(target["pre_tanh_mean"], captured[3]["pre_tanh_mean"])
    torch.testing.assert_close(
        target["log_std"], captured[3]["log_std"] - math.log(0.4),
    )
    engine.reset_for_evaluation(79, reuse_action_pool=True)
    agent.cfg.inner_execution_action = "mean"
    engine.act(root, t0=True, collect_diagnostics=False, capture_teacher_policy=True)
    _assert_tree_equal(engine.last_teacher_policy, target)
    engine.act(root, collect_diagnostics=False)
    assert engine.last_teacher_policy is None


@pytest.mark.parametrize("mix", [False, True])
def test_refresh_deduplicates_sequence_first_and_preserves_live_learner(models, monkeypatch, mix):
    model = models(inner_outer_replay_fraction=0.5 if mix else 0.0)
    agent = model.agent
    replay = _ReplayProbe(mix=mix)
    if mix:
        agent.inner_engine.outer_replay_buffer = replay
    agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    agent.prepare_training_resume_boundary()
    live_engine = agent.inner_engine
    last_metrics, last_lengths = agent.last_inner_metrics, agent.last_inner_rollout_lengths
    agent.model.train()
    agent.model._pi[0].eval()  # Preserve mixed child modes, not only the root flag.
    modes = [(module, module.training) for module in agent.modules()]
    for parameter in agent.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    grads = [parameter.grad.clone() for parameter in agent.model.parameters()]
    before = _clone_tree(agent.training_state_dict())
    obs, teacher = _packet(model.cfg.action_dim)
    replay.draws.clear()
    monkeypatch.setattr(model.env, "step", lambda *_: pytest.fail("Reanalysis interacted with the env"))
    global_rng = _globals()
    reanalyzer = LazyReanalyzer(agent)
    refreshed, metrics = reanalyzer.refresh(replay, obs, teacher, 11)
    _assert_globals(global_rng)
    assert agent.inner_engine is live_engine
    assert agent.last_inner_metrics is last_metrics
    assert agent.last_inner_rollout_lengths is last_lengths
    assert all(module.training == training for module, training in modes)
    _assert_tree_equal(agent.training_state_dict(), before)
    for parameter, grad in zip(agent.model.parameters(), grads):
        torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)
    assert replay.writes[0][0].tolist() == [[0, 0], [1, 0], [1, 1]]
    assert metrics["distillation_reanalyze_requested_roots"] == 4
    assert metrics["distillation_reanalyze_roots"] == 3
    assert metrics["distillation_reanalyze_duplicates"] == 1
    assert metrics["distillation_reanalyze_actor_updates"] == 3
    assert metrics["distillation_reanalyze_critic_updates"] == 3
    assert metrics["distillation_reanalyze_model_steps"] == 12
    assert reanalyzer.roots == 3 and reanalyzer.refreshes == 1
    assert refreshed["valid"][:, :2].all()
    assert not refreshed["valid"][:, 2].any()
    assert not teacher["valid"].any()
    torch.testing.assert_close(refreshed["mean"][0, 1], refreshed["mean"][1, 0], rtol=0, atol=0)
    assert bool(replay.draws) == mix
    if mix:
        assert all(generator is not live_engine.rng.generator("replay") for generator in replay.draws)


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_resume_reproduces_fresh_teacher_solves_and_reuses_allocations(models, adaptation):
    options = {"inner_critic_adaptation": adaptation}
    if adaptation == "lora_rl":
        options["inner_critic_lora_rank"] = 2
    model = models(**options)
    obs, teacher = _packet(model.cfg.action_dim)
    reanalyzer = LazyReanalyzer(model.agent)
    initial = _clone_tree(reanalyzer.state_dict())
    first, _ = reanalyzer.refresh(_ReplayProbe(), obs, teacher, 4)
    actor_id = id(reanalyzer.engine._action_pool.actor)
    regions_id = id(reanalyzer.engine._compile_regions)
    reanalyzer.load_state_dict(initial)
    second, _ = reanalyzer.refresh(_ReplayProbe(), obs, teacher, 4)
    _assert_tree_equal(first, second)
    assert id(reanalyzer.engine._action_pool.actor) == actor_id
    assert id(reanalyzer.engine._compile_regions) == regions_id
    assert reanalyzer.refreshes == 1 and reanalyzer.roots == 3


def test_each_reanalyzed_root_matches_a_fresh_standalone_solve(models):
    model = models()
    agent = model.agent
    agent.model.eval()
    obs, teacher = _packet(model.cfg.action_dim)
    reanalyzer = LazyReanalyzer(agent)
    seed_stream = torch.Generator().set_state(reanalyzer.state_dict()["rng_state"])
    torch.randint(0, 2**31 - 1, (), generator=seed_stream)  # Observation encoding.
    positions = torch.tensor([[0, 0], [1, 0], [1, 1]])
    with torch.no_grad():
        roots = agent.model.encode(obs[positions[:, 0], positions[:, 1]])
    expected = []
    for root in roots.split(1):
        seed = int(torch.randint(0, 2**31 - 1, (), generator=seed_stream))
        fresh = InnerImprovementEngine(agent)
        fresh.reset_for_evaluation(seed)
        fresh.act(root, t0=True, eval_mode=True, collect_diagnostics=False,
                  capture_teacher_policy=True)
        expected.append(fresh.last_teacher_policy)
    replay = _ReplayProbe()
    reanalyzer.refresh(replay, obs, teacher, 4)
    for name, actual in (("pre_tanh_mean", replay.writes[0][1]), ("log_std", replay.writes[0][2])):
        torch.testing.assert_close(actual, torch.stack([row[name] for row in expected]), rtol=0, atol=0)


def test_observation_augmentation_uses_private_rng_and_current_encoder(models, monkeypatch):
    model = models()
    original = model.agent.model.encode
    calls = []

    def noisy_encoder(obs):
        calls.append(obs.clone())
        encoded = original(obs)
        return encoded + torch.rand_like(encoded) * 0.01

    monkeypatch.setattr(model.agent.model, "encode", noisy_encoder)
    obs, teacher = _packet(model.cfg.action_dim)
    reanalyzer = LazyReanalyzer(model.agent)
    saved = _clone_tree(reanalyzer.state_dict())
    first, _ = reanalyzer.refresh(_ReplayProbe(), obs, teacher, 0)
    torch.rand(47)
    state_after_outer_draws = _globals()
    reanalyzer.load_state_dict(saved)
    second, _ = reanalyzer.refresh(_ReplayProbe(), obs, teacher, 0)
    _assert_tree_equal(first, second)
    _assert_globals(state_after_outer_draws)
    assert len(calls) == 2
    torch.testing.assert_close(calls[0], torch.stack([obs[0, 0], obs[1, 0], obs[1, 1]]))


def test_refresh_failure_restores_modes_global_rng_and_seed_stream(models, monkeypatch):
    model = models()
    reanalyzer = LazyReanalyzer(model.agent)
    model.agent.model.train()
    model.agent.model._pi[0].eval()
    modes = [(module, module.training) for module in model.agent.modules()]
    saved = _clone_tree(reanalyzer.state_dict())
    globals_before = _globals()
    obs, teacher = _packet(model.cfg.action_dim)
    replay = _ReplayProbe()

    def fail(*args, **kwargs):
        torch.rand(4)
        random.random()
        np.random.rand(2)
        raise RuntimeError("teacher failure")

    monkeypatch.setattr(reanalyzer.engine, "solve_teacher", fail)
    with pytest.raises(RuntimeError, match="teacher failure"):
        reanalyzer.refresh(replay, obs, teacher, 3)
    _assert_globals(globals_before)
    _assert_tree_equal(reanalyzer.state_dict(), saved)
    assert all(module.training == training for module, training in modes)
    assert not replay.writes


@pytest.mark.parametrize("field,value", [
    ("roots", -1), ("refreshes", True), ("rng_state", torch.zeros(3, dtype=torch.uint8)),
])
def test_resume_preflight_rejects_invalid_state_transactionally(models, field, value):
    reanalyzer = LazyReanalyzer(models().agent)
    saved = _clone_tree(reanalyzer.state_dict())
    invalid = _clone_tree(saved)
    invalid[field] = value
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        reanalyzer.load_state_dict(invalid)
    _assert_tree_equal(reanalyzer.state_dict(), saved)
