"""Independent controller UTD preserves the recurrent world-model contract."""

from copy import deepcopy

import pytest
import torch

from test_ambixqc_prior_checkpoint import wrappers
from test_ambixqc_replay_archive import _assert_equal, _resident_rows, _rng_state


class _Replay:
    """Fresh, reproducible sequence batches without consuming training RNG."""

    def __init__(self, agent, seed=41):
        self.agent = agent
        self.generator = torch.Generator().manual_seed(seed)
        self.batches = []

    def sample(self):
        cfg = self.agent.cfg
        h, b = cfg.train_unroll_horizon, cfg.batch_size

        def draw(*shape):
            return torch.randn(*shape, generator=self.generator).to(self.agent.device)

        batch = (
            draw(h + 1, b, cfg.obs_shape["state"][0]),
            draw(h, b, cfg.action_dim).tanh(),
            draw(h, b, 1),
            (draw(h, b, 1) > 0.5).float(),
            None,
        )
        self.batches.append(batch)
        return batch


def _world_state(agent):
    return deepcopy({
        "model": agent.model.state_dict(),
        "optimizer": agent.world_optimizer.state_dict(),
        "gradients": [parameter.grad for parameter in agent._world_params],
        "reward_statistics": agent.reward_normalizer.state_dict(),
    })


def _optimizer_steps(optimizer):
    return {int(state["step"].item()) for state in optimizer.state.values()}


@pytest.mark.parametrize("ratio", [True, False, 0, -1, 1.5, None, "bad", float("nan"), float("inf")])
def test_invalid_controller_ratio_is_rejected(wrappers, ratio):
    with pytest.raises(ValueError, match="xqc_utd"):
        wrappers(xqc_utd=ratio)


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_ratio_changes_only_outer_controller_schedule_configuration(wrappers, ratio):
    default = wrappers()
    selected = wrappers(xqc_utd=ratio)
    assert default.cfg.xqc_utd == 1
    assert selected.cfg.utd == 1
    assert selected.cfg.xqc_lr_transition_steps == selected.cfg.steps * ratio
    assert selected.cfg.lr == default.cfg.lr
    assert selected.cfg.inner_critic_updates_per_action == default.cfg.inner_critic_updates_per_action
    assert selected.cfg.inner_actor_lr == default.cfg.inner_actor_lr
    with pytest.raises(ValueError, match="Use xqc_utd"):
        wrappers(utd=2)


def test_explicit_one_matches_default_seeded_training(wrappers):
    outcomes = []
    for settings in ({}, {"xqc_utd": 1}):
        model = wrappers(inner_operator="none", **settings)
        model.env.reset(seed=3)
        model.env.action_space.seed(3)
        model.learn(total_timesteps=10)
        outcomes.append(deepcopy({
            "checkpoint": model.agent.checkpoint_state(),
            "replay": _resident_rows(model.buffer), "rng": _rng_state(model),
        }))
    _assert_equal(*outcomes)


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("auxiliary", [None, True, False])
def test_fresh_slots_drive_optimizer_delay_target_and_bn_clocks(
    wrappers, monkeypatch, ratio, auxiliary,
):
    settings = {} if auxiliary is None else {
        "aux_return_mode": "xqc", "aux_return_detach_representation": auxiliary,
    }
    model = wrappers(xqc_utd=ratio, xqc_target_update_interval=3, **settings)
    agent, workspace = model.agent, model.agent.xqc_workspace
    replay = _Replay(agent)
    agent.observe_reward(2.0, False, False)
    reward_before = deepcopy(agent.reward_normalizer.state_dict())
    targets = [agent.xqc_controller.critic_target]
    if agent.aux_return is not None:
        targets.append(agent.aux_return.critic_target)
    target_buffers = [deepcopy(dict(target.named_buffers())) for target in targets]
    critic_steps, actor_steps, actor_bn_changes = [], [], []
    original_critic = workspace.step_critic
    original_actor = workspace.step_actor_and_temperature
    original_objective = agent.xqc_controller.actor_objective

    def critic_step():
        before = deepcopy(dict(targets[0].named_parameters()))
        result = original_critic()
        changed = any(not torch.equal(value, before[key])
                      for key, value in targets[0].named_parameters())
        critic_steps.append((*result, changed))
        return result

    def actor_step(*args, **kwargs):
        result = original_actor(*args, **kwargs)
        actor_steps.append(result)
        return result

    def actor_objective(latents, **kwargs):
        assert latents.shape[:2] == (agent.cfg.train_unroll_horizon + 1, agent.cfg.batch_size)
        before = deepcopy(dict(agent.xqc_controller.actor.named_buffers()))
        result = original_objective(latents, **kwargs)
        actor_bn_changes.append(any(
            not torch.equal(value, before[key])
            for key, value in agent.xqc_controller.actor.named_buffers()
            if key.endswith(("running_mean", "running_var"))
        ))
        return result

    monkeypatch.setattr(workspace, "step_critic", critic_step)
    monkeypatch.setattr(workspace, "step_actor_and_temperature", actor_step)
    monkeypatch.setattr(agent.xqc_controller, "actor_objective", actor_objective)
    for world_step in range(1, 4):
        metrics = agent.update(replay)
        assert agent.num_updates == agent.outer_version == world_step
        assert workspace.update_step == world_step * ratio
        assert metrics["world_model_num_updates"] == world_step
        assert metrics["xqc_num_updates"] == world_step * ratio
        assert metrics["xqc_updates_per_world_update"] == ratio
        assert _optimizer_steps(agent.world_optimizer) == {world_step}
        assert _optimizer_steps(workspace.critic_optimizer) == {world_step * ratio}
        if agent.aux_return is not None:
            assert agent.aux_return.update_step == world_step * ratio
            assert _optimizer_steps(agent.aux_return.critic_optimizer) == {world_step * ratio}
            assert metrics["aux_return_learning_rate"] == metrics["critic_learning_rate"]
            assert metrics["aux_return_target_updated"] == metrics["target_updated"]

    count = 3 * ratio
    assert len(replay.batches) == len(critic_steps) == len(actor_steps) == count
    assert all(not torch.equal(left[0], right[0])
               for left, right in zip(replay.batches, replay.batches[1:]))
    expected_accepted = [int(slot % 3 == 0) for slot in range(count)]
    assert [step["actor_update_accepted"] for step in actor_steps] == expected_accepted
    assert all(actor_bn_changes)  # Even when that slot skips the actor optimizer.
    assert workspace.actor_optimizer_steps == workspace.temperature_optimizer_steps == sum(expected_accepted)
    assert _optimizer_steps(workspace.actor_optimizer) == {sum(expected_accepted)}
    assert _optimizer_steps(workspace.temperature_optimizer) == {sum(expected_accepted)}
    assert metrics["xqc_actor_num_updates"] == metrics["xqc_temperature_num_updates"] == sum(expected_accepted)
    for slot, (lr, updated, changed) in enumerate(critic_steps):
        expected_lr = agent.cfg.xqc_critic_lr + (
            agent.cfg.xqc_lr_end - agent.cfg.xqc_critic_lr
        ) * slot / (agent.cfg.steps * ratio)
        assert lr == pytest.approx(expected_lr)
        assert updated == changed == ((slot + 1) % 3 == 0)
    for index, step in enumerate(step for step in actor_steps if step["actor_update_accepted"]):
        expected_lr = agent.cfg.xqc_actor_lr + (
            agent.cfg.xqc_lr_end - agent.cfg.xqc_actor_lr
        ) * index / (agent.cfg.steps * ratio)
        assert step["actor_learning_rate"] == step["temperature_learning_rate"] == pytest.approx(expected_lr)
    for target, before in zip(targets, target_buffers):
        _assert_equal(dict(target.named_buffers()), before)
    _assert_equal(agent.reward_normalizer.state_dict(), reward_before)


@pytest.mark.parametrize("auxiliary", [None, True, False])
def test_extra_slots_preserve_joint_world_gradients_optimizer_and_reward_state(
    wrappers, monkeypatch, auxiliary,
):
    settings = {} if auxiliary is None else {
        "aux_return_mode": "xqc", "aux_return_detach_representation": auxiliary,
    }
    reference = wrappers(**settings).agent
    actual = wrappers(xqc_utd=3, **settings).agent
    initial_world = deepcopy(actual.model.state_dict())
    reference.update(_Replay(reference))
    original_extra = actual._update_xqc_only
    extra_calls = []

    def extra(*batch):
        before = _world_state(actual)
        # The joint optimizer has already run; extras use its updated encoder.
        _assert_equal(before, _world_state(reference))
        assert any(not torch.equal(value, initial_world[key])
                   for key, value in actual.model.state_dict().items())
        result = original_extra(*batch)
        _assert_equal(_world_state(actual), before)
        extra_calls.append(1)
        return result

    monkeypatch.setattr(actual, "_update_xqc_only", extra)
    actual.update(_Replay(actual))
    assert len(extra_calls) == 2
    _assert_equal(_world_state(actual), _world_state(reference))


@pytest.mark.parametrize("normalization", ["divide_horizon", "reference_weighted_mean"])
def test_extra_loss_uses_detached_recurrence_and_temporal_weights(wrappers, normalization):
    agent = wrappers(
        xqc_utd=2, aux_return_mode="xqc", aux_return_detach_representation=False,
        rho=0.25, temporal_loss_normalization=normalization,
    ).agent
    obs, actions, rewards, terminated, _ = _Replay(agent).sample()
    with torch.no_grad():
        expected_latents = [agent.model.encode(obs[0])]
        for action in actions:
            expected_latents.append(agent.model.next(expected_latents[-1], action))
        expected_latents = torch.stack(expected_latents)
    losses = agent._xqc_only_losses(obs, actions, rewards, terminated)
    _assert_equal(losses["latent_states"], expected_latents)
    assert losses["latent_states"].requires_grad is False
    horizon, reference_horizon, rho = agent.cfg.train_unroll_horizon, agent.cfg.temporal_loss_reference_horizon, agent.cfg.rho
    weights = torch.tensor([rho ** depth for depth in range(horizon)])
    if normalization == "divide_horizon":
        weights /= horizon
    else:
        weights *= sum(rho ** depth for depth in range(reference_horizon)) / reference_horizon / weights.sum()
    for key, objective in (("critic_loss", "critic"), ("aux_return_loss", "aux_return_critic")):
        per_time = losses[objective].per_sample_loss.mean(dim=1)
        torch.testing.assert_close(losses[key], (per_time * weights).sum())
        losses[key].backward()
    assert all(parameter.grad is None for parameter in agent.model.parameters())
    assert any(parameter.grad is not None for parameter in agent.xqc_controller.critic.parameters())
    assert any(parameter.grad is not None for parameter in agent.aux_return.critic.parameters())


def test_multislot_private_update_requires_sampler_before_mutation(wrappers):
    agent = wrappers(xqc_utd=2).agent
    batch = _Replay(agent).sample()[:4]
    before = deepcopy(agent.checkpoint_state())
    with pytest.raises(ValueError, match="fresh replay batch sampler"):
        agent._update(*batch)
    _assert_equal(before, agent.checkpoint_state())


@pytest.mark.parametrize("mode", ["off", "xqc"])
def test_ratio_checkpoint_round_trip_continues_identically(wrappers, tmp_path, mode):
    source = wrappers(xqc_utd=3, aux_return_mode=mode)
    source_replay = _Replay(source.agent)
    for _ in range(2):
        source.agent.update(source_replay)
    checkpoint = source.save(tmp_path, "ratio")
    saved = deepcopy(source.agent.checkpoint_state())
    assert saved["checkpoint_version"] == 7
    assert saved["semantic_signature"]["xqc_utd"] == 3
    restored = wrappers(xqc_utd=3, aux_return_mode=mode).load(checkpoint)
    _assert_equal(saved, restored.agent.checkpoint_state())
    restored_replay = _Replay(restored.agent)
    restored_replay.generator.set_state(source_replay.generator.get_state())
    source_metrics = source.agent.update(source_replay)
    restored_metrics = restored.agent.update(restored_replay)
    _assert_equal(source_metrics, restored_metrics)
    _assert_equal(source.agent.checkpoint_state(), restored.agent.checkpoint_state())


@pytest.mark.parametrize("mode", ["off", "xqc"])
def test_version_five_implies_ratio_one_including_auxiliary(wrappers, mode):
    source = wrappers(aux_return_mode=mode)
    source.agent.update(_Replay(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    saved["checkpoint_version"] = 5
    saved["semantic_signature"].pop("xqc_utd")
    saved["semantic_signature"].pop("inner_actor_bn_mode")
    restored = wrappers(aux_return_mode=mode).load(saved)
    _assert_equal(source.agent.checkpoint_state(), restored.agent.checkpoint_state())
    with pytest.raises(ValueError, match="semantics"):
        wrappers(xqc_utd=2, aux_return_mode=mode).load(saved, frozen_evaluation=True)


@pytest.mark.parametrize("problem", [
    "missing_ratio", "bool_ratio", "fractional_ratio", "zero_ratio", "different_ratio",
    "world_count", "critic_count", "auxiliary_count", "world_optimizer_count",
])
@pytest.mark.parametrize("frozen", [False, True])
def test_ratio_preflight_rejects_inconsistent_state_before_mutation(wrappers, problem, frozen):
    source = wrappers(xqc_utd=2, aux_return_mode="xqc")
    source.agent.update(_Replay(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    signature = saved["semantic_signature"]
    if problem == "missing_ratio":
        signature.pop("xqc_utd")
    elif problem.endswith("_ratio"):
        signature["xqc_utd"] = {
            "bool_ratio": True, "fractional_ratio": 2.5,
            "zero_ratio": 0, "different_ratio": 3,
        }[problem]
    elif problem == "world_count":
        saved["num_updates"] += 1
        saved["outer_version"] += 1
    elif problem == "critic_count":
        saved["xqc_workspace"]["update_step"] += 1
    elif problem == "auxiliary_count":
        saved["aux_return"]["update_step"] += 1
    else:
        for state in saved["world_optimizer"]["state"].values():
            state["step"].add_(1)
    target = wrappers(xqc_utd=2, aux_return_mode="xqc")
    before = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError):
        target.load(saved, frozen_evaluation=frozen)
    _assert_equal(before, target.agent.checkpoint_state())


def test_frozen_evaluation_preserves_ratio_and_all_outer_state(wrappers):
    source = wrappers(xqc_utd=2, aux_return_mode="xqc", inner_operator="none")
    source.agent.update(_Replay(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    target = wrappers(xqc_utd=2, aux_return_mode="xqc", inner_critic_source="aux_return")
    target.load(saved, frozen_evaluation=True)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["saved_semantic_signature"]["xqc_utd"] == 2
    assert provenance["evaluated_semantic_signature"]["xqc_utd"] == 2
    target.reset_for_evaluation(101)
    observation, _ = target.env.reset(seed=101)
    before = target.agent.frozen_outer_state()
    target.predict(observation)
    _assert_equal(before, target.agent.frozen_outer_state())
    replay = _Replay(target.agent)
    with pytest.raises(RuntimeError, match="cannot update"):
        target.agent.update(replay)
    assert replay.batches == []


def test_pretraining_applies_ratio_without_reobserving_rewards(wrappers):
    model = wrappers(xqc_utd=2, pretrain_steps=3, inner_operator="none")
    model.learn(total_timesteps=10)
    assert model.agent.num_updates == model._num_updates == 8
    assert model.agent.xqc_workspace.update_step == 16
    assert model.agent.reward_normalizer.count == 10
    assert model.buffer.total_transitions == 10


def test_extra_slots_reuse_compiled_loss_regions(wrappers, monkeypatch):
    agent = wrappers(xqc_utd=3, aux_return_mode="xqc", compile=True).agent
    regions = [agent.xqc_controller._critic_loss_region,
               agent.xqc_controller._actor_loss_region, agent.aux_return._critic_loss_region]
    assert all(not region.enabled for region in regions)  # CPU policy.
    for region in regions:
        region.enabled = True
    compilations, executions = [], []

    def compile_region(function, **kwargs):
        compilations.append(function)

        def execute(*args, **kw):
            executions.append(function)
            return function(*args, **kw)

        return execute

    monkeypatch.setattr(torch, "compile", compile_region)
    agent.update(_Replay(agent))
    assert len(compilations) == 3
    assert all(executions.count(function) == 3 for function in compilations)
    assert agent.xqc_controller.compile_status["fallback"] is False
    assert agent.aux_return.compile_status["fallback"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_ratio_compiled_updates_match_eager(wrappers):
    eager = wrappers(device="cuda", xqc_utd=2, aux_return_mode="xqc").agent
    compiled = wrappers(
        device="cuda", xqc_utd=2, aux_return_mode="xqc", compile=True, compile_strict=True,
    ).agent
    _assert_equal(eager.state_dict(), compiled.state_dict())
    eager_replay, compiled_replay = _Replay(eager), _Replay(compiled)
    for _ in range(2):
        eager.update(eager_replay)
        compiled.update(compiled_replay)
    for key, value in eager.state_dict().items():
        # As in the existing four-slot compile tests, Adam amplifies reduction
        # noise in near-zero BN-affine bias gradients. Keep the tighter bound
        # for every other parameter and buffer, including running statistics.
        atol, rtol = (1e-3, 1e-3) if "batch_norm.bias" in key else (3e-5, 3e-4)
        torch.testing.assert_close(
            value, compiled.state_dict()[key], rtol=rtol, atol=atol,
            msg=lambda message, name=key: f"{name}: {message}",
        )
    _assert_equal(eager_replay.batches, compiled_replay.batches)
    _assert_equal(eager_replay.generator.get_state(), compiled_replay.generator.get_state())
    _assert_equal(eager._outer_generator.get_state(), compiled._outer_generator.get_state())
    _assert_equal(eager.aux_return._generator.get_state(), compiled.aux_return._generator.get_state())
    for agent in (eager, compiled):
        assert agent.num_updates == agent.outer_version == 2
        assert agent.xqc_workspace.update_step == agent.aux_return.update_step == 4
        assert agent.xqc_workspace.actor_optimizer_steps == 2
        assert agent.xqc_workspace.temperature_optimizer_steps == 2

    observations = torch.randn(
        16, eager.cfg.obs_shape["state"][0],
        generator=torch.Generator().manual_seed(97),
    ).to(eager.device)
    probes = []
    for agent in (eager, compiled):
        before = deepcopy(dict(agent.named_buffers()))
        with torch.no_grad():
            latents = agent.model.encode(observations)
            actions, _ = agent.xqc_controller.sample_action(latents, deterministic=True)
            probe = {"latents": latents, "actions": actions}
            for name, critic in (("soft", agent.xqc_controller.critic),
                                 ("return", agent.aux_return.critic)):
                log_probs = critic.log_probs(latents, actions, bn_mode="running")
                probe[f"{name}_probabilities"] = log_probs.exp()
                probe[f"{name}_values"] = critic.values_from_log_probs(log_probs)
        _assert_equal(dict(agent.named_buffers()), before)
        probes.append(probe)
    for name, value in probes[0].items():
        # Check held-out behavior as well as parameter drift, using the existing
        # multi-slot compile tolerance for policy/Q-derived outputs.
        atol, rtol = (3e-5, 3e-4) if name == "latents" else (2e-4, 1e-3)
        torch.testing.assert_close(
            value, probes[1][name], atol=atol, rtol=rtol,
            msg=lambda message, name=name: f"held-out {name}: {message}",
        )
    assert compiled.xqc_controller.compile_status["critic_compiled"]
    assert compiled.xqc_controller.compile_status["actor_compiled"]
    assert compiled.aux_return.compile_status["critic_compiled"]
