from copy import deepcopy
import inspect
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.inner_xqc import InnerXQCEngine
from RL.tdmpc2_core.xqc_controller import LatentXQCConfig, LatentXQCController
from RL.xqc_core import DiscountedReturnNormalizer


def _assert_tree_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    else:
        assert left == right


class _FakeActor(nn.Module):
    def __init__(self, bias=0.0, *, use_noise=False):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.log_std = nn.Parameter(torch.tensor(0.0))
        self.use_noise = bool(use_noise)
        self.bn_modes = []

    def distribution(self, z, bn_mode="running"):
        self.bn_modes.append(bn_mode)
        return (
            self.bias.expand(z.shape[0], 1),
            self.log_std.expand(z.shape[0], 1),
        )

    def sample(
        self,
        z,
        *,
        deterministic=False,
        bn_mode="running",
        noise=None,
        **_,
    ):
        pre_tanh, _ = self.distribution(z, bn_mode=bn_mode)
        # Keep ordinary tests deterministic while allowing the rollout oracle
        # to make the private collection noise observable.
        if not deterministic:
            assert noise is not None and noise.shape == pre_tanh.shape
            pre_tanh = pre_tanh + noise * float(self.use_noise)
        return torch.tanh(pre_tanh), torch.zeros(z.shape[0], device=z.device)


class _FakeLocalController:
    def __init__(self, actor):
        self.actor = actor
        self.log_temperature = nn.Parameter(torch.tensor(-2.0))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def sample_action(self, z, *, deterministic=False, noise=None):
        return self.actor.sample(
            z,
            deterministic=deterministic,
            bn_mode="running",
            noise=noise,
        )


class _FakeWorkspace:
    def __init__(self, outer, rules):
        self._records = outer.records
        self.controller = _FakeLocalController(deepcopy(outer.actor))
        self.controller.config = rules
        self.update_step = 0
        self.actor_optimizer_steps = 0
        self.temperature_optimizer_steps = 0

    def reset_from_(self, outer):
        outer.reset_calls += 1
        self.controller.actor.load_state_dict(outer.actor.state_dict())
        self.controller.log_temperature.data.copy_(outer.log_temperature.data)
        self.update_step = 0
        self.actor_optimizer_steps = 0
        self.temperature_optimizer_steps = 0

    def update(
        self,
        batch,
        *,
        next_noise,
        actor_noise,
        reward_scale,
    ):
        assert next_noise.shape == actor_noise.shape == batch.actions.shape
        self._records.append(
            {
                "latents": batch.latents.detach().clone(),
                "actions": batch.actions.detach().clone(),
                "rewards": batch.rewards.detach().clone(),
                "next_latents": batch.next_latents.detach().clone(),
                "bootstrap_mask": batch.bootstrap_mask.detach().clone(),
                "discount": batch.discount.detach().clone(),
                "reward_scale": (
                    reward_scale.detach().clone()
                    if torch.is_tensor(reward_scale)
                    else float(reward_scale)
                ),
            }
        )
        accepted = self.update_step % self.controller.config.policy_delay == 0
        if accepted:
            with torch.no_grad():
                self.controller.actor.bias.add_(0.1)
            self.actor_optimizer_steps += 1
            self.temperature_optimizer_steps += 1
        self.update_step += 1
        return {
            "critic_loss": float(self.update_step),
            "actor_update_accepted": float(accepted),
        }


class _FakeOuterController:
    def __init__(self, *, use_noise=False):
        self.actor = _FakeActor(use_noise=use_noise)
        self.log_temperature = nn.Parameter(torch.tensor(-2.0))
        self.records = []
        self.clone_calls = 0
        self.reset_calls = 0

    def clone_for_inner(self, **kwargs):
        self.clone_calls += 1
        assert kwargs == {
            "actor_lr": 5e-5,
            "critic_lr": 7e-5,
            "transition_steps": 4,
        }
        return _FakeWorkspace(
            self,
            SimpleNamespace(policy_delay=3, target_update_interval=1),
        )


class _FakeTOLD(nn.Module):
    def __init__(
        self,
        *,
        terminate=False,
        mixed_termination=False,
        action_dependent=False,
    ):
        super().__init__()
        self.terminate = terminate
        self.mixed_termination = bool(mixed_termination)
        self.action_dependent = bool(action_dependent)
        if self.action_dependent:
            self.action_reward_scale = nn.Parameter(torch.tensor(0.5))
            self.action_transition_scale = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def joint_input(z, action):
        return torch.cat((z, action), dim=-1)

    def reward_from_joint(self, joint):
        reward = torch.ones(joint.shape[0], 1, device=joint.device)
        if self.action_dependent:
            reward = reward + self.action_reward_scale * joint[:, -1:]
        return reward

    def next_from_joint(self, joint):
        next_z = joint[:, :2] + 0.25
        if self.action_dependent:
            next_z = next_z + self.action_transition_scale * joint[:, -1:]
        return next_z

    def termination(self, z):
        if self.mixed_termination:
            # For N=2 this yields one terminated and one surviving branch on
            # the first step; the surviving singleton terminates next.
            values = torch.ones(z.shape[0], 1, device=z.device)
            values[1::2] = 0.0
            return values
        value = 1.0 if self.terminate else 0.0
        return torch.full((z.shape[0], 1), value, device=z.device)


def _agent(
    *,
    episodic=False,
    terminate=False,
    compile=False,
    compile_strict=False,
    oracle_rollout=False,
    mixed_termination=False,
    reward_mode="frozen_real_scale",
    reward_normalizer=None,
):
    cfg = SimpleNamespace(
        seed=11,
        latent_dim=2,
        action_dim=1,
        episodic=episodic,
        compile=compile,
        compile_strict=compile_strict,
        num_bins=0,
        vmin=-10.0,
        vmax=10.0,
        inner_rounds=2,
        inner_rollouts_per_round=2,
        inner_rollout_horizon=2,
        inner_updates_per_round=2,
        inner_batch_size=2,
        inner_replay_capacity=8,
        inner_replay_sampling="with_replacement",
        inner_actor_lr=5e-5,
        inner_critic_lr=7e-5,
        xqc_adam_eps=1e-8,
        xqc_policy_delay=3,
        xqc_tau=0.005,
        xqc_target_update_interval=1,
        inner_termination_threshold=0.5,
        inner_model_step_budget=8,
        inner_reward_normalization=reward_mode,
    )
    controller = _FakeOuterController(use_noise=oracle_rollout)
    agent = SimpleNamespace(
        cfg=cfg,
        device=torch.device("cpu"),
        model=_FakeTOLD(
            terminate=terminate,
            mixed_termination=mixed_termination,
            action_dependent=oracle_rollout,
        ),
        xqc_controller=controller,
        discount=0.9,
        reward_normalizer=(
            SimpleNamespace(scale=2.5, return_accumulator=4.0)
            if reward_normalizer is None
            else reward_normalizer
        ),
    )
    return agent, controller


@torch.no_grad()
def _legacy_stepwise_non_episodic_round(engine, root_z):
    """Independent oracle for the pre-dense non-episodic rollout."""

    cfg = engine.cfg
    count = int(cfg.inner_rollouts_per_round)
    horizon = int(cfg.inner_rollout_horizon)
    z = root_z.expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=engine.device)
    lengths = torch.zeros(count, dtype=torch.long, device=engine.device)
    reward_sums = torch.zeros(count, dtype=root_z.dtype, device=engine.device)
    discounted_rewards = torch.zeros_like(reward_sums)
    discount_weight = torch.ones_like(reward_sums)
    terminated_rollout = torch.zeros_like(alive)
    transition_fields = ([], [], [], [], [])

    for _ in range(horizon):
        active = torch.nonzero(alive, as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            break
        active_z = z.index_select(0, active)
        action = engine._sample_actor(active_z, stream="collection")
        joint = engine.model.joint_input(active_z, action)
        reward = td_math.two_hot_inv(
            engine.model.reward_from_joint(joint), cfg
        )
        next_z = engine.model.next_from_joint(joint)
        terminated = reward.new_zeros((active.numel(), 1))

        for values, value in zip(
            transition_fields,
            (active_z, action, reward, next_z, terminated),
        ):
            values.append(value)

        reward_vector = reward.squeeze(-1)
        lengths[active] += 1
        reward_sums[active] += reward_vector
        discounted_rewards[active] += discount_weight[active] * reward_vector
        discount_weight[active] *= float(engine.agent.discount)
        z[active] = next_z

    engine.state.replay.add_batch(
        *(torch.cat(values, dim=0) for values in transition_fields)
    )
    return {
        "lengths": lengths,
        "reward_sums": reward_sums,
        "discounted_rewards": discounted_rewards,
        "terminated": terminated_rollout,
        "reward_normalizer_returns": root_z.new_empty((0,)),
    }


def test_dense_non_episodic_round_matches_legacy_stepwise_oracle():
    dense_agent, dense_outer = _agent(oracle_rollout=True)
    legacy_agent, legacy_outer = _agent(oracle_rollout=True)
    dense = InnerXQCEngine(dense_agent)
    legacy = InnerXQCEngine(legacy_agent)
    root = torch.zeros(1, 2)

    with dense.rng.fork("initialization"):
        dense._prepare_action()
    with legacy.rng.fork("initialization"):
        legacy._prepare_action()
    dense_outer_before = deepcopy(dense_outer.actor.state_dict())
    legacy_outer_before = deepcopy(legacy_outer.actor.state_dict())
    dense_local_before = deepcopy(
        dense.state.workspace.controller.actor.state_dict()
    )
    legacy_local_before = deepcopy(
        legacy.state.workspace.controller.actor.state_dict()
    )
    dense_model_before = deepcopy(dense.model.state_dict())
    legacy_model_before = deepcopy(legacy.model.state_dict())
    global_rng = torch.random.get_rng_state().clone()

    dense_result = dense._collect_dense_round(
        root,
        count=int(dense.cfg.inner_rollouts_per_round),
        horizon=int(dense.cfg.inner_rollout_horizon),
    )
    legacy_result = _legacy_stepwise_non_episodic_round(legacy, root)

    assert dense_result.keys() == legacy_result.keys()
    for key in dense_result:
        torch.testing.assert_close(
            dense_result[key], legacy_result[key], atol=0, rtol=0
        )
    assert dense_result["lengths"].tolist() == [2, 2]
    assert not bool(dense_result["terminated"].any())
    assert dense.state.policy_evaluations == legacy.state.policy_evaluations == 4
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    _assert_tree_equal(dense_outer.actor.state_dict(), dense_outer_before)
    _assert_tree_equal(legacy_outer.actor.state_dict(), legacy_outer_before)
    _assert_tree_equal(
        dense.state.workspace.controller.actor.state_dict(), dense_local_before
    )
    _assert_tree_equal(
        legacy.state.workspace.controller.actor.state_dict(), legacy_local_before
    )
    _assert_tree_equal(dense.model.state_dict(), dense_model_before)
    _assert_tree_equal(legacy.model.state_dict(), legacy_model_before)
    assert all(
        parameter.grad is None
        for module in (
            dense_outer.actor,
            legacy_outer.actor,
            dense.state.workspace.controller.actor,
            legacy.state.workspace.controller.actor,
            dense.model,
            legacy.model,
        )
        for parameter in module.parameters()
    )

    dense_replay = dense.state.replay.state_dict()
    legacy_replay = legacy.state.replay.state_dict()
    _assert_tree_equal(dense_replay, legacy_replay)
    assert dense_replay["pos"] == 4
    assert dense_replay["full"] is False
    assert dense_replay["next_sample_id"] == 4
    torch.testing.assert_close(
        dense_replay["sample_id"], torch.arange(4), atol=0, rtol=0
    )
    actions = dense_replay["action"]
    assert actions.unique().numel() > 1
    assert bool(actions.abs().max() > 0)
    torch.testing.assert_close(
        dense_replay["reward"], 1.0 + 0.5 * actions, atol=0, rtol=0
    )
    # The first N rows are the common roots. Horizon-major ordering makes their
    # action-dependent successors both the next_z rows at h0 and the z rows at
    # h1; differing actions make this structure non-degenerate.
    torch.testing.assert_close(
        dense_replay["z"][:2],
        torch.zeros(2, 2),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        dense_replay["z"][2:], dense_replay["next_z"][:2], atol=0, rtol=0
    )
    assert not torch.equal(
        dense_replay["next_z"][0], dense_replay["next_z"][1]
    )
    _assert_tree_equal(
        dense.rng.training_state_dict(), legacy.rng.training_state_dict()
    )
    dense._release_action()
    legacy._release_action()


def test_rollout_compile_status_distinguishes_cpu_inactive_and_episodic():
    cpu_agent, _ = _agent(compile=True, compile_strict=True)
    cpu = InnerXQCEngine(cpu_agent)
    assert cpu.rollout_compile_status == {
        "requested": True,
        "applicable": True,
        "enabled": False,
        "strict": True,
        "compiled": False,
        "fallback": False,
    }
    cpu.act(torch.zeros(1, 2), collect_diagnostics=False)
    assert cpu.rollout_compile_status["compiled"] is False
    assert cpu.rollout_compile_status["fallback"] is False

    episodic_agent, _ = _agent(
        episodic=True,
        terminate=True,
        compile=True,
        compile_strict=True,
    )
    episodic = InnerXQCEngine(episodic_agent)
    assert episodic.rollout_compile_status == {
        "requested": True,
        "applicable": False,
        "enabled": False,
        "strict": True,
        "compiled": False,
        "fallback": False,
    }
    _, metrics, lengths = episodic.act(torch.zeros(1, 2))
    assert lengths.tolist() == [1, 1, 1, 1]
    assert metrics["inner_model_steps"] == 4
    assert metrics["inner_termination_rate"] == pytest.approx(1.0)
    assert metrics["inner_compile_rollout_fallback"] == 0.0


def test_action_local_xqc_uses_raw_imagination_and_frozen_real_reward_scale():
    agent, outer = _agent()
    engine = InnerXQCEngine(agent)
    root = torch.zeros(1, 2)
    outer_state = deepcopy(outer.actor.state_dict())
    global_rng = torch.random.get_rng_state().clone()

    action, metrics, lengths = engine.act(root, eval_mode=False)

    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert torch.allclose(action, torch.tanh(torch.tensor([0.2])))
    assert lengths == [2, 2, 2, 2]
    assert metrics["inner_model_steps"] == 8
    assert metrics["inner_update_slots"] == 4
    assert metrics["inner_actor_optimizer_steps"] == 2
    assert metrics["inner_temperature_optimizer_steps"] == 2
    assert metrics["inner_final_outer_policy_kl"] == pytest.approx(0.02)
    assert metrics["inner_reward_scale"] == pytest.approx(2.5)
    assert metrics["inner_reward_scale_initial"] == pytest.approx(2.5)
    assert metrics["inner_reward_scale_final"] == pytest.approx(2.5)
    assert metrics["inner_reward_scale_delta"] == pytest.approx(0.0)
    assert metrics["inner_reward_normalizer_count_initial"] == 0.0
    assert metrics["inner_reward_normalizer_count_final"] == 0.0
    assert metrics["inner_reward_normalizer_imagined_updates"] == 0.0
    assert metrics["inner_buffer_size"] == 8
    assert all(record["reward_scale"] == 2.5 for record in outer.records)
    assert all(torch.equal(record["rewards"], torch.ones(2, 1)) for record in outer.records)
    assert all(
        torch.equal(record["bootstrap_mask"], torch.ones(2, 1))
        for record in outer.records
    )
    assert all(float(record["discount"]) == pytest.approx(0.9) for record in outer.records)
    assert outer.actor.state_dict().keys() == outer_state.keys()
    assert all(
        torch.equal(outer.actor.state_dict()[key], value)
        for key, value in outer_state.items()
    )
    assert engine.state.workspace is None
    assert engine.state.replay is None
    assert set(engine._workspace_pool.controller.actor.bn_modes) == {"running"}
    assert torch.equal(engine._replay_pool.reward[:8], torch.ones(8, 1))


def _seeded_outer_reward_normalizer():
    normalizer = DiscountedReturnNormalizer(0.9)
    normalizer.update(2.0, False)
    normalizer.update(3.0, False)
    return normalizer


def _population_scale(values, epsilon):
    values = torch.as_tensor(values, dtype=torch.float64)
    return float(values.var(unbiased=False).sqrt()) + float(epsilon)


def test_action_local_imagined_reward_normalization_is_branchwise_round_local_and_precedes_updates():
    normalizer = _seeded_outer_reward_normalizer()
    outer_normalizer_before = deepcopy(normalizer.state_dict())
    agent, outer = _agent(
        oracle_rollout=True,
        reward_mode="action_local_imagined",
        reward_normalizer=normalizer,
    )
    engine = InnerXQCEngine(agent)

    _, metrics, lengths = engine.act(torch.zeros(1, 2), eval_mode=False)

    assert lengths == [2, 2, 2, 2]
    _assert_tree_equal(normalizer.state_dict(), outer_normalizer_before)
    raw_rewards = engine._replay_pool.reward[:8, 0].detach().double()
    assert raw_rewards.unique().numel() > 1

    all_returns = []
    expected_round_scales = []
    for rewards in raw_rewards.reshape(2, 2, 2):
        branch_returns = torch.full(
            (2,), normalizer.return_accumulator, dtype=torch.float64
        )
        for step_rewards in rewards:
            branch_returns = 0.9 * branch_returns + step_rewards
            all_returns.extend(branch_returns.tolist())
        expected_round_scales.append(
            _population_scale(all_returns, normalizer.epsilon)
        )

    # A single flattened recurrence would leak one branch into the next and
    # must not accidentally agree with the branchwise oracle.
    flat_accumulator = float(normalizer.return_accumulator)
    flattened_returns = []
    for reward in raw_rewards:
        flat_accumulator = 0.9 * flat_accumulator + float(reward)
        flattened_returns.append(flat_accumulator)
    assert expected_round_scales[-1] != pytest.approx(
        _population_scale(flattened_returns, normalizer.epsilon)
    )

    assert [record["reward_scale"] for record in outer.records] == pytest.approx(
        [
            expected_round_scales[0],
            expected_round_scales[0],
            expected_round_scales[1],
            expected_round_scales[1],
        ]
    )
    assert all(torch.is_tensor(record["reward_scale"]) for record in outer.records)
    assert all(
        bool(torch.isin(record["rewards"].reshape(-1).double(), raw_rewards).all())
        for record in outer.records
    )
    expected_final = expected_round_scales[-1]
    assert metrics["inner_reward_scale_initial"] == pytest.approx(normalizer.scale)
    assert metrics["inner_reward_scale"] == pytest.approx(expected_final)
    assert metrics["inner_reward_scale_final"] == pytest.approx(expected_final)
    assert metrics["inner_reward_scale_delta"] == pytest.approx(
        expected_final - normalizer.scale
    )
    assert metrics["inner_reward_normalizer_count_initial"] == 0.0
    assert metrics["inner_reward_normalizer_count_final"] == 8.0
    assert metrics["inner_reward_normalizer_imagined_updates"] == 8.0
    assert metrics["inner_reward_normalizer_imagined_updates"] == metrics[
        "inner_realized_model_steps"
    ]
    for key in (
        "inner_reward_scale",
        "inner_reward_scale_initial",
        "inner_reward_scale_final",
        "inner_reward_scale_delta",
        "inner_reward_normalizer_count_initial",
        "inner_reward_normalizer_count_final",
        "inner_reward_normalizer_imagined_updates",
    ):
        assert torch.is_tensor(metrics[key]), key
        assert metrics[key].shape == (), key


def test_action_local_reward_hot_path_has_no_host_scalar_extraction():
    source = "\n".join(
        (
            inspect.getsource(InnerXQCEngine._update_reward_normalizer),
            inspect.getsource(InnerXQCEngine.act),
        )
    )

    assert ".item(" not in source
    assert "float(normalizer.scale" not in source
    assert "float(local_reward_normalizer.count" not in source
    assert "float(self.state.reward_scale" not in source


def test_action_local_imagined_reward_normalization_resets_terminated_branches_before_reward():
    normalizer = _seeded_outer_reward_normalizer()
    outer_normalizer_before = deepcopy(normalizer.state_dict())
    agent, outer = _agent(
        episodic=True,
        mixed_termination=True,
        reward_mode="action_local_imagined",
        reward_normalizer=normalizer,
    )
    engine = InnerXQCEngine(agent)

    _, metrics, lengths = engine.act(torch.zeros(1, 2), eval_mode=False)

    assert lengths.tolist() == [1, 2, 1, 2]
    _assert_tree_equal(normalizer.state_dict(), outer_normalizer_before)
    assert engine._replay_pool.size == 6
    torch.testing.assert_close(
        engine._replay_pool.reward[:6], torch.ones(6, 1), atol=0, rtol=0
    )
    torch.testing.assert_close(
        engine._replay_pool.terminated[:6, 0],
        torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        atol=0,
        rtol=0,
    )
    per_round_returns = [
        1.0,
        0.9 * normalizer.return_accumulator + 1.0,
        1.0,
    ]
    expected_scale = _population_scale(
        per_round_returns * 2, normalizer.epsilon
    )
    assert [record["reward_scale"] for record in outer.records] == pytest.approx(
        [expected_scale] * 4
    )
    assert all(
        torch.equal(record["rewards"], torch.ones(2, 1))
        for record in outer.records
    )
    assert metrics["inner_reward_scale_final"] == pytest.approx(expected_scale)
    assert metrics["inner_reward_normalizer_count_final"] == 6.0
    assert metrics["inner_reward_normalizer_imagined_updates"] == 6.0
    assert metrics["inner_reward_normalizer_imagined_updates"] == metrics[
        "inner_realized_model_steps"
    ]


def test_final_policy_kl_uses_closed_form_direction_and_running_bn_without_rng():
    agent, outer = _agent()
    engine = InnerXQCEngine(agent)
    root = torch.zeros(1, 2)
    with engine.rng.fork("initialization"):
        engine._prepare_action()

    inner_actor = engine.state.workspace.controller.actor
    with torch.no_grad():
        outer.actor.bias.fill_(0.0)
        outer.actor.log_std.fill_(0.0)
        inner_actor.bias.fill_(1.0)
        inner_actor.log_std.fill_(math.log(2.0))
    outer_before = deepcopy(outer.actor.state_dict())
    inner_before = deepcopy(inner_actor.state_dict())
    global_rng = torch.random.get_rng_state().clone()

    metrics = engine._final_policy_diagnostics(root)

    assert metrics["inner_final_outer_policy_kl"] == pytest.approx(
        2.0 - math.log(2.0)
    )
    assert outer.actor.bn_modes[-1] == "running"
    assert inner_actor.bn_modes[-1] == "running"
    assert engine.state.policy_evaluations == 2
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    _assert_tree_equal(outer.actor.state_dict(), outer_before)
    _assert_tree_equal(inner_actor.state_dict(), inner_before)
    engine._release_action()


def test_unsampled_action_omits_final_policy_kl_and_diagnostic_work():
    agent, _ = _agent()
    engine = InnerXQCEngine(agent)

    _, metrics, _ = engine.act(torch.zeros(1, 2), collect_diagnostics=False)
    metrics = engine.finalize_timing_metrics(metrics)

    assert "inner_final_outer_policy_kl" not in metrics
    assert "inner_diagnostic_seconds" not in metrics
    assert metrics["inner_policy_evaluations"] == 25.0


def test_second_action_reuses_allocations_but_is_logically_fresh():
    normalizer = _seeded_outer_reward_normalizer()
    outer_normalizer_before = deepcopy(normalizer.state_dict())
    agent, outer = _agent(
        reward_mode="action_local_imagined",
        reward_normalizer=normalizer,
    )
    engine = InnerXQCEngine(agent)
    root = torch.zeros(1, 2)

    first_action, first_metrics, _ = engine.act(root)
    workspace_id = id(engine._workspace_pool)
    second_action, second_metrics, _ = engine.act(root)

    assert id(engine._workspace_pool) == workspace_id
    assert outer.clone_calls == 1
    assert outer.reset_calls == 1
    assert torch.equal(first_action, second_action)
    assert first_metrics["inner_actor_optimizer_steps"] == 2
    assert second_metrics["inner_actor_optimizer_steps"] == 2
    assert first_metrics["inner_reward_normalizer_count_initial"] == 0.0
    assert second_metrics["inner_reward_normalizer_count_initial"] == 0.0
    assert first_metrics["inner_reward_normalizer_count_final"] == 8.0
    assert second_metrics["inner_reward_normalizer_count_final"] == 8.0
    assert first_metrics["inner_reward_scale_final"] == pytest.approx(
        second_metrics["inner_reward_scale_final"]
    )
    _assert_tree_equal(normalizer.state_dict(), outer_normalizer_before)


def test_true_imagined_termination_masks_bootstrap_but_horizon_does_not():
    agent, outer = _agent(episodic=True, terminate=True)
    engine = InnerXQCEngine(agent)

    _, metrics, lengths = engine.act(torch.zeros(1, 2))

    assert lengths.tolist() == [1, 1, 1, 1]
    assert metrics["inner_model_steps"] == 4
    assert metrics["inner_termination_rate"] == pytest.approx(1.0)
    assert all(
        torch.equal(record["bootstrap_mask"], torch.zeros(2, 1))
        for record in outer.records
    )


def test_inner_boundary_state_round_trip_restores_only_persistent_rng_and_indices():
    agent, _ = _agent()
    source = InnerXQCEngine(agent)
    source.act(torch.zeros(1, 2))
    source.reset_episode()
    state = deepcopy(source.training_state_dict())

    restored = InnerXQCEngine(agent)
    restored.load_training_state_dict(state)

    assert restored.action_index == 1
    assert restored.episode_index == 1
    assert restored.state.workspace is None
    assert restored._workspace_pool is None
    _assert_tree_equal(restored.training_state_dict(), state)


def test_bad_inner_boundary_state_fails_before_mutation():
    agent, _ = _agent()
    engine = InnerXQCEngine(agent)
    engine.act(torch.zeros(1, 2))
    before = deepcopy(engine.training_state_dict())
    invalid = deepcopy(before)
    invalid["action_index"] = -1

    with pytest.raises(ValueError, match="action_index"):
        engine.load_training_state_dict(invalid)

    _assert_tree_equal(engine.training_state_dict(), before)


def test_real_inner_xqc_update_adapts_online_bn_and_keeps_official_target_bn_frozen():
    agent, _ = _agent()
    torch.manual_seed(19)
    controller = LatentXQCController(
        latent_dim=2,
        action_dim=1,
        config=LatentXQCConfig(
            actor_net_arch=(8,),
            critic_net_arch=(8,),
            num_atoms=11,
            target_entropy=-0.5,
            optimizer_backend="single_tensor",
        ),
    )
    agent.xqc_controller = controller
    outer_before = deepcopy(controller.state_dict())
    target_buffer_start = {
        name: value.detach().clone()
        for name, value in controller.critic.named_buffers()
    }
    actor_buffer_start = {
        name: value.detach().clone()
        for name, value in controller.actor.named_buffers()
    }
    engine = InnerXQCEngine(agent)
    global_rng = torch.random.get_rng_state().clone()

    action, metrics, _ = engine.act(torch.zeros(1, 2))

    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert torch.isfinite(action).all()
    assert all(
        torch.isfinite(torch.as_tensor(value)).all()
        for value in metrics.values()
    )
    _assert_tree_equal(controller.state_dict(), outer_before)
    inner = engine._workspace_pool.controller
    assert engine._workspace_pool.update_step == 4
    assert engine._workspace_pool.actor_optimizer_steps == 2
    assert engine._workspace_pool.temperature_optimizer_steps == 2
    for name, value in inner.critic_target.named_buffers():
        assert torch.equal(value, target_buffer_start[name])
    assert any(
        not torch.equal(value, target_buffer_start[name])
        for name, value in inner.critic.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    )
    assert any(
        not torch.equal(value, actor_buffer_start[name])
        for name, value in inner.actor.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    )


def test_failed_update_releases_all_logically_live_action_state():
    agent, _ = _agent()
    engine = InnerXQCEngine(agent)
    engine.act(torch.zeros(1, 2))

    def fail_update(*args, **kwargs):
        raise RuntimeError("synthetic update failure")

    engine._workspace_pool.update = fail_update
    with pytest.raises(RuntimeError, match="synthetic update failure"):
        engine.act(torch.zeros(1, 2))

    assert engine.state.workspace is None
    assert engine.state.replay is None
    assert engine._workspace_pool is not None
    assert engine._replay_pool is not None
