"""Matched observations/actions and reward-only routing in branch calibration."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

import evaluate_warm_actor_calibration as calibration
from RL.tdmpc2_core.inner_trace import evaluate_outer_tail
from tests.test_aux_return_inner import _prepared
from tests.test_ambi_real_calibration import AnalyticEnv
from utils.ambi_real_calibration import capture_simulator_snapshot


def test_noise_pairing_is_root_local_and_does_not_consume_global_rng():
    before_np = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    prefix, tail, seed = calibration.paired_noise("h3-j8-s101-d25", 3, 4, 32, 1)
    repeated = calibration.paired_noise("h3-j8-s101-d25", 3, 4, 32, 1)
    assert seed == repeated[2]
    np.testing.assert_array_equal(prefix, repeated[0])
    np.testing.assert_array_equal(tail, repeated[1])
    assert not np.array_equal(prefix, calibration.paired_noise("h3-j8-s101-d75", 3, 4, 32, 1)[0])
    assert prefix.dtype == tail.dtype == np.float32
    assert prefix.shape == (3, 32, 1) and tail.shape == (4, 32, 1)
    after_np = np.random.get_state()
    np.testing.assert_array_equal(before_np[1], after_np[1])
    assert before_np[0] == after_np[0] and before_np[2:] == after_np[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), before_torch, rtol=0, atol=0)


def test_boundary_diagnostics_selects_family_round_and_post_update_stage():
    def event(round_index, stage, inner, frozen):
        return dict(phase="transfer_probe", round_index=round_index, stage=stage,
                    metrics={"transfer_root_q_inner_advantage_mean_all": inner,
                             "transfer_root_q_frozen_advantage_mean_all": frozen,
                             "transfer_root_q_inner_actor_head_0": 1.,
                             "transfer_root_q_inner_actor_head_1": 3.})
    root = {"diagnostic_events": {
        "warm": [event(0, "initial", 1, 2), event(1, "before_first_actor_block", 40, 3),
                 event(1, "post_round", 5, 2), event(2, "post_round", 100, 1)],
        "cold": [event(1, "post_round", 9, 2)],
    }}
    result = calibration.boundary_diagnostics(root, "warm", 1)
    assert result["critic_preference_gap"] == 3
    assert result["critic_head_sd"] == 1
    assert "root_diagnostics" not in result
    assert calibration.boundary_diagnostics(root, "warm", 0)["critic_preference_gap"] == -1
    assert calibration.boundary_diagnostics(root, "cold", 1)["critic_preference_gap"] == 7
    assert calibration.boundary_diagnostics(root, "prior", 0) == {}


def test_frozen_callbacks_route_to_sac_actor_and_auxiliary_return_critic():
    with _prepared(inner_horizon_actor_source="sac", inner_horizon_critic_source="aux_return") as (holder, engine):
        observations = np.zeros((3, 3), dtype=np.float32)
        noise = np.array([[1.], [2.], [3.]], dtype=np.float32)
        sampled = calibration.FrozenCallbacks(holder, engine._horizon_actor,
            engine._horizon_actor_options, None)
        mean = calibration.FrozenCallbacks(holder, engine._horizon_actor,
            engine._horizon_actor_options, None, "mean")
        np.testing.assert_allclose(sampled.actor(observations, noise), np.tanh(-.3 + np.exp(-2) * noise), atol=1e-7)
        np.testing.assert_allclose(mean.actor(observations, noise), np.full((3, 1), np.tanh(-.3)), atol=1e-7)
        np.testing.assert_array_equal(sampled.q(observations, noise), [9., 9., 9.])
        np.testing.assert_array_equal(sampled.q_heads(observations, noise), np.full((3, 2), 9.))
        with pytest.raises(ValueError, match="Action mode"):
            calibration.FrozenCallbacks(holder, engine._horizon_actor, {}, None, "invalid")


@pytest.mark.parametrize("mode", ["sample", "mean"])
@pytest.mark.parametrize("reduction", ["mean_pair", "min_pair", "min_all"])
def test_model_prefix_matches_established_probe_and_keeps_tail_sampled(mode, reduction, monkeypatch):
    with _prepared(inner_horizon_actor_source="sac", inner_horizon_critic_source="aux_return",
                   mppi_terminal_q_reduction=reduction) as (holder, engine):
        with torch.no_grad():
            engine._horizon_critic[0][-1].bias.fill_(3.)
            engine._horizon_critic[1][-1].bias.fill_(11.)
        prefix, tail, _ = calibration.paired_noise("root", 2, 3, 4, 1)
        bounds = engine._inner_policy_kwargs()
        policy = engine.state.actor
        mode_before = [(module, module.training) for module in engine.model.modules()]
        rng = torch.random.get_rng_state().clone()
        calls = []
        original_pi = engine.model.pi

        def recording_pi(z, *args, **kwargs):
            calls.append((kwargs["policy"], kwargs["noise"].clone()))
            return original_pi(z, *args, **kwargs)

        monkeypatch.setattr(engine.model, "pi", recording_pi)
        actual = calibration.model_branches(holder, np.zeros(3), policy, bounds,
                                             prefix, tail, None, mode)
        assert len(calls) == 3
        for step in range(2):
            assert calls[step][0] is policy
            expected_noise = np.zeros_like(prefix[step]) if mode == "mean" else prefix[step]
            np.testing.assert_array_equal(calls[step][1].numpy(), expected_noise)
        assert calls[-1][0] is engine._horizon_actor
        np.testing.assert_array_equal(calls[-1][1].numpy(), tail[0])
        monkeypatch.setattr(engine.model, "pi", original_pi)
        model_noise = np.concatenate((np.zeros_like(prefix) if mode == "mean" else prefix, tail[:1]))
        z = engine.model.encode(torch.zeros(1, 3))
        expected = evaluate_outer_tail(engine, z, policy, torch.as_tensor(model_noise),
                                       policy_bounds=bounds)
        for index, row in enumerate(actual):
            assert row["model_prefix_reward"] == pytest.approx(expected["reward"][index].item())
            assert row["model_bootstrap"] == pytest.approx(expected["bootstrap"][index].item())
            assert row["predicted_model_return"] == pytest.approx(expected["total"][index].item())
            np.testing.assert_allclose(row["model_endpoint_action"], np.tanh(-.3 + np.exp(-2) * tail[0, index]), atol=1e-7)
            assert row["model_endpoint_q_heads"] == [3., 11.]
            assert row["model_endpoint_q_expected_mean_pair"] == 7
            assert row["model_endpoint_q_expected_min_pair"] == 3
            assert row["model_endpoint_q_min_all"] == 3
            assert row["predicted_model_return_expected_mean_pair"] == pytest.approx(row["model_prefix_reward"] + .99**2 * 7)
        assert all(module.training == flag for module, flag in mode_before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_five_head_distributional_model_uses_given_pair_and_all_head_expectations():
    with _prepared(inner_horizon_actor_source="sac", inner_horizon_critic_source="aux_return",
                   q_representation="distributional", num_q=5,
                   mppi_terminal_q_reduction="mean_pair") as (holder, engine):
        with torch.no_grad():
            for index, head in enumerate(engine._horizon_critic):
                head[-1].bias.zero_()
                head[-1].bias[index] = 5
        prefix, tail, _ = calibration.paired_noise("five-head", 2, 3, 4, 1)
        bounds = engine._inner_policy_kwargs()
        with pytest.raises(ValueError, match="explicit pair_indices"):
            calibration.model_branches(holder, np.zeros(3), engine.state.actor, bounds, prefix, tail, None, "sample")
        rng = torch.random.get_rng_state().clone()
        pair = torch.tensor([1, 4])
        rows = calibration.model_branches(holder, np.zeros(3), engine.state.actor, bounds, prefix, tail, pair, "sample")
        for row in rows:
            heads = np.array(row["model_endpoint_q_heads"])
            assert row["model_bootstrap"] == pytest.approx(.99**2 * heads[[1, 4]].mean())
            assert row["model_endpoint_q_expected_mean_pair"] == pytest.approx(heads.mean())
            assert row["model_endpoint_q_min_all"] == pytest.approx(heads.min())
            mins = [min(heads[a], heads[b]) for a in range(5) for b in range(a + 1, 5)]
            assert row["model_endpoint_q_expected_min_pair"] == pytest.approx(np.mean(mins))
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_join_rows_decomposition_and_incomplete_measurement_rejection():
    predicted = {"predicted_model_return": 12., "predicted_model_return_min_all": 8.}
    real = {"real_bootstrapped_return": 10., "real_mc_return": 7.,
            "real_bootstrapped_return_min_all": 6., "mc_complete": True,
            "episode_cutoff_complete": True}
    row = calibration.join_rows([predicted], [real], metadata={"root_id": "r1"})[0]
    assert (row["model_prefix_error"], row["terminal_value_error"], row["total_prediction_error"]) == (2, 3, 5)
    assert (row["model_prefix_error_min_all"], row["terminal_value_error_min_all"], row["total_prediction_error_min_all"]) == (2, -1, 1)
    assert row["root_id"] == "r1" and row["replicate"] == 0
    with pytest.raises(ValueError, match="coverage"):
        calibration.join_rows([], [real], metadata={})
    with pytest.raises(ValueError, match="complete"):
        calibration.join_rows([predicted], [dict(real, mc_complete=False)], metadata={})


class ThreeObservationEnv(AnalyticEnv):
    """Analytic branch task with the tiny checkpoint's three-input encoder."""

    def _observation(self):
        return np.array([float(self.position), 0., 0.], dtype=np.float32)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self._observation(), {}

    def step(self, action):
        _, reward, terminal, truncated, info = super().step(action)
        return self._observation(), reward, terminal, truncated, info

    def calibration_state(self):
        return dict(super().calibration_state(), observation=self._observation())

    def load_calibration_state(self, state):
        super().load_calibration_state(state)
        return self._observation()


def test_serialized_actor_snapshot_to_matched_model_and_real_records(tmp_path, monkeypatch):
    with _prepared(aux_return_mode="sac", inner_actor_source="sac",
                   inner_horizon_actor_source="sac", inner_horizon_critic_source="aux_return") as (holder, engine):
        holder.agent.model.eval()
        source = ThreeObservationEnv()
        observation, _ = source.reset(seed=101)
        checkpoint, matrix = tmp_path / "checkpoint.pt", tmp_path / "matrix.json"
        checkpoint.write_bytes(b"identity-only-checkpoint")
        matrix.write_text("{}")
        policy_path = tmp_path / "warm-r0.pt"
        torch.save(deepcopy(engine.state.actor).cpu(), policy_path)
        root = dict(root_id="tiny-r1", checkpoint=str(checkpoint), matrix=str(matrix), selector="test",
                    checkpoint_sha256=calibration._file_sha256(checkpoint), matrix_sha256=calibration._file_sha256(matrix),
                    H=2, J=8, seed=101, decision_index=499, episode_max_steps=500,
                    observation=observation.tolist(), snapshot=capture_simulator_snapshot(source).to_dict(),
                    actors=[dict(family="warm", round=0, path=policy_path.name,
                                 sha256=calibration._file_sha256(policy_path), bounds=engine._inner_policy_kwargs())])
        root_path, output = tmp_path / "root.json", tmp_path / "result.json"
        root_path.write_text(json.dumps(root))
        monkeypatch.setattr(calibration, "load_preset_matrix", lambda path: {})
        monkeypatch.setattr(calibration, "load_checkpoint_context", lambda path: {})
        monkeypatch.setattr(calibration, "resolve_preset", lambda *args, **kwargs: {"algorithm_config": {"alg_params": {}}})
        monkeypatch.setattr(calibration, "_make_env", lambda resolved: ThreeObservationEnv())

        def initialize(resolved, env, checkpoint, controller_seed, device):
            assert resolved["algorithm_config"]["alg_params"] == calibration.INFERENCE_OVERRIDES
            return holder, {}

        monkeypatch.setattr(calibration, "_initialize_frozen_model", initialize)
        result = calibration.evaluate_root(root_path, output, device="cpu", rollouts=3, tail_steps=2)
        restored = json.loads(output.read_text())
        assert restored == result and result["frozen_state_unchanged"]
        assert len(result["records"]) == 6
        assert {row["action_mode"] for row in result["records"]} == {"sample", "mean"}
        assert result["identity"]["inference_overrides"] == calibration.INFERENCE_OVERRIDES
        assert result["q_pair_indices"] is None
        assert result["horizon_actor_source"] == "sac" and result["horizon_critic_source"] == "aux_return"
        for row in result["records"]:
            assert row["model_prefix_error"] + row["terminal_value_error"] == pytest.approx(row["total_prediction_error"])
            assert row["real_endpoint_q_heads"] == [9., 9.]
            assert row["real_endpoint_q_expected_mean_pair"] == 9.
            assert row["simulator_decisions"] == 4
        # An exact completed artifact is reusable without reconstructing a model.
        monkeypatch.setattr(calibration, "_initialize_frozen_model", lambda *args, **kwargs: pytest.fail("cache miss"))
        assert calibration.evaluate_root(root_path, output, device="cpu", rollouts=3, tail_steps=2) == result
        matrix.write_text('{"changed":true}')
        with pytest.raises(ValueError, match="matrix mismatch"):
            calibration.evaluate_root(root_path, output, device="cpu", rollouts=3, tail_steps=2)
