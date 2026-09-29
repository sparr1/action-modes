"""Portable auxiliary critics are complete, validated, and frozen in evaluation."""

from copy import deepcopy

import pytest
import torch

from test_ambixqc_core import _batch
from test_ambixqc_prior_checkpoint import wrappers
from test_ambixqc_replay_archive import _assert_equal, _resident_rows, _view_rows
from utils.replay_archive import load_checkpoint_replay


def _settings(**overrides):
    return {
        "aux_return_mode": "xqc", "inner_critic_source": "aux_return",
        "inner_horizon_critic_source": "aux_return", "inner_terminal_bootstrap": "outer",
        **overrides,
    }


def test_auxiliary_checkpoint_round_trip_and_identical_next_update(wrappers, tmp_path):
    source = wrappers(**_settings())
    source.agent.observe_reward(4.0, False, False)
    source.agent._update(*_batch(source.agent))
    checkpoint = source.save(tmp_path, "auxiliary")
    saved = deepcopy(source.agent.checkpoint_state())
    assert saved["checkpoint_version"] == 7
    assert saved["aux_return"]["update_step"] == 1
    assert saved["semantic_signature"]["aux_return"]["target"] == "reward_only"
    restored = wrappers(**_settings()).load(checkpoint)
    _assert_equal(saved, restored.agent.checkpoint_state())
    source.agent._update(*_batch(source.agent))
    restored.agent._update(*_batch(restored.agent))
    _assert_equal(source.agent.checkpoint_state(), restored.agent.checkpoint_state())


def test_frozen_evaluation_allows_independent_inner_routes_and_preserves_auxiliary(wrappers):
    source = wrappers(aux_return_mode="xqc", inner_operator="none")
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    target = wrappers(**_settings())
    before = target.agent.frozen_outer_state()
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    _assert_equal(before, target.agent.frozen_outer_state())
    target.load(saved, frozen_evaluation=True)
    _assert_equal(source.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["saved_semantic_signature"]["inner_critic_source"] == "xqc"
    assert provenance["evaluated_semantic_signature"]["inner_critic_source"] == "aux_return"
    assert provenance["evaluated_semantic_signature"]["inner_critic_target"] == "reward_only"
    target.reset_for_evaluation(101)
    before = target.agent.frozen_outer_state()
    observation, _ = target.env.reset(seed=101)
    target.predict(observation)
    target.predict(observation)
    _assert_equal(before, target.agent.frozen_outer_state())


@pytest.mark.parametrize("problem", [
    "missing_state", "missing_tensor", "nonfinite_tensor", "counter", "optimizer",
    "nonfinite_optimizer", "generator", "source", "target", "scale", "mode", "coefficient",
    "online_support", "target_support", "primary_support",
])
def test_auxiliary_preflight_rejects_corrupt_state_before_mutation(wrappers, problem):
    source = wrappers(**_settings())
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    if problem == "missing_state":
        saved.pop("aux_return")
    elif problem in {"missing_tensor", "nonfinite_tensor"}:
        key = next(key for key, value in saved["module"].items()
                   if key.startswith("aux_return.") and value.is_floating_point())
        if problem == "missing_tensor":
            saved["module"].pop(key)
        else:
            saved["module"][key].fill_(float("nan"))
    elif problem == "counter":
        saved["aux_return"]["update_step"] += 1
    elif problem == "optimizer":
        saved["aux_return"]["critic_optimizer"]["state"].clear()
    elif problem == "nonfinite_optimizer":
        next(iter(saved["aux_return"]["critic_optimizer"]["state"].values()))["exp_avg"].fill_(float("inf"))
    elif problem == "generator":
        saved["aux_return"]["generator"] = torch.zeros(1, dtype=torch.uint8)
    elif problem in {"online_support", "target_support", "primary_support"}:
        key = {
            "online_support": "aux_return.critic.support",
            "target_support": "aux_return.critic_target.support",
            "primary_support": "xqc_controller.critic.support",
        }[problem]
        saved["module"][key].zero_()
    elif problem == "source":
        saved["semantic_signature"]["inner_critic_source"] = "missing"
    elif problem == "target":
        saved["semantic_signature"]["inner_critic_target"] = "missing"
    elif problem == "scale":
        saved["semantic_signature"]["reward_normalization"] = "real_discounted_return_plus_fresh_action_local_imagined_returns"
    elif problem == "mode":
        saved["semantic_signature"]["aux_return"]["mode"] = []
    else:
        saved["semantic_signature"]["aux_return"]["critic_coef"] = float("nan")
    target = wrappers(**_settings())
    before = target.agent.frozen_outer_state()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        target.load(saved, frozen_evaluation=True)
    _assert_equal(before, target.agent.frozen_outer_state())


@pytest.mark.parametrize("frozen", [False, True])
def test_missing_auxiliary_cannot_be_restored_as_random_weights(wrappers, frozen):
    saved = deepcopy(wrappers().agent.checkpoint_state())
    target = wrappers(aux_return_mode="xqc")
    before = target.agent.frozen_outer_state()
    with pytest.raises(ValueError, match="lacks a trained auxiliary"):
        target.load(saved, frozen_evaluation=frozen)
    _assert_equal(before, target.agent.frozen_outer_state())
    auxiliary = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError):
        wrappers().load(auxiliary, frozen_evaluation=frozen)


@pytest.mark.parametrize("ratio", [1, 2])
def test_auxiliary_checkpoint_retains_matching_replay_archive(wrappers, tmp_path, ratio):
    source = wrappers(aux_return_mode="xqc", inner_operator="none", xqc_utd=ratio)
    source.enable_replay_archive(tmp_path, "auxiliary")
    source.set_checkpointing(5, tmp_path, "auxiliary", save_strat=("all", "latest"))
    try:
        source.learn(total_timesteps=10)
        checkpoint = source.save(tmp_path, "auxiliary_final")
        restored = wrappers(aux_return_mode="xqc", inner_operator="none", xqc_utd=ratio).load(checkpoint)
        # Final saves may reuse the last periodic model bytes; compare the
        # serialized snapshot rather than post-episode lifecycle counters.
        _assert_equal(torch.load(checkpoint, weights_only=False), restored.agent.checkpoint_state())
        _assert_equal(source.agent.aux_return.training_state_dict(), restored.agent.aux_return.training_state_dict())
        _assert_equal(_view_rows(load_checkpoint_replay(checkpoint)), _resident_rows(source.buffer))
        assert load_checkpoint_replay(tmp_path / "auxiliary_5").num_rows == 6
    finally:
        source.close_replay_archive()


def test_cpu_frozen_load_validates_auxiliary_cuda_rng(wrappers):
    saved = deepcopy(wrappers(aux_return_mode="xqc", inner_operator="none").agent.checkpoint_state())
    cuda_rng = torch.tensor(list((42).to_bytes(8, "little") + (4).to_bytes(8, "little")), dtype=torch.uint8)
    saved["outer_generator"] = cuda_rng.clone()
    saved["inner"]["rng"]["device_type"] = "cuda"
    for name in saved["inner"]["rng"]["streams"]:
        saved["inner"]["rng"]["streams"][name] = cuda_rng.clone()
    saved["aux_return"].update(device_type="cuda", generator=cuda_rng.clone())
    target = wrappers(aux_return_mode="xqc", inner_operator="none")
    target.load(saved, frozen_evaluation=True)
    assert target.agent.checkpoint_evaluation_provenance["cross_device_rng_reset"]
    assert target.agent.aux_return.training_state_dict()["device_type"] == "cpu"
    before = target.agent.frozen_outer_state()
    malformed = deepcopy(saved)
    malformed["aux_return"]["generator"] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="RNG"):
        target.load(malformed, frozen_evaluation=True)
    _assert_equal(before, target.agent.frozen_outer_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_auxiliary_real_cross_device_frozen_round_trip(wrappers):
    cpu = wrappers(**_settings(inner_operator="none"))
    cpu.agent._update(*_batch(cpu.agent))
    cuda = wrappers(**_settings(device="cuda")).load(
        deepcopy(cpu.agent.checkpoint_state()), frozen_evaluation=True,
    )
    cuda.reset_for_evaluation(101)
    before = cuda.agent.frozen_outer_state()
    observation, _ = cuda.env.reset(seed=101)
    cuda.predict(observation)
    _assert_equal(before, cuda.agent.frozen_outer_state())
    restored = wrappers(**_settings()).load(deepcopy(cuda.agent.checkpoint_state()), frozen_evaluation=True)
    restored.reset_for_evaluation(101)
    restored.predict(observation)
    assert restored.agent.checkpoint_evaluation_provenance["cross_device_rng_reset"]
