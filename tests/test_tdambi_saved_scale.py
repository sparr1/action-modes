"""Frozen native scale transfer and independent per-decision inner estimates."""

from copy import deepcopy

import pytest
import torch

from tests.test_tdambi_checkpoint import make_tdambi, pair


@pytest.mark.parametrize("mode", ["checkpoint_or_calibrate", "checkpoint"])
def test_each_inner_solve_starts_from_saved_scale_without_calibration(pair, monkeypatch, mode):
    native, _ = pair
    native.agent.scale.value.fill_(37.5)
    saved = deepcopy(native.agent.checkpoint_state())
    adapted = make_tdambi(inner_updates_per_round=2, tdambi_scale_initialization=mode)
    try:
        adapted.agent.load(saved)
        # The loaded reference must own its storage, even for dictionary loads.
        saved["scale"]["value"].fill_(900.0)
        engine = adapted.agent.inner_engine
        original = engine._tdambi_actor_step
        starts = []

        def capture(batch):
            scale = engine.state.tdambi_scale
            assert scale.data_ptr() != adapted.agent.tdambi_checkpoint_scale.data_ptr()
            if engine.state.actor_steps == 0:
                starts.append(scale.detach().clone())
            return original(batch)

        monkeypatch.setattr(engine, "_tdambi_actor_step", capture)
        frozen = deepcopy(adapted.agent.model.state_dict())
        for episode_start in (True, False, True):
            adapted.predict([0.2, -0.3, 0.7], deterministic=True, episode_start=episode_start)
            metrics = adapted.agent.last_inner_metrics
            assert metrics["inner_tdambi_calibration_samples"] == 0
            assert metrics["inner_tdambi_scale_from_checkpoint"] == 1
            assert metrics["inner_tdambi_q_scale_initial"] == pytest.approx(37.5)
            assert metrics["inner_tdambi_q_scale_final"] != pytest.approx(37.5)
            torch.testing.assert_close(adapted.agent.tdambi_checkpoint_scale, torch.tensor([37.5]))
        assert len(starts) == 3
        for start in starts:
            torch.testing.assert_close(start, torch.tensor([37.5]))
        for key, value in frozen.items():
            torch.testing.assert_close(adapted.agent.model.state_dict()[key], value, rtol=0, atol=0)
    finally:
        adapted.close()
        adapted.env.close()


def test_legacy_load_clears_previous_saved_scale_and_uses_calibration(pair):
    native, _ = pair
    native.agent.scale.value.fill_(37.5)
    state = deepcopy(native.agent.checkpoint_state())
    adapted = make_tdambi(inner_updates_per_round=1)
    try:
        adapted.agent.load(state)
        assert adapted.agent.tdambi_scale_source == "checkpoint"
        del state["scale"]
        adapted.agent.load(state)
        assert adapted.agent.tdambi_checkpoint_scale is None
        assert adapted.agent.tdambi_scale_source == "first_collection_calibration"
        adapted.predict([0.2, -0.3, 0.7], deterministic=True, episode_start=True)
        metrics = adapted.agent.last_inner_metrics
        assert metrics["inner_tdambi_scale_from_checkpoint"] == 0
        assert metrics["inner_tdambi_calibration_samples"] == adapted.cfg.inner_batch_size
        assert metrics["inner_tdambi_q_scale_initial"] >= 1.0
    finally:
        adapted.close()
        adapted.env.close()


def test_calibration_ablation_ignores_available_checkpoint_scale(pair):
    native, _ = pair
    native.agent.scale.value.fill_(37.5)
    adapted = make_tdambi(inner_updates_per_round=1, tdambi_scale_initialization="calibrate")
    try:
        adapted.agent.load(deepcopy(native.agent.checkpoint_state()))
        assert adapted.agent.tdambi_scale_source == "first_collection_calibration"
        adapted.predict([0.2, -0.3, 0.7], deterministic=True, episode_start=True)
        metrics = adapted.agent.last_inner_metrics
        assert metrics["inner_tdambi_calibration_samples"] == adapted.cfg.inner_batch_size
        assert metrics["inner_tdambi_scale_from_checkpoint"] == 0
        assert metrics["inner_tdambi_q_scale_initial"] != pytest.approx(37.5)
        torch.testing.assert_close(adapted.agent.tdambi_checkpoint_scale, torch.tensor([37.5]))
    finally:
        adapted.close()
        adapted.env.close()


@pytest.mark.parametrize("missing", [False, True])
def test_scale_preflight_preserves_loaded_weights_scale_and_counters(pair, missing):
    native, _ = pair
    adapted = make_tdambi(tdambi_scale_initialization="checkpoint")
    try:
        native.agent.scale.value.fill_(37.5)
        state = deepcopy(native.agent.checkpoint_state())
        adapted.agent.load(state)
        broken = deepcopy(state)
        broken["num_updates"] = 100
        broken["model"]["_pi.2.bias"].add_(1)
        if missing:
            del broken["scale"]
        else:
            broken["scale"]["value"].fill_(float("nan"))
        before = deepcopy(adapted.agent.model.state_dict())
        with pytest.raises(ValueError, match="Q scale"):
            adapted.agent.load(broken)
        assert adapted.agent.num_updates == state["num_updates"]
        torch.testing.assert_close(adapted.agent.tdambi_checkpoint_scale, torch.tensor([37.5]))
        for key, value in before.items():
            torch.testing.assert_close(adapted.agent.model.state_dict()[key], value, rtol=0, atol=0)
    finally:
        adapted.close()
        adapted.env.close()


def test_invalid_scale_initialization_is_rejected():
    with pytest.raises(ValueError, match="tdambi_scale_initialization"):
        make_tdambi(tdambi_scale_initialization="last_inner_solve")
