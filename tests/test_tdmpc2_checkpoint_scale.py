"""Portable native checkpoints preserve the learned Q normalization scale."""

from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.common.scale import preflight_scale_state
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_tdmpc2_correctness import make_model


@pytest.fixture
def native_model():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    model = make_model(TDMPC2Baseline, env)
    try:
        yield model
    finally:
        model._checkpoint_writer.shutdown()
        env.close()


@pytest.mark.parametrize("from_file", [False, True])
def test_portable_native_checkpoint_restores_learned_scale(native_model, tmp_path, from_file):
    agent = native_model.agent
    agent.scale.value.fill_(37.5)
    agent.num_updates = 713
    expected = deepcopy(agent.checkpoint_state())
    assert set(expected["scale"]) == {"value", "percentiles"}
    preflight_scale_state(expected["scale"])
    if from_file:
        source = tmp_path / "model_25000"
        agent.save(source)
    else:
        source = expected
    with torch.no_grad():
        next(agent.model.parameters()).add_(1.0)
        agent.scale.value.fill_(9.0)
        agent.scale._percentiles.copy_(torch.tensor([1.0, 99.0]))
    agent.num_updates = 999
    agent.load(source)
    _assert_tree_equal(agent.checkpoint_state(), expected)
    assert not agent.scale.value.requires_grad


@pytest.mark.parametrize("tracked", [False, True])
def test_periodic_25000_checkpoint_freezes_scale_with_the_model(native_model, tmp_path, tracked):
    model = native_model
    if tracked:
        model.set_checkpointing(25_000, tmp_path, "model")
    else:
        model._checkpointing = (25_000, tmp_path, "model")
    model._global_step = 25_000
    model._num_updates = model.agent.num_updates = 777
    model.agent.scale.value.fill_(43.25)
    expected = deepcopy(model.agent.checkpoint_state())
    model._maybe_checkpoint()
    with torch.no_grad():
        model.agent.scale.value.fill_(1000.0)
        model.agent.scale._percentiles.copy_(torch.tensor([20.0, 80.0]))
        next(model.agent.model.parameters()).add_(7.0)
    model.flush_checkpoints()
    saved = torch.load(tmp_path / "model_25000", weights_only=False)
    _assert_tree_equal(saved, expected)
    torch.testing.assert_close(saved["scale"]["value"], torch.tensor([43.25]), rtol=0, atol=0)


@pytest.mark.parametrize("raw_model", [False, True])
def test_legacy_native_checkpoint_resets_missing_scale(native_model, raw_model):
    agent = native_model.agent
    agent.num_updates = 7
    checkpoint = deepcopy(agent.checkpoint_state())
    del checkpoint["scale"]
    if raw_model:
        checkpoint = checkpoint["model"]
    agent.scale.value.fill_(99.0)
    agent.scale._percentiles.copy_(torch.tensor([1.0, 99.0]))
    agent.load(checkpoint)
    torch.testing.assert_close(agent.scale.value, torch.tensor([1.0]), rtol=0, atol=0)
    torch.testing.assert_close(agent.scale._percentiles, torch.tensor([5.0, 95.0]), rtol=0, atol=0)
    assert agent.num_updates == (0 if raw_model else 7)


def _invalid_scale(case):
    state = {"value": torch.tensor([25.0]), "percentiles": torch.tensor([5.0, 95.0])}
    if case == "none":
        return None
    if case == "missing-key":
        del state["value"]
    elif case == "extra-key":
        state["tau"] = 0.01
    elif case == "value-number":
        state["value"] = 25.0
    elif case == "value-scalar":
        state["value"] = torch.tensor(25.0)
    elif case == "value-shape":
        state["value"] = torch.tensor([25.0, 25.0])
    elif case == "value-dtype":
        state["value"] = state["value"].double()
    elif case in {"nan", "inf", "below-floor", "negative"}:
        state["value"].fill_({"nan": float("nan"), "inf": float("inf"), "below-floor": 0.5, "negative": -1.0}[case])
    elif case == "percentiles-list":
        state["percentiles"] = [5.0, 95.0]
    elif case == "percentiles-shape":
        state["percentiles"] = state["percentiles"].reshape(1, 2)
    elif case == "percentiles-dtype":
        state["percentiles"] = state["percentiles"].long()
    elif case == "percentiles-values":
        state["percentiles"] = torch.tensor([10.0, 90.0])
    elif case == "percentiles-nan":
        state["percentiles"][0] = float("nan")
    return state


@pytest.mark.parametrize("case", [
    "none", "missing-key", "extra-key", "value-number", "value-scalar", "value-shape", "value-dtype",
    "nan", "inf", "below-floor", "negative", "percentiles-list", "percentiles-shape",
    "percentiles-dtype", "percentiles-values", "percentiles-nan",
])
def test_invalid_saved_scale_fails_before_any_native_agent_mutation(native_model, case):
    agent = native_model.agent
    agent.scale.value.fill_(11.0)
    agent.num_updates = 7
    pristine = deepcopy(agent.training_state_dict())
    incoming = deepcopy(agent.checkpoint_state())
    changed_key = next(key for key, value in incoming["model"].items() if value.is_floating_point())
    incoming["model"][changed_key].add_(1.0)
    incoming["num_updates"] = 99
    incoming["scale"] = _invalid_scale(case)
    with pytest.raises(ValueError, match="scale"):
        agent.load(incoming)
    _assert_tree_equal(agent.training_state_dict(), pristine)


def test_exact_training_scale_payload_keeps_its_schema_and_checks_floor(native_model):
    agent = native_model.agent
    agent.scale.value.fill_(17.0)
    state = deepcopy(agent.training_state_dict())
    assert state["schema"] == "tdmpc2-agent-training-state"
    assert state["version"] == 1
    _assert_tree_equal(state["scale"], agent.checkpoint_state()["scale"])
    invalid = deepcopy(state)
    invalid["scale"]["value"].fill_(0.5)
    with pytest.raises(ValueError, match="scale"):
        agent.load_training_state_dict(invalid)
    _assert_tree_equal(agent.training_state_dict(), state)
