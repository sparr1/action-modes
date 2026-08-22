import importlib
import sys
from types import ModuleType

import pytest


def _load_baselines_without_optional_sb3(monkeypatch):
    stable_baselines3 = ModuleType("stable_baselines3")
    common = ModuleType("stable_baselines3.common")
    callbacks = ModuleType("stable_baselines3.common.callbacks")
    logger = ModuleType("stable_baselines3.common.logger")
    callbacks.BaseCallback = type("BaseCallback", (), {})
    logger.KVWriter = type("KVWriter", (), {})
    stable_baselines3.common = common
    common.callbacks = callbacks
    common.logger = logger
    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.callbacks", callbacks)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.logger", logger)
    sys.modules.pop("RL.baselines", None)
    return importlib.import_module("RL.baselines")


@pytest.mark.parametrize("training_fails", [False, True], ids=["success", "failure"])
def test_baseline_finalizer_attempts_all_callbacks_and_preserves_primary(
    monkeypatch, training_fails
):
    module = _load_baselines_without_optional_sb3(monkeypatch)
    calls = []

    class Model:
        def learn(self, **_kwargs):
            if training_fails:
                raise RuntimeError("training failed")
            return "trained"

    class Callback:
        def __init__(self, name, error):
            self.name = name
            self.error = error

        def finish(self):
            calls.append(self.name)
            raise self.error

    monkeypatch.setattr(module, "WandbBaselineCallback", Callback)
    learner = object.__new__(module.Baseline)
    learner.model = Model()
    learner.callback = [
        Callback("first", OSError("first callback failed")),
        Callback("second", ValueError("second callback failed")),
    ]

    expected = RuntimeError if training_fails else OSError
    message = "training failed" if training_fails else "first callback failed"
    try:
        with pytest.raises(expected, match=message) as captured:
            learner.learn(total_timesteps=1)
    finally:
        sys.modules.pop("RL.baselines", None)

    assert calls == ["first", "second"]
    if training_fails:
        assert getattr(captured.value, "__notes__", ()) == [
            "Additional baseline cleanup failure: first callback failed",
            "Additional baseline cleanup failure: second callback failed",
        ]
    else:
        assert getattr(captured.value, "__notes__", ()) == [
            "Additional cleanup failure: second callback failed"
        ]
