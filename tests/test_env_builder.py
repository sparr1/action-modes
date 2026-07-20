import pytest

from utils import core


class _FakeEnv:
    def __init__(self, inner=None):
        self.inner = inner
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.inner is not None:
            self.inner.close()


def test_build_env_copies_params_and_applies_wrappers_in_declared_order(monkeypatch):
    base_env = _FakeEnv()
    make_calls = []
    wrapper_calls = []

    def fake_make(env_id, **params):
        make_calls.append((env_id, params))
        params["nested"]["value"] = "mutated by gym.make"
        return base_env

    def fake_setup_wrapper(domain, name, params):
        wrapper_calls.append((name, params, domain))
        return _FakeEnv(domain)

    monkeypatch.setattr(core.gym, "make", fake_make)
    monkeypatch.setattr(core, "setup_wrapper", fake_setup_wrapper)

    experiment_params = {
        "env_params": {
            "render_mode": "human",
            "nested": {"value": "original"},
        }
    }
    run_params = {
        "env": "Example-v0",
        "env_wrappers": [
            {"name": "AntPlane", "wrapper_params": {"first": True}},
            {
                "name": "example:ScaledStateWrapper",
                "wrapper_params": {"second": True},
            },
        ],
        "env_wrapper": {
            "name": "Subtask",
            "wrapper_params": {"third": True},
        },
    }

    result = core.build_env(run_params, experiment_params, render_mode=None)

    assert make_calls == [
        (
            "Example-v0",
            {
                "render_mode": None,
                "nested": {"value": "mutated by gym.make"},
            },
        )
    ]
    assert experiment_params["env_params"] == {
        "render_mode": "human",
        "nested": {"value": "original"},
    }
    assert [call[0] for call in wrapper_calls] == [
        "AntPlane",
        "example:ScaledStateWrapper",
        "Subtask",
    ]
    assert wrapper_calls[0][2] is base_env
    assert wrapper_calls[1][2] is not base_env
    assert wrapper_calls[2][2] is not wrapper_calls[1][2]
    assert result is not base_env


def test_build_env_preserves_configured_render_mode_when_override_is_omitted(
    monkeypatch,
):
    captured_params = []
    env = _FakeEnv()

    def fake_make(_env_id, **params):
        captured_params.append(params)
        return env

    monkeypatch.setattr(core.gym, "make", fake_make)

    assert core.build_env(
        {"env": "Example-v0"}, {"env_params": {"render_mode": "human"}}
    ) is env
    assert captured_params == [{"render_mode": "human"}]


def test_build_env_passes_requested_render_mode(monkeypatch):
    captured_params = []
    env = _FakeEnv()

    def fake_make(_env_id, **params):
        captured_params.append(params)
        return env

    monkeypatch.setattr(core.gym, "make", fake_make)

    assert core.build_env(
        {"env": "Example-v0"}, {}, render_mode="rgb_array"
    ) is env
    assert captured_params == [{"render_mode": "rgb_array"}]


@pytest.mark.parametrize(
    ("bad_wrapper", "message"),
    [
        ({}, "non-empty string 'name'"),
        (
            {"name": "example:UnknownWrapper", "wrapper_params": {}},
            "Unsupported wrapper",
        ),
        ({"name": "AntPlane"}, "must define 'wrapper_params'"),
        (
            {"name": "AntPlane", "wrapper_params": []},
            "'wrapper_params' must be a mapping",
        ),
    ],
)
def test_build_env_validates_wrappers_and_closes_the_environment(
    monkeypatch, bad_wrapper, message
):
    env = _FakeEnv()
    monkeypatch.setattr(core.gym, "make", lambda *_args, **_kwargs: env)

    with pytest.raises(ValueError, match=message):
        core.build_env(
            {"env": "Example-v0", "env_wrappers": [bad_wrapper]},
            {},
        )

    assert env.close_calls == 1


def test_build_env_closes_the_current_wrapper_when_later_setup_fails(monkeypatch):
    base_env = _FakeEnv()
    first_wrapper = _FakeEnv(base_env)
    setup_calls = 0

    monkeypatch.setattr(core.gym, "make", lambda *_args, **_kwargs: base_env)

    def fake_setup_wrapper(_domain, _name, _params):
        nonlocal setup_calls
        setup_calls += 1
        if setup_calls == 1:
            return first_wrapper
        raise RuntimeError("wrapper setup failed")

    monkeypatch.setattr(core, "setup_wrapper", fake_setup_wrapper)
    wrappers = [
        {"name": "AntPlane", "wrapper_params": {}},
        {"name": "Subtask", "wrapper_params": {}},
    ]

    with pytest.raises(RuntimeError, match="wrapper setup failed"):
        core.build_env({"env": "Example-v0", "env_wrappers": wrappers}, {})

    assert first_wrapper.close_calls == 1
    assert base_env.close_calls == 1


def test_build_env_rejects_non_sequence_wrapper_collection_and_closes(monkeypatch):
    env = _FakeEnv()
    monkeypatch.setattr(core.gym, "make", lambda *_args, **_kwargs: env)

    with pytest.raises(ValueError, match="'env_wrappers' must be a sequence"):
        core.build_env(
            {"env": "Example-v0", "env_wrappers": {"name": "AntPlane"}},
            {},
        )

    assert env.close_calls == 1
