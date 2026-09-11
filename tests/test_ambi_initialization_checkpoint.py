"""Scratch initialization is scientific state, including at empty boundaries."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture
def models():
    opened = []

    def create(**options):
        params = dict(
            inner_rounds=1, inner_rollouts_per_round=4,
            inner_updates_per_round=3,
        )
        params.update(options)
        if params.get("inner_critic_adaptation") == "lora_rl":
            params.setdefault("inner_critic_lora_rank", 4)
        model = _tiny_model(**params)
        opened.append(model)
        return model.agent

    yield create
    for model in opened:
        model.env.close()


@pytest.mark.parametrize("adaptation,version", [("clone", 1), ("lora_rl", 3)])
def test_explicit_prior_preserves_default_checkpoint_payload(models, adaptation, version):
    implicit = models(inner_critic_adaptation=adaptation)
    explicit = models(
        inner_actor_initialization="prior", inner_critic_initialization="prior",
        inner_critic_adaptation=adaptation,
    )
    assert implicit.inner_engine.training_state_dict()["version"] == version
    assert "initialization_spec" not in implicit.inner_engine.training_state_dict()
    _assert_tree_equal(implicit.training_state_dict(), explicit.training_state_dict())


@pytest.mark.parametrize("actor,critic", [
    ("random", "prior"), ("prior", "random"), ("random", "random"),
])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_random_checkpoint_reproduces_next_solve(models, actor, critic, adaptation):
    options = dict(
        inner_actor_initialization=actor, inner_critic_initialization=critic,
        inner_critic_adaptation=adaptation,
    )
    source = models(**options)
    source.act(torch.zeros(3), collect_diagnostics=False)
    source.prepare_training_resume_boundary()
    payload = deepcopy(source.training_state_dict())
    assert payload["inner"]["version"] == 4
    assert payload["inner"]["initialization_spec"] == {"actor": actor, "critic": critic}
    assert ("lora_rl_spec" in payload["inner"]) == (adaptation == "lora_rl")
    assert payload["inner"]["workspace"]["actor"] is None
    assert payload["inner"]["workspace"]["critic"] is None
    spec = payload["outer"]["critic_target_spec"]["inner_solve"]
    for component, choice in (("actor", actor), ("critic", critic)):
        assert spec.get(f"{component}_initialization", "prior") == choice

    direct = models(**options).inner_engine
    direct.load_training_state_dict(payload["inner"])
    _assert_tree_equal(direct.training_state_dict(), payload["inner"])

    restored = models(**options)
    global_rng = torch.random.get_rng_state().clone()
    restored.load_training_state_dict(payload)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    _assert_tree_equal(restored.training_state_dict(), payload)
    source.reset()
    restored.reset()
    for observation in (torch.tensor([0.2, -0.1, 0.3]), torch.zeros(3)):
        expected = source.act(observation, collect_diagnostics=False)
        actual = restored.act(observation, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for component in ("actor", "critic", "critic_target", "actor_optim", "critic_optim"):
            left = getattr(source.inner_engine._action_pool, component)
            right = getattr(restored.inner_engine._action_pool, component)
            _assert_tree_equal(left.state_dict(), right.state_dict())
        _assert_tree_equal(
            source.inner_engine.rng.training_state_dict(),
            restored.inner_engine.rng.training_state_dict(),
        )
    source.prepare_training_resume_boundary()
    restored.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())


@pytest.mark.parametrize("source_options,target_options", [
    ({}, {"inner_actor_initialization": "random"}),
    ({"inner_actor_initialization": "random"}, {}),
    ({"inner_actor_initialization": "random"}, {"inner_critic_initialization": "random"}),
    ({"inner_critic_initialization": "random"},
     {"inner_critic_initialization": "random", "inner_critic_adaptation": "lora_rl"}),
    ({"inner_critic_initialization": "random", "inner_critic_adaptation": "lora_rl"},
     {"inner_critic_initialization": "random"}),
])
@pytest.mark.parametrize("direct", [False, True])
def test_cross_initialization_or_adaptation_resume_rejected_before_mutation(
    models, source_options, target_options, direct,
):
    source, target = models(**source_options), models(**target_options)
    if direct:
        source, target = source.inner_engine, target.inner_engine
    payload = deepcopy(source.training_state_dict())
    before = deepcopy(target.training_state_dict())
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="incompatible|does not match"):
        target.load_training_state_dict(payload)
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)


@pytest.mark.parametrize("tamper", ["actor", "missing_critic", "extra", "lora_rank"])
def test_random_protocol_rejects_corruption_before_mutation(models, tamper):
    engine = models(
        inner_actor_initialization="random", inner_critic_initialization="random",
        inner_critic_adaptation="lora_rl",
    ).inner_engine
    before = deepcopy(engine.training_state_dict())
    payload = deepcopy(before)
    if tamper == "actor":
        payload["initialization_spec"]["actor"] = True
    elif tamper == "missing_critic":
        del payload["initialization_spec"]["critic"]
    elif tamper == "extra":
        payload["initialization_spec"]["refresh"] = "action"
    else:
        payload["lora_rl_spec"]["rank"] += 1
    with pytest.raises((ValueError, TypeError)):
        engine.load_training_state_dict(payload)
    _assert_tree_equal(engine.training_state_dict(), before)


@pytest.mark.parametrize("source_random", [False, True])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_portable_outer_weights_allow_changing_initialization(models, source_random, adaptation):
    source = models(
        inner_actor_initialization="random" if source_random else "prior",
        inner_critic_initialization="random" if source_random else "prior",
    )
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.add_(0.01)
    target = models(
        inner_actor_initialization="prior" if source_random else "random",
        inner_critic_initialization="prior" if source_random else "random",
        inner_critic_adaptation=adaptation,
    )
    payload = deepcopy(source.checkpoint_state())
    target.load(payload)
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    assert target.cfg.inner_actor_initialization == ("prior" if source_random else "random")
    outer_before = deepcopy(target.model.state_dict())
    action = target.act(torch.zeros(3), collect_diagnostics=False)
    assert torch.isfinite(action).all()
    _assert_tree_equal(target.model.state_dict(), outer_before)
