"""Temperature-LR interventions preserve the engine and change only scalar Adam LR."""
from copy import copy, deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_ambi_alpha_lr as cli
from tests.test_ambi_entropy_probe_invariants import _fixture, _snapshot, _assert_unchanged
from tests.test_ambi_prior_kl_cuda import _PlainWithSnapshots
from tests.test_ambi_prior_kl_probe import _saved
from utils import ambi_alpha_lr as probe
from utils.ambi_entropy_probe import _hash, _preserve_environment


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_full_lr_arms_match_plain_engines_and_preserve_live_state(representation):
    wrapper = _fixture(representation, ent_coef="auto_0.0033654973")
    observation = np.asarray([.2, -.1, .7], dtype=np.float32)
    try:
        # Also preserve live episode state, existing grad buffers, and mixed flags.
        wrapper.agent.act(torch.as_tensor(observation), t0=True, eval_mode=True)
        params = list(wrapper.agent.model.parameters())
        params[0].requires_grad_(False)
        params[1].grad = torch.full_like(params[1], .125)
        before = _snapshot(wrapper)
        config_before = deepcopy(vars(wrapper.cfg))
        result = probe.probe_alpha_lr_root(wrapper, observation, seed=23, anchor_samples=8)
        _assert_unchanged(wrapper, before)
        assert vars(wrapper.cfg) == config_before
        assert result["paired_first_phase"] and result["paired_first_actor_update"]
        assert result["paired_private_rng_draw_counts"]
        assert not result["counterfactuals_are_closed_loop_evaluations"]
        assert list(result["full_solves"]) == ["lr_0p0003", "lr_0p003", "lr_0p03", "lr_0p3"]
        json.dumps(result, allow_nan=False)
        for rate in probe.DEFAULT_TEMPERATURE_LRS:
            arm = result["full_solves"][probe.temperature_lr_key(rate)]
            assert arm["config_overrides"] == {"inner_temperature_lr": rate}
            assert arm["optimizer_temperature_lr"] == rate
            assert arm["full_solve_performed"] and arm["temperature_allocated"]
            assert arm["actor_steps"] == arm["temperature_steps"] == arm["critic_steps"] == 4
            assert arm["actor_updates"][0]["temperature_before"]["adam_step"] == 0
            assert arm["actor_updates"][0]["alpha_before"] == result["initial_alpha"]
            proxy = copy(wrapper.agent)
            proxy.cfg = deepcopy(wrapper.agent.cfg)
            proxy.cfg.inner_temperature_lr = rate
            with _preserve_environment(wrapper.agent.model, wrapper.agent.device), torch.enable_grad():
                with torch.no_grad():
                    root = wrapper.agent.model.encode(torch.as_tensor(observation)[None]).detach()
                plain = _PlainWithSnapshots(proxy)
                plain.rng = plain._new_rng(23)
                action, metrics, _ = plain.act(root, t0=True, eval_mode=True, collect_diagnostics=False)
                np.testing.assert_array_equal(arm["action"], action.detach().numpy().squeeze())
                assert arm["first_replay_sha256"] == _hash(plain.initial_replay)
                assert arm["initial_actor_sha256"] == _hash(plain.initial_actor_state)
                assert arm["first_critic_sha256"] == _hash(plain.first_critic_state)
                for name in ("actor", "critic"):
                    assert arm[f"final_{name}_sha256"] == _hash(getattr(plain._action_pool, name).state_dict())
                assert arm["final_private_rng_sha256"] == _hash(plain.rng.training_state_dict())
                for name in ("inner_alpha_final", "inner_actor_entropy", "inner_temperature_loss"):
                    assert arm["metrics"][name] == float(metrics[name])
            _assert_unchanged(wrapper, before)

            # Replay the measured scalar entropy residual through fresh Adam;
            # this checks the actual response, not a loss-value interpretation.
            log_alpha = torch.nn.Parameter(torch.tensor(result["initial_alpha"]).log())
            optimizer = torch.optim.Adam([log_alpha], lr=rate, eps=wrapper.cfg.inner_adam_eps)
            for step, row in enumerate(arm["actor_updates"], 1):
                residual = row["metrics"]["actor_entropy"] - result["target_entropy"]
                assert row["temperature_entropy_residual"] == pytest.approx(residual)
                optimizer.zero_grad(set_to_none=True)
                (log_alpha * residual).backward()
                torch.nn.utils.clip_grad_norm_([log_alpha], wrapper.cfg.inner_temperature_grad_clip_norm)
                optimizer.step()
                with torch.no_grad():
                    log_alpha.clamp_(min=np.log(1e-8))
                assert row["temperature_after"]["adam_step"] == step
                assert row["temperature_after"]["log_alpha"] == pytest.approx(float(log_alpha.detach()), abs=2e-6)
    finally:
        wrapper.env.close()


def test_failed_lr_probe_restores_state_rng_and_configuration(monkeypatch):
    wrapper = _fixture()
    try:
        before = _snapshot(wrapper)
        config_before = deepcopy(vars(wrapper.cfg))
        original = wrapper.agent.model.pi
        calls = 0

        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected temperature-LR failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(wrapper.agent.model, "pi", interrupted)
        with pytest.raises(RuntimeError, match="injected temperature-LR failure"):
            probe.probe_alpha_lr_root(wrapper, [.2, -.1, .7], seed=23, anchor_samples=8)
        _assert_unchanged(wrapper, before)
        assert vars(wrapper.cfg) == config_before
    finally:
        wrapper.env.close()


@pytest.mark.parametrize("rates", [[], [0.], [-.3], [float("inf")], [float("nan")], [.003, .003]])
def test_invalid_lr_sweeps_rejected(rates):
    with pytest.raises(ValueError, match="Temperature learning rates"):
        probe.validate_temperature_lrs(rates)


def test_cli_plan_validates_original_saved_bank_without_loading_model(tmp_path, monkeypatch):
    path, profile, resolved, bank, _ = _saved(tmp_path)
    profile.update(temperature_lrs=list(probe.DEFAULT_TEMPERATURE_LRS), actor_steps=32, anchor_samples=128)
    selected = {**profile["checkpoints"]["test"], "step": 100000, "source_run": "original/run"}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    args = SimpleNamespace(profile=profile_path, root_bank=path, checkpoint_id="test", environment_seed=101,
                           plan_only=True)
    monkeypatch.setattr(cli, "_prepare", lambda args: (profile, selected, tmp_path / "checkpoint", None, resolved, [0]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Plan-only must not load a model or simulator")

    monkeypatch.setattr(cli, "_make_env", forbidden)
    monkeypatch.setattr(cli, "_initialize_frozen_model", forbidden)
    plan = cli.run(args)
    assert plan["root_bank_id"] == bank["id"]
    assert plan["original_source_git_commit"] == "original"
    assert plan["source_controller_solves"] == plan["source_simulator_decisions"] == 0
    assert plan["full_solves_per_root"] == 4
    assert plan["temperature_lrs"] == list(probe.DEFAULT_TEMPERATURE_LRS)
    assert plan["changed_parameter"] == "inner_temperature_lr"
