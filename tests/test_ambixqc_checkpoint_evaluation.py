"""Frozen XQC episode evaluation preserves priors and portable diagnostics."""

import gzip
import json
from pathlib import Path

import gymnasium as gym
import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.AMBIXQC import AMBIXQC


@pytest.fixture
def checkpoint_case(tmp_path):
    params = {
        "device": "cpu", "model_size": None, "enc_dim": 16, "mlp_dim": 16,
        "latent_dim": 8, "num_enc_layers": 2, "simnorm_dim": 4,
        "num_bins": 5, "vmin": -5, "vmax": 5, "batch_size": 2,
        "train_unroll_horizon": 2, "buffer_size": 32, "seed_steps": 4,
        "pretrain_steps": 1, "utd": 1, "compile": False, "episodic": False,
        "discount": 0.99, "wandb": False, "xqc_actor_net_arch": [8, 8],
        "xqc_critic_net_arch": [8, 8], "xqc_num_atoms": 11,
        "xqc_vmin": -2, "xqc_vmax": 2, "xqc_optimizer_backend": "single_tensor",
        "inner_operator": "none", "inner_rounds": 1,
        "inner_rollouts_per_round": 2, "inner_rollout_horizon": 2,
        "inner_updates_per_round": 4, "inner_batch_size": 2,
        "inner_replay_capacity": 4, "inner_diagnostics_every": 1,
    }
    run = {"alg": "AMBIXQC/AMBIXQC", "env": "Pendulum-v1", "seed": 3,
           "total_steps": 10, "device": "cpu", "alg_params": params}
    environment = {"env_params": {"max_episode_steps": 3}}
    env = gym.make("Pendulum-v1", max_episode_steps=3)
    model = AMBIXQC("AMBIXQC", env, params, run, environment)
    # Non-empty optimizers, target statistics, normalizer, and learner RNG make
    # the invariant check cover a learned checkpoint rather than initial weights.
    generator = torch.Generator().manual_seed(87)
    for _ in range(4):
        model.agent._update(
            torch.randn(3, 2, 3, generator=generator),
            torch.randn(2, 2, 1, generator=generator).tanh(),
            torch.randn(2, 2, 1, generator=generator), torch.zeros(2, 2, 1),
        )
    model.agent.observe_reward(2, False, False)
    model.agent.observe_reward(3, False, True)
    checkpoint = tmp_path / "prior.pt"
    model.agent.save(str(checkpoint))
    env.close()
    metadata = {
        "schema_version": 1, "trial_run_params": run, "experiment_params": environment,
        "checkpoint": {"kind": "step", "step": 4, "episode": 1,
                       "best_score": None, "best_window": 100},
    }
    Path(f"{checkpoint}.metadata.json").write_text(json.dumps(metadata))
    matrix = {
        "schema_version": 1, "base_alg_config": "checkpoint",
        "evaluation": {"seeds": [101, 102], "controller_seed": 55,
                       "max_steps": 2, "default_presets": ["controller/prior"]},
        "comparisons": {"controller": {"reference": "prior", "variants": {
            "prior": {"alg_params": {"inner_operator": "none"}},
            "xqc": {"alg_params": {"inner_operator": "xqc"}},
            "imagined": {"alg_params": {"inner_operator": "xqc",
                "inner_reward_normalization": "action_local_imagined"}},
        }}},
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    return matrix_path, checkpoint


def _events(path, run):
    return [json.loads(line) for relative in run["trace_files"]
            for line in gzip.decompress((path / relative).read_bytes()).decode().splitlines()]


def _scientific_episodes(result):
    # Wall times are observations, not deterministic scientific outputs.
    def strip_timings(value):
        if isinstance(value, dict):
            return {key: strip_timings(item) for key, item in value.items() if not key.endswith("_seconds")}
        return value
    return [strip_timings(episode) for episode in result["episodes"]]


def test_paired_controllers_frozen_bundle_and_actual_delayed_counts(checkpoint_case, tmp_path):
    matrix, checkpoint = checkpoint_case
    bundle = tmp_path / "paired"
    payload = evaluator.evaluate_matrix(matrix, checkpoint, comparisons=["controller"], bundle_dir=bundle)
    assert len(payload["results"]) == 3
    for result in payload["results"]:
        assert result["outer_state_unchanged"]
        assert result["outer_updates_before"] == result["outer_updates_after"] == 4
        assert result["environment_seeds"] == [101, 102]
        assert result["paired_return_delta_vs_reference"]["count"] == 2
        assert result["saved_algorithm_config"]["alg_params"]["inner_operator"] == "none"
        assert result["checkpoint_evaluation_provenance"]["checkpoint_version"] == 4
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    for run in manifest["runs"]:
        events = _events(bundle, run)
        assert len(events) == 4
        assert all(event["phase"] == "decision" for event in events)
        expected = (0, 0, 0) if run["selector"] == "controller/prior" else (4, 2, 2)
        assert all(tuple(event[key] for key in ("critic_updates", "actor_updates", "temperature_updates"))
                   == expected for event in events)
        assert all("decision/reward" in event["metrics"] and
                   event["metrics"]["decision/control_seconds"] >= 0 for event in events)


def test_episode_seeds_are_order_independent_and_warmup_is_unscored(checkpoint_case, tmp_path):
    matrix, checkpoint = checkpoint_case
    first = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/xqc"])["results"][0]
    warm = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/xqc"],
                                     bundle_dir=tmp_path / "warm")["results"][0]
    reversed_seeds = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/xqc"],
                                               seeds=[102, 101])["results"][0]
    assert _scientific_episodes(first) == _scientific_episodes(warm)
    assert _scientific_episodes(first) == list(reversed(_scientific_episodes(reversed_seeds)))


def test_prior_reference_reuse_and_protocol_mismatch_preflight(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    reference = tmp_path / "prior"
    prior = evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=reference)["results"][0]
    result = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/xqc"],
                                      bundle_dir=tmp_path / "inner", reference_bundle=reference)["results"][0]
    expected = [episode["return"] - baseline["return"]
                for episode, baseline in zip(result["episodes"], prior["episodes"])]
    assert result["paired_return_delta_vs_prior"]["mean"] == pytest.approx(sum(expected) / len(expected))
    monkeypatch.setattr(evaluator, "_make_env", lambda _: pytest.fail("environment created before preflight"))
    with pytest.raises(ValueError, match="protocol"):
        evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=tmp_path / "bad", max_steps=1,
                                  reference_bundle=reference)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("options", [{"bank_only": True}, {"root_bank_path": "missing.json"},
                                    {"save_root_bank": "bank.json"}, {"bank_repetitions": 3}])
def test_bank_requests_fail_before_environments_or_bundles(checkpoint_case, tmp_path, monkeypatch, options):
    matrix, checkpoint = checkpoint_case
    monkeypatch.setattr(evaluator, "_make_env", lambda _: pytest.fail("environment created before rejection"))
    with pytest.raises(ValueError, match="unsupported"):
        evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=tmp_path / "bad", **options)
    assert not (tmp_path / "bad").exists()


def test_completed_episodes_survive_later_failure_with_frozen_state(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    make_env = evaluator._make_env

    class FailSecondSeed(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            self.seed_value = seed
            return self.env.reset(seed=seed, options=options)

        def step(self, action):
            if self.seed_value == 102:
                raise RuntimeError("intentional later episode failure")
            return self.env.step(action)

    monkeypatch.setattr(evaluator, "_make_env", lambda resolved: FailSecondSeed(make_env(resolved)))
    bundle = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="intentional later"):
        evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/xqc"], bundle_dir=bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    run = manifest["runs"][0]
    assert manifest["status"] == run["status"] == "failed"
    assert run["outer_state_unchanged"]
    assert run["episodes"][0]["seed"] == 101
    assert run["episodes"][0]["length"] == 2
    assert run["episodes"][1]["status"] == "failed"
    assert len(_events(bundle, run)) == 2


def test_legacy_json_rng_contract_and_new_bundle_preflight(tmp_path, monkeypatch):
    from types import SimpleNamespace

    resolved = {"selector": "controller/prior", "comparison": "controller", "variant": "prior",
                "reference": "prior", "description": "",
                "algorithm_config": {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {"inner_operator": "none"}},
                "environment": {"id": "Pendulum-v1", "params": {"max_episode_steps": 1}}}
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"fixture")

    class Model:
        def __init__(self):
            self.agent = SimpleNamespace(num_updates=0, device="cpu", last_inner_metrics={},
                                         inner_engine=object(), model=SimpleNamespace(critic_signature={}))
            self.cfg = SimpleNamespace(inner_operator="none")
            self.calls = 0

        def predict(self, observation, **kwargs):
            self.calls += 1
            return [0.0], None

    model = Model()
    monkeypatch.setattr(evaluator, "_initialize_frozen_model", lambda *args, **kwargs: (model, {"alg_params": {}}))
    monkeypatch.setattr(evaluator, "_outer_state_digest", lambda model: "frozen")
    result = evaluator.evaluate_preset(resolved, checkpoint, [101, 102], controller_seed=55)
    assert model.calls == 2
    assert result["seed_scheme"] == "constructor_seed_continuous_stream"
    assert all(episode["solver_seed"] is None for episode in result["episodes"])
    monkeypatch.setattr(evaluator, "_make_env", lambda resolved: pytest.fail("bundle preflight ran late"))
    with pytest.raises(ValueError, match="AMBI-XQC only"):
        evaluator.evaluate_preset(resolved, checkpoint, [101], controller_seed=55, bundle=object())


@pytest.mark.parametrize("allow", [False, True])
def test_nonfinite_diagnostics_are_explicit_and_strict_json(checkpoint_case, tmp_path, monkeypatch, allow):
    matrix, checkpoint = checkpoint_case
    initialize = evaluator._initialize_frozen_model

    def with_nonfinite(*args, **kwargs):
        model, run = initialize(*args, **kwargs)
        predict = model.predict

        def instrumented(*args, **kwargs):
            action = predict(*args, **kwargs)
            model.agent.last_inner_metrics["diagnostic_nan"] = float("nan")
            return action

        model.predict = instrumented
        return model, run

    monkeypatch.setattr(evaluator, "_initialize_frozen_model", with_nonfinite)
    bundle = tmp_path / "nonfinite"
    options = dict(bundle_dir=bundle, allow_nonfinite_metrics=allow)
    if allow:
        result = evaluator.evaluate_matrix(matrix, checkpoint, **options)["results"][0]
        assert result["nonfinite_model_metrics"] == {"diagnostic_nan": 4}
    else:
        with pytest.raises(RuntimeError, match="Non-finite model metrics"):
            evaluator.evaluate_matrix(matrix, checkpoint, **options)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["status"] == ("complete" if allow else "failed")
    events = _events(bundle, manifest["runs"][0])
    assert all(event["metrics"]["decision/diagnostic_nan"] is None for event in events)
    assert all(event["nonfinite"]["decision/diagnostic_nan"] == "nan" for event in events)


def test_frozen_invariant_failure_is_saved(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    initialize = evaluator._initialize_frozen_model

    def with_bad_mutation(*args, **kwargs):
        model, run = initialize(*args, **kwargs)
        predict = model.predict

        def instrumented(*args, **kwargs):
            action = predict(*args, **kwargs)
            with torch.no_grad():
                next(model.agent.xqc_controller.actor.parameters()).add_(0.001)
            return action

        model.predict = instrumented
        return model, run

    monkeypatch.setattr(evaluator, "_initialize_frozen_model", with_bad_mutation)
    bundle = tmp_path / "mutation"
    with pytest.raises(RuntimeError, match="Frozen evaluation invariant"):
        evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["outer_state_unchanged"] is False
