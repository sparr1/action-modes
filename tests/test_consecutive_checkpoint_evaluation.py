"""Scientific coverage of consecutive resets, RNG continuity and publication."""

import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.tdmpc2_core import frozen_graph
from slurm import graph_checkpoint_campaign as campaign
from utils import ambi_benchmark as benchmark, checkpoint_context
from utils import eval_series_data as data
from utils.ambi_research import load_preset_matrix, validate_preset_matrix
from utils.checkpoint_return_plot import episode_return_rows
from utils.evaluation_episodes import consecutive_protocol, validate_paired_starts


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_checkpoint_graph_eval_consecutive.json"


class StreamEnv:
    def __init__(self):
        self.action_space = self.observation_space = SimpleNamespace(seed=lambda seed: None)
        self.reset_seeds, self.starts, self.actions = [], [], []
        self.closed = False

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.start = float(self.rng.uniform())
        self.starts.append(self.start)
        self.steps = 0
        return np.array([self.start], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(float(action))
        self.steps += 1
        return np.array([self.start], dtype=np.float32), self.start + float(action), False, self.steps == 2, {}

    def close(self):
        self.closed = True


class StreamEngine:
    def __init__(self):
        self.rng = np.random.default_rng(999)
        self.resets, self.boundaries = [], 0

    def reset_for_evaluation(self, seed, **kwargs):
        self.resets.append(seed)
        self.rng = np.random.default_rng(seed)

    def reset_episode(self):
        self.boundaries += 1


class StreamModel:
    def __init__(self, resolved):
        self.cfg = SimpleNamespace(**{**resolved["algorithm_config"]["alg_params"], "device": "cpu"})
        self.agent = SimpleNamespace(inner_engine=StreamEngine(), num_updates=0, device="cpu",
                                     model=SimpleNamespace(critic_signature={}), last_inner_metrics={})
        self.graph, self.saved_graph = None, None
        self.global_seeds, self.predictions = [], []
        self.closed = False

    def _set_seed(self, seed):
        self.global_seeds.append(seed)

    def predict(self, observation, *, deterministic, episode_start, **kwargs):
        assert deterministic
        self.predictions.append((observation.copy(), episode_start))
        if episode_start:
            self.agent.inner_engine.reset_episode()
        if self.graph is not None:
            self.graph.runner.calls += 1
            action = self.graph.rng.uniform()
        else:
            action = self.agent.inner_engine.rng.uniform()
        if self.cfg.inner_operator == "none":
            action = 0.
        self.agent.last_inner_metrics = {
            "inner_cuda_graph_used": 1, "inner_critic_optimizer_steps": 128,
            "inner_actor_optimizer_steps": 32, "inner_model_steps": 3072, "inner_compile_fallback": 0,
        }
        return action, None

    def close(self):
        self.closed = True


class StreamGraph:
    """Exercise evaluator reset orchestration without pretending to run CUDA."""
    def __init__(self, model, observation, seed):
        self.model = model
        self.runner = SimpleNamespace(calls=0)
        self.rng = np.random.default_rng(seed)
        self.rng.uniform(size=4)  # capture/warmup draws must be discarded
        self.resets = []
        model.graph = model.saved_graph = self

    def reset_episode(self, seed):
        self.resets.append(seed)
        self.rng = np.random.default_rng(seed)

    def close(self):
        self.model.graph = None


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    envs, models = [], []
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    saved = json.loads((ROOT / "configs/dmcontrol/algs/ambi_gsde_squashed_real_steps.json").read_text())
    context = SimpleNamespace(source=tmp_path / "metadata.json", trial_run_params={**saved, "env": "DMControl-v0"},
                              experiment_params={"env_params": {"task": "humanoid-walk", "obs": "state", "render_mode": None}},
                              metadata={"checkpoint": {"step": 25000}})
    inventory = tmp_path / "checkpoint-inventory.json"
    inventory.write_text(json.dumps({"source_run": "entity/project/source", "checkpoints": [
        {"step": 25000, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]}))

    def make_env(resolved):
        envs.append(StreamEnv())
        return envs[-1]

    def make_model(resolved, *args, **kwargs):
        models.append(StreamModel(resolved))
        return models[-1], resolved["algorithm_config"]

    monkeypatch.setattr(evaluator, "_make_env", make_env)
    monkeypatch.setattr(evaluator, "_initialize_frozen_model", make_model)
    monkeypatch.setattr(evaluator, "_outer_state_digest", lambda model: "frozen")
    monkeypatch.setattr(evaluator, "_validate_checkpoint_contract", lambda *args: None)
    monkeypatch.setattr(checkpoint_context, "load_checkpoint_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(frozen_graph, "FrozenGraphController", StreamGraph)
    # These tests isolate episode/publication logic; no fictitious GPU measurements
    # are staged or published, and no fake model state reaches a real checkpoint.
    return SimpleNamespace(envs=envs, models=models, checkpoint=checkpoint, inventory=inventory)


def run_matrix(runtime, tmp_path, *, matrix=MATRIX, name="evaluation", bundle=True, **kwargs):
    directory = tmp_path / name
    payload = evaluator.evaluate_matrix(matrix, runtime.checkpoint,
        bundle_dir=directory if bundle else None, max_steps=2,
        checkpoint_inventory=runtime.inventory, source_run="entity/project/source", stage_results=False, **kwargs)
    return payload, directory


@pytest.mark.parametrize("bundle", [False, True])
def test_ten_consecutive_starts_and_controller_draws_with_no_warmup_leak(fake_runtime, tmp_path, bundle):
    payload, directory = run_matrix(fake_runtime, tmp_path, bundle=bundle)
    prior, graph = payload["results"]
    sampling = consecutive_protocol(load_preset_matrix(MATRIX)["evaluation"])
    campaign.validate_result(payload, 2, sampling=sampling)
    for env, model in zip(fake_runtime.envs, fake_runtime.models):
        assert env.reset_seeds == [55] + [None] * 9
        assert len(set(env.starts)) == 10
        assert env.closed and model.closed
        assert model.agent.inner_engine.resets == [55]
        assert model.global_seeds == [55]
        assert model.agent.inner_engine.boundaries == 10 + int(bundle)
        assert len(env.actions) == 20  # compilation/warmup is not an episode
    assert fake_runtime.envs[0].starts == fake_runtime.envs[1].starts
    np.testing.assert_array_equal(fake_runtime.envs[1].actions, np.random.default_rng(55).uniform(size=20))
    assert fake_runtime.models[1].saved_graph.resets == [55]
    assert graph["graph_replays"] == 20
    assert graph["return"]["count"] == 10
    assert "environment_seeds" not in graph
    assert [e["episode_index"] for e in graph["episodes"]] == list(range(10))
    assert all("seed" not in e and "solver_seed" not in e for e in graph["episodes"])
    assert graph["paired_return_delta_vs_reference"]["mean"] == pytest.approx(
        statistics.mean(g["return"] - p["return"] for p, g in zip(prior["episodes"], graph["episodes"])))
    if bundle:
        records = data.normalize_bundle(directory, checkpoint_inventory=fake_runtime.inventory)
        for record in records:
            assert record["metrics"]["eval/episodes"] == 10
            assert record["metrics"]["eval/return_sample_std"] == pytest.approx(
                statistics.stdev(e["return"] for e in record["episodes"]))
        rows = episode_return_rows(records)
        assert len(rows) == 20
        assert [row[3] for row in rows[:10]] == list(range(1, 11))
        assert all(row[2] == 55 and row[-1] == 10 for row in rows)
        assert records[1]["metrics"]["eval/paired_gain_mean"] == pytest.approx(
            graph["paired_return_delta_vs_reference"]["mean"])


def test_streams_restart_at_each_checkpoint_and_repeat_reproducibly(fake_runtime, tmp_path):
    first, _ = run_matrix(fake_runtime, tmp_path, bundle=False)
    second, _ = run_matrix(fake_runtime, tmp_path, bundle=False)
    for a, b in zip(first["results"], second["results"]):
        for key in ("return", "paired_return_delta_vs_reference", "episode_sampling"):
            assert a[key] == b[key]
        assert [ep["initial_observation_sha256"] for ep in a["episodes"]] == [ep["initial_observation_sha256"] for ep in b["episodes"]]


def test_native_inner_solver_also_advances_across_episodes(fake_runtime, tmp_path):
    matrix = load_preset_matrix(MATRIX)
    matrix["comparisons"]["controller"]["variants"]["graph"]["evaluation"]["execution_backend"] = "native"
    path = tmp_path / "native.json"
    path.write_text(json.dumps(matrix))
    payload, _ = run_matrix(fake_runtime, tmp_path, matrix=path)
    model = fake_runtime.models[1]
    assert model.saved_graph is None
    assert model.agent.inner_engine.resets == [55]
    np.testing.assert_array_equal(fake_runtime.envs[1].actions, np.random.default_rng(55).uniform(size=20))
    assert payload["results"][1]["return"]["count"] == 10


def test_legacy_five_seeds_still_reseed_each_environment_and_solver(fake_runtime, tmp_path):
    payload, directory = run_matrix(fake_runtime, tmp_path, matrix=campaign.MATRIX)
    seeds = [101, 102, 103, 104, 105]
    campaign.validate_result(payload, 2, seeds)
    graph = fake_runtime.models[1]
    solver_seeds = [benchmark.solver_seed(55, "episode", seed) for seed in seeds]
    assert fake_runtime.envs[1].reset_seeds == [101] + seeds  # old warmup convention
    assert graph.agent.inner_engine.resets == solver_seeds
    assert graph.saved_graph.resets == solver_seeds
    expected = [draw for seed in solver_seeds for draw in np.random.default_rng(seed).uniform(size=2)]
    np.testing.assert_array_equal(fake_runtime.envs[1].actions, expected)
    assert graph.global_seeds == []
    records = data.normalize_bundle(directory, checkpoint_inventory=fake_runtime.inventory)
    assert "episode_sampling" not in records[0]["identity"]["protocol"]
    assert len(episode_return_rows(records)) == 10


def test_matrix_overrides_preserve_consecutive_pairing(fake_runtime, tmp_path):
    payload, _ = run_matrix(fake_runtime, tmp_path, num_episodes=3, environment_seed=77)
    assert all(env.reset_seeds == [77, None, None] for env in fake_runtime.envs)
    assert all(result["episode_sampling"]["num_episodes"] == 3 for result in payload["results"])
    validate_paired_starts(*(result["episodes"] for result in payload["results"]))


@pytest.mark.parametrize("changes", [
    {"seeds": [55]}, {"environment_seed": True}, {"environment_seed": -1},
    {"num_episodes": 0}, {"num_episodes": 2.5}, {"episode_sampling": "unknown"},
])
def test_invalid_protocol_is_rejected_before_any_environment(changes):
    matrix = load_preset_matrix(MATRIX)
    matrix["evaluation"].update(changes)
    with pytest.raises(ValueError):
        validate_preset_matrix(matrix)


def test_seed_cli_cannot_silently_replace_consecutive_protocol(fake_runtime, tmp_path):
    with pytest.raises(ValueError, match="--seeds"):
        run_matrix(fake_runtime, tmp_path, seeds=[55])
    with pytest.raises(ValueError, match="require consecutive"):
        run_matrix(fake_runtime, tmp_path, matrix=campaign.MATRIX, num_episodes=10)
    assert not fake_runtime.envs


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reset_seed", "initial_state", "gain", "returned_seed"])
def test_publication_rejects_incomplete_or_mispaired_streams(fake_runtime, tmp_path, corruption):
    _, directory = run_matrix(fake_runtime, tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    run = manifest["runs"][1]
    if corruption == "missing":
        run["episodes"].pop()
    elif corruption == "duplicate":
        run["episodes"][-1]["episode_index"] = 0
    elif corruption == "reset_seed":
        run["episodes"][0]["environment_seed"] = 56
    elif corruption == "initial_state":
        run["episodes"][0]["initial_observation_sha256"] = "a" * 64
    elif corruption == "gain":
        run["episodes"][0]["paired_return_delta"] += 1
    else:
        run["result"]["episodes"][0]["controller_stream_seed"] = 56
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        data.normalize_bundle(path, checkpoint_inventory=fake_runtime.inventory)


def test_new_identity_matches_preflight_and_rejects_old_prior_reference(fake_runtime, tmp_path, monkeypatch):
    _, new_dir = run_matrix(fake_runtime, tmp_path)
    _, old_dir = run_matrix(fake_runtime, tmp_path, matrix=campaign.MATRIX, name="legacy")
    records = data.normalize_bundle(new_dir, checkpoint_inventory=fake_runtime.inventory)
    old_records = data.normalize_bundle(old_dir, checkpoint_inventory=fake_runtime.inventory)
    assert records[0]["identity"]["protocol"] != old_records[0]["identity"]["protocol"]
    manifest = json.loads((new_dir / "manifest.json").read_text())
    run = manifest["runs"][1]
    monkeypatch.setattr(data, "resolved_checkpoint_config", lambda *args, **kwargs: run["resolved_config"])
    identity = data.identity_for_ambi_checkpoint(
        manifest["checkpoint"], {"algorithm_config": run["config"],
        "evaluation": {"execution_backend": frozen_graph.PROTOCOL}}, manifest["protocol"], [], manifest["code"],
        path=new_dir / "manifest.json", inventory_path=fake_runtime.inventory)
    assert identity == records[1]["identity"]
    with pytest.raises(ValueError, match="protocol does not match"):
        benchmark.reference_returns(old_dir, manifest["checkpoint"]["sha256"], manifest["protocol"])
    with pytest.raises(ValueError, match="mix episode sampling"):
        episode_return_rows([records[0], old_records[0]])


def test_consecutive_reference_is_portable_and_paired(fake_runtime, tmp_path):
    _, prior_dir = run_matrix(fake_runtime, tmp_path, selectors=["controller/prior"], name="prior")
    _, graph_dir = run_matrix(fake_runtime, tmp_path, selectors=["controller/graph"],
                             name="graph", reference_bundle=prior_dir)
    record, = data.normalize_bundle(graph_dir, checkpoint_inventory=fake_runtime.inventory)
    assert record["metrics"]["eval/paired_episodes"] == 10
    assert record["provenance"]["reference"]["format"] == "ambi_consecutive_prior-v1"
    assert (graph_dir / "reference/manifest.json").is_file()


def test_campaign_pins_new_matrix_without_relabeling_old_campaigns():
    settings = campaign.evaluation_settings("consecutive")
    assert campaign.campaign_matrix(settings) == MATRIX
    assert settings["episode_sampling"]["num_episodes"] == 10
    assert settings["episode_sampling"]["environment_seed"] == 55
    assert settings["max_steps"] == 500
    assert campaign.campaign_matrix({}) == campaign.MATRIX
    assert "episode_sampling" not in campaign.evaluation_settings("independent_seeds")


def test_actual_inner_engine_episode_boundary_preserves_random_stream():
    from tests.test_frozen_graph_evaluation import make_agent

    agent = make_agent()
    engine = agent.inner_engine
    engine.reset_for_evaluation(55)
    reference = engine._new_rng(55)
    for _ in range(10):
        agent.reset()
        for name in engine.rng.STREAMS:
            actual = torch.rand(4, generator=engine.rng.generator(name))
            expected = torch.rand(4, generator=reference.generator(name))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("AMBI_RUN_REAL_DMCONTROL_TESTS") != "1",
                    reason="Set AMBI_RUN_REAL_DMCONTROL_TESTS=1 for real Humanoid reset validation")
def test_real_humanoid_reset_sequence_is_paired_despite_different_actions():
    from domains.dmcontrol import DMControlEnv

    envs = [DMControlEnv(task="humanoid-walk", obs="state") for _ in range(2)]
    starts = []
    try:
        for index in range(10):
            observations = [env.reset(**({"seed": 55} if index == 0 else {}))[0] for env in envs]
            np.testing.assert_array_equal(*observations)
            starts.append(observations[0].tobytes())
            # A short simulator check, without models, rendering or W&B.
            for _ in range(3):
                for env, action in zip(envs, (0., .1)):
                    env.step(np.full(env.action_space.shape, action, dtype=np.float32))
        assert len(set(starts)) == 10
    finally:
        for env in envs:
            env.close()
