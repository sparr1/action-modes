"""Protocol identity, prefix isolation and immutable entropy worker output."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import evaluate_ambi_entropy_probe as cli
from utils.ambi_benchmark import canonical_hash, read_json, solver_seed


ROOT = Path(__file__).resolve().parents[1]


class FakeEnv:
    def __init__(self, end_at=None):
        self.index = 0
        self.end_at = end_at
        self.actions = []
        self.action_space = SimpleNamespace(seed=lambda seed: None)
        self.observation_space = SimpleNamespace(seed=lambda seed: None)

    def reset(self, seed):
        self.index = 0
        return np.array([0, seed], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.index += 1
        return (np.array([self.index, 101], dtype=np.float32), float(self.index),
                self.index == self.end_at, False, {})

    def close(self):
        pass


class FakeModel:
    def __init__(self):
        self.calls = []
        self.resets = []
        self.cfg = SimpleNamespace(inner_rounds=8)
        self.agent = SimpleNamespace(inner_engine=SimpleNamespace(reset_for_evaluation=self._reset))

    def _reset(self, seed, **kwargs):
        self.resets.append((seed, kwargs))

    def predict(self, observation, **kwargs):
        self.calls.append((observation.copy(), kwargs))
        return np.array([observation[0] / 100], dtype=np.float32), None

    def close(self):
        pass


def test_prefix_roots_precede_actions_and_last_root_has_no_action(tmp_path):
    env, model = FakeEnv(), FakeModel()
    bank, timing = cli.collect_roots(env, model, environment_seed=101, controller_seed=55,
                                    decisions=[0, 2, 5], identity={"pinned": True})
    assert len(model.calls) == 6  # One warmup plus exactly five source decisions.
    assert len(env.actions) == timing["source_controller_solves"] == 5
    assert [root["observation"][0] for root in bank["roots"]] == [0, 2, 5]
    assert [root["return_before"] for root in bank["roots"]] == [0, 3, 15]
    assert model.resets == [(solver_seed(55, "episode", 101), {"reuse_action_pool": True})]
    assert all(options["deterministic"] for _, options in model.calls)
    assert model.calls[1][1]["episode_start"] is True
    assert all(not options["episode_start"] for _, options in model.calls[2:])
    assert bank["historical_trajectory_replay"] is False
    path = tmp_path / "bank.json"
    cli.atomic_json(path, bank)
    assert cli.load_root_bank(path, identity={"pinned": True}, environment_seed=101,
                              decisions=[0, 2, 5]) == bank
    with pytest.raises(ValueError, match="identity"):
        cli.load_root_bank(path, identity={"pinned": False}, environment_seed=101, decisions=[0, 2, 5])
    with pytest.raises(ValueError, match="coverage"):
        cli.load_root_bank(path, identity={"pinned": True}, environment_seed=101, decisions=[0, 2])
    bank["roots"][0]["observation"][0] = 999
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="hash"):
        cli.load_root_bank(path, identity={"pinned": True}, environment_seed=101, decisions=[0, 2, 5])


def test_early_termination_never_silently_drops_requested_roots():
    with pytest.raises(RuntimeError, match="cannot collect all requested roots"):
        cli.collect_roots(FakeEnv(end_at=2), FakeModel(), environment_seed=101,
                          controller_seed=55, decisions=[0, 5], identity={})


def _prepared(tmp_path, monkeypatch, source_name="ambi_full_h3_j8_fresh"):
    profile = read_json(cli.DEFAULT_PROFILE)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")
    profile["checkpoints"] = {"test": dict(checkpoint_sha256=cli._file_sha256(checkpoint),
        metadata_sha256=cli._file_sha256(metadata), step=100000, source_run="test/source")}
    profile["matrix"] = str(ROOT / "configs/research/ambi_early_h3_j8.json")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    source = read_json(ROOT / f"configs/dmcontrol/algs/{source_name}.json")
    context = SimpleNamespace(source=metadata, trial_run_params=source,
        experiment_params={"env_params": {"domain_name": "humanoid", "task_name": "walk"}},
        metadata={"checkpoint": {"step": 100000}})
    monkeypatch.setattr(cli, "load_checkpoint_context", lambda *args: context)
    args = cli.build_parser().parse_args(["--profile", str(profile_path), "--checkpoint-id", "test",
        "--checkpoint", str(checkpoint), "--environment-seed", "101", "--output", str(tmp_path / "out"),
        "--decisions", "0", "10", "--device", "cpu"])
    return args, profile, context


@pytest.mark.parametrize("source_name", ["ambi_full_h3_j8_fresh", "ambi_full_h3_mppi_aux_return"])
def test_pinned_recipe_preserves_outer_and_exact_inner_settings(tmp_path, monkeypatch, source_name):
    args, _, context = _prepared(tmp_path, monkeypatch, source_name)
    before = deepcopy(context.trial_run_params)
    _, _, _, _, resolved, decisions = cli._prepare(args)
    params = resolved["algorithm_config"]["alg_params"]
    assert decisions == [0, 10]
    assert params["inner_operator"] == "sac"
    assert (params["inner_rollout_horizon"], params["inner_rounds"]) == (3, 8)
    assert (params["inner_critic_updates_per_round"], params["inner_actor_updates_per_round"]) == (16, 4)
    assert params["inner_rollouts_per_round"] == 128
    assert params["inner_batch_size"] == 256
    assert params["inner_temperature_mode"] == "auto"
    assert params["inner_temperature_initialization"] == params["inner_target_entropy"] == "inherit_outer"
    assert params["inner_sac_critic_target"] == "reward_only"
    assert params["inner_critic_source"] == params["inner_horizon_critic_source"] == "aux_return"
    assert params["inner_actor_initialization"] == "prior"
    assert params["inner_execution_action"] == "policy_sample"
    assert cli.frozen_execution_action(resolved) == "mean"
    for key, value in before["alg_params"].items():
        if not key.startswith("inner_") and key != "wandb":
            assert params[key] == value
    assert context.trial_run_params == before
    args.plan_only = True
    assert cli.run_probe(args)["source_controller_solves"] == 10


def test_preflight_rejects_wrong_checkpoint_and_unapproved_roots(tmp_path, monkeypatch):
    args, _, _ = _prepared(tmp_path, monkeypatch)
    args.decisions = [9]
    with pytest.raises(ValueError, match="subset"):
        cli._prepare(args)
    args.decisions = [0]
    args.checkpoint.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="Checkpoint SHA256"):
        cli._prepare(args)


def test_worker_saves_complete_bank_before_diagnostics_and_enforces_freeze(tmp_path, monkeypatch):
    args, _, _ = _prepared(tmp_path, monkeypatch)
    env, model, calls = FakeEnv(), FakeModel(), []
    monkeypatch.setattr(cli, "_make_env", lambda resolved: env)
    monkeypatch.setattr(cli, "_initialize_frozen_model", lambda *a, **k: (model, {}))
    monkeypatch.setattr(cli, "_science_identity", lambda: dict(sha256="code", git_commit="commit", files={}))
    monkeypatch.setattr(cli, "_outer_state_digest", lambda model: "complete-outer-state")
    from utils import ambi_entropy_probe as core
    def probe(model_arg, observation, **options):
        assert model_arg is model
        assert len(model.calls) == 11  # Diagnostics cannot run before source prefix completes.
        bank = read_json(args.output / "root-bank.json")
        assert len(bank["transitions"]) == 10
        calls.append((observation.copy(), options))
        return dict(outer_state_unchanged=True, global_rng_unchanged=True)
    monkeypatch.setattr(core, "probe_entropy_root", probe)
    result = cli.run_probe(args)
    assert result["complete"] and result["outer_state_unchanged"]
    assert result["outer_state_sha256_before"] == result["outer_state_sha256_after_diagnostics"]
    assert len(calls) == 2
    assert [options["seed"] for _, options in calls] == [solver_seed(55, "entropy_probe", 101, x) for x in (0, 10)]
    assert all(options["exact_engine"] and options["actor_steps"] == 32 for _, options in calls)
    assert result["root_bank_id"] == read_json(args.output / "root-bank.json")["id"]
    assert result["sha256"] == canonical_hash({key: value for key, value in result.items() if key != "sha256"})
    with pytest.raises(FileExistsError, match="immutable"):
        cli.run_probe(args)

    args.output = tmp_path / "changed-outer"
    digest_calls = iter(["before", "mutated"])
    monkeypatch.setattr(cli, "_outer_state_digest", lambda model: next(digest_calls))
    with pytest.raises(RuntimeError, match="root collection"):
        cli.run_probe(args)
    assert not (args.output / "result.json").exists()
    assert read_json(args.output / "failure.json")["complete"] is False


def test_worker_rejects_probe_rng_failure(tmp_path, monkeypatch):
    args, _, _ = _prepared(tmp_path, monkeypatch)
    args.decisions = [0]
    monkeypatch.setattr(cli, "_make_env", lambda resolved: FakeEnv())
    monkeypatch.setattr(cli, "_initialize_frozen_model", lambda *a, **k: (FakeModel(), {}))
    monkeypatch.setattr(cli, "_science_identity", lambda: dict(sha256="code", git_commit="commit", files={}))
    monkeypatch.setattr(cli, "_outer_state_digest", lambda model: "same")
    from utils import ambi_entropy_probe as core
    monkeypatch.setattr(core, "probe_entropy_root", lambda *a, **k:
                        dict(outer_state_unchanged=True, global_rng_unchanged=False))
    with pytest.raises(RuntimeError, match="RNG preservation"):
        cli.run_probe(args)
    assert (args.output / "root-bank.json").is_file()
    assert not (args.output / "result.json").exists()
