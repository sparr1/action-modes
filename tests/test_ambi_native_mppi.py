"""Frozen AMBI adaptation of the native TD-MPC2 planning protocol."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.ambi_mppi import FrozenAMBIMPPIController, resolve_mppi_settings
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.mppi import MPPIModelCallbacks, mppi_plan
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_ambi_benchmark_evaluation import checkpoint_matrix
from tests.test_ambi_inner_benchmark_launcher import launch_env


SMALL = {"horizon": 2, "num_samples": 8, "num_elites": 3,
         "num_pi_trajs": 2, "iterations": 2}


def _prior(**overrides):
    model = _tiny_model(inner_operator="none", inner_rounds=None,
                        inner_rollouts_per_round=None, inner_updates_per_round=None,
                        inner_temperature_mode="inherit_outer", **overrides)
    model.agent.model.eval()
    return model


@pytest.fixture
def prior():
    model = _prior()
    yield model
    model.env.close()


@pytest.mark.parametrize("seed", [1, 7, 49])
def test_native_action_matches_direct_upstream_equations(seed):
    """Verify weighted elite sampling, including its private RNG consumption."""
    expected_generator = torch.Generator().manual_seed(seed)
    mean, std = torch.zeros(2, 1), torch.full((2, 1), 2.0)
    for _ in range(2):
        actions = (mean[:, None] + std[:, None] *
                   torch.randn(2, 8, 1, generator=expected_generator)).clamp(-1, 1)
        values = actions[0] + 0.99 * actions[1]
        indices = values.squeeze(1).topk(3).indices
        elite_values, elite_actions = values[indices], actions[:, indices]
        score = (0.5 * (elite_values - elite_values.max(0).values)).exp()
        score = score / score.sum(0)
        mean = (score[None] * elite_actions).sum(1) / (score.sum(0) + 1e-9)
        std = ((score[None] * (elite_actions - mean[:, None]).square()).sum(1) /
               (score.sum(0) + 1e-9)).sqrt().clamp(0.05, 2)
    gumbels = -torch.empty_like(score.squeeze(1)).exponential_(
        generator=expected_generator).log()
    selected = (score.squeeze(1).log() + gumbels).softmax(0).argmax(-1)

    callbacks = MPPIModelCallbacks(
        action_dim=1, dynamics=lambda z, a: z, reward=lambda z, a: a,
        policy=lambda z, *, generator: z.new_zeros((z.shape[0], 1)),
        terminal_q=lambda z, a, *, reduction, generator: z.new_zeros((z.shape[0], 1)),
    )
    generator = torch.Generator().manual_seed(seed)
    result = mppi_plan(torch.zeros(1, 2), callbacks=callbacks, horizon=2,
                       iterations=2, num_samples=8, num_elites=3, num_pi_trajs=0,
                       temperature=0.5, min_std=0.05, max_std=2, discount=0.99,
                       q_reduction="mean_pair", generator=generator, eval_mode=True,
                       action_selection="tdmpc2")
    torch.testing.assert_close(result.action, elite_actions[0, selected], rtol=0, atol=0)
    torch.testing.assert_close(result.next_mean, mean, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_humanoid_defaults_keep_authored_and_effective_iterations_distinct():
    assert resolve_mppi_settings(None, action_dim=21) == {
        "horizon": 3, "iterations": 6, "effective_iterations": 8,
        "num_samples": 512, "num_elites": 64, "num_pi_trajs": 24,
        "min_std": 0.05, "max_std": 2.0, "temperature": 0.5,
    }
    assert resolve_mppi_settings({"iterations": 2}, action_dim=20)["effective_iterations"] == 4
    assert resolve_mppi_settings(None, action_dim=19)["effective_iterations"] == 6


@pytest.mark.parametrize("settings", [
    {"iterations": True}, {"iterations": 0}, {"horizon": 1.2},
    {"num_samples": 2, "num_elites": 3}, {"num_pi_trajs": 513},
    {"temperature": float("nan")}, {"min_std": 4}, {"max_std": 0},
    {"effective_iterations": 8}, {"q_reduction": "min_all"},
])
def test_settings_reject_invalid_or_unsupported_semantics(settings):
    with pytest.raises(ValueError):
        resolve_mppi_settings(settings, action_dim=21)


@pytest.mark.parametrize(("representation", "num_q"), [("scalar", 2), ("distributional", 5)])
def test_frozen_planning_preserves_outer_inner_modes_and_every_global_rng(representation, num_q):
    model = _prior(q_representation=representation, num_q=num_q, dropout=0.2)
    try:
        agent = model.agent
        planner = FrozenAMBIMPPIController(agent, SMALL)
        observation, _ = model.env.reset(seed=17)
        outer_before = deepcopy(agent.checkpoint_state())
        inner_before = deepcopy(agent.inner_engine.training_state_dict())
        modes_before = [module.training for module in agent.model.modules()]
        python_before, numpy_before = random.getstate(), np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        def episode(seed):
            planner.reset(seed)
            assert planner.previous_mean is None and planner.action_index == 0
            actions = torch.stack([planner.act(observation) for _ in range(3)])
            assert planner.previous_mean is not None and planner.action_index == 3
            return actions

        first = episode(101)
        episode(500)
        repeated = episode(101)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        assert first.device.type == "cpu" and first.shape == (3, 1)
        assert torch.isfinite(first).all() and (first.abs() <= 1).all()
        _assert_tree_equal(agent.checkpoint_state(), outer_before)
        _assert_tree_equal(agent.inner_engine.training_state_dict(), inner_before)
        assert [module.training for module in agent.model.modules()] == modes_before
        assert random.getstate() == python_before
        assert np.random.get_state()[0] == numpy_before[0]
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        assert np.random.get_state()[2:] == numpy_before[2:]
        assert torch.equal(torch.get_rng_state(), torch_before)
        metrics = agent.last_inner_metrics
        assert metrics["inner_model_steps"] == 2 * (2 - 1) + 2 * 8 * 2 == 34
        assert metrics["planner_policy_model_steps"] == 2
        assert metrics["planner_candidate_model_steps"] == 32
        assert metrics["inner_mppi_iterations"] == 2
        assert metrics["inner_policy_evaluations"] == 2 * 2 + 2 * 8
        assert metrics["inner_q_evaluations"] == 2 * 8
        for key in ("inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                    "inner_temperature_optimizer_steps", "inner_critic_target_updates"):
            assert metrics[key] == 0
        assert agent.last_inner_rollout_lengths == []
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        model.env.close()


def test_terminal_q_uses_online_pair_mean_sampled_prior_and_raw_reward_without_entropy(prior, monkeypatch):
    agent = prior.agent
    planner = FrozenAMBIMPPIController(agent, SMALL)
    q_calls, policy_calls = [], []

    def reward_logits(joint):
        logits = joint.new_full((joint.shape[0], agent.cfg.num_bins), -100)
        logits[:, -2] = 100
        return logits

    def sampled_policy(z, task=None, *, generator=None, deterministic=False, **kwargs):
        assert not deterministic
        assert generator is planner.generator
        policy_calls.append(z.shape[0])
        return z.new_full((z.shape[0], agent.cfg.action_dim), 0.375)

    def q(z, action, task=None, *, reduction=None, target=False, generator=None, **kwargs):
        assert reduction == "mean_pair" and target is False
        assert generator is planner.generator
        torch.testing.assert_close(action, torch.full_like(action, 0.375))
        q_calls.append(z.shape[0])
        return z.new_full((z.shape[0], 1), 3)

    monkeypatch.setattr(agent.model, "reward_from_joint", reward_logits)
    monkeypatch.setattr(agent.model, "pi_action", sampled_policy)
    monkeypatch.setattr(agent.model, "pi", lambda *a, **k: pytest.fail("No entropy/log-probability evaluation"))
    monkeypatch.setattr(agent.model, "Q", q)
    raw_reward = float(td_math.two_hot_inv(reward_logits(torch.zeros(1, 1)), agent.cfg).item())
    observation, _ = prior.env.reset(seed=13)
    planner.act(observation)
    expected = raw_reward * (1 + agent.discount) + agent.discount ** 2 * 3
    assert agent.last_inner_metrics["planner_value_mean"] == pytest.approx(expected, rel=1e-6)
    assert policy_calls == [2, 2, 8, 8]
    assert q_calls == [8, 8]
    assert planner.protocol["terminal_value_source"] == "online_ambi_q_mean_pair"
    assert planner.protocol["terminal_value_semantics"] == "learned_soft_q_tail_without_entropy_correction"
    assert planner.protocol["reward_units"] == "raw_environment_reward"
    assert planner.protocol["terminal_action"] == "frozen_prior_sample"


def test_controller_guards_training_submodules_and_observation_shape(prior):
    planner = FrozenAMBIMPPIController(prior.agent, SMALL)
    with pytest.raises(ValueError, match="shape"):
        planner.act(torch.zeros(2))
    child = next(module for module in prior.agent.model.modules() if isinstance(module, torch.nn.Linear))
    child.train()
    with pytest.raises(ValueError, match="eval mode"):
        FrozenAMBIMPPIController(prior.agent, SMALL)
    with pytest.raises(RuntimeError, match="training-mode"):
        planner.act(torch.zeros(3))
    assert child.training  # Reject rather than silently change caller modes.
    prior.agent.model.eval()
    prior.agent.cfg.inner_operator = "sac"
    with pytest.raises(ValueError, match="frozen prior"):
        FrozenAMBIMPPIController(prior.agent, SMALL)


@pytest.mark.parametrize("seed", [True, 1.5, "101"])
def test_controller_rejects_invalid_episode_seeds(prior, seed):
    with pytest.raises(ValueError, match="integer"):
        FrozenAMBIMPPIController(prior.agent, SMALL).reset(seed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_planning_preserves_device_rng_and_repeatability():
    model = _prior(device="cuda", dropout=0.2)
    try:
        planner = FrozenAMBIMPPIController(model.agent, SMALL)
        observation, _ = model.env.reset(seed=17)
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        planner.reset(101)
        first = torch.stack([planner.act(observation) for _ in range(3)])
        planner.reset(101)
        repeated = torch.stack([planner.act(observation) for _ in range(3)])
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        assert first.device.type == "cpu"
        assert planner.previous_mean.device.type == "cuda"
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    finally:
        model.env.close()


@pytest.fixture
def mppi_matrix(checkpoint_matrix):
    checkpoint, path = checkpoint_matrix
    matrix = json.loads(path.read_text())
    prior_variant = deepcopy(matrix["comparisons"]["budget"]["variants"]["prior"])
    mppi_variant = deepcopy(prior_variant)
    mppi_variant["evaluation_controller"] = {"type": "mppi", "params": SMALL}
    matrix["comparisons"] = {"controller": {"reference": "prior", "variants": {
        "prior": prior_variant, "mppi": mppi_variant}}}
    matrix["evaluation"]["default_presets"] = ["controller/mppi"]
    path.write_text(json.dumps(matrix))
    return checkpoint, path


def test_native_evaluator_pairs_prior_and_preserves_prior_actions(mppi_matrix, tmp_path):
    import evaluate_ambi_checkpoint as evaluator
    from utils.ambi_benchmark import read_json

    checkpoint, matrix = mppi_matrix
    baseline = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/prior"])
    bundle = tmp_path / "paired"
    paired = evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=bundle,
                                       selectors=["controller/mppi", "controller/prior"])
    runs = read_json(bundle / "manifest.json")["runs"]
    assert [run["selector"] for run in runs] == ["controller/prior", "controller/mppi"]
    assert all(result["outer_state_unchanged"] for result in paired["results"])
    assert [episode["return"] for episode in runs[0]["episodes"]] == [
        episode["return"] for episode in baseline["results"][0]["episodes"]]
    mppi = runs[1]
    assert mppi["action_rule"] == "weighted_elite_gumbel_no_execution_noise"
    assert mppi["evaluation_controller"]["settings"]["effective_iterations"] == 2
    assert mppi["evaluation_controller"]["protocol"]["terminal_value_source"] == "online_ambi_q_mean_pair"
    assert mppi["result"]["paired_return_delta_vs_prior"]["count"] == 2
    for episode, reference in zip(mppi["episodes"], runs[0]["episodes"]):
        assert episode["paired_return_delta"] == pytest.approx(episode["return"] - reference["return"])
        assert episode["length"] == 3 and episode["truncated"]
    from report_ambi_benchmark import load_bundles
    load_bundles([bundle])


def test_native_evaluator_episode_order_does_not_change_results(mppi_matrix, tmp_path):
    import evaluate_ambi_checkpoint as evaluator

    checkpoint, matrix = mppi_matrix
    returns = []
    for index, seeds in enumerate(([101, 102], [102, 101])):
        result = evaluator.evaluate_matrix(matrix, checkpoint, seeds=seeds,
                                           bundle_dir=tmp_path / str(index))
        returns.append({row["seed"]: row["return"] for row in result["results"][0]["episodes"]})
    assert returns[0] == returns[1]


@pytest.mark.parametrize("options", [
    {"bank_only": True}, {"save_root_bank": "bank.json"}, {"root_bank_path": "bank.json"},
])
def test_native_evaluator_rejects_bank_paths_before_environment_creation(mppi_matrix, tmp_path, monkeypatch, options):
    import evaluate_ambi_checkpoint as evaluator

    checkpoint, matrix = mppi_matrix
    monkeypatch.setattr(evaluator, "_make_env", lambda *a, **k: pytest.fail("Invalid request made environment"))
    with pytest.raises(ValueError, match="observation-bank probes are unsupported"):
        evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=tmp_path / "bad", **options)
    assert not (tmp_path / "bad").exists()


def test_native_matrix_rejects_training_materialization_and_active_inner_learner(mppi_matrix, tmp_path):
    from utils.ambi_research import PresetMatrixError, materialize_presets, resolve_preset
    from utils.checkpoint_context import load_checkpoint_context

    checkpoint, path = mppi_matrix
    context = load_checkpoint_context(checkpoint)
    with pytest.raises(PresetMatrixError, match="cannot be materialized as a training"):
        materialize_presets(path, tmp_path / "materialized", selectors=["controller/mppi"],
                            checkpoint_context=context)
    assert not (tmp_path / "materialized").exists()
    matrix = json.loads(path.read_text())
    matrix["comparisons"]["controller"]["variants"]["mppi"]["alg_params"]["inner_operator"] = "sac"
    with pytest.raises(PresetMatrixError, match="inner_operator='none'"):
        resolve_preset(path, "controller/mppi", matrix=matrix, checkpoint_context=context)


@pytest.mark.parametrize("controller", [
    {"type": "sac"}, {"type": "mppi", "params": {"q_reduction": "min_all"}},
    {"type": "mppi", "params": {"iterations": True}},
])
def test_native_matrix_rejects_unsupported_controller_settings(mppi_matrix, controller):
    from utils.ambi_research import PresetMatrixError, load_preset_matrix

    _, path = mppi_matrix
    matrix = json.loads(path.read_text())
    matrix["comparisons"]["controller"]["variants"]["mppi"]["evaluation_controller"] = controller
    path.write_text(json.dumps(matrix))
    with pytest.raises(PresetMatrixError):
        load_preset_matrix(path)


def test_checked_in_native_matrix_pins_backbone_and_five_episode_protocol():
    from utils.ambi_research import load_preset_matrix

    path = Path(__file__).resolve().parents[1] / "configs/research/ambi_humanoid_native_mppi_benchmark.json"
    matrix = load_preset_matrix(path)
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/u13m14st"
    assert matrix["base_alg_config"] == "checkpoint"
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["max_steps"] == 500
    assert matrix["evaluation"]["default_presets"] == ["controller/mppi"]
    params = matrix["shared_alg_params"]
    assert params["inner_operator"] == "none"
    assert params["inner_actor_writeback_coef"] == params["inner_critic_writeback_coef"] == 0
    selected = matrix["comparisons"]["controller"]["variants"]["mppi"]["evaluation_controller"]
    settings = resolve_mppi_settings(selected["params"], action_dim=21)
    assert (settings["horizon"], settings["num_samples"], settings["effective_iterations"]) == (3, 512, 8)


def test_native_evaluator_reuses_external_prior_without_repeating_or_changing_it(mppi_matrix, tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    from utils.ambi_benchmark import read_json

    checkpoint, matrix = mppi_matrix
    prior_bundle = tmp_path / "prior"
    baseline = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/prior"],
                                         bundle_dir=prior_bundle)

    def hashes():
        return {str(path.relative_to(prior_bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in prior_bundle.rglob("*") if path.is_file()}

    prior_hashes = hashes()
    initialized = []
    initialize = evaluator._initialize_frozen_model

    def record_initialization(resolved, *args, **kwargs):
        initialized.append(resolved["selector"])
        return initialize(resolved, *args, **kwargs)

    monkeypatch.setattr(evaluator, "_initialize_frozen_model", record_initialization)
    candidate_bundle = tmp_path / "mppi"
    candidate = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/mppi"],
                                           bundle_dir=candidate_bundle, reference_bundle=prior_bundle)
    assert initialized == ["controller/mppi"]
    assert hashes() == prior_hashes
    manifest = read_json(candidate_bundle / "manifest.json")
    assert manifest["status"] == "complete"
    assert [run["selector"] for run in manifest["runs"]] == ["controller/mppi"]
    assert candidate["results"][0]["paired_return_delta_vs_prior"]["count"] == 2
    references = {episode["seed"]: episode["return"]
                  for episode in baseline["results"][0]["episodes"]}
    for episode in manifest["runs"][0]["episodes"]:
        assert episode["paired_return_delta"] == pytest.approx(episode["return"] - references[episode["seed"]])


def test_native_metadata_only_spec_matches_executed_record_identity(mppi_matrix, tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    from RL.AMBITDMPC2 import AMBITDMPC2
    from utils import ambi_benchmark, eval_series_data as data

    checkpoint, matrix = mppi_matrix
    bundle = tmp_path / "executed"
    evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    executed = manifest["runs"][0]
    # This real-model fixture is Pendulum. Use the actually executed resolver
    # output in place of the production DMControl metadata-only shape adapter.
    monkeypatch.setattr(data, "resolved_checkpoint_config", lambda *a, **k: executed["resolved_config"])
    monkeypatch.setattr(data, "_source", lambda *a, **k: (
        "rwgao_b-brown-university/ambi/u13m14st", {"checkpoint_source_verified": True}, {}))
    monkeypatch.setattr(data, "scientific_identity", lambda algorithm, controller, *a, **k: {
        "algorithm": algorithm, "controller": controller, "implementation": "fixture"})
    record, = data.normalize_bundle(bundle)
    monkeypatch.setattr(ambi_benchmark, "code_identity", lambda: {"commit": "fixture", "dirty": False})
    monkeypatch.setattr(evaluator, "_make_env", lambda *a, **k: pytest.fail("Specification constructed environment"))
    monkeypatch.setattr(evaluator, "_initialize_frozen_model", lambda *a, **k: pytest.fail("Specification loaded checkpoint"))
    monkeypatch.setattr(AMBITDMPC2, "__init__", lambda *a, **k: pytest.fail("Specification constructed learner"))
    monkeypatch.setattr(FrozenAMBIMPPIController, "__init__", lambda *a, **k: pytest.fail("Specification constructed planner"))
    result = evaluator.evaluate_matrix(matrix, checkpoint, eval_series_spec_dir=tmp_path / "specs")
    assert result["mode"] == "evaluation_series_specifications"
    spec = json.loads(Path(result["specs"]["controller/mppi"]).read_text())
    assert spec["identity"] == record["identity"]
    assert spec["identity"]["planner"]["type"] == "mppi"
    assert spec["identity"]["planner"]["action_rule"] == "weighted_elite_gumbel_no_execution_noise"
    assert spec["identity"]["planner"]["semantics"]["discount"] == executed["evaluation_controller"]["protocol"]["discount"]


NATIVE_LAUNCHER = Path(__file__).resolve().parents[1] / "slurm/run_ambi_prior_mppi_eval_oscar.sbatch"


@pytest.mark.parametrize("checkpoint_index", [1, 3, 5])
def test_native_production_launcher_runs_only_mppi_with_existing_reference_and_run_assignment(launch_env, checkpoint_index):
    env = launch_env
    env["SLURM_ARRAY_TASK_ID"] = str(checkpoint_index)
    subprocess.run(["bash", str(NATIVE_LAUNCHER)], env=env, check=True, capture_output=True, text=True)
    evaluation, report = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    step = checkpoint_index * 100000
    assert evaluation[0] == "evaluate_ambi_checkpoint.py"
    assert [evaluation[i + 1] for i, arg in enumerate(evaluation) if arg == "--preset"] == ["controller/mppi"]
    assert evaluation[evaluation.index("--checkpoint") + 1] == f"{env['AMBI_CHECKPOINT_PREFIX']}{step}"
    assert evaluation[evaluation.index("--checkpoint-inventory") + 1] == env["CHECKPOINT_MANIFEST"]
    assert evaluation[evaluation.index("--seeds") + 1:evaluation.index("--seeds") + 6] == ["101", "102", "103", "104", "105"]
    assert evaluation[evaluation.index("--max-steps") + 1] == "500"
    reference = f"{env['AMBI_BENCHMARK_REFERENCE_ROOT']}/step_{step}/prior"
    assert evaluation[evaluation.index("--reference-bundle") + 1] == reference
    assert evaluation[evaluation.index("--eval-run-map") + 1] == env["EVAL_RUN_MAP"]
    assert "--wandb" not in evaluation
    assert report[0] == "report_ambi_benchmark.py"
    assert [report[i + 1] for i, arg in enumerate(report) if arg == "--bundle"] == [
        reference, f"{env['AMBI_BENCHMARK_OUTPUT_ROOT']}/step_{step}/bundle"]


def test_native_smoke_launcher_runs_short_pair_without_publication_or_reference(launch_env):
    env = launch_env
    env.pop("EVAL_RUN_MAP")
    env.pop("AMBI_BENCHMARK_REFERENCE_ROOT")
    subprocess.run(["bash", str(NATIVE_LAUNCHER), "--smoke"], env=env, check=True, capture_output=True, text=True)
    evaluation, report = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert [evaluation[i + 1] for i, arg in enumerate(evaluation) if arg == "--preset"] == [
        "controller/prior", "controller/mppi"]
    assert evaluation[evaluation.index("--seeds") + 1:evaluation.index("--seeds") + 3] == ["101", "102"]
    assert evaluation[evaluation.index("--max-steps") + 1] == "3"
    assert not {"--eval-run-map", "--wandb", "--reference-bundle"} & set(evaluation)
    assert report.count("--bundle") == 1


@pytest.mark.parametrize("missing", ["EVAL_RUN_MAP", "AMBI_BENCHMARK_REFERENCE_ROOT", "reference_manifest"])
def test_native_production_launcher_rejects_missing_run_or_reference_before_evaluation(launch_env, missing):
    env = launch_env
    if missing == "reference_manifest":
        (Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / "step_300000/prior/manifest.json").unlink()
    else:
        env.pop(missing)
    result = subprocess.run(["bash", str(NATIVE_LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not Path(env["TEST_CALLS"]).exists()
