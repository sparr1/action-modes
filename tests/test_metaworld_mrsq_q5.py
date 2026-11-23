"""Assembly full-ensemble aggregation and isolated resumable launch contract."""
from copy import deepcopy

import pytest
import torch

from slurm.metaworld_campaign import CONFIG_ROOT, main, production_algorithm, read_json
from tests.test_metaworld_ambi_config import MetaWorldSpaces
from tests.test_metaworld_oscar_launcher import launch  # noqa: F401


VARIANT = "mrsq_q5"
ALGORITHM = "ambi_state_mrsq_q5"
CHANGES = {
    "q_pair_size": 5,
    "outer_q_actor_reduction": "mean_all",
    "outer_q_target_reduction": "min_all",
    "inner_q_actor_reduction": "mean_all",
    "inner_q_target_reduction": "min_all",
    "mppi_terminal_q_reduction": "min_all",
}


def test_recipe_changes_only_the_six_reviewed_ensemble_fields():
    baseline = read_json(CONFIG_ROOT / "algs/ambi_state.json")
    expected = deepcopy(baseline)
    expected["alg_params"].update(CHANGES)
    actual = read_json(CONFIG_ROOT / f"algs/{ALGORITHM}.json")
    assert actual == expected
    assert actual["alg_params"]["num_q"] == 5
    assert baseline["alg_params"]["q_pair_size"] == 2
    assert actual["alg_params"]["wandb"] is False


def test_manifest_selects_one_separate_assembly_algorithm_without_protocol_changes():
    baseline = read_json(CONFIG_ROOT / "experiments/assembly.json")
    actual = read_json(CONFIG_ROOT / "experiments/assembly_mrsq_q5.json")
    expected = {**baseline, "configs": [ALGORITHM]}
    # Descriptive study text may identify the new comparison; execution must match.
    descriptive = {"study_type", "study_note"}
    assert {k: v for k, v in actual.items() if k not in descriptive} == {
        k: v for k, v in expected.items() if k not in descriptive
    }
    assert actual["trials"] == 1 and actual["env_params"]["task"] == "assembly"
    assert "overrides_alg" not in actual


@pytest.mark.parametrize("seed", [55, 56])
def test_online_variant_keeps_scientific_recipe_and_has_distinct_identity(tmp_path, seed):
    output = tmp_path / f"{ALGORITHM}.json"
    main(["production-config", "--task", "assembly", "--variant", VARIANT,
          "--seed", str(seed), "--output", str(output)])
    run = read_json(output)
    assert run == production_algorithm("assembly", seed=seed, variant=VARIANT)
    reference = read_json(CONFIG_ROOT / f"algs/{ALGORITHM}.json")
    reference["seed"] = seed
    assert {k: v for k, v in run.items() if k != "alg_params"} == {
        k: v for k, v in reference.items() if k != "alg_params"
    }
    params = run["alg_params"]
    assert {k: v for k, v in params.items() if not k.startswith("wandb")} == {
        k: v for k, v in reference["alg_params"].items() if not k.startswith("wandb")
    }
    assert params["wandb"] and params["wandb_mode"] == "online"
    assert params["wandb_event_indexed"] and params["wandb_step_every"] == 1000
    assert params["wandb_group"] == f"metaworld-assembly-mrsq-q5-s{seed}"
    assert params["wandb_run_name"] == f"Meta-World Assembly | Q5 actor mean / target min | seed{seed}"
    assert {"metaworld", "tdmpc2-04be337", "post-tanh", "sampled", "assembly",
            f"seed{seed}", "mrsq-q5", "mean_all", "min_all"} <= set(params["wandb_tags"])
    baseline = production_algorithm("assembly", seed=seed)
    assert baseline == production_algorithm("assembly", seed=seed, variant="baseline")
    assert baseline["alg_params"]["wandb_group"] != params["wandb_group"]
    assert baseline["alg_params"]["q_pair_size"] == 2


def test_real_wandb_adapter_receives_new_variant_identity_without_network(tmp_path):
    from tests.test_wandb_resume import _FakeWandb
    from utils.wandb_resume import WandbResumeContext
    from utils.wandb_utils import init_wandb

    params = production_algorithm("assembly", variant=VARIANT)["alg_params"]
    fake = _FakeWandb()
    run = init_wandb(params, default_project="unset", run_name="unset", wandb_module=fake,
                     resume_context=WandbResumeContext.new(run_id="assembly-q5", directory=tmp_path))
    try:
        call, = fake.init_calls
        assert (call["entity"], call["project"]) == ("rwgao_b-brown-university", "ambi")
        assert call["group"] == "metaworld-assembly-mrsq-q5-s55"
        assert call["name"] == "Meta-World Assembly | Q5 actor mean / target min | seed55"
        assert call["id"] == "assembly-q5" and call["resume"] == "never"
    finally:
        run.abort()


@pytest.mark.parametrize("variant", ["", "mrsq-q5", "mrsq_q15", "../baseline"])
def test_unknown_variant_is_rejected_before_config_output(tmp_path, variant):
    with pytest.raises(ValueError):
        production_algorithm("assembly", variant=variant)
    output = tmp_path / "not-created" / "config.json"
    with pytest.raises((ValueError, SystemExit)):
        main(["production-config", "--task", "assembly", "--variant", variant,
              "--output", str(output)])
    assert not output.parent.exists()


@pytest.mark.parametrize("task,variant", [("hand-insert", VARIANT), ("push", VARIANT),
                                           ("stick-pull", VARIANT), ("hand-insert", "mrsq_q10")])
def test_ensemble_variants_reject_other_tasks_before_config_output(tmp_path, task, variant):
    with pytest.raises(ValueError):
        production_algorithm(task, variant=variant)
    output = tmp_path / "not-created" / "config.json"
    with pytest.raises(ValueError):
        main(["production-config", "--task", task, "--variant", variant,
              "--output", str(output)])
    assert not output.parent.exists()


@pytest.mark.parametrize("variant", [VARIANT, "mrsq_q10"])
def test_ensemble_launcher_uses_new_lineage_then_resumes_it_without_touching_baseline(launch, variant):
    run, default_lineage = launch
    campaign = default_lineage.parents[3]
    baseline = campaign / "assembly/ambi_state/seed_55/lineage"
    baseline.mkdir(parents=True)
    (baseline / "LATEST").write_text("existing-baseline-generation")
    algorithm = f"ambi_state_{variant}"
    lineage = campaign / f"assembly/{algorithm}/seed_55/lineage"
    overrides = dict(AMBI_METAWORLD_VARIANT=variant, SLURM_ARRAY_TASK_ID="0")
    status, calls = run(**overrides)
    assert status == 0
    assert f"--variant {variant}" in calls
    assert f"/{algorithm}.json" in calls
    assert f"--run configs/metaworld/experiments/assembly_{variant}.json" in calls
    assert f"--lineage-dir {lineage} --resume-mode new" in calls
    assert "--resume-checkpoint-steps 25000" in calls and "--resume-checkpoint-minutes 30" in calls
    assert "--num-runs 1" in calls and "--trial-index" not in calls
    assert str(baseline) not in calls and calls.endswith("scheduler requeue 741_0\n")
    lineage.mkdir()
    (lineage / "LATEST").write_text("new-q5-generation")
    status, calls = run(**overrides, SLURM_RESTART_COUNT="1", TRAIN_STATUS="0")
    assert status == 0 and f"--lineage-dir {lineage} --resume-mode required" in calls
    assert "scheduler requeue" not in calls
    (lineage / "DONE").touch()
    status, calls = run(**overrides, SLURM_RESTART_COUNT="2")
    assert status == 0 and " done " in calls and "train " not in calls
    assert (baseline / "LATEST").read_text() == "existing-baseline-generation"


@pytest.mark.parametrize("variant,index", [("invalid", "0"), ("../baseline", "0"),
                                           (VARIANT, "1"), (VARIANT, "2"), (VARIANT, "3"),
                                           ("mrsq_q10", "1")])
def test_invalid_launcher_variant_or_task_fails_before_any_write(launch, variant, index):
    run, default_lineage = launch
    status, calls = run(AMBI_METAWORLD_VARIANT=variant, SLURM_ARRAY_TASK_ID=index)
    assert status != 0 and calls == ""
    assert not default_lineage.parents[3].exists()


def test_q5_unexpected_training_failure_does_not_requeue(launch):
    run, _ = launch
    status, calls = run(AMBI_METAWORLD_VARIANT=VARIANT, SLURM_ARRAY_TASK_ID="0", TRAIN_STATUS="6")
    assert status == 6 and "train " in calls and "scheduler" not in calls


def test_resolved_metaworld_ensemble_uses_all_heads_without_subset_sampling(monkeypatch):
    from RL.AMBITDMPC2 import AMBITDMPC2
    from RL.tdmpc2_core.common.q_representation import QRepresentation

    recipe = read_json(CONFIG_ROOT / f"algs/{ALGORITHM}.json")
    env = MetaWorldSpaces()
    try:
        learner = object.__new__(AMBITDMPC2)
        learner.env = env
        learner.run_params = {**recipe, "device": "cpu"}
        learner.custom_params = deepcopy(recipe["alg_params"])
        cfg = learner._build_cfg({**learner.custom_params, "device": "cpu"})
        assert cfg.action_dim == 4 and cfg.obs_shape == {"state": (39,)}
        assert cfg.num_q == cfg.q_pair_size == 5
        assert all(getattr(cfg, key) == value for key, value in CHANGES.items())
        assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
        assert cfg.critic_geometry_diagnostics and cfg.ent_coef == pytest.approx(0.0004)
        backend = QRepresentation.from_config(cfg)

        def forbidden(*args, **kwargs):
            pytest.fail("Full-ensemble reductions must not sample critic subsets")

        monkeypatch.setattr(backend, "_pair_indices", forbidden)
        values = torch.tensor([5., -1., 4., 2., 0.]).reshape(5, 1, 1).requires_grad_()
        torch.testing.assert_close(backend.reduce(values, "min_all"), values.min(0).values)
        mean = backend.reduce(values, "mean_all")
        torch.testing.assert_close(mean, values.mean(0))
        mean.sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, 0.2))
        assert backend.sample_pair_indices("cpu") is None
    finally:
        env.close()


def test_q10_recipe_changes_only_ensemble_size_and_uses_its_own_wandb_group():
    q5 = production_algorithm("assembly", variant=VARIANT)
    q10 = production_algorithm("assembly", variant="mrsq_q10")
    expected = {k: v for k, v in q5["alg_params"].items() if not k.startswith("wandb")}
    expected.update(num_q=10, q_pair_size=10)
    assert {k: v for k, v in q10["alg_params"].items() if not k.startswith("wandb")} == expected
    assert q10["alg_params"]["wandb_group"] == "metaworld-assembly-mrsq-q10-s55"
    q5_manifest = read_json(CONFIG_ROOT / "experiments/assembly_mrsq_q5.json")
    q10_manifest = read_json(CONFIG_ROOT / "experiments/assembly_mrsq_q10.json")
    for name in ("study_type", "study_note", "configs"):
        q5_manifest.pop(name)
    assert q10_manifest.pop("configs") == ["ambi_state_mrsq_q10"]
    for name in ("study_type", "study_note"):
        q10_manifest.pop(name)
    assert q10_manifest == q5_manifest


@pytest.mark.parametrize("task,index", [("push", "2"), ("stick-pull", "3")])
def test_q10_hydra_cells_preserve_task_and_shared_resume_lineage(launch, task, index):
    run, default_lineage = launch
    root = default_lineage.parents[3]
    q10 = production_algorithm(task, variant="mrsq_q10")
    params = q10["alg_params"]
    assert q10["seed"] == 55 and q10["total_steps"] == 1_000_000
    assert params["num_q"] == params["q_pair_size"] == 10
    assert params["wandb_group"] == f"metaworld-{task}-mrsq-q10-s55"
    baseline = read_json(CONFIG_ROOT / f"experiments/{task}.json")
    manifest = read_json(CONFIG_ROOT / f"experiments/{task}_mrsq_q10.json")
    assert manifest["configs"] == ["ambi_state_mrsq_q10"]
    for key in baseline.keys() - {"configs", "study_type", "study_note"}:
        assert baseline[key] == manifest[key]
    lineage = root / task / "ambi_state_mrsq_q10/seed_55/lineage"
    overrides = dict(AMBI_METAWORLD_VARIANT="mrsq_q10", SLURM_ARRAY_TASK_ID=index,
                     AMBI_METAWORLD_RESUME_MINUTES="15")
    status, calls = run(**overrides)
    assert status == 0 and f"--lineage-dir {lineage} --resume-mode new" in calls
    assert f"--run configs/metaworld/experiments/{task}_mrsq_q10.json" in calls
    assert "--resume-checkpoint-minutes 15" in calls
    lineage.mkdir()
    (lineage / "LATEST").write_text("committed")
    status, calls = run(**overrides, SLURM_RESTART_COUNT="1", TRAIN_STATUS="0")
    assert status == 0 and f"--lineage-dir {lineage} --resume-mode required" in calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires scheduled CUDA GPU")
@pytest.mark.parametrize("heads", [5, 10])
def test_cuda_full_ensemble_metaworld_capture_matches_native_solve(monkeypatch, heads):
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from tests import test_inner_entropy_graph as checks

    original = checks.make_agent

    def metaworld_agent(entropy, scope, update, *, device, compiled):
        cfg = deepcopy(original(entropy, scope, update).cfg)
        cfg.action_dim = 4
        cfg.action_dims = [4]
        cfg.obs_shape = {"state": (39,)}
        cfg.obs_shapes = [{"state": (39,)}]
        cfg.ent_coef = cfg.inner_temperature = 0.0004
        cfg.target_entropy = cfg.inner_target_entropy = -4.
        cfg.num_q = heads
        for key, value in CHANGES.items():
            setattr(cfg, key, value)
        cfg.q_pair_size = heads
        cfg.device = device
        cfg.compile = cfg.compile_strict = compiled
        agent = AMBITDMPC2Agent(cfg)
        agent.model.eval()
        agent.actor_loss_scale.fill_(4.)
        return agent

    monkeypatch.setattr(checks, "make_agent", metaworld_agent)
    checks.test_cuda_capture_matches_uncaptured_normalized_solve("squashed", "action", "per_update")
