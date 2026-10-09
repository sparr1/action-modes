"""Campaign scheduling, provenance and full-episode publication gates."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_curve_campaign as campaign
from utils.ambi_benchmark import canonical_hash, solver_seed


CONFIG = campaign.ROOT / "configs/research/ambi_transfer_checkpoint_curves.json"
J6_CONFIG = CONFIG.parent / "ambi_transfer_checkpoint_j6_after500k_curves.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    rows = []
    for step in (25000, 575000, 2000000):
        checkpoint = tmp_path / "checkpoints" / str(step)
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_text(f"checkpoint {step}")
        metadata_path = Path(str(checkpoint) + ".metadata.json")
        save(metadata_path, dict(schema_version=1,
            checkpoint=dict(kind="periodic", step=step, episode=1, best_score=None, best_window=100),
            trial_run_params=dict(alg="AMBITDMPC2/AMBITDMPC2", env="DMControl/DMControl", alg_params={}),
            experiment_params=dict(env_params=dict(task="humanoid-walk"))))
        rows.append(dict(step=step, checkpoint=str(checkpoint), checkpoint_sha256=campaign.digest(checkpoint),
                         metadata_sha256=campaign.digest(metadata_path), prior_reference=f"prior/{step}",
                         prior_pin=dict(selector="reference/prior", manifest_sha256="prior-digest")))
    monkeypatch.setattr(campaign, "ANCHOR_SHA", rows[1]["checkpoint_sha256"])
    monkeypatch.setattr(campaign, "ANCHOR_METADATA_SHA", rows[1]["metadata_sha256"])
    inventory = tmp_path / "inventory.json"
    save(inventory, dict(schema_version=1, source_run=campaign.SOURCE_RUN, checkpoints=rows))
    source = dict(source_commit="tested-commit", source_tree="tested-tree", source_dir=str(campaign.ROOT),
                  scientific_source=campaign.science_identity())
    monkeypatch.setattr(campaign, "source", lambda expected_sha=None: deepcopy(source))
    args = SimpleNamespace(root=tmp_path / "campaign", config=CONFIG, inventory=inventory,
                           settings=["h1_j2_bernoulli_a075_c075", "h1_j2_fresh"],
                           expected_source_sha="tested-commit", reuse_575k_root=None)
    value = campaign.prepare(args)
    return args, value


def write_result(root, definition, cell, *, smoke=False):
    """Construct complete saved evidence; mutations test each independent gate."""
    checkpoint = definition["checkpoints"][cell["checkpoint_index"]]
    directory = root / "smoke" / cell["name"] if smoke else Path(cell["result_dir"])
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.checkpoint_context import load_checkpoint_context
    from utils.transfer_campaign import load_campaign, resolved_cell
    recipe = load_campaign(cell["config_path"])
    context = load_checkpoint_context(checkpoint["checkpoint"], metadata_path=checkpoint["metadata"])
    matrix = load_preset_matrix(recipe["base_matrix_path"])
    base = resolve_preset(recipe["base_matrix_path"], recipe["base_preset"], matrix=matrix, checkpoint_context=context)
    resolved = resolved_cell(base, recipe, horizon=cell["H"], rounds=cell["J"], arm=cell["arm"])
    assert canonical_hash(resolved) == cell["resolved_sha256"]
    seeds = definition["smoke_seeds"] if smoke else definition["seeds"]
    limit = definition["smoke_steps"] if smoke else definition["max_steps"]
    shared = dict(protocol="inner-sac-transfer-discovery-v1", cell_id=cell["setting_id"], arm=cell["arm"],
                  horizon=cell["H"], rounds=cell["J"], checkpoint_sha256=checkpoint["checkpoint_sha256"],
                  seeds=seeds, smoke=smoke, source=dict(git_head=definition["source_commit"], **definition["scientific_source"]))
    manifest = dict(**shared, checkpoint_step=cell["step"], controller_seed=55, max_steps=limit,
                    compile=True, arm_definition=cell["arm_definition"], resolved=resolved,
                    campaign_sha256=campaign.digest(cell["config_path"]), diagnostics=definition["diagnostics"],
                    metric_policy="all_scalars", metric_coverage=dict(policy="all_scalars", computed_scalars_only=True,
                    extra_solver_probes=False, per_update_traces=False))
    episodes = []
    for seed in seeds:
        episode = dict(seed=seed, solver_seed=solver_seed(55, "episode", seed), episode_solver_seed=solver_seed(55, "episode", seed),
                       length=limit, steps=limit, smoke=smoke, truncated=not smoke, terminated=False,
                       truncated_by_evaluator=smoke, control_seconds=float(limit), **{"return": float(limit)})
        from utils.transfer_campaign_diagnostics import FAMILIES
        sampled = sorted(value for value in set(definition["diagnostics"]["decisions"]) | ({0, 1} if smoke else set()) if value < limit)
        stationary = sorted(set(sampled) & (set(definition["diagnostics"]["stationary_decisions"]) | ({1} if smoke else set())))
        summary = {f"{stage}_{metric}": 0. for stage in ("prior", "initial", "final") for metric in
            ("critic_rmse", "policy_kl", "action_gradient_gain", "actor_feature_effective_rank_fraction")}
        episode.update(diagnostic_seconds=1., diagnostic_isolation_verified=smoke,
            diagnostics=dict(complete=True, completed_decisions=sampled, samples=len(sampled),
                completed_stationary_decisions=stationary, families=list(FAMILIES), summary=summary))
        metrics = dict(inner_actor_optimizer_steps=4 * cell["J"], inner_critic_optimizer_steps=16 * cell["J"],
                       inner_model_steps_budget=128 * cell["H"] * cell["J"], inner_actor_q_mean=0., inner_actor_entropy=0.,
                       inner_q_mean=0., inner_q_target_mean=0., inner_td_error_abs_mean=0., inner_temperature_optimizer_steps=4 * cell["J"])
        rows = [dict(seed=seed, decision=i, reward=1., cumulative_reward=i + 1, control_seconds=1., action=[0.], metrics=metrics)
                for i in range(limit)]
        for row in rows:
            if row["decision"] in sampled:
                row["diagnostics"] = dict(decision=row["decision"], summary=summary)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"decisions-seed-{seed}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        episodes.append(episode)
    save(directory / "manifest.json", manifest)
    save(directory / "results.json", dict(**shared, complete=True, frozen_outer_verified=True, episodes=episodes))
    return directory


def test_preparation_pins_checkpoint_grid_and_smokes_without_launch(prepared):
    args, definition = prepared
    assert len(definition["cells"]) == 6
    assert definition["smoke_indices"] == definition["production_indices"] == list(range(6))
    assert definition["smoke_checkpoint_steps"] == [25000, 575000, 2000000]
    assert definition["reused_indices"] == []
    assert len({cell["result_dir"] for cell in definition["cells"]}) == 6
    for cell in definition["cells"]:
        recipe = campaign.read_json(cell["config_path"])
        checkpoint = definition["checkpoints"][cell["checkpoint_index"]]
        assert recipe["checkpoint_contract"] == dict(step=cell["step"], sha256=checkpoint["checkpoint_sha256"])
        assert checkpoint["prior_pin"]["selector"] == "reference/prior"
        assert recipe["critic_updates"] == 16 and recipe["actor_updates"] == 4
        assert recipe["rollouts"] == 128 and recipe["batch_size"] == 256
    with pytest.raises(ValueError, match="overwrite"):
        campaign.prepare(args)


@pytest.mark.parametrize("changes,match", [
    ({"seeds": [101, 102]}, "protocol"), ({"critic_updates": 32}, "protocol"),
    ({"max_steps": 3}, "protocol"), ({"controller_seed": 54}, "protocol"),
])
def test_changed_scientific_recipe_is_rejected(changes, match):
    config = campaign.read_json(CONFIG)
    config.update(changes)
    with pytest.raises(ValueError, match=match):
        campaign.validate_configuration(config)


@pytest.mark.parametrize("selected", [[], ["unknown"], ["h1_j2_fresh"] * 2])
def test_bad_setting_subset_is_rejected(selected):
    with pytest.raises(ValueError):
        campaign.validate_configuration(campaign.read_json(CONFIG), selected)


def test_inventory_rejects_different_backbone_and_modified_weights(prepared):
    args, definition = prepared
    inventory = campaign.read_json(args.inventory)
    first = inventory["checkpoints"][0]
    metadata = Path(first["checkpoint"] + ".metadata.json")
    changed = campaign.read_json(metadata)
    changed["trial_run_params"]["alg_params"]["outer_learning_rate"] = 9.
    save(metadata, changed)
    first["metadata_sha256"] = campaign.digest(metadata)
    with pytest.raises(ValueError, match="different backbone"):
        campaign.validate_inventory(inventory)
    first["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checkpoint hash"):
        campaign.validate_inventory(inventory)


def test_validated_receipt_pins_results_and_all_trace_files(prepared):
    args, definition = prepared
    cell = definition["cells"][0]
    directory = write_result(args.root, definition, cell)
    receipt = campaign.make_receipt(args.root, definition, cell, directory, hardware="NVIDIA L40S, uuid, driver")
    campaign.write(campaign.receipt_path(args.root, cell), receipt)
    assert campaign.validate_receipt(args.root, definition, cell) == receipt
    assert len(receipt["file_sha256"]) == 7
    assert receipt["metric_coverage"] == "all_computed_inner_scalars_and_sampled_diagnostics"
    with (directory / "decisions-seed-101.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="incomplete"):
        campaign.validate_receipt(args.root, definition, cell)


@pytest.mark.parametrize("target,mutation,match", [
    ("results.json", lambda value: value.update(frozen_outer_verified=False), "frozen"),
    ("results.json", lambda value: value["episodes"][0].update(length=499), "full requested"),
    ("results.json", lambda value: value["episodes"][0].update(solver_seed=12), "solver stream"),
    ("results.json", lambda value: value["episodes"][0].update(control_seconds=499.), "totals"),
    ("manifest.json", lambda value: value.update(checkpoint_step=575000), "budget"),
    ("manifest.json", lambda value: value.update(compile=False), "budget"),
    ("manifest.json", lambda value: value.update(metric_policy="legacy"), "scalar measurement"),
    ("manifest.json", lambda value: value["arm_definition"].update(actor_rho=0.), "semantics"),
])
def test_corrupt_results_cannot_be_published(prepared, target, mutation, match):
    args, definition = prepared
    cell = definition["cells"][0]
    directory = write_result(args.root, definition, cell)
    value = campaign.read_json(directory / target)
    mutation(value)
    save(directory / target, value)
    with pytest.raises(ValueError, match=match):
        campaign.validate_result(directory, definition, cell)


def test_production_requires_all_smoke_receipts_before_gpu_use(prepared, monkeypatch):
    args, definition = prepared
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k: pytest.fail("GPU inspected before smoke gate"))
    with pytest.raises(FileNotFoundError):
        campaign.worker(SimpleNamespace(root=args.root, index=0, smoke=False, expected_source_sha="tested-commit"))


def test_source_rejects_dirty_or_wrong_checkout(monkeypatch):
    monkeypatch.setattr(campaign, "git", lambda *args: "dirty" if args[0] == "status" else "head")
    with pytest.raises(ValueError, match="clean"):
        campaign.source("head")
    monkeypatch.setattr(campaign, "git", lambda *args: "" if args[0] == "status" else "actual")
    with pytest.raises(ValueError, match="expected commit"):
        campaign.source("expected")


def test_historical_transfer_reuse_is_rejected(prepared):
    args, definition = prepared
    cell = definition["cells"][0]
    directory = write_result(args.root, definition, cell)
    with pytest.raises(ValueError, match="Historical transfer reuse"):
        campaign.validate_result(directory, definition, cell, allow_historical=True)
    cell["historical_reuse"] = True
    with pytest.raises(ValueError, match="Historical transfer reuse"):
        campaign.validate_result(directory, definition, cell)


def test_source_fingerprint_matches_actual_evaluator():
    from evaluate_ambi_transfer_campaign import source_identity
    value = source_identity()
    assert campaign.science_identity() == {key: value[key] for key in ("files", "sha256")}
    assert "utils/spectral_transfer.py" in value["files"]
    assert "utils/transfer_campaign_diagnostics.py" in value["files"]


def test_shortlist_preserves_all_parameter_bernoulli_and_matrix_shrink():
    config = campaign.read_json(CONFIG)
    assert config["seeds"] == [101, 102, 103, 104, 105]
    assert len(campaign.validate_configuration(config)) == 5
    recipe = campaign.read_json(CONFIG.parent / config["discovery_template"])
    assert recipe["horizons"] == [1] and recipe["rounds"] == [2, 4]
    assert recipe["arms"]["bernoulli_a075_c075"]["parameter_scope"] == "all"
    assert recipe["arms"]["bernoulli_a075_c075"]["actor_bernoulli_p"] == .75
    assert recipe["arms"]["bernoulli_a075_c075"]["critic_bernoulli_p"] == .75
    assert recipe["arms"]["bernoulli_a0_c05"]["parameter_scope"] == "all"
    assert recipe["arms"]["bernoulli_a0_c05"]["critic_bernoulli_p"] == .5
    assert recipe["arms"]["matrix_blend05_actor"]["parameter_scope"] == "matrices"
    assert recipe["arms"]["matrix_blend05_actor"]["actor_rho"] == .5
    assert recipe["diagnostics"]["enabled"] is True


def test_j6_selection_preserves_recipe_and_requires_all_matched_controls():
    config = campaign.read_json(J6_CONFIG)
    candidates = campaign.validate_configuration(config)
    assert {row["setting_id"] for row in candidates} == campaign.J6_SETTINGS
    assert all((row["H"], row["J"]) == (1, 6) for row in candidates)
    original = campaign.read_json(CONFIG)
    original_recipe = campaign.read_json(CONFIG.parent / original["discovery_template"])
    recipe = campaign.read_json(CONFIG.parent / config["discovery_template"])
    assert recipe["rounds"] == [6] and recipe["horizons"] == [1]
    assert recipe["arms"] == {name: original_recipe["arms"][name] for name in
                              ("fresh", "bernoulli_a0_c05", "matrix_blend05_actor")}
    for key in ("base_matrix", "base_preset", "checkpoint_contract", "seeds", "controller_seed", "max_steps",
                "critic_updates", "actor_updates", "rollouts", "batch_size", "diagnostics"):
        assert recipe[key] == original_recipe[key]
    with pytest.raises(ValueError, match="all three"):
        campaign.validate_configuration(config, ["h1_j6_fresh"])
    with pytest.raises(ValueError, match="matched settings"):
        campaign.validate_configuration(config, list(campaign.J6_SETTINGS) + ["h1_j6_fresh"])


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.pop("selection_id"), "explicit versioned selection"),
    (lambda value: value.update(selection_id="unreviewed"), "Unknown versioned"),
    (lambda value: value["checkpoint_range"].update(start=500000), "J6 checkpoint range"),
    (lambda value: value["checkpoint_range"].update(stop=1975000), "J6 checkpoint range"),
    (lambda value: value["checkpoint_range"].update(step=50000), "J6 checkpoint range"),
    (lambda value: value["candidates"].pop(), "J6 selection requires"),
    (lambda value: value["candidates"][1].update(fresh_setting_id="h1_j4_fresh"), "matching fresh"),
])
def test_j6_scope_drift_is_rejected(mutation, match):
    config = campaign.read_json(J6_CONFIG)
    mutation(config)
    with pytest.raises(ValueError, match=match):
        campaign.validate_configuration(config)


def test_j6_does_not_expand_legacy_shortlist_budgets():
    config = campaign.read_json(CONFIG)
    config["candidates"] = [dict(setting_id="h1_j6_fresh", H=1, J=6, arm="fresh", role="fresh")]
    with pytest.raises(ValueError, match="budget"):
        campaign.validate_configuration(config)


def test_j6_range_requires_every_selected_checkpoint_and_excludes_500k():
    config = campaign.read_json(J6_CONFIG)
    inventory = [dict(step=step) for step in range(25000, 2000001, 25000)]
    selected = campaign.select_checkpoints(inventory, config)
    assert [row["step"] for row in selected] == list(range(525000, 2000001, 25000))
    assert len(selected) == 60 and all(row["step"] > 500000 for row in selected)
    assert campaign.select_checkpoints(inventory, campaign.read_json(CONFIG)) is inventory
    with pytest.raises(ValueError, match="every checkpoint"):
        campaign.select_checkpoints([row for row in inventory if row["step"] != 525000], config)
    config["checkpoint_range"]["step"] = 25000.0
    with pytest.raises(ValueError, match="Malformed checkpoint range"):
        campaign.select_checkpoints(inventory, config)


def test_j6_preparation_creates_180_cells_and_nine_smokes_from_full_bank(prepared):
    original_args, original_definition = prepared
    inventory = campaign.read_json(original_args.inventory)
    present = {row["step"] for row in inventory["checkpoints"]}
    anchor = next(row for row in inventory["checkpoints"] if row["step"] == 575000)
    metadata = campaign.read_json(anchor["checkpoint"] + ".metadata.json")
    for step in range(25000, 2000001, 25000):
        if step in present:
            continue
        checkpoint = Path(anchor["checkpoint"]).parent / str(step)
        checkpoint.write_text(f"checkpoint {step}")
        sidecar = Path(str(checkpoint) + ".metadata.json")
        value = deepcopy(metadata)
        value["checkpoint"]["step"] = step
        save(sidecar, value)
        inventory["checkpoints"].append(dict(step=step, checkpoint=str(checkpoint),
            checkpoint_sha256=campaign.digest(checkpoint), metadata_sha256=campaign.digest(sidecar),
            prior_reference=f"prior/{step}", prior_pin=dict(selector="reference/prior", manifest_sha256="prior-digest")))
    save(original_args.inventory, inventory)
    args = SimpleNamespace(**vars(original_args))
    args.root = original_args.root.parent / "j6-campaign"
    args.config, args.settings = J6_CONFIG, None
    definition = campaign.prepare(args)
    assert len(definition["checkpoints"]) == 60 and len(definition["cells"]) == 180
    assert definition["selection_id"] == campaign.J6_SELECTION
    assert definition["checkpoint_range"] == campaign.J6_CHECKPOINT_RANGE
    assert definition["smoke_checkpoint_steps"] == [525000, 575000, 2000000]
    assert len(definition["smoke_indices"]) == 9
    assert definition["production_indices"] == list(range(180)) and definition["reused_indices"] == []
    assert len({cell["result_dir"] for cell in definition["cells"]}) == 180
    for cell in definition["cells"]:
        assert (cell["H"], cell["J"]) == (1, 6) and cell["step"] > 500000
        assert definition["checkpoints"][cell["checkpoint_index"]]["step"] == cell["step"]
    assert definition["diagnostics"] == original_definition["diagnostics"]
    assert definition["scientific_source"] == original_definition["scientific_source"]
    for cell in definition["cells"][:3]:
        directory = write_result(args.root, definition, cell, smoke=True)
        campaign.validate_result(directory, definition, cell, smoke=True)


def test_diagnostic_coverage_and_isolation_are_required(prepared):
    args, definition = prepared
    cell = definition["cells"][0]
    directory = write_result(args.root, definition, cell, smoke=True)
    result_path = directory / "results.json"
    value = campaign.read_json(result_path)
    value["episodes"][0]["diagnostic_isolation_verified"] = False
    save(result_path, value)
    with pytest.raises(ValueError, match="isolation"):
        campaign.validate_result(directory, definition, cell, smoke=True)
    value["episodes"][0]["diagnostic_isolation_verified"] = True
    value["episodes"][0]["diagnostics"]["completed_decisions"] = []
    save(result_path, value)
    with pytest.raises(RuntimeError, match="coverage"):
        campaign.validate_result(directory, definition, cell, smoke=True)


def test_launcher_is_gpu_worker_only_and_pins_source():
    text = (campaign.ROOT / "slurm/run_ambi_transfer_curves_oscar.sbatch").read_text()
    assert "--expected-source-sha" in text and "--smoke" in text
    assert "WANDB_MODE=disabled" in text and "--no-requeue" in text
    assert "sbatch" not in text and "uv sync" not in text
