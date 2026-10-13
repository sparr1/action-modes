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
J8_CONFIG = CONFIG.parent / "ambi_transfer_checkpoint_j8_after500k_curves.json"
J10_CONFIG = CONFIG.parent / "ambi_transfer_800k_bernoulli_j10_curves.json"
H2_J8_CONFIG = CONFIG.parent / "ambi_transfer_checkpoint_h2_j8_after500k_curves.json"
HORIZON_J8_CONFIG = CONFIG.parent / "ambi_transfer_checkpoint_horizon_j8_after500k_curves.json"
EXTENSIONS = [(6, J6_CONFIG), (8, J8_CONFIG)]


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


@pytest.mark.parametrize("rounds,config_path", EXTENSIONS)
def test_extension_selection_preserves_recipe_and_requires_all_matched_controls(rounds, config_path):
    config = campaign.read_json(config_path)
    candidates = campaign.validate_configuration(config)
    required_settings = getattr(campaign, f"J{rounds}_SETTINGS")
    assert {row["setting_id"] for row in candidates} == required_settings
    assert all((row["H"], row["J"]) == (1, rounds) for row in candidates)
    original = campaign.read_json(CONFIG)
    original_recipe = campaign.read_json(CONFIG.parent / original["discovery_template"])
    recipe = campaign.read_json(CONFIG.parent / config["discovery_template"])
    assert recipe["rounds"] == [rounds] and recipe["horizons"] == [1]
    assert recipe["arms"] == {name: original_recipe["arms"][name] for name in
                              ("fresh", "bernoulli_a0_c05", "matrix_blend05_actor")}
    for key in ("base_matrix", "base_preset", "checkpoint_contract", "seeds", "controller_seed", "max_steps",
                "critic_updates", "actor_updates", "rollouts", "batch_size", "diagnostics"):
        assert recipe[key] == original_recipe[key]
    with pytest.raises(ValueError, match="all requested settings"):
        campaign.validate_configuration(config, [f"h1_j{rounds}_fresh"])
    with pytest.raises(ValueError, match="all requested settings"):
        campaign.validate_configuration(config, list(required_settings) + [f"h1_j{rounds}_fresh"])


@pytest.mark.parametrize("rounds,config_path", EXTENSIONS)
@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.pop("selection_id"), "explicit versioned selection"),
    (lambda value: value.update(selection_id="unreviewed"), "Unknown versioned"),
    (lambda value: value["checkpoint_range"].update(start=500000), "checkpoint range"),
    (lambda value: value["checkpoint_range"].update(stop=1975000), "checkpoint range"),
    (lambda value: value["checkpoint_range"].update(step=50000), "checkpoint range"),
    (lambda value: value["candidates"].pop(), "selection requires"),
    (lambda value: value["candidates"][1].update(fresh_setting_id="h1_j4_fresh"), "matching fresh"),
])
def test_extension_scope_drift_is_rejected(rounds, config_path, mutation, match):
    config = campaign.read_json(config_path)
    mutation(config)
    with pytest.raises(ValueError, match=match):
        campaign.validate_configuration(config)


@pytest.mark.parametrize("rounds,config_path", EXTENSIONS)
def test_extension_does_not_expand_legacy_shortlist_budgets(rounds, config_path):
    config = campaign.read_json(CONFIG)
    config["candidates"] = [dict(setting_id=f"h1_j{rounds}_fresh", H=1, J=rounds, arm="fresh", role="fresh")]
    with pytest.raises(ValueError, match="budget"):
        campaign.validate_configuration(config)


@pytest.mark.parametrize("rounds,config_path", EXTENSIONS)
def test_extension_profiles_cannot_exchange_settings(rounds, config_path):
    config = campaign.read_json(config_path)
    config["selection_id"] = campaign.J8_SELECTION if rounds == 6 else campaign.J6_SELECTION
    with pytest.raises(ValueError, match="selection requires"):
        campaign.validate_configuration(config)


@pytest.mark.parametrize("rounds,config_path", EXTENSIONS)
def test_extension_range_requires_every_selected_checkpoint_and_excludes_500k(rounds, config_path):
    config = campaign.read_json(config_path)
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


@pytest.mark.parametrize("rounds,config_path", EXTENSIONS)
def test_extension_preparation_creates_180_cells_and_nine_smokes_from_full_bank(prepared, rounds, config_path):
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
    args.root = original_args.root.parent / f"j{rounds}-campaign"
    args.config, args.settings = config_path, None
    definition = campaign.prepare(args)
    assert len(definition["checkpoints"]) == 60 and len(definition["cells"]) == 180
    assert definition["selection_id"] == getattr(campaign, f"J{rounds}_SELECTION")
    assert definition["checkpoint_range"] == getattr(campaign, f"J{rounds}_CHECKPOINT_RANGE")
    assert f"H1 J{rounds} matched-budget" in definition["selection"]
    assert definition["smoke_checkpoint_steps"] == [525000, 575000, 2000000]
    assert len(definition["smoke_indices"]) == 9
    assert definition["production_indices"] == list(range(180)) and definition["reused_indices"] == []
    assert len({cell["result_dir"] for cell in definition["cells"]}) == 180
    for cell in definition["cells"]:
        assert (cell["H"], cell["J"]) == (1, rounds) and cell["step"] > 500000
        assert definition["checkpoints"][cell["checkpoint_index"]]["step"] == cell["step"]
    assert definition["diagnostics"] == original_definition["diagnostics"]
    assert definition["scientific_source"] == original_definition["scientific_source"]
    for cell in definition["cells"][:3]:
        directory = write_result(args.root, definition, cell, smoke=True)
        campaign.validate_result(directory, definition, cell, smoke=True)


def test_j10_preserves_j8_recipe_with_only_requested_rounds_and_arm():
    config = campaign.read_json(J10_CONFIG)
    candidates = campaign.validate_configuration(config)
    assert len(candidates) == 1
    assert candidates[0]["setting_id"] == "h1_j10_bernoulli_a0_c05"
    assert (candidates[0]["H"], candidates[0]["J"], candidates[0]["role"]) == (1, 10, "transfer")
    assert "fresh_setting_id" not in candidates[0]
    assert campaign.validate_configuration(config, [candidates[0]["setting_id"]]) == candidates
    j8 = campaign.read_json(J8_CONFIG)
    old = campaign.read_json(CONFIG.parent / j8["discovery_template"])
    new = campaign.read_json(CONFIG.parent / config["discovery_template"])
    assert new["rounds"] == [10]
    assert new["arms"] == {"bernoulli_a0_c05": old["arms"]["bernoulli_a0_c05"]}
    normalized = deepcopy(new)
    for key in ("rounds", "arms", "description"):
        normalized[key] = old[key]
    assert normalized == old


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.pop("selection_id"), "explicit versioned selection"),
    (lambda value: value["checkpoint_range"].update(start=575000), "checkpoint range"),
    (lambda value: value["candidates"].append(dict(setting_id="h1_j10_fresh", H=1, J=10,
                                                 arm="fresh", role="fresh")), "exact versioned settings"),
    (lambda value: value["candidates"][0].update(H=2), "identity mismatch"),
    (lambda value: value["candidates"][0].update(J=8), "identity mismatch"),
    (lambda value: value["candidates"][0].update(arm="fresh"), "identity mismatch"),
    (lambda value: value["candidates"][0].update(role="fresh"), "role differs"),
    (lambda value: value["candidates"][0].update(fresh_setting_id="h1_j8_fresh"), "no matching fresh J10"),
    (lambda value: value.update(seeds=[101]), "protocol"),
    (lambda value: value.update(critic_updates=32), "protocol"),
])
def test_j10_scope_drift_is_rejected(mutation, match):
    config = campaign.read_json(J10_CONFIG)
    mutation(config)
    with pytest.raises(ValueError, match=match):
        campaign.validate_configuration(config)


def test_j10_selection_requires_800k_and_does_not_relax_other_profiles():
    config = campaign.read_json(J10_CONFIG)
    inventory = [dict(step=575000), dict(step=800000), dict(step=2000000)]
    assert campaign.select_checkpoints(inventory, config) == [dict(step=800000)]
    with pytest.raises(ValueError, match="every checkpoint"):
        campaign.select_checkpoints([dict(step=575000)], config)
    with pytest.raises(ValueError, match="all requested settings"):
        campaign.validate_configuration(config, [])
    legacy = campaign.read_json(CONFIG)
    legacy["candidates"] = deepcopy(config["candidates"])
    with pytest.raises(ValueError, match="budget"):
        campaign.validate_configuration(legacy)
    legacy["checkpoint_range"] = config["checkpoint_range"]
    with pytest.raises(ValueError, match="audited 575k anchor"):
        campaign.select_checkpoints(inventory, legacy)


def test_j10_preparation_keeps_anchor_validation_and_full_seed_receipts(prepared):
    original_args, original_definition = prepared
    inventory = campaign.read_json(original_args.inventory)
    anchor = next(row for row in inventory["checkpoints"] if row["step"] == 575000)
    checkpoint = Path(anchor["checkpoint"]).parent / "800000"
    checkpoint.write_text("checkpoint 800000")
    metadata = campaign.read_json(anchor["checkpoint"] + ".metadata.json")
    metadata["checkpoint"]["step"] = 800000
    sidecar = Path(str(checkpoint) + ".metadata.json")
    save(sidecar, metadata)
    inventory["checkpoints"].append(dict(step=800000, checkpoint=str(checkpoint),
        checkpoint_sha256=campaign.digest(checkpoint), metadata_sha256=campaign.digest(sidecar)))
    without_anchor = deepcopy(inventory)
    without_anchor["checkpoints"] = [row for row in without_anchor["checkpoints"] if row["step"] != 575000]
    with pytest.raises(ValueError, match="audited 575k anchor"):
        campaign.validate_inventory(without_anchor)
    save(original_args.inventory, inventory)
    args = SimpleNamespace(**vars(original_args))
    args.root = original_args.root.parent / "j10-campaign"
    args.config, args.settings = J10_CONFIG, None
    definition = campaign.prepare(args)
    assert [row["step"] for row in definition["checkpoints"]] == [800000]
    assert len(definition["cells"]) == 1
    assert definition["smoke_indices"] == definition["production_indices"] == [0]
    assert definition["smoke_checkpoint_steps"] == [800000] and definition["reused_indices"] == []
    assert definition["seeds"] == [101, 102, 103, 104, 105]
    assert definition["diagnostics"] == original_definition["diagnostics"]
    assert definition["scientific_source"] == original_definition["scientific_source"]
    cell = definition["cells"][0]
    assert cell["name"] == "step-800000/h1_j10_bernoulli_a0_c05"
    assert "fresh_setting_id" not in cell
    for smoke in (True, False):
        directory = write_result(args.root, definition, cell, smoke=smoke)
        receipt = campaign.make_receipt(args.root, definition, cell, directory,
            smoke=smoke, hardware="NVIDIA L40S, uuid, driver")
        campaign.write(campaign.receipt_path(args.root, cell, smoke=smoke), receipt)
        assert campaign.validate_receipt(args.root, definition, cell, smoke=smoke) == receipt
        assert receipt["episodes"] == (2 if smoke else 5)
    from utils.transfer_campaign import load_campaign, resolved_cell
    manifest = campaign.read_json(Path(cell["result_dir"]) / "manifest.json")
    j10_resolved = manifest["resolved"]
    j8_config = campaign.read_json(J8_CONFIG)
    j8_recipe = load_campaign(CONFIG.parent / j8_config["discovery_template"])
    j8_resolved = resolved_cell(j10_resolved, j8_recipe, horizon=1, rounds=8, arm=cell["arm"])
    expected = deepcopy(j8_resolved)
    expected["algorithm_config"]["alg_params"]["inner_rounds"] = 10
    expected["selector"] = "transfer_discovery/bernoulli_a0_c05/H1/J10"
    assert j10_resolved == expected
    assert j10_resolved["algorithm_config"]["alg_params"]["inner_replay_capacity"] >= 1280
    partial = campaign.read_json(Path(cell["result_dir"]) / "results.json")
    partial["episodes"].pop()
    save(Path(cell["result_dir"]) / "results.json", partial)
    with pytest.raises(ValueError, match="seed panel"):
        campaign.validate_result(cell["result_dir"], definition, cell)


def test_h2_j8_selects_only_existing_bernoulli_critic_recipe_with_new_horizon():
    from utils.transfer_campaign import load_campaign
    from utils.transfer_campaign_diagnostics import diagnostic_settings
    config = campaign.read_json(H2_J8_CONFIG)
    candidates = campaign.validate_configuration(config)
    assert {row["setting_id"] for row in candidates} == campaign.H2_J8_SETTINGS
    assert len(candidates) == 1 and (candidates[0]["H"], candidates[0]["J"]) == (2, 8)
    assert candidates[0]["arm"] == "bernoulli_a0_c05" and candidates[0]["role"] == "transfer"
    assert "fresh_setting_id" not in candidates[0]
    recipe = load_campaign(CONFIG.parent / config["discovery_template"])
    recipe["diagnostics"] = diagnostic_settings(recipe["diagnostics"])
    original = campaign.validate_h2_j8_template(recipe)
    assert recipe["arms"] == {"bernoulli_a0_c05": original["arms"]["bernoulli_a0_c05"]}
    normalized = deepcopy(recipe)
    for key in ("horizons", "description", "arms"):
        normalized[key] = original[key]
    assert normalized == original
    assert recipe["arms"]["bernoulli_a0_c05"]["critic_bernoulli_p"] == .5
    assert recipe["arms"]["bernoulli_a0_c05"]["parameter_scope"] == "all"
    assert campaign.validate_configuration(config, [candidates[0]["setting_id"]]) == candidates
    with pytest.raises(ValueError, match="all requested settings"):
        campaign.validate_configuration(config, [])


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.pop("selection_id"), "explicit versioned selection"),
    (lambda value: value["checkpoint_range"].update(start=500000), "checkpoint range"),
    (lambda value: value["checkpoint_range"].update(stop=1975000), "checkpoint range"),
    (lambda value: value["checkpoint_range"].update(step=50000), "checkpoint range"),
    (lambda value: value["candidates"].append(dict(setting_id="h2_j8_fresh", H=2, J=8,
                                                 arm="fresh", role="fresh")), "exact versioned settings"),
    (lambda value: value["candidates"][0].update(H=1), "identity mismatch"),
    (lambda value: value["candidates"][0].update(J=6), "identity mismatch"),
    (lambda value: value["candidates"][0].update(role="fresh"), "role differs"),
    (lambda value: value["candidates"][0].update(fresh_setting_id="h1_j8_fresh"), "no matching fresh H2 J8"),
    (lambda value: value.update(seeds=[101]), "protocol"),
    (lambda value: value.update(critic_updates=32), "protocol"),
    (lambda value: value.update(selection_id=campaign.J8_SELECTION), "exact versioned settings"),
])
def test_h2_j8_scope_drift_is_rejected(mutation, match):
    config = campaign.read_json(H2_J8_CONFIG)
    mutation(config)
    with pytest.raises(ValueError, match=match):
        campaign.validate_configuration(config)


@pytest.mark.parametrize("mutation", [
    lambda recipe: recipe["arms"]["bernoulli_a0_c05"].update(actor_bernoulli_p=.5),
    lambda recipe: recipe["arms"]["bernoulli_a0_c05"].update(parameter_scope="matrices"),
    lambda recipe: recipe["arms"]["bernoulli_a0_c05"].update(critic_bernoulli_p=.75),
    lambda recipe: recipe["arms"]["bernoulli_a0_c05"].update(compile=False),
    lambda recipe: recipe.update(actor_updates=8),
    lambda recipe: recipe["diagnostics"].update(mc_rollouts=4),
])
def test_h2_j8_rejects_transfer_training_or_diagnostic_recipe_changes(mutation):
    from utils.transfer_campaign import load_campaign
    from utils.transfer_campaign_diagnostics import diagnostic_settings
    config = campaign.read_json(H2_J8_CONFIG)
    recipe = load_campaign(CONFIG.parent / config["discovery_template"])
    recipe["diagnostics"] = diagnostic_settings(recipe["diagnostics"])
    mutation(recipe)
    with pytest.raises(ValueError, match="may only change the horizon"):
        campaign.validate_h2_j8_template(recipe)


def test_h2_j8_does_not_relax_legacy_or_other_horizon_profiles():
    h2 = campaign.read_json(H2_J8_CONFIG)
    legacy = campaign.read_json(CONFIG)
    legacy["candidates"] = deepcopy(h2["candidates"])
    with pytest.raises(ValueError, match="budget"):
        campaign.validate_configuration(legacy)
    inventory = [dict(step=step) for step in range(25000, 2000001, 25000)]
    selected = campaign.select_checkpoints(inventory, h2)
    assert [row["step"] for row in selected] == list(range(525000, 2000001, 25000))
    assert len(selected) == 60
    with pytest.raises(ValueError, match="every checkpoint"):
        campaign.select_checkpoints([row for row in inventory if row["step"] != 525000], h2)


def test_h2_j8_prepares_sixty_cells_three_smokes_and_only_changes_resolved_horizon(prepared):
    from utils.transfer_campaign import load_campaign, resolved_cell
    original_args, original_definition = prepared
    inventory = campaign.read_json(original_args.inventory)
    present = {row["step"] for row in inventory["checkpoints"]}
    anchor = next(row for row in inventory["checkpoints"] if row["step"] == 575000)
    metadata = campaign.read_json(anchor["checkpoint"] + ".metadata.json")
    for step in range(525000, 2000001, 25000):
        if step in present:
            continue
        checkpoint = Path(anchor["checkpoint"]).parent / str(step)
        checkpoint.write_text(f"checkpoint {step}")
        sidecar = Path(str(checkpoint) + ".metadata.json")
        value = deepcopy(metadata)
        value["checkpoint"]["step"] = step
        save(sidecar, value)
        inventory["checkpoints"].append(dict(step=step, checkpoint=str(checkpoint),
            checkpoint_sha256=campaign.digest(checkpoint), metadata_sha256=campaign.digest(sidecar)))
    save(original_args.inventory, inventory)
    args = SimpleNamespace(**vars(original_args))
    args.root = original_args.root.parent / "h2-j8-campaign"
    args.config, args.settings = H2_J8_CONFIG, None
    definition = campaign.prepare(args)
    assert len(definition["checkpoints"]) == len(definition["cells"]) == 60
    assert definition["selection_id"] == campaign.H2_J8_SELECTION
    assert definition["checkpoint_range"] == campaign.H2_J8_CHECKPOINT_RANGE
    assert "no fresh H2 J8 control" in definition["selection"]
    assert definition["smoke_checkpoint_steps"] == [525000, 575000, 2000000]
    assert definition["smoke_indices"] == [0, 2, 59]
    assert definition["production_indices"] == list(range(60)) and definition["reused_indices"] == []
    assert len({cell["result_dir"] for cell in definition["cells"]}) == 60
    assert definition["seeds"] == [101, 102, 103, 104, 105]
    assert definition["diagnostics"] == original_definition["diagnostics"]
    assert definition["scientific_source"] == original_definition["scientific_source"]
    for cell in definition["cells"]:
        assert (cell["H"], cell["J"], cell["setting_id"]) == (2, 8, "h2_j8_bernoulli_a0_c05")
        assert "fresh_setting_id" not in cell
        assert definition["checkpoints"][cell["checkpoint_index"]]["step"] == cell["step"]
    for index in definition["smoke_indices"]:
        cell = definition["cells"][index]
        directory = write_result(args.root, definition, cell, smoke=True)
        receipt = campaign.make_receipt(args.root, definition, cell, directory,
            smoke=True, hardware="NVIDIA L40S, uuid, driver")
        campaign.write(campaign.receipt_path(args.root, cell, smoke=True), receipt)
        assert campaign.validate_receipt(args.root, definition, cell, smoke=True) == receipt
    cell = definition["cells"][0]
    directory = write_result(args.root, definition, cell)
    result, manifest = campaign.validate_result(directory, definition, cell)
    h2_resolved = manifest["resolved"]
    h1 = load_campaign(CONFIG.parent / campaign.read_json(J8_CONFIG)["discovery_template"])
    h1_resolved = resolved_cell(h2_resolved, h1, horizon=1, rounds=8, arm=cell["arm"])
    expected = deepcopy(h1_resolved)
    expected["algorithm_config"]["alg_params"]["inner_rollout_horizon"] = 2
    expected["selector"] = "transfer_discovery/bernoulli_a0_c05/H2/J8"
    assert h2_resolved == expected
    params = h2_resolved["algorithm_config"]["alg_params"]
    assert params["inner_replay_capacity"] == 3072
    assert params["compile"] is params["compile_strict"] is True
    assert len(result["episodes"]) == 5
    rows = [json.loads(line) for line in (directory/"decisions-seed-101.jsonl").read_text().splitlines()]
    assert rows[0]["metrics"]["inner_model_steps_budget"] == 2048
    assert rows[0]["metrics"]["inner_actor_optimizer_steps"] == 32
    assert rows[0]["metrics"]["inner_critic_optimizer_steps"] == 128
    # A stale H1 work count must not pass the H2 completion/publication gate.
    rows[0]["metrics"]["inner_model_steps_budget"] = 1024
    (directory/"decisions-seed-101.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="budget differs"):
        campaign.validate_result(directory, definition, cell)


def test_horizon_followup_contains_exactly_h3_transfer_and_h2_fresh():
    from utils.transfer_campaign import load_campaign, needs_donor
    from utils.transfer_campaign_diagnostics import diagnostic_settings
    config = campaign.read_json(HORIZON_J8_CONFIG)
    candidates = campaign.validate_configuration(config)
    assert [(row["H"], row["J"], row["arm"], row["role"]) for row in candidates] == [
        (3, 8, "bernoulli_a0_c05", "transfer"), (2, 8, "fresh", "fresh")]
    assert {row["setting_id"] for row in candidates} == campaign.HORIZON_J8_SETTINGS
    assert all("fresh_setting_id" not in row for row in candidates)
    recipe = load_campaign(CONFIG.parent / config["discovery_template"])
    recipe["diagnostics"] = diagnostic_settings(recipe["diagnostics"])
    reference = campaign.validate_horizon_j8_template(recipe)
    h2 = load_campaign(CONFIG.parent / campaign.read_json(H2_J8_CONFIG)["discovery_template"])
    h2["diagnostics"] = diagnostic_settings(h2["diagnostics"])
    for key in set(h2) - {"description", "horizons", "arms"}:
        assert recipe[key] == h2[key]
    assert recipe["arms"]["bernoulli_a0_c05"] == h2["arms"]["bernoulli_a0_c05"]
    assert recipe["arms"]["fresh"] == reference["arms"]["fresh"]
    assert not needs_donor(recipe["arms"]["fresh"])
    assert needs_donor(recipe["arms"]["bernoulli_a0_c05"])
    assert recipe["arms"]["bernoulli_a0_c05"].get("actor_bernoulli_p", 0.) == 0.
    assert recipe["arms"]["bernoulli_a0_c05"]["critic_bernoulli_p"] == .5
    assert recipe["horizons"] == [2, 3] and recipe["rounds"] == [8]
    assert campaign.validate_configuration(config, sorted(campaign.HORIZON_J8_SETTINGS)) == candidates
    with pytest.raises(ValueError, match="all requested settings"):
        campaign.validate_configuration(config, ["h3_j8_bernoulli_a0_c05"])


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.pop("selection_id"), "explicit versioned selection"),
    (lambda value: value["checkpoint_range"].update(start=500000), "checkpoint range"),
    (lambda value: value["checkpoint_range"].update(stop=1975000), "checkpoint range"),
    (lambda value: value["checkpoint_range"].update(step=50000), "checkpoint range"),
    (lambda value: value["candidates"].pop(), "exact versioned settings"),
    (lambda value: value["candidates"][0].update(H=2), "identity mismatch"),
    (lambda value: value["candidates"][1].update(H=3), "identity mismatch"),
    (lambda value: value["candidates"][0].update(J=6), "identity mismatch"),
    (lambda value: value["candidates"][0].update(role="fresh"), "role differs"),
    (lambda value: value["candidates"][0].update(fresh_setting_id="h2_j8_fresh"), "no matching fresh H3"),
    (lambda value: value["candidates"][1].update(fresh_setting_id="h2_j8_fresh"), "no matching fresh H3"),
    (lambda value: value.update(seeds=[101]), "protocol"),
    (lambda value: value.update(selection_id=campaign.H2_J8_SELECTION), "exact versioned settings"),
])
def test_horizon_followup_rejects_scope_drift(mutation, match):
    config = campaign.read_json(HORIZON_J8_CONFIG)
    mutation(config)
    with pytest.raises(ValueError, match=match):
        campaign.validate_configuration(config)


@pytest.mark.parametrize("mutation", [
    lambda recipe: recipe["arms"]["bernoulli_a0_c05"].update(actor_bernoulli_p=.5),
    lambda recipe: recipe["arms"]["bernoulli_a0_c05"].update(critic_bernoulli_p=.75),
    lambda recipe: recipe["arms"]["fresh"].update(critic_rho=.5),
    lambda recipe: recipe["arms"]["fresh"].update(full_state=True),
    lambda recipe: recipe["arms"]["fresh"].update(compile=False),
    lambda recipe: recipe.update(actor_updates=8),
    lambda recipe: recipe["diagnostics"].update(mc_rollouts=4),
])
def test_horizon_followup_rejects_recipe_changes(mutation):
    from utils.transfer_campaign import load_campaign
    from utils.transfer_campaign_diagnostics import diagnostic_settings
    config = campaign.read_json(HORIZON_J8_CONFIG)
    recipe = load_campaign(CONFIG.parent / config["discovery_template"])
    recipe["diagnostics"] = diagnostic_settings(recipe["diagnostics"])
    mutation(recipe)
    with pytest.raises(ValueError, match="may only change horizons"):
        campaign.validate_horizon_j8_template(recipe)


def test_horizon_followup_prepares_120_interleaved_cells_and_six_smokes(prepared):
    from utils.transfer_campaign import load_campaign, resolved_cell
    original_args, original_definition = prepared
    inventory = campaign.read_json(original_args.inventory)
    present = {row["step"] for row in inventory["checkpoints"]}
    anchor = next(row for row in inventory["checkpoints"] if row["step"] == 575000)
    metadata = campaign.read_json(anchor["checkpoint"] + ".metadata.json")
    for step in range(525000, 2000001, 25000):
        if step in present:
            continue
        checkpoint = Path(anchor["checkpoint"]).parent / str(step)
        checkpoint.write_text(f"checkpoint {step}")
        sidecar = Path(str(checkpoint) + ".metadata.json")
        value = deepcopy(metadata)
        value["checkpoint"]["step"] = step
        save(sidecar, value)
        inventory["checkpoints"].append(dict(step=step, checkpoint=str(checkpoint),
            checkpoint_sha256=campaign.digest(checkpoint), metadata_sha256=campaign.digest(sidecar)))
    save(original_args.inventory, inventory)
    args = SimpleNamespace(**vars(original_args))
    args.root = original_args.root.parent / "horizon-j8-campaign"
    args.config, args.settings = HORIZON_J8_CONFIG, None
    definition = campaign.prepare(args)
    assert len(definition["checkpoints"]) == 60 and len(definition["cells"]) == 120
    assert definition["selection_id"] == campaign.HORIZON_J8_SELECTION
    assert definition["checkpoint_range"] == campaign.HORIZON_J8_CHECKPOINT_RANGE
    assert definition["smoke_checkpoint_steps"] == [525000, 575000, 2000000]
    assert definition["smoke_indices"] == [0, 1, 4, 5, 118, 119]
    assert definition["production_indices"] == list(range(120)) and definition["reused_indices"] == []
    assert len({cell["result_dir"] for cell in definition["cells"]}) == 120
    assert definition["seeds"] == [101, 102, 103, 104, 105]
    assert definition["diagnostics"] == original_definition["diagnostics"]
    assert definition["scientific_source"] == original_definition["scientific_source"]
    expected_settings = ["h3_j8_bernoulli_a0_c05", "h2_j8_fresh"]
    for index, cell in enumerate(definition["cells"]):
        assert cell["index"] == index and cell["checkpoint_index"] == index // 2
        assert cell["setting_id"] == expected_settings[index % 2]
        assert cell["step"] == 525000 + 25000 * (index // 2)
        assert "fresh_setting_id" not in cell
    for index in definition["smoke_indices"]:
        cell = definition["cells"][index]
        directory = write_result(args.root, definition, cell, smoke=True)
        receipt = campaign.make_receipt(args.root, definition, cell, directory,
            smoke=True, hardware="NVIDIA L40S, uuid, driver")
        campaign.write(campaign.receipt_path(args.root, cell, smoke=True), receipt)
        assert campaign.validate_receipt(args.root, definition, cell, smoke=True) == receipt
    reference = load_campaign(CONFIG.parent / campaign.read_json(J8_CONFIG)["discovery_template"])
    for cell in definition["cells"][:2]:
        directory = write_result(args.root, definition, cell)
        result, manifest = campaign.validate_result(directory, definition, cell)
        resolved = manifest["resolved"]
        expected = resolved_cell(resolved, reference, horizon=1, rounds=8, arm=cell["arm"])
        expected["algorithm_config"]["alg_params"]["inner_rollout_horizon"] = cell["H"]
        expected["selector"] = f"transfer_discovery/{cell['arm']}/H{cell['H']}/J8"
        assert resolved == expected
        params = resolved["algorithm_config"]["alg_params"]
        assert params["inner_replay_capacity"] == 3072
        assert params["compile"] is params["compile_strict"] is True
        assert all(params[f"inner_{name}_scope"] == "action" for name in
            ("actor", "critic", "replay", "temperature", "actor_optimizer", "critic_optimizer", "temperature_optimizer"))
        assert len(result["episodes"]) == 5
        path = directory / "decisions-seed-101.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["metrics"]["inner_model_steps_budget"] == 1024 * cell["H"]
        assert rows[0]["metrics"]["inner_actor_optimizer_steps"] == 32
        assert rows[0]["metrics"]["inner_critic_optimizer_steps"] == 128
        rows[0]["metrics"]["inner_model_steps_budget"] = 1024
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError, match="budget differs"):
            campaign.validate_result(directory, definition, cell)


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
