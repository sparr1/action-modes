"""The four smoothness cells preserve their independently audited baselines."""

from copy import deepcopy
from itertools import product
from pathlib import Path
import json

import pytest

from tests.test_ambi_config_decoupling import _build_cfg
from utils.checkpointing import resolve_checkpoint_config
from utils.pave_identity import PAVE_COEFFICIENT_KEYS, pave_training_signature
from utils.resume_identity import (
    canonical_json, scientific_trial_parameters, validate_resume_selection,
)


ROOT = Path(__file__).resolve().parents[1] / "configs/dmcontrol"
BASELINES = {
    "return_only": ("de37a4eec8094c67a2a691db04e47a2b", ("outer_pave",)),
    "alpha_carry": ("774f452fdda045aaa835dec670cb3584", ("outer_pave", "aux_return_pave")),
}
TERMS = ("mpr", "action_smoothness")
CELLS = tuple(product(BASELINES, TERMS))
LABEL_FIELDS = ("wandb_group", "wandb_run_name", "wandb_tags")
GROUP = "ambi-critic-smoothness-hydra-20261030"


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _read(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_keys)


def _name(baseline, term):
    return f"ambi_smoothness_{baseline}_{term}"


def _cell(baseline, term):
    name = _name(baseline, term)
    return (_read(ROOT / "algs" / f"{name}.json"),
            _read(ROOT / "experiments" / f"{name}.json"))


def _without_labels(config):
    result = deepcopy(config)
    for key in LABEL_FIELDS:
        result["alg_params"].pop(key, None)
    return result


@pytest.mark.parametrize("baseline,term", CELLS)
def test_only_accepted_scientific_changes_from_frozen_source(baseline, term):
    sources = _read(ROOT / "critic_smoothness_sources.json")
    assert set(sources) == {value[0] for value in BASELINES.values()}
    source_id, prefixes = BASELINES[baseline]
    source = sources[source_id]
    actual, manifest = _cell(baseline, term)
    expected = deepcopy(source["run_params"])
    expected["total_steps"] = 1_000_000
    expected["alg_params"].update({
        **dict.fromkeys(PAVE_COEFFICIENT_KEYS, 0.),
        **{f"{prefix}_{term}_coef": .004 for prefix in prefixes},
        "outer_pave_batch_size": None,
        "outer_pave_obs_noise_std": .01,
        "outer_pave_action_noise_std": .02,
    })
    if baseline == "return_only":
        # The historical source predates this observational diagnostic.
        assert "inner_root_entropy_samples" not in source["run_params"]["alg_params"]
        expected["alg_params"]["inner_root_entropy_samples"] = 0
    assert _without_labels(actual) == _without_labels(expected)
    assert manifest["source_run_id"] == source_id
    assert manifest["source_commit"] == source["commit"]
    assert manifest["regularizer"] == term
    assert manifest["target_critic"] == ("both" if len(prefixes) == 2 else prefixes[0])


@pytest.mark.parametrize("baseline,term", CELLS)
def test_selected_critics_full_batch_and_coefficient_convention(baseline, term):
    source_id, prefixes = BASELINES[baseline]
    config, _ = _cell(baseline, term)
    params = config["alg_params"]
    active = {f"{prefix}_{term}_coef" for prefix in prefixes}
    assert {key: params[key] for key in PAVE_COEFFICIENT_KEYS} == {
        key: .004 if key in active else 0. for key in PAVE_COEFFICIENT_KEYS
    }
    assert params["num_q"] == 5
    # Each selected ensemble gets its own .2 coefficient ratio. Selecting both
    # does not halve the individual strength or combine their gradients first.
    for prefix in prefixes:
        td_coefficient = params["critic_coef" if prefix == "outer_pave" else "aux_return_critic_coef"]
        assert params[f"{prefix}_{term}_coef"] * params["num_q"] / td_coefficient == pytest.approx(.2)
    assert params["outer_pave_batch_size"] is None
    assert params["batch_size"] * params["train_unroll_horizon"] == 768
    assert params["outer_pave_obs_noise_std"] == .01
    assert params["outer_pave_action_noise_std"] == .02
    assert config["seed"] == 55 and config["total_steps"] == 1_000_000
    assert (params["train_unroll_horizon"], params["outer_planning_horizon"],
            params["inner_rollout_horizon"], params["inner_rounds"],
            params["inner_solve_interval"]) == (3, 3, 3, 10, 3)
    assert (params["inner_critic_updates_per_round"], params["inner_actor_updates_per_round"],
            params["inner_rollouts_per_round"], params["inner_batch_size"]) == (16, 4, 128, 256)
    assert params["inner_sac_critic_target"] == "reward_only"
    assert params["wandb_group"] == GROUP
    assert f"source-{source_id}" in params["wandb_tags"]
    assert {baseline, term, "seed55", "1m-decisions", "2m-environment-steps"} <= set(params["wandb_tags"])

    # Resolve through the real wrapper without allocating networks or rendering.
    resolved = _build_cfg(**params)
    assert all(getattr(resolved, key) == .004 for key in active)
    assert resolved.outer_pave_batch_size is None
    assert resolved.num_q == 5
    signature = pave_training_signature(resolved)
    assert signature["version"] == (1 if term == "mpr" else 2)
    assert all(signature["controls"][key] == .004 for key in active)


@pytest.mark.parametrize("term", TERMS)
def test_distinct_baseline_replay_temperature_and_checkpoint_contracts(term):
    single, single_manifest = _cell("return_only", term)
    dual, dual_manifest = _cell("alpha_carry", term)
    for config, manifest in ((single, single_manifest), (dual, dual_manifest)):
        checkpoint = resolve_checkpoint_config(config, manifest)
        assert checkpoint.enabled and checkpoint.every == 25_000
        assert checkpoint.strategies == ("all",)
        assert config["total_steps"] // checkpoint.every == 40
    one, two = single["alg_params"], dual["alg_params"]
    assert one["buffer_size"] == 500_000 and two["buffer_size"] == 1_000_000
    assert one["aux_return_mode"] == "off" and two["aux_return_mode"] == "sac"
    assert one["outer_critic_target"] == "reward_only"
    assert two["outer_critic_target"] == "entropy_augmented"
    assert one["inner_critic_source"] == one["inner_horizon_critic_source"] == "sac"
    assert two["inner_critic_source"] == two["inner_horizon_critic_source"] == "aux_return"
    assert one["inner_temperature_scope"] == one["inner_temperature_optimizer_scope"] == "action"
    assert two["inner_temperature_scope"] == two["inner_temperature_optimizer_scope"] == "run"
    assert two["aux_return_detach_representation"] is True
    assert (one["inner_root_entropy_samples"], two["inner_root_entropy_samples"]) == (0, 128)
    assert "checkpoint_replay" not in single and "checkpoint_replay" not in single_manifest
    assert resolve_checkpoint_config(single, single_manifest).replay is False
    assert dual["checkpoint_replay"] is True
    assert resolve_checkpoint_config(dual, dual_manifest).replay is True


def test_master_and_single_manifests_cover_exactly_four_independent_lineages():
    master = _read(ROOT / "experiments/ambi_critic_smoothness_training.json")
    names = [_name(*cell) for cell in CELLS]
    assert master["configs"] == names
    assert master["trials"] == 1 and "overrides_alg" not in master
    identities, run_names = set(), set()
    for index, (baseline, term) in enumerate(CELLS):
        source_id, prefixes = BASELINES[baseline]
        config, manifest = _cell(baseline, term)
        name = names[index]
        assert master["cells"][index] == {
            "index": index, "config": name, "baseline": baseline,
            "regularizer": term, "target_critic": "both" if len(prefixes) == 2 else prefixes[0],
            "source_run_id": source_id,
        }
        assert manifest["configs"] == [name] and manifest["trials"] == 1
        assert "overrides_alg" not in manifest
        for key in ("study_type", "env_params", "logs", "save_trials", "checkpoint_every",
                    "save_strat", "log_info", "log_type"):
            assert manifest[key] == master[key]
        assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
        assert manifest["logs"] == "timestamp" and manifest["save_trials"] == "none"
        validate_resume_selection(
            algorithm=config["alg"], observation_mode=config["alg_params"]["obs"],
            num_runs=1, save_trials=manifest["save_trials"],
            checkpoint_minutes=15, drain_after_seconds=3600,
        )
        assert config["alg_params"]["wandb"] is True
        assert config["alg_params"]["wandb_mode"] == "online"
        assert config["total_steps"] % 500 == 0
        identities.add(canonical_json(scientific_trial_parameters(config)))
        run_names.add(config["alg_params"]["wandb_run_name"])
    assert len(master["cells"]) == len(identities) == len(run_names) == 4


@pytest.mark.parametrize("baseline", BASELINES)
def test_each_pair_changes_only_regularizer_and_publication_labels(baseline):
    mpr, _ = _cell(baseline, "mpr")
    action, _ = _cell(baseline, "action_smoothness")
    mpr, action = _without_labels(mpr), _without_labels(action)
    for config in (mpr, action):
        for key in PAVE_COEFFICIENT_KEYS:
            config["alg_params"].pop(key)
    assert mpr == action
