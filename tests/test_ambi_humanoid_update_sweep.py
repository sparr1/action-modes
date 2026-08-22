import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "configs/ambi/algs/ambi_anchor.json"
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
UPDATES = (2, 4, 8, 16)
BASE_G4 = "ambi_humanoid_walk_updates_g4"
BASE_V1 = "ambi_humanoid_walk_base"
BASE_V1_LORA_R8 = "ambi_humanoid_walk_base_lora_r8"
BASE_V1_LORA_R16 = "ambi_humanoid_walk_base_lora_r16"
BASE_V1_ACTOR_MEAN_PAIR = "ambi_humanoid_walk_base_actor_mean_pair"
BASE_V1_MIN_ALL = "ambi_humanoid_walk_base_min_all"
BASE_V1_PERCENTILE_NORMALIZED = "ambi_humanoid_walk_base_percentile_normalized"
BASE_V1_STUDY_NOTE = (
    "Single-seed exploratory 14-million-decision Humanoid Walk base-v1 run "
    "using the G4 inner schedule and selected inner critic learning rate with "
    "TD-MPC2-aligned outer recipe parameters; make no confidence, "
    "significance, or confirmatory claims."
)
BASE_V1_LORA_R8_STUDY_NOTE = (
    "Single-seed exploratory 14-million-decision Humanoid Walk base-v1 G4 run "
    "changing only the inner actor and critic from full cloned modules to "
    "rank-8 LoRA adapters with alpha=16 (direct scale=2); all other training "
    "recipe parameters match the base. Make no confidence, significance, or "
    "confirmatory claims."
)
BASE_V1_LORA_R16_STUDY_NOTE = (
    "Single-seed exploratory 14-million-decision Humanoid Walk base-v1 G4 run "
    "changing only the inner actor and critic from full cloned modules to "
    "rank-16 LoRA adapters with alpha=32 (direct scale=2); all other training "
    "recipe parameters match the base. Make no confidence, significance, or "
    "confirmatory claims."
)
BASE_V1_ACTOR_MEAN_PAIR_STUDY_NOTE = (
    "Single-seed exploratory 14-million-decision Humanoid Walk base-v1 run "
    "changing only the outer and inner policy-learning Q reductions from a "
    "random-pair minimum to a random-pair mean; target-Q reductions remain "
    "min_pair. Make no confidence, significance, or confirmatory claims."
)
BASE_V1_MIN_ALL_STUDY_NOTE = (
    "Single-seed exploratory 14-million-decision Humanoid Walk base-v1 G4 "
    "anchor with five Q heads, min_all for all outer and inner actor/target "
    "reductions, entropy-augmented outer and inner critic targets, an "
    "adaptable cloned inner critic, automatic outer and inner entropy "
    "coefficients, and no actor-loss percentile scaling. Make no confidence, "
    "significance, or confirmatory claims."
)
BASE_V1_PERCENTILE_NORMALIZED_STUDY_NOTE = (
    "Single-seed exploratory 14-million-decision Humanoid Walk base-v1 G4 "
    "run changing only SAC actor-loss preconditioning to the TD-MPC2 P5-P95 "
    "running range with tau=0.01; rewards, soft Bellman targets, and automatic "
    "entropy losses remain in raw units. Make no confidence, significance, or "
    "confirmatory claims."
)
BASE_V1_TAGS = [
    "ambi",
    "dmcontrol",
    "humanoid-walk",
    "state",
    "inner-alpha-auto",
    "base-v1",
    "tdmpc2-aligned-recipe",
    "14m-decisions",
    "g4",
    "critic-lr1e-4",
    "actor-lr5e-5",
]
BASE_V1_VARIANTS = {
    "ambi_humanoid_walk_base_j2_n32_g8": (2, 32, 8),
    "ambi_humanoid_walk_base_j4_n32_g2": (4, 32, 2),
    "ambi_humanoid_walk_base_j4_n32_g4": (4, 32, 4),
    "ambi_humanoid_walk_base_j2_n64_g2": (2, 64, 2),
    "ambi_humanoid_walk_base_j2_n64_g4": (2, 64, 4),
    "ambi_humanoid_walk_base_j2_n64_g8": (2, 64, 8),
    "ambi_humanoid_walk_base_j2_n128_g4": (2, 128, 4),
}
LR_CELLS = {
    "ambi_humanoid_walk_updates_g4_critic_lr1e4": {
        "inner_actor_lr": 5e-5,
        "inner_critic_lr": 1e-4,
        "wandb_run_name": (
            "AMBITDMPC2-humanoid-walk-g4-critic-lr1e-4-seed55"
        ),
        "wandb_tags": [
            "ambi",
            "dmcontrol",
            "humanoid-walk",
            "state",
            "inner-alpha-auto",
            "inner-learning-rate-screen",
            "g4",
            "critic-lr1e-4",
            "actor-lr5e-5",
        ],
        "study_note": (
            "Single-seed exploratory Humanoid Walk G4 run doubling only the "
            "AMBI inner critic learning rate; make no confidence, significance, "
            "or confirmatory claims."
        ),
    },
    "ambi_humanoid_walk_updates_g4_both_lr1e4": {
        "inner_actor_lr": 1e-4,
        "inner_critic_lr": 1e-4,
        "wandb_run_name": "AMBITDMPC2-humanoid-walk-g4-both-lr1e-4-seed55",
        "wandb_tags": [
            "ambi",
            "dmcontrol",
            "humanoid-walk",
            "state",
            "inner-alpha-auto",
            "inner-learning-rate-screen",
            "g4",
            "critic-lr1e-4",
            "actor-lr1e-4",
        ],
        "study_note": (
            "Single-seed exploratory Humanoid Walk G4 run doubling both AMBI "
            "inner actor and critic learning rates; make no confidence, "
            "significance, or confirmatory claims."
        ),
    },
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _name(updates: int) -> str:
    return f"ambi_humanoid_walk_updates_g{updates}"


def _base_variant_tags(rounds: int, rollouts: int, updates: int) -> list[str]:
    return [
        "ambi",
        "dmcontrol",
        "humanoid-walk",
        "state",
        "inner-alpha-auto",
        "base-v1",
        "tdmpc2-aligned-recipe",
        "14m-decisions",
        "runtime-variant",
        f"j{rounds}",
        f"n{rollouts}",
        f"g{updates}",
        "critic-lr1e-4",
        "actor-lr5e-5",
    ]


def _base_variant_study_note(rounds: int, rollouts: int, updates: int) -> str:
    return (
        "Single-seed exploratory 14-million-decision Humanoid Walk base-v1 "
        f"runtime variant with J={rounds}, N={rollouts}, H=3, and G={updates}; "
        "all other recipe parameters match the base. Make no confidence, "
        "significance, or confirmatory claims."
    )


def test_humanoid_update_sweep_changes_only_task_and_update_dose_from_anchor():
    anchor = _load(ANCHOR)
    run_names = set()

    for updates in UPDATES:
        name = _name(updates)
        actual = _load(ALGORITHM_ROOT / f"{name}.json")
        expected = copy.deepcopy(anchor)
        expected["env"] = "DMControl-v0"
        expected_params = expected["alg_params"]
        expected_params["obs"] = "state"
        expected_params["inner_updates_per_round"] = updates
        expected_params["wandb_mode"] = "online"
        expected_params["wandb_run_name"] = (
            f"AMBITDMPC2-humanoid-walk-g{updates}-seed55"
        )
        expected_params["wandb_tags"] = [
            "ambi",
            "dmcontrol",
            "humanoid-walk",
            "state",
            "inner-alpha-auto",
            "inner-updates-screen",
            f"g{updates}",
        ]

        assert actual == expected
        assert "wandb_group" not in actual["alg_params"]
        run_names.add(actual["alg_params"]["wandb_run_name"])

    assert len(run_names) == len(UPDATES)


def test_humanoid_update_sweep_is_log_spaced_and_keeps_anchor_work_budget():
    total_slots = []
    for updates in UPDATES:
        params = _load(ALGORITHM_ROOT / f"{_name(updates)}.json")["alg_params"]
        assert (
            params["inner_rounds"],
            params["inner_rollouts_per_round"],
            params["inner_rollout_horizon"],
            params["inner_batch_size"],
            params["inner_replay_capacity"],
        ) == (2, 32, 3, 64, 192)
        assert params["inner_temperature_mode"] == "auto"
        assert params["inner_temperature_initialization"] == "inherit_outer"
        assert params["inner_target_entropy"] == "inherit_outer"
        assert params["q_representation"] == "distributional"
        assert params["q_pair_size"] == 2
        assert params["compile"] is True
        total_slots.append(params["inner_rounds"] * updates)

    assert total_slots == [4, 8, 16, 32]


def test_each_humanoid_update_cell_is_an_independent_resumable_screen():
    for updates in UPDATES:
        name = _name(updates)
        algorithm = _load(ALGORITHM_ROOT / f"{name}.json")
        experiment = _load(EXPERIMENT_ROOT / f"{name}.json")

        assert algorithm["seed"] == 55
        assert algorithm["total_steps"] == 1_000_000
        assert algorithm["total_steps"] % 500 == 0
        assert experiment["study_type"] == (
            "single_seed_exploratory_inner_update_screening"
        )
        assert "no confidence" in experiment["study_note"].lower()
        assert experiment["overrides_alg"] == {
            "seed": 55,
            "device": "cuda",
            "env": "DMControl-v0",
            "total_steps": 1_000_000,
            "episodes": None,
        }
        assert experiment["env_params"] == {
            "task": "humanoid-walk",
            "obs": "state",
            "render_mode": None,
        }
        assert experiment["trials"] == 1
        assert experiment["configs"] == [name]
        assert experiment["logs"] == "timestamp"
        assert experiment["save_trials"] == "none"
        assert experiment["checkpoint_every"] == 100_000
        assert experiment["save_strat"] == ["best", "latest"]
        assert experiment["checkpoint_best_window"] == 100
        assert experiment["log_type"] == "summary"


def test_g4_learning_rate_cells_change_only_the_selected_inner_optimizers():
    baseline = _load(ALGORITHM_ROOT / f"{BASE_G4}.json")

    for name, cell in LR_CELLS.items():
        actual = _load(ALGORITHM_ROOT / f"{name}.json")
        expected = copy.deepcopy(baseline)
        expected["alg_params"].update(
            {
                "inner_actor_lr": cell["inner_actor_lr"],
                "inner_critic_lr": cell["inner_critic_lr"],
                "wandb_run_name": cell["wandb_run_name"],
                "wandb_tags": cell["wandb_tags"],
            }
        )

        # This full equality keeps the cell a deep copy of G4 except for the
        # two selected optimizer rates and its explicit W&B identity.
        assert actual == expected
        assert actual["alg_params"]["actor_lr"] == 3e-4
        assert actual["alg_params"]["critic_lr"] == 3e-4
        assert actual["alg_params"]["inner_temperature_lr"] == 5e-5
        assert actual["alg_params"]["inner_updates_per_round"] == 4


def test_g4_learning_rate_cells_have_distinct_single_run_wandb_identities():
    algorithms = [
        _load(ALGORITHM_ROOT / f"{name}.json")
        for name in (BASE_G4, *LR_CELLS)
    ]
    run_names = [algorithm["alg_params"]["wandb_run_name"] for algorithm in algorithms]

    assert len(run_names) == len(set(run_names))
    for algorithm in algorithms[1:]:
        params = algorithm["alg_params"]
        tags = set(params["wandb_tags"])
        assert params["wandb"] is True
        assert params["wandb_mode"] == "online"
        assert "wandb_group" not in params
        assert {"ambi", "dmcontrol", "humanoid-walk", "state", "g4"} <= tags
        assert any("lr" in tag.lower() for tag in tags)


def test_g4_learning_rate_cells_reuse_the_resumable_humanoid_protocol():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE_G4}.json")

    for name, cell in LR_CELLS.items():
        actual = _load(EXPERIMENT_ROOT / f"{name}.json")
        expected = copy.deepcopy(baseline)
        expected["study_type"] = (
            "single_seed_exploratory_inner_learning_rate_screening"
        )
        expected["study_note"] = cell["study_note"]
        expected["configs"] = [name]

        assert actual == expected


def test_humanoid_base_v1_changes_only_the_selected_tdmpc2_recipe_parameters():
    selected = _load(
        ALGORITHM_ROOT / "ambi_humanoid_walk_updates_g4_critic_lr1e4.json"
    )
    actual = _load(ALGORITHM_ROOT / f"{BASE_V1}.json")
    expected = copy.deepcopy(selected)
    expected["alg_params"].update(
        {
            "rho": 0.5,
            "critic_coef": 0.1,
            "tau": 0.01,
            "actor_adam_eps": 1e-5,
            "wandb_run_name": "AMBITDMPC2-humanoid-walk-base-v1-g4-seed55",
            "wandb_tags": BASE_V1_TAGS,
        }
    )
    expected["total_steps"] = 14_000_000

    assert actual == expected
    params = actual["alg_params"]
    assert actual["total_steps"] == 14_000_000
    assert params["inner_updates_per_round"] == 4
    assert params["inner_actor_lr"] == 5e-5
    assert params["inner_critic_lr"] == 1e-4
    assert params["adam_eps"] == 1e-8
    assert params["inner_adam_eps"] == 1e-8
    assert params["inner_critic_target_tau"] == 0.005


def test_humanoid_base_v1_has_a_matching_runnable_manifest():
    selected = _load(
        EXPERIMENT_ROOT / "ambi_humanoid_walk_updates_g4_critic_lr1e4.json"
    )
    actual = _load(EXPERIMENT_ROOT / f"{BASE_V1}.json")
    expected = copy.deepcopy(selected)
    expected.update(
        {
            "study_type": "single_seed_exploratory_base",
            "study_note": BASE_V1_STUDY_NOTE,
            "configs": [BASE_V1],
        }
    )
    expected["overrides_alg"]["total_steps"] = 14_000_000

    assert actual == expected
    assert actual["overrides_alg"]["total_steps"] == 14_000_000


def test_humanoid_base_v1_lora_r8_changes_only_inner_adaptation():
    baseline = _load(ALGORITHM_ROOT / f"{BASE_V1}.json")
    actual = _load(ALGORITHM_ROOT / f"{BASE_V1_LORA_R8}.json")
    expected = copy.deepcopy(baseline)
    expected["alg_params"].update(
        {
            "inner_actor_adaptation": "lora",
            "inner_critic_adaptation": "lora",
            "inner_actor_lora_rank": 8,
            "inner_actor_lora_scale": 2.0,
            "inner_actor_lora_dropout": 0.0,
            "inner_critic_lora_rank": 8,
            "inner_critic_lora_scale": 2.0,
            "inner_critic_lora_dropout": 0.0,
            "wandb_run_name": (
                "AMBITDMPC2-humanoid-walk-base-v1-g4-"
                "lora-r8-alpha16-seed55"
            ),
            "wandb_tags": [
                *BASE_V1_TAGS,
                "inner-lora",
                "lora-r8",
                "lora-alpha16",
                "lora-scale2",
            ],
        }
    )

    assert actual == expected
    params = actual["alg_params"]
    assert (
        params["inner_actor_lora_scale"] * params["inner_actor_lora_rank"] == 16
    )
    assert (
        params["inner_critic_lora_scale"] * params["inner_critic_lora_rank"]
        == 16
    )
    assert actual["seed"] == 55
    assert actual["total_steps"] == 14_000_000


def test_humanoid_base_v1_lora_r8_has_a_matching_manifest():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE_V1}.json")
    actual = _load(EXPERIMENT_ROOT / f"{BASE_V1_LORA_R8}.json")
    expected = copy.deepcopy(baseline)
    expected.update(
        {
            "study_type": "single_seed_exploratory_inner_adaptation",
            "study_note": BASE_V1_LORA_R8_STUDY_NOTE,
            "configs": [BASE_V1_LORA_R8],
        }
    )

    assert actual == expected


def test_humanoid_base_v1_lora_r16_changes_only_rank_from_r8():
    rank8 = _load(ALGORITHM_ROOT / f"{BASE_V1_LORA_R8}.json")
    actual = _load(ALGORITHM_ROOT / f"{BASE_V1_LORA_R16}.json")
    expected = copy.deepcopy(rank8)
    expected["alg_params"].update(
        {
            "inner_actor_lora_rank": 16,
            "inner_critic_lora_rank": 16,
            "wandb_run_name": (
                "AMBITDMPC2-humanoid-walk-base-v1-g4-"
                "lora-r16-alpha32-seed55"
            ),
            "wandb_tags": [
                tag.replace("lora-r8", "lora-r16").replace(
                    "lora-alpha16", "lora-alpha32"
                )
                for tag in rank8["alg_params"]["wandb_tags"]
            ],
        }
    )

    assert actual == expected
    params = actual["alg_params"]
    assert params["inner_actor_lora_scale"] == 2.0
    assert params["inner_critic_lora_scale"] == 2.0
    assert (
        params["inner_actor_lora_scale"] * params["inner_actor_lora_rank"] == 32
    )
    assert (
        params["inner_critic_lora_scale"] * params["inner_critic_lora_rank"]
        == 32
    )
    assert actual["seed"] == 55
    assert actual["total_steps"] == 14_000_000


def test_humanoid_base_v1_lora_r16_has_a_matching_manifest():
    rank8 = _load(EXPERIMENT_ROOT / f"{BASE_V1_LORA_R8}.json")
    actual = _load(EXPERIMENT_ROOT / f"{BASE_V1_LORA_R16}.json")
    expected = copy.deepcopy(rank8)
    expected.update(
        {
            "study_note": BASE_V1_LORA_R16_STUDY_NOTE,
            "configs": [BASE_V1_LORA_R16],
        }
    )

    assert actual == expected


def test_humanoid_base_v1_actor_mean_pair_changes_only_policy_q_reductions():
    baseline = _load(ALGORITHM_ROOT / f"{BASE_V1}.json")
    actual = _load(ALGORITHM_ROOT / f"{BASE_V1_ACTOR_MEAN_PAIR}.json")
    expected = copy.deepcopy(baseline)
    expected["alg_params"].update(
        {
            "outer_q_actor_reduction": "mean_pair",
            "inner_q_actor_reduction": "mean_pair",
            "wandb_run_name": (
                "AMBITDMPC2-humanoid-walk-base-v1-g4-"
                "actor-mean-pair-seed55"
            ),
            "wandb_tags": [*BASE_V1_TAGS, "actor-q-mean-pair"],
        }
    )

    assert actual == expected
    params = actual["alg_params"]
    assert params["outer_q_target_reduction"] == "min_pair"
    assert params["inner_q_target_reduction"] == "min_pair"
    assert params["q_pair_size"] == 2
    assert params["inner_temperature_mode"] == "auto"
    assert params["inner_temperature_initialization"] == "inherit_outer"
    assert params["inner_target_entropy"] == "inherit_outer"
    assert actual["seed"] == 55
    assert actual["total_steps"] == 14_000_000


def test_humanoid_base_v1_actor_mean_pair_has_a_matching_manifest():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE_V1}.json")
    actual = _load(EXPERIMENT_ROOT / f"{BASE_V1_ACTOR_MEAN_PAIR}.json")
    expected = copy.deepcopy(baseline)
    expected.update(
        {
            "study_type": "single_seed_exploratory_actor_q_reduction",
            "study_note": BASE_V1_ACTOR_MEAN_PAIR_STUDY_NOTE,
            "configs": [BASE_V1_ACTOR_MEAN_PAIR],
        }
    )

    assert actual == expected


def test_humanoid_base_v1_min_all_pins_full_critic_recipe():
    baseline = _load(ALGORITHM_ROOT / f"{BASE_V1}.json")
    actual = _load(ALGORITHM_ROOT / f"{BASE_V1_MIN_ALL}.json")
    expected = copy.deepcopy(baseline)
    expected["alg_params"].update(
        {
            "num_q": 5,
            "outer_q_target_reduction": "min_all",
            "outer_q_actor_reduction": "min_all",
            "inner_q_target_reduction": "min_all",
            "inner_q_actor_reduction": "min_all",
            "outer_critic_target": "entropy_augmented",
            "inner_sac_critic_target": "entropy_augmented",
            "sac_actor_loss_scale_mode": "none",
            "wandb_run_name": (
                "AMBITDMPC2-humanoid-walk-base-v1-g4-min-all-seed55"
            ),
            "wandb_tags": [
                *BASE_V1_TAGS,
                "q-min-all",
                "q-heads-5",
                "critic-target-entropy-augmented",
                "inner-critic-clone",
                "outer-alpha-auto",
            ],
        }
    )

    assert actual == expected
    params = actual["alg_params"]
    assert params["q_representation"] == "distributional"
    assert params["q_pair_size"] == 2
    assert {
        params["outer_q_target_reduction"],
        params["outer_q_actor_reduction"],
        params["inner_q_target_reduction"],
        params["inner_q_actor_reduction"],
    } == {"min_all"}
    assert actual["seed"] == 55
    assert actual["total_steps"] == 14_000_000


def test_humanoid_base_v1_min_all_has_a_matching_manifest():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE_V1}.json")
    actual = _load(EXPERIMENT_ROOT / f"{BASE_V1_MIN_ALL}.json")
    expected = copy.deepcopy(baseline)
    expected.update(
        {
            "study_type": "single_seed_exploratory_all_head_q_minimum",
            "study_note": BASE_V1_MIN_ALL_STUDY_NOTE,
            "configs": [BASE_V1_MIN_ALL],
        }
    )

    assert actual == expected


def test_humanoid_base_v1_percentile_variant_changes_only_actor_loss_scaling():
    baseline = _load(ALGORITHM_ROOT / f"{BASE_V1}.json")
    actual = _load(
        ALGORITHM_ROOT / f"{BASE_V1_PERCENTILE_NORMALIZED}.json"
    )
    expected = copy.deepcopy(baseline)
    expected["alg_params"].update(
        {
            "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
            "sac_actor_loss_scale_tau": 0.01,
            "wandb_run_name": (
                "AMBITDMPC2-humanoid-walk-base-v1-g4-"
                "percentile-normalized-seed55"
            ),
            "wandb_tags": [*BASE_V1_TAGS, "tdmpc2-percentile-range"],
        }
    )

    assert actual == expected
    assert actual["seed"] == 55
    assert actual["total_steps"] == 14_000_000


def test_humanoid_base_v1_percentile_variant_has_a_matching_manifest():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE_V1}.json")
    actual = _load(
        EXPERIMENT_ROOT / f"{BASE_V1_PERCENTILE_NORMALIZED}.json"
    )
    expected = copy.deepcopy(baseline)
    expected.update(
        {
            "study_type": "single_seed_exploratory_actor_loss_scaling",
            "study_note": BASE_V1_PERCENTILE_NORMALIZED_STUDY_NOTE,
            "configs": [BASE_V1_PERCENTILE_NORMALIZED],
        }
    )

    assert actual == expected


def test_humanoid_base_v1_runtime_variants_change_only_j_n_g_and_capacity():
    baseline = _load(ALGORITHM_ROOT / f"{BASE_V1}.json")
    run_names = {baseline["alg_params"]["wandb_run_name"]}

    for name, (rounds, rollouts, updates) in BASE_V1_VARIANTS.items():
        actual = _load(ALGORITHM_ROOT / f"{name}.json")
        expected = copy.deepcopy(baseline)
        expected["alg_params"].update(
            {
                "inner_rounds": rounds,
                "inner_rollouts_per_round": rollouts,
                "inner_updates_per_round": updates,
                "inner_replay_capacity": rounds * rollouts * 3,
                "wandb_run_name": (
                    "AMBITDMPC2-humanoid-walk-base-v1-"
                    f"j{rounds}-n{rollouts}-g{updates}-seed55"
                ),
                "wandb_tags": _base_variant_tags(rounds, rollouts, updates),
            }
        )

        assert actual == expected
        params = actual["alg_params"]
        assert actual["total_steps"] == 14_000_000
        assert params["inner_replay_capacity"] == rounds * rollouts * 3
        assert rounds * updates > 0
        assert "wandb_group" not in params
        run_names.add(params["wandb_run_name"])

    assert len(run_names) == len(BASE_V1_VARIANTS) + 1


def test_humanoid_base_v1_runtime_variants_have_matching_manifests():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE_V1}.json")

    for name, (rounds, rollouts, updates) in BASE_V1_VARIANTS.items():
        actual = _load(EXPERIMENT_ROOT / f"{name}.json")
        expected = copy.deepcopy(baseline)
        expected.update(
            {
                "study_type": "single_seed_exploratory_base_runtime_variant",
                "study_note": _base_variant_study_note(
                    rounds, rollouts, updates
                ),
                "configs": [name],
            }
        )

        assert actual == expected
        assert actual["overrides_alg"]["total_steps"] == 14_000_000
