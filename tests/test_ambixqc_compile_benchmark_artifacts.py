import copy
import importlib.util
import json
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALG_DIR = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_DIR = ROOT / "configs/dmcontrol/experiments"
EAGER_PRODUCTION = ALG_DIR / "ambixqc_humanoid_walk_state.json"
COMPILED_PRODUCTION = ALG_DIR / "ambixqc_humanoid_walk_state_compiled.json"
COMPILED_PRODUCTION_MANIFEST = (
    EXPERIMENT_DIR / "ambixqc_humanoid_walk_state_compiled.json"
)
TIMING_EAGER = ALG_DIR / "ambixqc_humanoid_walk_state_timing_eager.json"
TIMING_COMPILED = ALG_DIR / "ambixqc_humanoid_walk_state_timing_compiled.json"
TIMING_MANIFEST = (
    EXPERIMENT_DIR / "ambixqc_humanoid_walk_state_timing_pair.json"
)
BENCHMARK = ROOT / "tests/benchmarks/ambixqc_compute_throughput.py"
LAUNCHER = ROOT / "slurm/run_ambixqc_compile_pair_oscar.sbatch"
SUBMITTER = ROOT / "slurm/submit_ambixqc_compile_pair_oscar.sh"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _without_compile(config):
    config = copy.deepcopy(config)
    params = config["alg_params"]
    params.pop("compile")
    params.pop("compile_strict")
    return config


def test_production_compiled_sibling_changes_only_compile_and_wandb_identity():
    eager = _load_json(EAGER_PRODUCTION)
    compiled = _load_json(COMPILED_PRODUCTION)

    assert eager["alg_params"]["compile"] is False
    assert eager["alg_params"]["compile_strict"] is False
    assert compiled["alg_params"]["compile"] is True
    assert compiled["alg_params"]["compile_strict"] is True

    eager_scientific = _without_compile(eager)
    compiled_scientific = _without_compile(compiled)
    eager_params = eager_scientific["alg_params"]
    compiled_params = compiled_scientific["alg_params"]
    eager_group = eager_params.pop("wandb_group")
    compiled_group = compiled_params.pop("wandb_group")
    eager_tags = eager_params.pop("wandb_tags")
    compiled_tags = compiled_params.pop("wandb_tags")

    assert compiled_scientific == eager_scientific
    assert eager_group == "ambixqc-humanoid-walk-state-1m"
    assert compiled_group == "ambixqc-humanoid-walk-state-1m-compiled"
    assert set(compiled_tags) == set(eager_tags) | {"torch-compile-strict"}

    manifest = _load_json(COMPILED_PRODUCTION_MANIFEST)
    assert manifest["configs"] == ["ambixqc_humanoid_walk_state_compiled"]
    assert manifest["trials"] == 1
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["checkpoint_every"] == 100_000
    assert manifest["save_strat"] == ["best", "latest"]
    assert "strictly compiled sibling" in manifest["study_note"].lower()
    assert "rather than a statistical or paper reproduction" in manifest["study_note"]


def test_timing_pair_is_one_axis_and_keeps_exact_production_compute_shape():
    eager = _load_json(TIMING_EAGER)
    compiled = _load_json(TIMING_COMPILED)
    eager_params = eager["alg_params"]
    compiled_params = compiled["alg_params"]

    assert _without_compile(eager) == _without_compile(compiled)
    assert (eager_params["compile"], eager_params["compile_strict"]) == (
        False,
        False,
    )
    assert (compiled_params["compile"], compiled_params["compile_strict"]) == (
        True,
        True,
    )
    for config in (eager, compiled):
        params = config["alg_params"]
        assert config["seed"] == 55
        assert config["device"] == "cuda"
        assert config["total_steps"] == 1_502
        assert params["model_size"] == 5
        assert params["buffer_size"] == 2_000
        assert params["batch_size"] == 256
        assert params["seed_steps"] == 500
        assert params["pretrain_steps"] == 1
        assert params["utd"] == 1
        assert params["eval_freq"] is None
        assert params["wandb"] is False
        assert params["xqc_actor_net_arch"] == [256] * 4
        assert params["xqc_critic_net_arch"] == [512] * 4
        assert params["xqc_num_atoms"] == 101
        assert (
            params["inner_rounds"],
            params["inner_rollouts_per_round"],
            params["inner_rollout_horizon"],
            params["inner_updates_per_round"],
            params["inner_batch_size"],
            params["inner_replay_capacity"],
        ) == (2, 32, 3, 4, 64, 192)

    manifest = _load_json(TIMING_MANIFEST)
    assert manifest["study_type"] == "ambixqc_eager_compiled_paired_timing_canary"
    assert manifest["configs"] == [
        "ambixqc_humanoid_walk_state_timing_eager",
        "ambixqc_humanoid_walk_state_timing_compiled",
    ]
    assert manifest["trials"] == 1
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 1_502
    assert manifest["save_strat"] == ["latest"]
    assert manifest["checkpoint_best_window"] == 1
    assert "501 random actions" in manifest["study_note"]
    assert "1,001 planned inner-XQC actions" in manifest["study_note"]
    assert "1,002 outer updates" in manifest["study_note"]
    assert "one final portable latest checkpoint" in manifest["study_note"]


def test_exact_compute_benchmark_freezes_shape_workload_and_safety_contract():
    spec = importlib.util.spec_from_file_location(
        "ambixqc_compute_throughput", BENCHMARK
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defaults = module._parser().parse_args([])

    assert module.PRODUCTION_OBS_DIM == 67
    assert module.PRODUCTION_ACTION_DIM == 21
    assert module.PRODUCTION_LATENT_DIM == 512
    assert module.PRODUCTION_OUTER_BATCH == 256
    assert module.PRODUCTION_TRAIN_HORIZON == 3
    assert module.CANONICAL_INNER == {
        "rounds": 2,
        "rollouts_per_round": 32,
        "horizon": 3,
        "updates_per_round": 4,
        "batch_size": 64,
        "replay_capacity": 192,
    }
    assert defaults.device == "cuda"
    assert defaults.warmup == 10
    assert defaults.measured == 50
    assert defaults.compile is defaults.compile_strict is False

    contents = BENCHMARK.read_text(encoding="utf-8")
    assert "if args.warmup < 2:" in contents
    assert "_percentile(warmup_seconds[1:], 0.50)" in contents
    assert "_ShapeOnlyHumanoidEnv" in contents
    assert "agent.act(observation, collect_diagnostics=False)" in contents
    assert "agent.update(buffer)" in contents
    assert '"outer_compile_status"' in contents
    assert '"inner_compile_status"' in contents
    assert '"outer_optimizer_backend"' in contents
    assert '"inner_optimizer_backend"' in contents
    assert '"cold_cycle_seconds"' in contents
    assert '"cold_wall_seconds"' in contents
    assert '"cold_cuda_event_seconds"' in contents
    assert "torch.cuda.synchronize(device)" in contents
    assert '"iteration_p50_seconds"' in contents
    assert '"iteration_p95_seconds"' in contents
    assert '"actions_per_second"' in contents
    assert '"outer_updates_per_second"' in contents
    assert '"peak_cuda_allocated_bytes"' in contents
    assert '"peak_cuda_reserved_bytes"' in contents
    assert 'path.open("x"' in contents
    assert "import domains" not in contents
    assert "gym.make" not in contents
    assert BENCHMARK.stat().st_mode & stat.S_IXUSR


def test_oscar_pair_launcher_runs_both_arms_on_one_guarded_gpu():
    contents = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu" in contents
    assert "#SBATCH --gres=gpu:l40s:1" in contents
    assert "#SBATCH --cpus-per-task=6" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=02:00:00" in contents
    assert "#SBATCH --array" not in contents
    assert "#SBATCH --nodelist" not in contents
    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "AMBIXQC_ACTION_MODES_DIR" in contents
    assert "SLURM_SUBMIT_DIR" not in contents
    assert "^[0-9a-f]{40}$" in contents
    assert contents.count(
        'require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"'
    ) == 2
    assert "status --porcelain=v1 --untracked-files=all" in contents
    assert "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6" in contents
    assert "durable results must be outside the Git checkout" in contents
    assert '[[ ! -e "$JOB_ROOT" && ! -L "$JOB_ROOT" ]]' in contents
    assert 'readonly JOB_ROOT="$RESULTS_ROOT_REAL/$SLURM_JOB_ID"' in contents
    assert "/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-compile/$EXPECTED_ACTION_MODES_SHA" in contents
    assert "rm -rf" not in contents
    assert "git pull" not in contents
    assert "git fetch" not in contents
    assert "uv sync" not in contents
    assert "WANDB_MODE=disabled" in contents
    assert "WANDB_MODE=online" not in contents

    assert "ambixqc_compute_throughput.py" in contents
    assert "tests/test_ambixqc_compile.py" in contents
    assert "tests/test_compile_regions.py" in contents
    assert 'configure_process_cache "cuda-correctness-gate"' in contents
    assert "cuda-correctness-gate.log" in contents
    assert "--warmup 10" in contents
    assert "--measured 50" in contents
    assert "--no-compile --no-compile-strict" in contents
    assert "--compile --compile-strict --require-compiled" in contents
    assert "for repetition in 1 2 3 4 5" in contents
    assert "repetition % 2 == 1" in contents
    assert '[[ "$PAIR_ORDER" == eager-first ]]' in contents
    assert 'configure_process_cache "compute-$mode-rep$repetition"' in contents
    assert "TDMPC2_COMPUTE_TIMING_OUTPUT" in contents
    assert 'training-$mode-compute.json' in contents
    assert "XDG_CACHE_HOME" in contents
    assert "TORCHINDUCTOR_CACHE_DIR" in contents
    assert "TRITON_CACHE_DIR" in contents
    assert "CUDA_CACHE_PATH" in contents
    assert "compute-aggregate.json" in contents
    assert '"compiled_p50_ratio_at_most_0p90"' in contents
    assert '"compiled_p95_ratio_at_most_0p95"' in contents
    assert '"maximum_p50_cv_at_most_0p05"' in contents
    assert '"paired_compiled_p50_ratio_at_most_1p10"' in contents
    assert '"peak_allocation_ratio_at_most_1p10"' in contents
    assert '"compile_cost_break_even_actions"' in contents
    assert '"cold_process_elapsed_seconds"' in contents
    assert '"warmed_compute_seconds"' in contents
    assert '"exclude-first-planned-action-and-first-outer-update"' in contents
    assert '"warmed_canary_compute_speedup_at_least_1p05"' in contents
    assert '"cold_process_speedup"' in contents
    assert '"correctness-pass-performance-miss"' in contents
    assert 'root / "PERFORMANCE_MISS"' in contents
    assert "ambixqc_humanoid_walk_state_timing_pair.json" in contents
    assert "index=0" in contents and "index=1" in contents
    assert 'expected_counters = (1002, 1002, 1002, 334, 334, 1001)' in contents
    assert 'checkpoint.get("step") != 1502' in contents
    assert "p50_ratio_compiled_over_eager" in contents
    assert "warmed_canary_compute_speedup" in contents
    assert 'printf \'PASS\\n\' > "$JOB_ROOT/PASS"' in contents


def test_oscar_submitter_requires_git_transport_and_new_external_artifacts():
    contents = SUBMITTER.read_text(encoding="utf-8")

    assert "--expected-action-modes-sha" in contents
    assert "--results-root" in contents
    assert "--pair-id" not in contents
    assert "--order eager-first|compiled-first" in contents
    assert "--dry-run" in contents
    assert "submission requires a clean checkout" in contents
    assert "durable results must be outside the Git checkout" in contents
    assert "/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-compile/$EXPECTED_ACTION_MODES_SHA" in contents
    assert "AMBIXQC_BENCHMARK_RESULTS_ROOT=$AMBIXQC_BENCHMARK_RESULTS_ROOT" in contents
    assert "AMBIXQC_ACTION_MODES_DIR=$PROJECT_DIR" in contents
    assert "AMBIXQC_PAIR_ORDER=$AMBIXQC_PAIR_ORDER" in contents
    assert "--parsable" in contents
    assert '--chdir="$PROJECT_DIR"' in contents
    assert "sbatch" in contents
    assert "scancel" not in contents
    assert "rm -rf" not in contents


def test_new_scripts_are_executable_valid_bash_and_documented():
    for path in (LAUNCHER, SUBMITTER):
        assert path.stat().st_mode & stat.S_IXUSR
        subprocess.run(["bash", "-n", str(path)], check=True)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "### AMBI-XQC compiled execution and paired timing" in readme
    assert "ambixqc_compute_throughput.py" in readme
    assert "ambixqc_humanoid_walk_state_timing_pair.json" in readme
    assert "submit_ambixqc_compile_pair_oscar.sh" in readme
    assert "First commit and push" in readme
    assert "/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-compile/<full-SHA>/<job-id>" in readme
    assert "five independent compute processes per mode" in readme
    assert "mandatory AMBI-XQC and shared compile-region CUDA correctness suites" in readme
    assert "cold whole-process wall time" in readme
    assert "excludes the first planned" in readme
    assert "inner action and the first outer update independently" in readme
    assert "not used for the five-percent" in readme
    assert "classification" in readme
    assert "greater than one favor compilation" in readme.lower()
