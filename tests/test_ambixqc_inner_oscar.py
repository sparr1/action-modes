from pathlib import Path
import subprocess


def test_inner_oscar_launcher_preserves_runtime_and_reference_contract():
    path = Path(__file__).resolve().parents[1] / "slurm/run_ambixqc_inner_eval_oscar.sbatch"
    content = path.read_text()
    for guard in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=6",
                  "#SBATCH --mem=32G", "#SBATCH --time=02:00:00", "#SBATCH --no-requeue",
                  "EXPECTED_ACTION_MODES_SHA", "--untracked-files=all", "LOCK_SHA",
                  "run_ambixqc_inner_evaluation.py", "--wandb", "WANDB_CACHE_DIR",
                  "tests/test_ambixqc_inner_j6.py", "smoke_reference_manifest_sha256",
                  "--smoke-reference-manifest-sha256", "export CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        assert guard in content
    assert "pip install" not in content and "uv sync" not in content
    assert "run_ambixqc_mppi_evaluation.py" not in content
    subprocess.run(["/bin/bash", "-n", str(path)], check=True, close_fds=False)


def test_source_fingerprint_covers_inner_campaign_entrypoints(tmp_path, monkeypatch):
    from utils import ambi_benchmark as storage
    module = tmp_path / "utils/ambi_benchmark.py"
    module.parent.mkdir()
    module.write_text("# identity fixture\n")
    monkeypatch.setattr(storage, "__file__", str(module))
    monkeypatch.setattr(storage.subprocess, "check_output", lambda *args, **kwargs: b"")
    before = storage.code_identity()["source_sha256"]
    for name in ("run_ambixqc_inner_evaluation.py", "summarize_ambixqc_inner_eval.py"):
        (tmp_path / name).write_text("# changed evaluation semantics\n")
        after = storage.code_identity()["source_sha256"]
        assert after != before
        before = after
