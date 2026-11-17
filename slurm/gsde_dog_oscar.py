"""Prepare the two seed-55 Dog transfers of the pinned gSDE8 learner."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import gsde_noise_scopes_oscar as base

SOURCE_COMMIT = "edec58cf7ec2b51bd54834aa4d6f806715075502"
SOURCE_ROOT = Path("/oscar/scratch/rgao48/ambi/gsde-noise-scopes-20261105")


def read(path):
    return json.loads(Path(path).read_text())


def verify_recipe():
    variant = os.environ["AMBI_GSDE_VARIANT"]
    assert variant in {"dog_trot", "dog_run"}
    folder = ROOT / "configs/dmcontrol"
    source = read(folder / "algs/ambi_gsde_squashed_real_steps.json")
    target = read(folder / f"algs/ambi_gsde_squashed_real_steps_{variant}.json")
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items()
                                if not k.startswith("wandb")}
    assert source == target
    original = read(folder / "experiments/ambi_gsde_noise_scopes.json")
    original["env_params"]["task"] = variant.replace("_", "-")
    manifest = read(base.MANIFEST)
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert original[key] == manifest[key], key
    assert manifest["configs"] == [f"ambi_gsde_squashed_real_steps_{variant}"]


def prepare():
    verify_recipe()
    campaign = base.CAMPAIGN_ROOT
    assert campaign == Path(os.environ["AMBI_GSDE_CAMPAIGN"])
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT)
    protected = ["RL", "domains", "utils", "environments", "main.py", base.GPU_PARITY_TEST]
    subprocess.run(["git", "diff", "--exit-code", SOURCE_COMMIT, "HEAD", "--", *protected],
                   cwd=ROOT, check=True)
    old = read(SOURCE_ROOT / "gpu-parity.json")
    assert old["binding"]["commit"] == SOURCE_COMMIT
    base.validate_global_parity(old, old["binding"], old["runtime"])
    for relative, expected in [(old["binding"]["manifest"], old["binding"]["manifest_sha256"]),
                               ("environments/dmcontrol/uv.lock", old["binding"]["lock_sha256"]),
                               *[(f"configs/dmcontrol/algs/{name}.json", digest)
                                 for name, digest in old["binding"]["config_sha256"].items()]]:
        content = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    junit = Path(old["junit_path"])
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == old["junit_sha256"]
    assert base.parity_cases(junit) == old["case_names"]
    campaign.mkdir(parents=True, exist_ok=False)
    for name in ("gates", "production", "logs"):
        (campaign / name).mkdir()
    copied = campaign / "reused-gpu-parity.xml"
    with copied.open("xb") as stream:
        stream.write(junit.read_bytes())
    evidence = {"kind": "reused_unchanged_learner", "original_commit": SOURCE_COMMIT,
                "original_receipt": str(SOURCE_ROOT / "gpu-parity.json"),
                "protected_source_paths": protected,
                "new_validation": "Fresh full-size Dog gate on one/two CPUs and native execution, including geometry and checkpoint roundtrip."}
    bound = base.binding(base.MANIFEST)
    base.write_new(campaign / "gpu-parity.json", {
        **old, "binding": bound, "junit_path": str(copied),
        "original_binding": old["binding"], "source_evidence": evidence})
    reference = read(SOURCE_ROOT / "gates/5/cpu-1.json")
    space = base.checkpoint_space(campaign, 2 * 40 * reference["checkpoint_bytes"])
    base.write_new(campaign / "preflight.json", {"binding": bound, "source_evidence": evidence,
                                              "space": space})
    variant = os.environ["AMBI_GSDE_VARIANT"]
    task_label = variant.replace("_", " ").title()
    control = {"dog_trot": "409d693629b04b6393ee277443e02a86",
               "dog_run": "def5f8fd7b294f53ab3b1e666f34f040"}[variant]
    state = {"commit": bound["commit"], "output": str(campaign),
             "workspace_name": f"gSDE8 sampled — {task_label} seed55 — Oscar",
             "protocol": f"**{task_label}: gSDE8 sampled, seed55, 1,000,000 decisions.**\n\n"
             "Same pinned learner and recipe as the completed Humanoid Walk gSDE8 reference; "
             "only the task and publication identity change. Joint post-tanh entropy alpha0.0021, "
             "H3/J8/K1/N128/C16/A4/B256, reward-only critics, action-local adaptation, "
             "real_steps gSDE noise persistence and execution resampling every8 decisions. "
             "All40 checkpoints retained every25k; critic geometry every1000 updates. "
             "GPU gates have W&B disabled; full training starts only after validation. "
             "CPU sizing compares one/two cores; walltime uses measured throughput and "
             "completed reference overhead with a15% margin plus30minutes. "
             "The existing post-tanh sampled seed55 run is included for comparison.",
             "runs": [{"id": uuid.uuid4().hex, "label": f"gSDE8 · sampled · {task_label} · seed55",
                       "seed": 55, "task": variant.replace("_", "-"), "color": "#2c7fb8"}],
             "controls": [{"id": control, "label": f"Post-tanh · sampled · {task_label} · seed55",
                           "color": "#e7893c"}]}
    base.write_new(campaign / "submission.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    assert sys.argv[1:] == ["prepare"]
    prepare()
