import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "slurm"))
spec = importlib.util.spec_from_file_location("aux_repro", ROOT / "slurm/reproduce_original_prior_aux.py")
aux = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aux)


def test_auxiliary_recipe_preserves_every_other_historical_setting():
    historical = json.loads((ROOT / aux.base.ALG).read_text())
    run, manifest = aux.recipe("test")
    params = run["alg_params"]
    assert params["aux_return_mode"] == "sac"
    assert params["aux_return_detach_representation"] is True
    assert params["aux_return_critic_coef"] == historical["alg_params"]["critic_coef"] == .1
    assert params["aux_return_critic_lr"] == historical["alg_params"]["critic_lr"] == .0003
    assert params["target_entropy"] == -21
    for key in list(params):
        if key.startswith("aux_return_"):
            params.pop(key)
    for key in (*aux.base.PUBLICATION_KEYS, "target_entropy"):
        params[key] = historical["alg_params"][key]
    assert run == historical
    expected = json.loads((ROOT / aux.base.MANIFEST).read_text())
    for key in ("study_type", "study_note", "configs"):
        manifest[key] = expected[key]
    assert manifest == expected


def test_resource_estimate_uses_slowest_occupancy_and_checkpoint_io():
    cases = [dict(cores=n, early_seconds_per_decision=e, full_seconds_per_decision=f,
                  startup_seconds=200, checkpoint_seconds=2)
             for n,e,f in [(1,.05,.09),(2,.06,.07),(4,.06,.069)]]
    result = aux.choose_resources(cases)
    assert result["cpus"] == 2
    assert result["seconds_per_decision"] == .07
    assert result["walltime_minutes"] * 60 >= .07 * 1_500_000 * 1.2 + 200 + 120 + 300


def test_launcher_syntax_and_shared_runtime_contract():
    path = ROOT / "slurm/run_original_prior_aux_hydra.sbatch"
    subprocess.run(["bash", "-n", str(path)], check=True)
    script = path.read_text()
    assert "WANDB__REQUIRE_CORE=true" in script
    assert "unset WANDB_RESUME" in script
    assert "AMBI_AUX_REPRO_GATE" in script
