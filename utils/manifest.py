import sys, json

def make_manifest(abs_path_python, abs_path_main, abs_path_experiment_config, abs_path_manifest):
    with open(abs_path_experiment_config, "r") as f:
        experiment_params = json.load(f)
    
    with open(abs_path_manifest, "a") as g:
        for i, alg in enumerate(experiment_params["configs"]):
            for j in range(experiment_params["trials"]):
                g.write(abs_path_python, abs_path_main, '-r', abs_path_experiment_config, '--alg-index', str(i), '--trial-index', str(j), '--num-runs 1')
            

    
    # example: "python main.py -r configs/experiments/AntPlaneMoveResets6.0.json --alg-index 0 --trial-index 1 --num-runs 1"


if __name__ == "__main__":
    PYTHON = sys.executable()
    MAIN = ...
    CONFIG = ...
    MANIFEST = ...
    make_manifest(PYTHON, MAIN, CONFIG, MANIFEST)