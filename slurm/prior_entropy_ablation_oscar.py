"""GPU parity, measured sizing, and training for the three prior-entropy arms."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import inner_entropy_oscar as campaign


def configure():
    campaign.MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_prior_entropy_ablation.json"
    campaign.CAMPAIGN_ROOT = Path("/oscar/scratch/rgao48/ambi/prior-entropy-ablation-20261103")
    campaign.GPU_PARITY_TEST = "tests/test_prior_entropy_ablation_graph.py::test_cuda_recipe_parity"
    campaign.GPU_PARITY_CASE_COUNT = 3
    campaign.ENTRYPOINT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    campaign.main()
