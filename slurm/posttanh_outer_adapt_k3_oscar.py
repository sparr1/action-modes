"""GPU parity, measured sizing, and training for the K3 post-tanh outer-adapt arm."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import inner_entropy_oscar as campaign


def configure():
    campaign.MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_posttanh_outer_adapt_k3.json"
    campaign.CAMPAIGN_ROOT = Path("/oscar/scratch/rgao48/ambi/posttanh-outer-adapt-k3-20261103")
    campaign.GPU_PARITY_TEST = "tests/test_k3_entropy_graph.py::test_cuda_k3_capture_continuation"
    campaign.GPU_PARITY_CASE_COUNT = 2
    # Numerical parity and production-size timing can run on separate GPUs.
    # The train entry point still requires both receipts from the exact commit.
    campaign.INLINE_GPU_PARITY_INDEX = None
    campaign.ENTRYPOINT = Path(__file__).resolve()


if __name__ == "__main__":
    configure()
    campaign.main()
