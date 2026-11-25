import os
from pathlib import Path
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[1] / "slurm/hydra_q10_local_cache.sh"

SHELL_SETUP = """
set -Eeuo pipefail
umask 077
source "$1"
mount=$2
findmnt() { printf '%s\\n' "$MOCK_MOUNT"; }
stat() {
    if [[ "$4" == / ]]; then
        printf '1\\n'
    else
        printf '%s\\n' "$MOCK_DEVICE"
    fi
}
df() {
    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'
    printf 'mock 10000000 900000 %s 50%% %s\\n' "$MOCK_FREE" "$mount"
}
rm() {
    [[ "$1" == -rf && "$2" == --one-file-system && "$3" == -- && "$#" == 4 ]]
    command rm -rf -- "$4"
}
"""


def invoke(mount, statements, **overrides):
    env = {**os.environ, "MOCK_MOUNT": "/dev/nvme0n1p1 rw,nodev,nosuid,relatime",
           "MOCK_DEVICE": "2", "MOCK_FREE": "4194304", "CHECK_PYTHON": sys.executable, **overrides}
    return subprocess.run(["bash", "-c", SHELL_SETUP + statements, "cache-test", str(HELPER), str(mount)],
                          env=env, text=True, capture_output=True, check=False)


def test_cache_routes_all_temporary_paths_and_removes_its_unique_directory(tmp_path):
    sibling = tmp_path / "keep"
    sibling.mkdir()
    statements = """
    setup_hydra_q10_local_cache "$mount" 501.1
    for var in TMPDIR TEMP TMP TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR TRITON_HOME \
        TRITON_CACHE_DIR TRITON_DUMP_DIR TRITON_OVERRIDE_DIR CUDA_CACHE_PATH XDG_CACHE_HOME \
        XDG_CONFIG_HOME XDG_RUNTIME_DIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR \
        WANDB_DATA_DIR WANDB_ARTIFACT_DIR; do
        [[ "${!var}" == "$hydra_q10_cache"* && -d "${!var}" ]]
    done
    "$CHECK_PYTHON" -B -c 'import os, tempfile; assert tempfile.gettempdir() == os.environ["TMPDIR"]'
    printf '%s\\n' "$hydra_q10_cache"
    """
    first = invoke(tmp_path, statements)
    second = invoke(tmp_path, statements)
    assert first.returncode == second.returncode == 0, (first.stderr, second.stderr)
    assert first.stdout != second.stdout
    for result in (first, second):
        cache = Path(result.stdout.strip())
        assert cache.parent == tmp_path and not cache.exists()
    assert list(tmp_path.iterdir()) == [sibling]


@pytest.mark.parametrize("exit_status,command", [(23, "exit 23"), (143, 'kill -TERM "$$"')])
def test_cache_is_removed_after_error_or_term(tmp_path, exit_status, command):
    result = invoke(tmp_path, f'setup_hydra_q10_local_cache "$mount" 501.1\n{command}\n')
    assert result.returncode == exit_status, result.stderr
    assert not list(tmp_path.iterdir())


def test_cache_is_removed_after_partial_setup_failure(tmp_path):
    result = invoke(tmp_path, 'mkdir() { command mkdir "$2"; return 1; }\n'
                              'setup_hydra_q10_local_cache "$mount" 501.1\n')
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("overrides", [
    {"MOCK_MOUNT": "nfs:/ltmp rw,nosuid"},
    {"MOCK_MOUNT": "/dev/nvme0n1p1 rw,noexec"},
    {"MOCK_MOUNT": "/dev/nvme0n1p1 ro,nosuid"},
    {"MOCK_DEVICE": "1"},
    {"MOCK_FREE": "4194303"},
])
def test_unsafe_local_paths_are_rejected_before_cache_creation(tmp_path, overrides):
    result = invoke(tmp_path, 'setup_hydra_q10_local_cache "$mount" 501.1\n', **overrides)
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())


def test_missing_mount_does_not_fall_back_to_inherited_temp(tmp_path):
    result = invoke(tmp_path / "missing-ltmp", 'setup_hydra_q10_local_cache "$mount" 501.1\n',
                    SLURM_TMPDIR=str(tmp_path), TMPDIR=str(tmp_path))
    assert result.returncode != 0
    assert "physical local mount" in result.stderr
    assert not list(tmp_path.iterdir())
