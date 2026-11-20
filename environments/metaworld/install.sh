#!/usr/bin/env bash
set -euo pipefail

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "This SDK and lock require Linux x86-64." >&2
    exit 1
fi
runtime=$(cd -- "$(dirname -- "$0")" && pwd)
for tool in uv cc patchelf curl sha256sum tar; do
    command -v "$tool" >/dev/null || { echo "Missing $tool; see $runtime/README.md" >&2; exit 1; }
done

sdk_dir="$runtime/.mujoco"
archive=mujoco210-linux-x86_64.tar.gz
if [[ ! -d "$sdk_dir/mujoco210" ]]; then
    mkdir -p "$sdk_dir"
    stage=$(mktemp -d "$sdk_dir/setup.XXXXXXXX")
    trap 'rm -rf -- "$stage"' EXIT
    (
        cd "$stage"
        curl -fLsS --retry 3 -o "$archive" \
            "https://github.com/google-deepmind/mujoco/releases/download/2.1.0/$archive"
        echo "a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012  $archive" | sha256sum -c -
        tar -xzf "$archive"
        mv -Tn mujoco210 "$sdk_dir/mujoco210"
    )
fi
# Check reused installations as well as new extractions.
(cd "$sdk_dir/mujoco210/bin" && echo "9e7ffba152bff0e4e889e2fddf3850785fb5da28ebd05adbb6f4daaa34e3bd21  libmujoco210.so" | sha256sum -c -)

export UV_PROJECT_ENVIRONMENT="$runtime/.venv-tdmpc2"
uv sync --project "$runtime" --locked --python 3.10 --link-mode copy --no-install-package gym
"$runtime/python" -m pip install --disable-pip-version-check --require-hashes \
    --no-binary gym --no-deps --no-build-isolation -r "$runtime/gym-requirement.txt"

# Compile the binding and compare one full episode per task with the raw backend.
cd "$runtime/../.."
WANDB_MODE=disabled AMBI_RUN_REAL_METAWORLD_TESTS=1 "$runtime/python" -m pytest -q \
    tests/test_metaworld_env.py::test_real_task_matches_tdmpc2_raw_trajectory
