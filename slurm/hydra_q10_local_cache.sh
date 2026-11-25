#!/usr/bin/env bash

cleanup_hydra_q10_local_cache() {
    if [[ -n "${hydra_q10_cache:-}" ]]; then
        if rm -rf --one-file-system -- "$hydra_q10_cache"; then
            hydra_q10_cache=
        else
            printf 'Could not remove Hydra job cache %s\n' "$hydra_q10_cache" >&2
        fi
    fi
}

setup_hydra_q10_local_cache() {
    local mount=$1 allocation=$2 mount_info source options mount_device root_device available
    [[ "$allocation" =~ ^[0-9]+\.[0-9]+$ ]] || return 1
    if [[ ! -d "$mount" || -L "$mount" || ! -w "$mount" || ! -x "$mount" ]]; then
        printf 'Hydra needs a writable, physical local mount at %s\n' "$mount" >&2
        return 1
    fi
    mount_info=$(findmnt -n --mountpoint "$mount" -o SOURCE,OPTIONS) || {
        printf 'Hydra local cache path %s is not a mountpoint\n' "$mount" >&2
        return 1
    }
    read -r source options <<< "$mount_info"
    mount_device=$(stat -c %d -- "$mount") || return 1
    root_device=$(stat -c %d -- /) || return 1
    if [[ "$source" != /dev/* || "$mount_device" == "$root_device" ||
          ",$options," != *,rw,* || ",$options," == *,noexec,* ]]; then
        printf 'Hydra requires a writable local device mounted with execution enabled at %s\n' "$mount" >&2
        return 1
    fi
    available=$(df -Pk -- "$mount" | awk 'NR == 2 {print $4}') || return 1
    if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < 4 * 1024 * 1024 )); then
        printf 'Hydra requires at least 4 GiB free on %s (available KiB: %s)\n' "$mount" "$available" >&2
        return 1
    fi

    hydra_q10_cache=
    trap cleanup_hydra_q10_local_cache EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    hydra_q10_cache=$(mktemp -d -- "$mount/ambi-q10-$UID-$allocation.XXXXXXXX") || return 1
    mkdir -- "$hydra_q10_cache/"{tmp,inductor,triton,triton-dump,triton-override,extensions,cuda,xdg,xdg-config,xdg-runtime,wandb,wandb-cache,data,artifacts} || return 1

    export TMPDIR="$hydra_q10_cache/tmp" TMP="$hydra_q10_cache/tmp" TEMP="$hydra_q10_cache/tmp"
    export TORCHINDUCTOR_CACHE_DIR="$hydra_q10_cache/inductor" TORCH_EXTENSIONS_DIR="$hydra_q10_cache/extensions"
    export TRITON_HOME="$hydra_q10_cache" TRITON_CACHE_DIR="$hydra_q10_cache/triton"
    export TRITON_DUMP_DIR="$hydra_q10_cache/triton-dump" TRITON_OVERRIDE_DIR="$hydra_q10_cache/triton-override"
    export CUDA_CACHE_PATH="$hydra_q10_cache/cuda"
    export XDG_CACHE_HOME="$hydra_q10_cache/xdg" XDG_CONFIG_HOME="$hydra_q10_cache/xdg-config" XDG_RUNTIME_DIR="$hydra_q10_cache/xdg-runtime"
    export WANDB_DIR="$hydra_q10_cache/wandb" WANDB_CACHE_DIR="$hydra_q10_cache/wandb-cache"
    export WANDB_CONFIG_DIR="$hydra_q10_cache/xdg-config" WANDB_DATA_DIR="$hydra_q10_cache/data"
    export WANDB_ARTIFACT_DIR="$hydra_q10_cache/artifacts"
}
