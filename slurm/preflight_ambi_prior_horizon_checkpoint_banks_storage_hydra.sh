#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

readonly CAMPAIGN_ROOT="/cs/home/rgao48/projects/ambi-runs/ambi-prior-horizon-checkpoint-banks-1p5m"
readonly EXPECTED_CAMPAIGN_BANKS=3
# The existing 60-snapshot model-size-5 banks occupy approximately 4.1 GiB.
readonly ESTIMATED_BANK_MIB=4200
readonly CAMPAIGN_HEADROOM_MIB=4096
case "${1:---aggregate}" in
  --aggregate)
    readonly PREFLIGHT_BANKS="$EXPECTED_CAMPAIGN_BANKS"
    readonly PREFLIGHT_SCOPE="all three banks"
    ;;
  --single-bank)
    readonly PREFLIGHT_BANKS=1
    readonly PREFLIGHT_SCOPE="one remaining bank"
    ;;
  *)
    echo "Usage: $0 [--aggregate|--single-bank]" >&2
    exit 2
    ;;
esac
readonly MIN_CAMPAIGN_FREE_KIB=$((
  (PREFLIGHT_BANKS * ESTIMATED_BANK_MIB + CAMPAIGN_HEADROOM_MIB) * 1024
))

if [[ -L "$CAMPAIGN_ROOT" ]]; then
  echo "Refusing symlinked campaign artifact root: $CAMPAIGN_ROOT" >&2
  exit 1
fi
mkdir -p "$CAMPAIGN_ROOT"

available_kib="$(df -Pk "$CAMPAIGN_ROOT" | awk 'NR == 2 {print $4}')"
if [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
  echo "Could not determine free space for $CAMPAIGN_ROOT." >&2
  exit 1
fi
if (( available_kib < MIN_CAMPAIGN_FREE_KIB )); then
  required_gib="$(awk -v kib="$MIN_CAMPAIGN_FREE_KIB" 'BEGIN {printf "%.2f", kib / 1024 / 1024}')"
  available_gib="$(awk -v kib="$available_kib" 'BEGIN {printf "%.2f", kib / 1024 / 1024}')"
  echo "Checkpoint-bank campaign storage preflight failed." >&2
  echo "Required free: ${required_gib} GiB for ${PREFLIGHT_SCOPE} at ~4.1 GiB each plus 4 GiB headroom." >&2
  echo "Available free: ${available_gib} GiB at $CAMPAIGN_ROOT." >&2
  exit 1
fi

available_gib="$(awk -v kib="$available_kib" 'BEGIN {printf "%.2f", kib / 1024 / 1024}')"
required_gib="$(awk -v kib="$MIN_CAMPAIGN_FREE_KIB" 'BEGIN {printf "%.2f", kib / 1024 / 1024}')"
echo "Campaign storage preflight passed: ${available_gib} GiB free; ${required_gib} GiB required."
echo "Capacity model: ${PREFLIGHT_BANKS} bank(s) x ${ESTIMATED_BANK_MIB} MiB + ${CAMPAIGN_HEADROOM_MIB} MiB headroom (${PREFLIGHT_SCOPE})."
echo "Campaign root: $CAMPAIGN_ROOT"
