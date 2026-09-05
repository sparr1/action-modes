#!/usr/bin/env bash

set -Eeuo pipefail

if [[ -z "${EXPECTED_ACTION_MODES_SHA:-}" ]]; then
  echo "EXPECTED_ACTION_MODES_SHA is required for production training." >&2
  echo "Pass the full committed SHA through sbatch --export." >&2
  exit 1
fi
if [[ ! "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_ACTION_MODES_SHA must be one full lowercase 40-character Git SHA." >&2
  exit 1
fi

actual_action_modes_sha="$(git rev-parse HEAD)"
if [[ "$actual_action_modes_sha" != "$EXPECTED_ACTION_MODES_SHA" ]]; then
  echo "Refusing production training from the wrong action-modes commit." >&2
  echo "Expected: $EXPECTED_ACTION_MODES_SHA" >&2
  echo "Actual:   $actual_action_modes_sha" >&2
  exit 1
fi

printf '%s\n' "$actual_action_modes_sha"
