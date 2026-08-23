#!/usr/bin/env python3
"""Compare an XQC oracle with exact structure and tolerant numeric leaves."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class NumericDifference:
    path: str
    expected: float
    actual: float
    absolute: float
    relative: float
    allowed: float


@dataclass
class Comparison:
    numeric_count: int = 0
    numeric_failures: list[NumericDifference] = field(default_factory=list)
    structural_failures: list[str] = field(default_factory=list)
    max_absolute: NumericDifference | None = None
    max_relative: NumericDifference | None = None

    @property
    def ok(self) -> bool:
        return not self.numeric_failures and not self.structural_failures


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_fixture_data(expected, actual, *, atol: float, rtol: float) -> Comparison:
    result = Comparison()

    def walk(reference, candidate, path: str) -> None:
        if isinstance(reference, dict) and isinstance(candidate, dict):
            reference_keys = set(reference)
            candidate_keys = set(candidate)
            for key in sorted(reference_keys - candidate_keys):
                result.structural_failures.append(
                    f'{path}: missing key {json.dumps(key)}'
                )
            for key in sorted(candidate_keys - reference_keys):
                result.structural_failures.append(
                    f'{path}: unexpected key {json.dumps(key)}'
                )
            for key in sorted(reference_keys & candidate_keys):
                walk(reference[key], candidate[key], f'{path}[{json.dumps(key)}]')
            return

        if isinstance(reference, list) and isinstance(candidate, list):
            if len(reference) != len(candidate):
                result.structural_failures.append(
                    f"{path}: list length {len(candidate)} != {len(reference)}"
                )
            for index, (left, right) in enumerate(zip(reference, candidate)):
                walk(left, right, f"{path}[{index}]")
            return

        if _is_number(reference) and _is_number(candidate):
            expected_value = float(reference)
            actual_value = float(candidate)
            result.numeric_count += 1
            if math.isfinite(expected_value) and math.isfinite(actual_value):
                absolute = abs(actual_value - expected_value)
                relative = (
                    0.0
                    if absolute == 0.0
                    else absolute / abs(expected_value)
                    if expected_value != 0.0
                    else math.inf
                )
                allowed = atol + rtol * abs(expected_value)
            else:
                absolute = relative = allowed = math.inf
            difference = NumericDifference(
                path, expected_value, actual_value, absolute, relative, allowed
            )
            if (
                result.max_absolute is None
                or absolute > result.max_absolute.absolute
            ):
                result.max_absolute = difference
            if (
                result.max_relative is None
                or relative > result.max_relative.relative
            ):
                result.max_relative = difference
            if not (
                math.isfinite(expected_value)
                and math.isfinite(actual_value)
                and absolute <= allowed
            ):
                result.numeric_failures.append(difference)
            return

        if type(reference) is not type(candidate):
            result.structural_failures.append(
                f"{path}: type {type(candidate).__name__} != "
                f"{type(reference).__name__}"
            )
        elif reference != candidate:
            result.structural_failures.append(
                f"{path}: {candidate!r} != {reference!r}"
            )

    walk(expected, actual, "$")
    return result


def _load_json(path: Path):
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _format_maximum(label: str, difference: NumericDifference | None) -> str:
    if difference is None:
        return f"{label}: n/a"
    value = (
        difference.absolute
        if label == "max absolute error"
        else difference.relative
    )
    return (
        f"{label}: {value:.9g} at {difference.path} "
        f"(expected={difference.expected:.9g}, actual={difference.actual:.9g})"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if not (
        math.isfinite(args.atol)
        and math.isfinite(args.rtol)
        and args.atol >= 0.0
        and args.rtol >= 0.0
    ):
        parser.error("tolerances must be finite and non-negative")

    try:
        expected = _load_json(args.expected)
        actual = _load_json(args.actual)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"fixture comparison input error: {exc}", file=sys.stderr)
        return 2

    result = compare_fixture_data(
        expected, actual, atol=args.atol, rtol=args.rtol
    )
    status = "passed" if result.ok else "FAILED"
    output = sys.stdout if result.ok else sys.stderr
    print(
        f"XQC fixture comparison {status}: {result.numeric_count} numeric leaves; "
        f"atol={args.atol:g}, rtol={args.rtol:g}; "
        f"structure/metadata failures={len(result.structural_failures)}, "
        f"numeric failures={len(result.numeric_failures)}",
        file=output,
    )
    print(_format_maximum("max absolute error", result.max_absolute), file=output)
    print(_format_maximum("max relative error", result.max_relative), file=output)

    if result.ok:
        print("Nonnumeric metadata and recursive structure are identical.")
        return 0

    for issue in result.structural_failures[:20]:
        print(f"structure/metadata: {issue}", file=sys.stderr)
    for difference in result.numeric_failures[:20]:
        print(
            f"numeric: {difference.path}: abs={difference.absolute:.9g} "
            f"> allowed={difference.allowed:.9g}; "
            f"expected={difference.expected:.9g}, actual={difference.actual:.9g}",
            file=sys.stderr,
        )
    omitted = max(0, len(result.structural_failures) - 20) + max(
        0, len(result.numeric_failures) - 20
    )
    if omitted > 0:
        print(f"... {omitted} additional differences omitted", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
