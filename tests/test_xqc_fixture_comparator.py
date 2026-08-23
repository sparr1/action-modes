import json
import subprocess
import sys
from pathlib import Path


COMPARATOR = (
    Path(__file__).resolve().parent / "oracles" / "compare_xqc_fixture.py"
)


def _run_comparison(tmp_path, expected, actual):
    expected_path = tmp_path / "expected.json"
    actual_path = tmp_path / "actual.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    actual_path.write_text(json.dumps(actual), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(COMPARATOR),
            str(expected_path),
            str(actual_path),
            "--atol",
            "1e-6",
            "--rtol",
            "1e-5",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_fixture_comparator_accepts_tolerance_and_reports_worst_paths(tmp_path):
    expected = {"metadata": {"commit": "pinned"}, "values": [1.0, 0.0]}
    actual = {"metadata": {"commit": "pinned"}, "values": [1.00001, 5e-7]}

    completed = _run_comparison(tmp_path, expected, actual)

    assert completed.returncode == 0, completed.stderr
    assert "2 numeric leaves" in completed.stdout
    assert "max absolute error" in completed.stdout
    assert '$["values"][0]' in completed.stdout
    assert "max relative error" in completed.stdout
    assert '$["values"][1]' in completed.stdout
    assert "metadata and recursive structure are identical" in completed.stdout


def test_fixture_comparator_rejects_numeric_metadata_and_structure_changes(tmp_path):
    expected = {
        "metadata": {"commit": "pinned"},
        "values": [1.0, {"x": 2.0}],
    }
    actual = {
        "metadata": {"commit": "wrong"},
        "values": [1.1],
        "extra": True,
    }

    completed = _run_comparison(tmp_path, expected, actual)

    assert completed.returncode == 1
    assert "XQC fixture comparison FAILED" in completed.stderr
    assert "structure/metadata failures=3" in completed.stderr
    assert "numeric failures=1" in completed.stderr
    assert "max absolute error" in completed.stderr
    assert '$["values"][0]' in completed.stderr
    assert 'unexpected key "extra"' in completed.stderr
    assert '$["metadata"]["commit"]' in completed.stderr
    assert "list length 1 != 2" in completed.stderr
    assert "numeric:" in completed.stderr
