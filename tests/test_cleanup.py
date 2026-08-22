import pytest

from utils.cleanup import add_cleanup_notes, raise_cleanup_errors


def test_cleanup_notes_fall_back_to_manual_notes_for_python_310_contract():
    primary = RuntimeError("primary")
    primary.add_note = None

    add_cleanup_notes(
        primary,
        (OSError("first cleanup"), ValueError("second cleanup")),
    )

    assert primary.__notes__ == [
        "Additional cleanup failure: first cleanup",
        "Additional cleanup failure: second cleanup",
    ]


def test_raise_cleanup_errors_keeps_later_failures_on_the_first():
    first = OSError("first cleanup")
    second = ValueError("second cleanup")

    with pytest.raises(OSError, match="first cleanup") as captured:
        raise_cleanup_errors((first, second))

    assert captured.value is first
    assert captured.value.__notes__ == [
        "Additional cleanup failure: second cleanup"
    ]
