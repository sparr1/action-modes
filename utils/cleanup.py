"""Small helpers for preserving primary failures during resource cleanup."""


def add_cleanup_notes(primary_error, cleanup_errors, *, prefix="Additional cleanup failure"):
    """Attach cleanup failures on Python 3.10+ without masking ``primary_error``."""

    for cleanup_error in cleanup_errors:
        note = f"{prefix}: {cleanup_error}"
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(note)
        else:
            notes = list(getattr(primary_error, "__notes__", ()))
            notes.append(note)
            primary_error.__notes__ = notes


def raise_cleanup_errors(cleanup_errors):
    """Raise the first cleanup failure and retain every later one as a note."""

    errors = tuple(cleanup_errors)
    if not errors:
        return
    add_cleanup_notes(errors[0], errors[1:])
    raise errors[0]
