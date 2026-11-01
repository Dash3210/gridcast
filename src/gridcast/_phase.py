def todo(phase: str, what: str) -> None:
    """Fail loudly for work a later phase owns."""
    raise SystemExit(
        f"gridcast: {what} is not implemented yet - arrives in {phase} (see README status table)"
    )
