"""Pytest configuration and global fixtures.

Enforces offline execution: raises if any network socket attempt is made during tests.
"""

import socket

import pytest


class NetworkAttemptForbidden(RuntimeError):
    """Raised when a test attempts network communication."""


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any test from opening a socket connection."""

    def guarded_connect(*args: object, **kwargs: object) -> None:
        raise NetworkAttemptForbidden(
            "Network access is strictly forbidden in the test suite."
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
