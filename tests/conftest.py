"""Pytest configuration and global fixtures.

Enforces offline execution: raises if any network socket attempt is made during tests.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

_tests_dir = str(Path(__file__).parent.resolve())
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from helpers import assert_injection_changed  # noqa: E402

__all__ = ["assert_injection_changed"]


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
