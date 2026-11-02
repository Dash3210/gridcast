"""Smoke tests for gridcast setup and CLI gating."""

import importlib
import importlib.metadata
import runpy
import sys

import pytest

import gridcast
from gridcast import _phase, cli


def test_version_matches_metadata() -> None:
    assert isinstance(gridcast.__version__, str)
    assert len(gridcast.__version__) > 0
    installed_version = importlib.metadata.version("gridcast")
    assert gridcast.__version__ == installed_version


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["version"])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"gridcast {gridcast.__version__}"


def test_cli_features(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["features"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Feature frame constructed successfully" in captured.out
    assert "Distinct issues" in captured.out


def test_cli_unknown_subcommand_exits_two() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["unknown-subcommand"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("subcommand", "expected_phase"),
    [
        ("serve", "P4"),
    ],
)
def test_cli_unimplemented_subcommands_raise_phase_exit(
    subcommand: str, expected_phase: str
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([subcommand])
    assert expected_phase in str(exc_info.value)
    assert subcommand in str(exc_info.value)


@pytest.mark.parametrize(
    "module_name",
    [
        "gridcast.serve",
    ],
)
def test_stub_modules_expose_empty_all(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert getattr(mod, "__all__", None) == []


def test_models_module_exposes_non_empty_all() -> None:
    mod = importlib.import_module("gridcast.models")
    all_symbols = getattr(mod, "__all__", None)
    assert isinstance(all_symbols, list)
    assert len(all_symbols) > 0
    for symbol in all_symbols:
        assert hasattr(mod, symbol)


def test_backtest_module_exposes_non_empty_all() -> None:
    mod = importlib.import_module("gridcast.backtest")
    all_symbols = getattr(mod, "__all__", None)
    assert isinstance(all_symbols, list)
    assert len(all_symbols) > 0
    for symbol in all_symbols:
        assert hasattr(mod, symbol)


def test_features_module_exposes_non_empty_all() -> None:
    mod = importlib.import_module("gridcast.features")
    all_symbols = getattr(mod, "__all__", None)
    assert isinstance(all_symbols, list)
    assert len(all_symbols) > 0
    for symbol in all_symbols:
        assert hasattr(mod, symbol)


def test_ingest_module_exposes_non_empty_all() -> None:
    mod = importlib.import_module("gridcast.ingest")
    all_symbols = getattr(mod, "__all__", None)
    assert isinstance(all_symbols, list)
    assert len(all_symbols) > 0
    for symbol in all_symbols:
        assert hasattr(mod, symbol)


def test_version_fallback_when_package_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_version(package_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    reloaded = importlib.reload(gridcast)
    try:
        assert reloaded.__version__ == "0.0.0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(gridcast)


def test_run_module_as_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gridcast", "version"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("gridcast", run_name="__main__")
    assert exc_info.value.code == 0


def test_console_script_entry_point() -> None:
    eps = importlib.metadata.entry_points(group="console_scripts")
    matches = [ep for ep in eps if ep.name == "gridcast"]
    assert len(matches) == 1
    assert matches[0].value == "gridcast.cli:main"


def test_phase_todo_message_content() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _phase.todo("P2", "feature-extraction")
    message = str(exc_info.value)
    assert "P2" in message
    assert "feature-extraction" in message
    assert "see README status table" in message
