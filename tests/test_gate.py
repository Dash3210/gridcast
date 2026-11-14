"""Tests for the CI accuracy gate, threshold derivation, provenance, and negative control.

Implements P5-a gate verification:
1. Committed threshold artifact data/eval/gate_threshold.json and arithmetic invariant.
2. Provenance block exposition and PASS verdict on pinned fixture v2.
3. Negative control rejection on data/eval/negative_control.parquet.
4. CLI gate-negative subcommand behavior and failure when gate is weakened.
5. Hermetic execution: no network calls and no writes under data/.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from gridcast.backtest import (
    DEFAULT_EVAL_FIXTURE_PARQUET,
    DEFAULT_EVAL_FIXTURE_SIDECAR,
    DEFAULT_GATE_THRESHOLD_PATH,
    PINNED_EVAL_FIXTURE_SHA256,
    evaluate_from_fixture,
    evaluate_gate,
    load_and_verify_gate_threshold,
)
from gridcast.cli import run_eval_cli, run_gate_negative_cli

NEGATIVE_CONTROL_PARQUET = Path("data/eval/negative_control.parquet")


def test_gate_threshold_artifact_exists_and_arithmetic_valid() -> None:
    """Requirement 1: Verify data/eval/gate_threshold.json exists and arithmetic holds."""
    assert DEFAULT_GATE_THRESHOLD_PATH.exists(), (
        f"Threshold artifact missing at {DEFAULT_GATE_THRESHOLD_PATH}"
    )

    meta = load_and_verify_gate_threshold(DEFAULT_GATE_THRESHOLD_PATH)
    assert meta["rule"] == "candidate closes >= 30% of the persistence-to-operator MAE gap"
    assert meta["fraction"] == 0.30
    assert meta["persistence_mae_validation"] == 57.776
    assert meta["operator_mae_validation"] == 9.649
    assert meta["required_mae"] == 43.3379
    assert meta["adr"] == "ADR-021"

    derived = meta["derived_on"]
    assert derived["split"] == "validation"
    assert derived["start"] == "2026-06-05"
    assert derived["end"] == "2026-08-30"
    assert derived["origins"] == 15

    # Direct assertion of arithmetic: required = persistence - fraction * (persistence - operator)
    p = meta["persistence_mae_validation"]
    o = meta["operator_mae_validation"]
    frac = meta["fraction"]
    recomputed = p - frac * (p - o)
    assert round(recomputed, 4) == meta["required_mae"]


def test_gate_threshold_tamper_rejected(tmp_path: Path) -> None:
    """Requirement 1: If arithmetic changes without updating required_mae, gate refuses."""
    tampered_data: dict[str, Any] = {
        "rule": "candidate closes >= 30% of the persistence-to-operator MAE gap",
        "fraction": 0.30,
        "derived_on": {
            "split": "validation",
            "start": "2026-06-05",
            "end": "2026-08-30",
            "origins": 15,
        },
        "persistence_mae_validation": 57.776,
        "operator_mae_validation": 9.649,
        "required_mae": 40.0000,  # Deliberately tampered value != 43.3379
        "adr": "ADR-021",
    }
    tampered_path = tmp_path / "tampered_threshold.json"
    tampered_path.write_text(json.dumps(tampered_data, indent=2), encoding="utf-8")

    # Precondition assertion (ADR-015): prove tampered value differs from recomputed
    recomputed = 57.776 - 0.30 * (57.776 - 9.649)
    assert round(recomputed, 4) != tampered_data["required_mae"]

    with pytest.raises(AssertionError, match=r"Gate threshold arithmetic mismatch"):
        load_and_verify_gate_threshold(tampered_path)


def test_evaluate_gate_on_pinned_fixture_passes_with_provenance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 2: make eval evaluates candidate against required_mae and prints provenance."""
    results = evaluate_from_fixture(
        parquet_path=DEFAULT_EVAL_FIXTURE_PARQUET,
        sidecar_path=DEFAULT_EVAL_FIXTURE_SIDECAR,
        verbose=True,
        pinned_sha256=PINNED_EVAL_FIXTURE_SHA256,
    )

    assert results.verdict == "PASS"
    assert results.gate_value < 43.3379

    captured = capsys.readouterr().out

    # Provenance block assertions
    assert "GRIDCAST ACCURACY GATE EVALUATION" in captured
    assert "candidate closes >= 30% of the persistence-to-operator MAE gap (ADR-021)" in captured
    assert "Gate Fraction:               0.30" in captured
    assert "Threshold (required MAE):    43.3379 gCO2/kWh" in captured
    assert "validation (2026-06-05 to 2026-08-30, 15 origins)" in captured
    assert "Persistence MAE (val):     57.776 gCO2/kWh" in captured
    assert "Operator MAE (val):        9.649 gCO2/kWh" in captured
    assert f"Fixture SHA-256:             {PINNED_EVAL_FIXTURE_SHA256}" in captured
    assert "Archive commit:              3617143" in captured
    assert "Candidate GATE_VALUE:        39.118 gCO2/kWh" in captured
    assert "Operator GATE_VALUE (NESO):  11.666 gCO2/kWh" in captured
    assert "Persistence GATE_VALUE:      53.741 gCO2/kWh" in captured
    assert "Seasonal-Naive GATE_VALUE:   51.495 gCO2/kWh" in captured
    assert "GATE VERDICT: NONE - threshold invalid (ADR-020)" in captured
    assert "GATE VERDICT: PASS" in captured

    # Direct unit test calling evaluate_gate on results
    direct_verdict = evaluate_gate(
        results,
        threshold_path=DEFAULT_GATE_THRESHOLD_PATH,
        verbose=False,
    )
    assert direct_verdict == "PASS"


def test_negative_control_fixture_exists_and_rejects() -> None:
    """Requirement 3: negative_control.parquet has candidate_pred = persistence and fails gate."""
    assert NEGATIVE_CONTROL_PARQUET.exists(), (
        f"Negative control artifact missing at {NEGATIVE_CONTROL_PARQUET}"
    )

    import pandas as pd
    ctrl_df = pd.read_parquet(NEGATIVE_CONTROL_PARQUET)
    fixture_df = pd.read_parquet(DEFAULT_EVAL_FIXTURE_PARQUET)

    assert len(ctrl_df) == 1440
    # Truth columns match fixture exactly
    truth_cols = [
        "issue_time",
        "horizon",
        "target_time",
        "actual",
        "neso_forecast",
        "persistence_pred",
        "seasonal_naive_pred",
    ]
    for col in truth_cols:
        assert ctrl_df[col].equals(fixture_df[col]), f"Truth column {col} differs from fixture"

    # Precondition assertion (ADR-015): prove candidate_pred was replaced by persistence_pred
    assert ctrl_df["candidate_pred"].equals(ctrl_df["persistence_pred"]), (
        "Precondition failed: candidate_pred must equal persistence_pred in negative control"
    )
    assert not ctrl_df["candidate_pred"].equals(fixture_df["candidate_pred"]), (
        "Precondition failed: candidate_pred must differ from original candidate"
    )

    results = evaluate_from_fixture(
        parquet_path=NEGATIVE_CONTROL_PARQUET,
        sidecar_path=None,
        verbose=False,
        pinned_sha256=None,
    )
    assert results.verdict == "FAIL"
    assert round(results.gate_value, 3) == 53.741


def test_run_gate_negative_cli_success(capsys: pytest.CaptureFixture[str]) -> None:
    """Requirement 3: make gate-negative runs gate on negative control and exits 0."""
    rc = run_gate_negative_cli(
        fixture_path=NEGATIVE_CONTROL_PARQUET,
        threshold_path=DEFAULT_GATE_THRESHOLD_PATH,
        verbose=True,
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "GATE VERDICT: FAIL" in captured.out
    assert "Negative control verified: gate rejected candidate with verdict FAIL" in captured.out


def test_run_gate_negative_cli_fails_if_gate_weakened(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requirement 3: Demonstrate that gate-negative fails when gate is weakened to always pass."""
    # Temporarily monkeypatch evaluate_gate to unconditionally return PASS
    monkeypatch.setattr(
        "gridcast.backtest.evaluate_gate",
        lambda *args, **kwargs: "PASS",
    )

    rc = run_gate_negative_cli(
        fixture_path=NEGATIVE_CONTROL_PARQUET,
        threshold_path=DEFAULT_GATE_THRESHOLD_PATH,
        verbose=False,
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "Gate negative control failed: expected verdict 'FAIL', got 'PASS'" in captured.err


def test_run_eval_cli_exits_nonzero_on_gate_fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """make eval / run_eval_cli must exit non-zero (1) whenever verdict != 'PASS'."""
    neg_sha = hashlib.sha256(NEGATIVE_CONTROL_PARQUET.read_bytes()).hexdigest()
    rc_neg = run_eval_cli(
        fixture_path=NEGATIVE_CONTROL_PARQUET,
        sidecar_path=None,
        verbose=True,
        pinned_sha256=neg_sha,
    )
    assert rc_neg == 1
    captured_neg = capsys.readouterr()
    assert "GATE VERDICT: FAIL" in captured_neg.out
    assert "Accuracy gate failed: expected verdict 'PASS', got 'FAIL'" in captured_neg.err

    monkeypatch.setattr(
        "gridcast.backtest.evaluate_gate",
        lambda *args, **kwargs: "FAIL",
    )
    rc_monkey = run_eval_cli(verbose=False)
    assert rc_monkey == 1
    captured_monkey = capsys.readouterr()
    assert "Accuracy gate failed: expected verdict 'PASS', got 'FAIL'" in captured_monkey.err


def test_hermetic_data_check_offline_and_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 6: Test suite makes no network calls and writes nothing under data/."""
    data_dir = Path("data")
    assert data_dir.exists()

    def snapshot_data_tree() -> dict[str, tuple[int, str]]:
        state: dict[str, tuple[int, str]] = {}
        for p in data_dir.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                rel_path = str(p.relative_to(data_dir))
                size = p.stat().st_size
                with open(p, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                state[rel_path] = (size, file_hash)
        return state

    initial_snapshot = snapshot_data_tree()

    # Run evaluations
    evaluate_from_fixture(
        parquet_path=DEFAULT_EVAL_FIXTURE_PARQUET,
        sidecar_path=DEFAULT_EVAL_FIXTURE_SIDECAR,
        verbose=False,
    )
    run_gate_negative_cli(
        fixture_path=NEGATIVE_CONTROL_PARQUET,
        verbose=False,
    )

    final_snapshot = snapshot_data_tree()

    # Assert no files added, modified, or deleted under data/
    assert initial_snapshot == final_snapshot, (
        "Evaluation run modified or created files under data/"
    )

    # Assert socket connection is blocked (offline hermetic execution)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(RuntimeError, match=r"Network access is strictly forbidden"):
        s.connect(("1.1.1.1", 80))


def test_published_gate_closure_matches_fixture() -> None:
    """The published closure percentage must be recomputed from the fixture, not asserted.

    The first version of this claim was wrong: 39.118 closes 34.75% of the 42.075 gap to the
    operator, not the 30.38% that appeared in two documents. That number is derived, so it is
    checked as derived.
    """
    res = evaluate_from_fixture(verbose=False)
    gap = res.persistence_gate_value - res.operator_gate_value
    closed = (res.persistence_gate_value - res.gate_value) / gap * 100.0
    expected = f"{closed:.2f}%"

    for doc in ("MODEL_CARD.md", "docs/decisions.md"):
        text = Path(doc).read_text(encoding="utf-8")
        assert expected in text, f"{doc} does not state the recomputed closure {expected}"
        assert "30.38%" not in text, f"{doc} still carries the wrong 30.38%"

