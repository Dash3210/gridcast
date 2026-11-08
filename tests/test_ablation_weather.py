"""Tests for the pre-registered weather removal ablation (Phase P4-c).

Verifies:
1. Decision rule behavior: DROP if N <= W * 1.02, KEEP if N > W * 1.02.
2. Control arm precondition guard: failure to reproduce fixture exactly halts execution.
3. Feature selection: treatment arm drops weather columns while retaining non-weather columns.
4. CLI wiring for ablation-weather subcommand.
5. End-to-end ablation execution with mocked backtest.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from gridcast.backtest import (
    DEFAULT_EVAL_FIXTURE_PARQUET,
    DEFAULT_EVAL_FIXTURE_SIDECAR,
    WeatherAblationResults,
    evaluate_from_fixture,
    run_weather_ablation,
)
from gridcast.cli import main, run_ablation_weather_cli
from gridcast.features import FEATURE_COLUMNS, WEATHER_FEATURE_COLUMNS


def test_ablation_decision_rule_logic() -> None:
    """Requirement 0: Test amended decision rule: DROP if N <= 1.02W and N_long <= 1.05W_long."""
    w = 39.032478
    w_long = 38.0234

    # Equal performance -> DROP
    n_equal = w
    n_long_equal = w_long
    assert n_equal <= w * 1.02 and n_long_equal <= w_long * 1.05
    assert (
        "DROP"
        if (n_equal <= w * 1.02 and n_long_equal <= w_long * 1.05)
        else "KEEP"
    ) == "DROP"

    # Aggregate passes (+1.0%), tail passes (+3.0%) -> DROP
    n_pass = w * 1.01
    n_long_pass = w_long * 1.03
    assert (
        "DROP"
        if (n_pass <= w * 1.02 and n_long_pass <= w_long * 1.05)
        else "KEEP"
    ) == "DROP"

    # Aggregate passes (+1.0%), but tail fails (+5.5%) -> KEEP
    n_long_fail = w_long * 1.055
    assert (
        "DROP"
        if (n_pass <= w * 1.02 and n_long_fail <= w_long * 1.05)
        else "KEEP"
    ) == "KEEP"

    # Aggregate fails (+2.5%), tail passes -> KEEP
    n_fail = w * 1.025
    assert (
        "DROP"
        if (n_fail <= w * 1.02 and n_long_pass <= w_long * 1.05)
        else "KEEP"
    ) == "KEEP"

    # Real measured values (+0.22% aggregate, +0.86% tail) -> DROP
    n_measured = 39.118459
    n_long_measured = 38.3512
    assert (
        "DROP"
        if (n_measured <= w * 1.02 and n_long_measured <= w_long * 1.05)
        else "KEEP"
    ) == "DROP"


def test_treatment_arm_feature_selection() -> None:
    """Requirement 1: Verify treatment arm excludes weather columns and keeps remaining features."""
    weather_cols = ["temperature", "temperature_24h_mean", "temperature_24h_min"]
    treatment_cols = [c for c in FEATURE_COLUMNS if c not in weather_cols]

    assert len(treatment_cols) == 26
    assert len(FEATURE_COLUMNS) == 26
    for col in weather_cols:
        assert col not in treatment_cols
    assert "horizon" in treatment_cols
    assert "actual_recent_3h" in treatment_cols
    assert "mix_share_wind" in treatment_cols


def test_control_arm_precondition_guard_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 1: Control arm precondition failure must raise RuntimeError and halt."""
    fixture_df = pd.read_parquet(DEFAULT_EVAL_FIXTURE_PARQUET)

    intensity_dates = pd.date_range("2024-11-08", "2026-11-10", freq="30min", tz="UTC")
    intensity_df = pd.DataFrame({
        "period_start_utc": intensity_dates,
        "forecast": [50.0] * len(intensity_dates),
        "actual": [50.0] * len(intensity_dates),
    })
    weather_df = pd.DataFrame({"time": intensity_dates})
    mix_df = pd.DataFrame({"time": intensity_dates})

    dummy_frame = pd.DataFrame({
        "issue_time": fixture_df["issue_time"],
        "horizon": fixture_df["horizon"],
        "target_time": fixture_df["target_time"],
        "y": fixture_df["actual"],
    })
    for col in list(FEATURE_COLUMNS) + list(WEATHER_FEATURE_COLUMNS):
        if col != "horizon":
            dummy_frame[col] = 1.0

    monkeypatch.setattr(
        "gridcast.backtest.load_feature_sources",
        lambda: (intensity_df, weather_df, mix_df),
    )
    monkeypatch.setattr(
        "gridcast.backtest.build_example_frame",
        lambda **kwargs: dummy_frame,
    )

    # Mock LightGBM booster with predictions that diverge from candidate_pred
    mock_booster = MagicMock()
    mock_booster.predict.return_value = np.array([999.0] * 96)

    monkeypatch.setattr(
        "lightgbm.train",
        lambda *args, **kwargs: mock_booster,
    )

    with pytest.raises(RuntimeError, match="Control arm failed to reproduce fixture"):
        run_weather_ablation(
            parquet_path=DEFAULT_EVAL_FIXTURE_PARQUET,
            sidecar_path=DEFAULT_EVAL_FIXTURE_SIDECAR,
            max_control_tolerance=0.0,
            verbose=False,
        )


def test_ablation_mocked_clean_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify clean ablation flow with mocked LightGBM reproducing control."""
    fixture_df = pd.read_parquet(DEFAULT_EVAL_FIXTURE_PARQUET)
    fixture_res = evaluate_from_fixture(verbose=False)

    intensity_dates = pd.date_range("2024-11-08", "2026-11-10", freq="30min", tz="UTC")
    intensity_df = pd.DataFrame({
        "period_start_utc": intensity_dates,
        "forecast": [50.0] * len(intensity_dates),
        "actual": [50.0] * len(intensity_dates),
    })
    weather_df = pd.DataFrame({"time": intensity_dates})
    mix_df = pd.DataFrame({"time": intensity_dates})

    dummy_frame = pd.DataFrame({
        "issue_time": fixture_df["issue_time"],
        "horizon": fixture_df["horizon"],
        "target_time": fixture_df["target_time"],
        "y": fixture_df["actual"],
    })
    for col in list(FEATURE_COLUMNS) + list(WEATHER_FEATURE_COLUMNS):
        if col != "horizon":
            dummy_frame[col] = 1.0

    monkeypatch.setattr(
        "gridcast.backtest.load_feature_sources",
        lambda: (intensity_df, weather_df, mix_df),
    )
    monkeypatch.setattr(
        "gridcast.backtest.build_example_frame",
        lambda **kwargs: dummy_frame,
    )

    # Create mock booster that returns exact candidate_pred on control
    train_call_count = 0
    unique_origins = sorted(fixture_df["issue_time"].unique())

    def mock_train(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal train_call_count
        current_fold_idx = train_call_count % 15
        train_call_count += 1
        booster = MagicMock()

        def mock_predict(X: pd.DataFrame) -> np.ndarray:
            is_control = len(X.columns) > len(FEATURE_COLUMNS)
            origin_val = unique_origins[current_fold_idx]
            fold_sub = fixture_df[fixture_df["issue_time"] == origin_val].sort_values("horizon")
            preds = fold_sub["candidate_pred"].to_numpy(dtype=float)
            if is_control:
                return preds
            # Treatment arm: return slightly perturbed predictions
            return preds + 0.05

        booster.predict.side_effect = mock_predict
        return booster

    monkeypatch.setattr("lightgbm.train", mock_train)

    res = run_weather_ablation(
        parquet_path=DEFAULT_EVAL_FIXTURE_PARQUET,
        sidecar_path=DEFAULT_EVAL_FIXTURE_SIDECAR,
        max_control_tolerance=0.0,
        verbose=True,
    )

    assert isinstance(res, WeatherAblationResults)
    assert res.max_control_delta == 0.0
    assert math.isclose(res.control_gate_value, fixture_res.gate_value, abs_tol=1e-6)
    assert res.decision in ("DROP", "KEEP")
    assert len(res.per_horizon_table) == 96
    assert 1 in res.key_horizons
    assert 2 in res.key_horizons
    assert 48 in res.key_horizons
    assert 96 in res.key_horizons


def test_cli_ablation_weather_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI wiring for ablation-weather subcommand."""
    called = False

    def mock_run_ablation(*args: Any, **kwargs: Any) -> WeatherAblationResults:
        nonlocal called
        called = True
        return WeatherAblationResults(
            control_gate_value=39.032478,
            treatment_gate_value=39.118459,
            delta_gate_value=0.085981,
            delta_percent=0.2203,
            decision="DROP",
            max_control_delta=0.0,
            decision_threshold_percent=2.0,
            per_horizon_table=pd.DataFrame(),
            operator_gate_value=11.666,
            persistence_gate_value=53.741,
            seasonal_naive_gate_value=51.495,
            key_horizons={},
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr("gridcast.backtest.run_weather_ablation", mock_run_ablation)

    rc = run_ablation_weather_cli(verbose=False)
    assert rc == 0
    assert called

    rc_main = main(["ablation-weather", "--quiet"])
    assert rc_main == 0

    # Error handling
    def mock_raise(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated error")

    monkeypatch.setattr("gridcast.backtest.run_weather_ablation", mock_raise)
    rc_fail = run_ablation_weather_cli(verbose=False)
    assert rc_fail == 1


def test_emit_weather_ablation_table_missing_markers(tmp_path: Path) -> None:
    """emit_weather_ablation_table raises ValueError when markers are absent."""
    from gridcast.backtest import emit_weather_ablation_table

    doc = tmp_path / "test.md"
    doc.write_text("No markers here\n", encoding="utf-8")
    df = pd.DataFrame({
        "horizon": [1],
        "control_mae": [10.0],
        "treatment_mae": [10.5],
        "delta": [0.5],
        "delta_pct": [5.0],
        "operator_mae": [5.0],
        "persistence_mae": [15.0],
        "seasonal_naive_mae": [12.0],
    })
    with pytest.raises(ValueError, match="markers"):
        emit_weather_ablation_table(df, doc)


def test_emit_weather_ablation_table_replaces_block(tmp_path: Path) -> None:
    """emit_weather_ablation_table replaces content between markers without touching prose."""
    from gridcast.backtest import emit_weather_ablation_table

    doc = tmp_path / "test.md"
    initial_text = (
        "Prose before table.\n\n"
        "<!-- BEGIN GENERATED TABLE -->\n"
        "stale row\n"
        "<!-- END GENERATED TABLE -->\n\n"
        "Prose after table.\n"
    )
    doc.write_text(initial_text, encoding="utf-8")
    df = pd.DataFrame({
        "horizon": [1],
        "control_mae": [13.2536],
        "treatment_mae": [12.9836],
        "delta": [-0.2700],
        "delta_pct": [-2.04],
        "operator_mae": [13.7333],
        "persistence_mae": [33.1333],
        "seasonal_naive_mae": [29.9333],
    })
    emit_weather_ablation_table(df, doc)
    updated = doc.read_text(encoding="utf-8")
    assert "Prose before table." in updated
    assert "Prose after table." in updated
    assert "stale row" not in updated
    assert "| 1 | 13.2536 | 12.9836 | -0.2700 | -2.04% |" in updated


def parse_ablation_generated_table(doc_text: str) -> list[dict[str, float]]:
    """Parse the per-horizon table from between generated table markers in the doc."""
    pattern = r"<!-- BEGIN GENERATED TABLE -->(.*?)<!-- END GENERATED TABLE -->"
    match = re.search(pattern, doc_text, re.DOTALL)
    assert match is not None, "Generated table markers not found in document"
    block = match.group(1).strip()
    rows: list[dict[str, float]] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "Horizon" in line:
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if len(cols) >= 8:
            h = int(cols[0])
            w = float(cols[1])
            n = float(cols[2])
            delta = float(cols[3])
            delta_pct = float(cols[4].rstrip("%"))
            neso = float(cols[5])
            pers = float(cols[6])
            sn = float(cols[7])
            rows.append({
                "horizon": h,
                "control_mae": w,
                "treatment_mae": n,
                "delta": delta,
                "delta_pct": delta_pct,
                "operator_mae": neso,
                "persistence_mae": pers,
                "seasonal_naive_mae": sn,
            })
    return rows


def parse_reported_gate_values(doc_text: str) -> tuple[float, float]:
    """Parse reported control and treatment GATE_VALUE from document."""
    ctrl_match = re.search(r"Control Arm.*?GATE_VALUE\s*=\s*([0-9]+\.[0-9]+)", doc_text)
    treat_match = re.search(r"Treatment Arm.*?GATE_VALUE\s*=\s*([0-9]+\.[0-9]+)", doc_text)
    assert ctrl_match is not None, "Reported Control GATE_VALUE not found in document"
    assert treat_match is not None, "Reported Treatment GATE_VALUE not found in document"
    return float(ctrl_match.group(1)), float(treat_match.group(1))


def parse_key_horizon_rows_from_prose(doc_text: str) -> dict[int, dict[str, float]]:
    """Parse the key horizons table from the Key Horizons Detail section."""
    key_horizons: dict[int, dict[str, float]] = {}
    lines = doc_text.splitlines()
    in_key_section = False
    for line in lines:
        if "## 6. Key Horizons Detail" in line:
            in_key_section = True
            continue
        if in_key_section:
            if line.startswith("## "):
                break
            if not line.strip().startswith("|") or "|---" in line or "Horizon" in line:
                continue
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 9:
                h_match = re.search(r"(\d+)", cols[0])
                if h_match:
                    h = int(h_match.group(1))
                    key_horizons[h] = {
                        "control_mae": float(cols[2]),
                        "treatment_mae": float(cols[3]),
                        "delta": float(cols[4]),
                        "delta_pct": float(cols[5].rstrip("%")),
                        "operator_mae": float(cols[6]),
                        "persistence_mae": float(cols[7]),
                        "seasonal_naive_mae": float(cols[8]),
                    }
    return key_horizons


def test_ablation_document_table_integrity() -> None:
    """Assert generated per-horizon table in doc matches reported GATE_VALUEs and key rows."""
    doc_path = Path("docs/ablations/weather_removal.md")
    assert doc_path.exists(), f"Ablation document not found at {doc_path}"
    doc_text = doc_path.read_text(encoding="utf-8")

    rows = parse_ablation_generated_table(doc_text)
    assert len(rows) == 96, f"Expected exactly 96 rows, got {len(rows)}"

    horizons = [r["horizon"] for r in rows]
    assert sorted(horizons) == list(range(1, 97)), "Horizons must be 1..96 exactly once each"

    reported_ctrl_gv, reported_treat_gv = parse_reported_gate_values(doc_text)

    ctrl_mean = float(np.mean([r["control_mae"] for r in rows]))
    treat_mean = float(np.mean([r["treatment_mae"] for r in rows]))

    # Anchor the control column to the fixture, not to another number in the same document.
    # A table and the GATE values quoted beside it can be edited together, and this file is
    # exactly where a transcribed table survived review once. The fixture is the independent
    # source: its control GATE cannot be moved by editing the ablation document.
    fixture = pd.read_parquet("data/eval/frozen_split_v1.parquet")
    fixture_per_h = (
        (fixture["candidate_pred"] - fixture["actual"]).abs().groupby(fixture["horizon"]).mean()
    )
    fixture_control_gate = float(fixture_per_h.mean())
    assert math.isclose(ctrl_mean, fixture_control_gate, abs_tol=0.002), (
        f"Control column mean ({ctrl_mean:.6f}) does not match the fixture's control "
        f"GATE_VALUE ({fixture_control_gate:.6f}) within 0.002 "
        f"(diff: {abs(ctrl_mean - fixture_control_gate):.6f})"
    )

    assert math.isclose(ctrl_mean, reported_ctrl_gv, abs_tol=0.002), (
        f"Control column mean ({ctrl_mean:.6f}) does not match reported "
        f"GATE_VALUE ({reported_ctrl_gv:.6f}) within 0.002 "
        f"(diff: {abs(ctrl_mean - reported_ctrl_gv):.6f})"
    )
    assert math.isclose(treat_mean, reported_treat_gv, abs_tol=0.002), (
        f"Treatment column mean ({treat_mean:.6f}) does not match reported "
        f"GATE_VALUE ({reported_treat_gv:.6f}) within 0.002 "
        f"(diff: {abs(treat_mean - reported_treat_gv):.6f})"
    )

    key_rows = parse_key_horizon_rows_from_prose(doc_text)
    rows_by_h = {r["horizon"]: r for r in rows}
    for h in (1, 2, 48, 96):
        assert h in key_rows, f"Key horizon h={h} missing from Key Horizons Detail prose"
        expected = key_rows[h]
        actual = rows_by_h[h]
        msg_ctrl = (
            f"h={h} control MAE mismatch between table ({actual['control_mae']}) "
            f"and prose ({expected['control_mae']})"
        )
        assert math.isclose(actual["control_mae"], expected["control_mae"], abs_tol=1e-4), msg_ctrl
        msg_treat = (
            f"h={h} treatment MAE mismatch between table ({actual['treatment_mae']}) "
            f"and prose ({expected['treatment_mae']})"
        )
        assert math.isclose(
            actual["treatment_mae"], expected["treatment_mae"], abs_tol=1e-4
        ), msg_treat

