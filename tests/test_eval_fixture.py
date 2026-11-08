"""Tests for the frozen evaluation fixture, integrity guards, and offline evaluation.

Verifies:
1. Frozen fixture exists, has exactly 1440 rows and exact required columns.
2. Sidecar metadata carries all required provenance fields and matching SHA-256.
3. Tampered fixture or missing sidecar hard-fails with loud integrity errors.
4. Independent naive MAE recomputation matches GATE_VALUE to 6 decimal places.
5. Evaluation runs from fixture alone without data/curated.
6. Refresh eligibility guard refuses too-recent windows, naming both dates and shortfall.
7. Fixture evaluation is strictly deterministic across repeated runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gridcast.backtest import (
    DEFAULT_EVAL_FIXTURE_PARQUET,
    DEFAULT_EVAL_FIXTURE_SIDECAR,
    PINNED_EVAL_FIXTURE_SHA256,
    REQUIRED_FIXTURE_COLUMNS,
    BacktestResults,
    assert_refresh_window_eligibility,
    check_sidecar_library_versions,
    evaluate_from_fixture,
    refresh_eval_fixture,
    reproduce_eval_fixture,
    verify_fixture_integrity,
)
from gridcast.cli import main, run_eval_cli, run_eval_refresh_cli, run_eval_reproduce_cli


def test_frozen_fixture_structure_and_schema() -> None:
    """Requirement 1: Verify structure, row count, and columns of frozen_split_v1."""
    assert DEFAULT_EVAL_FIXTURE_PARQUET.exists(), (
        f"Fixture parquet not found at {DEFAULT_EVAL_FIXTURE_PARQUET}"
    )
    assert DEFAULT_EVAL_FIXTURE_SIDECAR.exists(), (
        f"Fixture sidecar not found at {DEFAULT_EVAL_FIXTURE_SIDECAR}"
    )

    df = pd.read_parquet(DEFAULT_EVAL_FIXTURE_PARQUET)
    assert len(df) == 1440, f"Expected 1440 rows (15 origins x 96 horizons), got {len(df)}"

    expected_cols = [
        "issue_time",
        "horizon",
        "target_time",
        "actual",
        "neso_forecast",
        "persistence_pred",
        "seasonal_naive_pred",
        "candidate_pred",
    ]
    assert list(df.columns) == expected_cols, (
        f"Parquet columns {list(df.columns)} do not match expected {expected_cols}"
    )
    assert expected_cols == REQUIRED_FIXTURE_COLUMNS

    # Verify no nulls in any scored column
    for col in expected_cols:
        assert df[col].notna().all(), f"Column '{col}' in fixture contains nulls"

    # Verify origins count and horizons
    assert len(df["issue_time"].unique()) == 15, "Expected exactly 15 unique issue times"
    for _, group in df.groupby("issue_time"):
        assert sorted(group["horizon"].tolist()) == list(range(1, 97)), (
            "Each origin must have horizons 1..96"
        )


def test_frozen_fixture_sidecar_metadata() -> None:
    """Requirement 1: Verify sidecar carries all required provenance fields."""
    with open(DEFAULT_EVAL_FIXTURE_SIDECAR, encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = [
        "schema_version",
        "created_at",
        "archive_commit",
        "archive_newest_period",
        "window_start",
        "window_end",
        "origins",
        "row_count",
        "sha256",
        "model_artifact_sha256",
        "training_window",
        "lgbm_params",
        "num_boost_round",
        "versions",
    ]
    for k in required_keys:
        assert k in meta, f"Sidecar missing required key '{k}'"

    assert meta["row_count"] == 1440
    assert len(meta["origins"]) == 15
    assert meta["num_boost_round"] == 100

    # Verify version metadata contains python, lightgbm, pandas, numpy
    versions = meta["versions"]
    for pkg in ("python", "lightgbm", "pandas", "numpy"):
        assert pkg in versions and versions[pkg], f"Versions missing package '{pkg}'"

    # Verify sha256 matches parquet bytes
    actual_sha = hashlib.sha256(DEFAULT_EVAL_FIXTURE_PARQUET.read_bytes()).hexdigest()
    assert meta["sha256"] == actual_sha, (
        f"Sidecar SHA-256 {meta['sha256']} does not match parquet hash {actual_sha}"
    )


def test_verify_fixture_integrity_guards(tmp_path: Path) -> None:
    """Requirement 2: verify_fixture_integrity hard-fails on missing sidecar or SHA mismatch."""
    # 1. Missing sidecar raises FileNotFoundError
    missing_sidecar = tmp_path / "nonexistent.json"
    dummy_parquet = tmp_path / "dummy.parquet"
    dummy_parquet.write_bytes(b"dummy")

    with pytest.raises(FileNotFoundError, match="Sidecar fixture not found"):
        verify_fixture_integrity(dummy_parquet, missing_sidecar)

    # 2. Missing parquet raises FileNotFoundError
    valid_sidecar = tmp_path / "sidecar.json"
    valid_sidecar.write_text(json.dumps({"sha256": "abc"}), encoding="utf-8")
    missing_parquet = tmp_path / "nonexistent.parquet"
    with pytest.raises(FileNotFoundError, match="Parquet fixture not found"):
        verify_fixture_integrity(missing_parquet, valid_sidecar)

    # 3. Tampered parquet raises ValueError
    # Precondition assertion (brief template section 5): prove hash genuinely differs
    corrupted_parquet = tmp_path / "corrupted.parquet"
    corrupted_parquet.write_bytes(b"corrupted content")
    corrupted_sha = hashlib.sha256(b"corrupted content").hexdigest()
    recorded_sha = hashlib.sha256(b"original content").hexdigest()
    assert corrupted_sha != recorded_sha, "Precondition failed: hashes must differ"

    bad_sidecar = tmp_path / "bad_sidecar.json"
    bad_sidecar.write_text(json.dumps({"sha256": recorded_sha}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_fixture_integrity(corrupted_parquet, bad_sidecar)


def test_evaluate_from_fixture_matches_independent_naive_mae() -> None:
    """Requirement 2 & Verify item 4: Recompute MAEs via independent naive loop."""
    results = evaluate_from_fixture(
        parquet_path=DEFAULT_EVAL_FIXTURE_PARQUET,
        sidecar_path=DEFAULT_EVAL_FIXTURE_SIDECAR,
        verbose=False,
    )

    df = pd.read_parquet(DEFAULT_EVAL_FIXTURE_PARQUET)

    # Independent, naive recomputation: explicit dictionary loops without helper functions
    horizon_records = df.to_dict(orient="records")
    horizon_groups: dict[int, list[dict]] = {}
    for r in horizon_records:
        h = int(r["horizon"])
        horizon_groups.setdefault(h, []).append(r)

    assert len(horizon_groups) == 96, "Expected all 96 horizons present"

    naive_maes = []
    for h in range(1, 97):
        rows = horizon_groups[h]
        errors = [abs(float(row["candidate_pred"]) - float(row["actual"])) for row in rows]
        naive_mae_h = sum(errors) / len(errors)
        naive_maes.append(naive_mae_h)

    naive_gate_val = sum(naive_maes) / 96.0

    # Assert independent recomputation matches printed/computed GATE_VALUE to 6 decimal places
    assert math.isclose(results.gate_value, naive_gate_val, rel_tol=0.0, abs_tol=1e-6), (
        f"GATE_VALUE mismatch: results={results.gate_value:.8f}, naive={naive_gate_val:.8f}"
    )


def test_evaluate_from_fixture_offline_without_curated(tmp_path: Path) -> None:
    """Requirement 2: make eval succeeds with data/curated renamed away."""
    # Ensure evaluate_from_fixture does not touch data/curated
    results = evaluate_from_fixture(
        parquet_path=DEFAULT_EVAL_FIXTURE_PARQUET,
        sidecar_path=DEFAULT_EVAL_FIXTURE_SIDECAR,
        verbose=False,
    )
    assert results.gate_value > 0
    assert results.intersection_size == 1440


def test_eval_refresh_refuses_too_recent_window() -> None:
    """Requirement 3 & Verify 5: Refuse unless last target is >= 3 days behind newest period."""
    # Case 1: Shortfall scenario
    newest_period = pd.Timestamp("2026-11-07 23:30:00+00:00")
    # Last target is 2026-11-06 00:00 (only 1.98 days behind, shortfall of ~1.02 days)
    too_recent_last_target = pd.Timestamp("2026-11-06 00:00:00+00:00")

    with pytest.raises(ValueError) as exc_info:
        assert_refresh_window_eligibility(
            newest_archive_period=newest_period,
            window_last_target=too_recent_last_target,
            reconcile_window_days=3,
        )

    msg = str(exc_info.value)
    # Must name both dates and the shortfall
    assert "2026-11-07 23:30" in msg, f"Message must name newest archive period: {msg}"
    assert "2026-11-06 00:00" in msg, f"Message must name window last target: {msg}"
    assert "shortfall" in msg.lower(), f"Message must state shortfall: {msg}"

    # Case 2: Eligible scenario (e.g. 2026-11-02 00:00 is 5.98 days behind >= 3 days)
    eligible_target = pd.Timestamp("2026-11-02 00:00:00+00:00")
    # Must not raise
    assert_refresh_window_eligibility(
        newest_archive_period=newest_period,
        window_last_target=eligible_target,
        reconcile_window_days=3,
    )


def test_evaluate_from_fixture_deterministic() -> None:
    """Verify item 2: Repeated evaluation from fixture yields identical numbers to 6 dp."""
    res1 = evaluate_from_fixture(verbose=False)
    res2 = evaluate_from_fixture(verbose=False)

    assert math.isclose(res1.gate_value, res2.gate_value, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(
        res1.operator_gate_value, res2.operator_gate_value, rel_tol=0.0, abs_tol=1e-6
    )
    assert math.isclose(
        res1.persistence_gate_value, res2.persistence_gate_value, rel_tol=0.0, abs_tol=1e-6
    )
    assert math.isclose(
        res1.seasonal_naive_gate_value, res2.seasonal_naive_gate_value, rel_tol=0.0, abs_tol=1e-6
    )


def test_evaluate_from_fixture_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify verbose printing produces table and gate verdict."""
    results = evaluate_from_fixture(verbose=True)
    captured = capsys.readouterr().out
    assert "PER-HORIZON PERFORMANCE (h=1..96)" in captured
    assert "GATE VERDICT: NONE - threshold invalid (ADR-020)" in captured
    assert f"Gate Scalar (GATE_VALUE):    {results.gate_value:.3f} gCO2/kWh" in captured


def test_verify_fixture_integrity_missing_sha_field(tmp_path: Path) -> None:
    """Integrity check fails if sidecar lacks sha256."""
    dummy_p = tmp_path / "dummy.parquet"
    dummy_p.write_bytes(b"content")
    bad_sidecar = tmp_path / "bad.json"
    bad_sidecar.write_text(json.dumps({"some_key": "val"}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain 'sha256' field"):
        verify_fixture_integrity(dummy_p, bad_sidecar, pinned_sha256=None)


def test_evaluate_from_fixture_missing_columns(tmp_path: Path) -> None:
    """Evaluation fails if fixture misses required columns."""
    df = pd.DataFrame({"issue_time": [pd.Timestamp("2026-09-05")]})
    p_path = tmp_path / "test.parquet"
    s_path = tmp_path / "test.json"
    df.to_parquet(p_path)
    sha = hashlib.sha256(p_path.read_bytes()).hexdigest()
    s_path.write_text(json.dumps({"sha256": sha}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_from_fixture(p_path, s_path, pinned_sha256=None)


def test_evaluate_from_fixture_invalid_row_count(tmp_path: Path) -> None:
    """Evaluation fails if fixture does not have exactly 1440 rows."""
    cols = {col: [1.0] * 10 for col in REQUIRED_FIXTURE_COLUMNS}
    cols["issue_time"] = [pd.Timestamp("2026-09-05")] * 10
    cols["target_time"] = [pd.Timestamp("2026-09-05 00:30")] * 10
    cols["horizon"] = list(range(1, 11))
    df = pd.DataFrame(cols)
    p_path = tmp_path / "test.parquet"
    s_path = tmp_path / "test.json"
    df.to_parquet(p_path)
    sha = hashlib.sha256(p_path.read_bytes()).hexdigest()
    s_path.write_text(json.dumps({"sha256": sha}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected fixture to have 1440 rows"):
        evaluate_from_fixture(p_path, s_path, pinned_sha256=None)


def test_cli_eval_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI eval entry point executes cleanly and prints verdict."""
    rc = run_eval_cli(verbose=True)
    assert rc == 0
    captured = capsys.readouterr().out
    assert "GATE VERDICT: NONE - threshold invalid (ADR-020)" in captured

    rc_main = main(["eval"])
    assert rc_main == 0
    captured_main = capsys.readouterr().out
    assert "GATE VERDICT: NONE - threshold invalid (ADR-020)" in captured_main


def test_refresh_eval_fixture_immutability_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse to edit v1 if model artifact hash differs."""
    monkeypatch.setattr(
        "gridcast.backtest.get_model_artifact_hash",
        lambda: "tampered_diff_hash_0000000000",
    )
    with pytest.raises(RuntimeError, match="frozen_split_v1 is immutable"):
        refresh_eval_fixture(version="v1")


def test_refresh_eval_fixture_mocked_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify refresh_eval_fixture writes atomic parquet and sidecar cleanly."""
    intensity_dates = pd.date_range("2024-11-08", "2026-11-10", freq="30min", tz="UTC")
    intensity_df = pd.DataFrame({
        "period_start_utc": intensity_dates,
        "forecast": [50.0] * len(intensity_dates),
        "actual": [50.0] * len(intensity_dates),
    })
    weather_df = pd.DataFrame({"time": intensity_dates})
    mix_df = pd.DataFrame({"time": intensity_dates})

    monkeypatch.setattr(
        "gridcast.backtest.load_feature_sources",
        lambda: (intensity_df, weather_df, mix_df),
    )
    monkeypatch.setattr(
        "gridcast.backtest.build_example_frame",
        lambda **kwargs: pd.DataFrame({"dummy": [1]}),
    )

    origins = [pd.Timestamp(f"2026-09-{i+1:02d} 00:00:00+00:00") for i in range(15)]
    pred_rows: list[dict[str, Any]] = []
    for origin in origins:
        for h in range(1, 97):
            pred_rows.append({
                "issue_time": origin,
                "horizon": h,
                "target_time": origin + pd.Timedelta(minutes=30 * h),
                "actual": 40.0,
                "operator_forecast": 42.0,
                "persistence_forecast": 41.0,
                "seasonal_naive_forecast": 41.5,
                "forecast": 39.5,
            })
    mock_preds = pd.DataFrame(pred_rows)

    dummy_results = BacktestResults(
        predictions=mock_preds,
        per_horizon_table=pd.DataFrame({"horizon": list(range(1, 97))}),
        gate_value=39.032,
        operator_gate_value=42.0,
        persistence_gate_value=41.0,
        seasonal_naive_gate_value=41.5,
        overall_skill_vs_operator=0.05,
        skill_h2=0.05,
        skill_h48=0.05,
        skill_h96=0.05,
        intersection_size=1440,
        excluded_counts={},
        origins_evaluated=origins,
    )

    monkeypatch.setattr(
        "gridcast.backtest.run_rolling_origin_backtest",
        lambda **kwargs: dummy_results,
    )

    pq_path, sc_path, pq_sha, gate_val = refresh_eval_fixture(
        version="vtest",
        output_dir=tmp_path,
        verbose=True,
    )
    assert pq_path.exists()
    assert sc_path.exists()
    assert len(pq_sha) == 64
    assert gate_val == 39.032

    # Also test CLI wrapper
    rc = run_eval_refresh_cli(
        version="vtest2",
        output_dir=tmp_path,
        verbose=False,
    )
    assert rc == 0
    assert (tmp_path / "frozen_split_vtest2.parquet").exists()

    rc_cli = main(["eval-refresh", "--version", "vtest3", "--output-dir", str(tmp_path)])
    assert rc_cli == 0
    assert (tmp_path / "frozen_split_vtest3.parquet").exists()


def test_pinned_eval_fixture_sha256_refusal_on_tamper(tmp_path: Path) -> None:
    """Section 1: Refusal with expected/actual hashes when one byte is flipped; clean passes."""
    tmp_parquet = tmp_path / "frozen_split_v1.parquet"
    tmp_sidecar = tmp_path / "frozen_split_v1.json"

    # Copy fixture and sidecar to temp path
    tmp_parquet.write_bytes(DEFAULT_EVAL_FIXTURE_PARQUET.read_bytes())
    tmp_sidecar.write_text(
        DEFAULT_EVAL_FIXTURE_SIDECAR.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Injection precondition (brief template section 5): prove copied file matches pinned constant
    initial_sha = hashlib.sha256(tmp_parquet.read_bytes()).hexdigest()
    assert initial_sha == PINNED_EVAL_FIXTURE_SHA256, (
        f"Precondition failed: copy {initial_sha} != pinned {PINNED_EVAL_FIXTURE_SHA256}"
    )

    # 1. Unmodified file passes
    results = evaluate_from_fixture(
        parquet_path=tmp_parquet,
        sidecar_path=tmp_sidecar,
        verbose=False,
        pinned_sha256=PINNED_EVAL_FIXTURE_SHA256,
    )
    assert results.gate_value > 0

    # 2. Flip exactly one byte
    corrupted_bytes = bytearray(tmp_parquet.read_bytes())
    corrupted_bytes[0] ^= 0x01
    tmp_parquet.write_bytes(bytes(corrupted_bytes))

    # Precondition assertion (brief template section 5): prove mutation took effect and hash differs
    corrupted_sha = hashlib.sha256(tmp_parquet.read_bytes()).hexdigest()
    assert corrupted_sha != PINNED_EVAL_FIXTURE_SHA256, (
        "Precondition failed: flipped-byte hash must differ from pinned hash"
    )

    # 3. Assert refusal with both values named
    with pytest.raises(ValueError) as exc_info:
        evaluate_from_fixture(
            parquet_path=tmp_parquet,
            sidecar_path=tmp_sidecar,
            verbose=False,
            pinned_sha256=PINNED_EVAL_FIXTURE_SHA256,
        )

    msg = str(exc_info.value)
    assert PINNED_EVAL_FIXTURE_SHA256 in msg, (
        f"Refusal message must name expected hash ({PINNED_EVAL_FIXTURE_SHA256}): {msg}"
    )
    assert corrupted_sha in msg, (
        f"Refusal message must name actual hash ({corrupted_sha}): {msg}"
    )

    # 4. Assert CLI eval refusal exits non-zero
    rc = run_eval_cli(
        fixture_path=tmp_parquet,
        sidecar_path=tmp_sidecar,
        verbose=False,
        pinned_sha256=PINNED_EVAL_FIXTURE_SHA256,
    )
    assert rc != 0, "CLI run_eval_cli must exit non-zero on hash refusal"


def test_check_sidecar_library_versions() -> None:
    """Section 2: Check library versions against sidecar metadata."""
    with open(DEFAULT_EVAL_FIXTURE_SIDECAR, encoding="utf-8") as f:
        meta = json.load(f)

    # Clean case in current test environment
    clean_mismatches = check_sidecar_library_versions(meta)
    assert clean_mismatches == [], f"Expected clean version check, got: {clean_mismatches}"

    # Mutated case
    tampered_meta = dict(meta)
    tampered_meta["versions"] = dict(meta["versions"])
    tampered_meta["versions"]["lightgbm"] = "99.99.99"

    # Precondition check: version was mutated
    assert tampered_meta["versions"]["lightgbm"] != meta["versions"]["lightgbm"]

    mismatches = check_sidecar_library_versions(tampered_meta)
    assert len(mismatches) == 1
    assert "lightgbm" in mismatches[0]
    assert "99.99.99" in mismatches[0]


def test_sidecar_python_patch_tolerated_but_minor_rejected() -> None:
    """A different Python patch is fine; a different minor version is not.

    The CI runner serves whatever 3.10.x is current, while the fixture records the patch that ran
    when it was built. Demanding the exact patch failed every run for a difference that cannot move
    a float. A minor difference still fails, because that can.
    """
    current = sys.version.split()[0]
    major, minor = current.split(".")[:2]

    same_minor = check_sidecar_library_versions({"versions": {"python": f"{major}.{minor}.0"}})
    assert same_minor == [], f"Python patch difference must be tolerated, got {same_minor}"

    other_minor_version = f"{major}.{int(minor) + 1}.0"
    other_minor = check_sidecar_library_versions({"versions": {"python": other_minor_version}})
    assert len(other_minor) == 1, f"Python minor difference must be rejected, got {other_minor}"
    assert "python" in other_minor[0]



def test_reproduce_eval_fixture_mocked_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Section 2: Fast unit test of reproduce_eval_fixture logic with mocked backtest."""
    df_fixture = pd.read_parquet(DEFAULT_EVAL_FIXTURE_PARQUET)

    intensity_dates = pd.date_range("2024-11-08", "2026-11-10", freq="30min", tz="UTC")
    intensity_df = pd.DataFrame({
        "period_start_utc": intensity_dates,
        "forecast": [50.0] * len(intensity_dates),
        "actual": [50.0] * len(intensity_dates),
    })
    weather_df = pd.DataFrame({"time": intensity_dates})
    mix_df = pd.DataFrame({"time": intensity_dates})

    monkeypatch.setattr(
        "gridcast.backtest.load_feature_sources",
        lambda: (intensity_df, weather_df, mix_df),
    )
    monkeypatch.setattr(
        "gridcast.backtest.build_example_frame",
        lambda **kwargs: pd.DataFrame({"dummy": [1]}),
    )

    # Clean reproduction mock
    fixture_eval = evaluate_from_fixture(verbose=False)
    preds_clean = pd.DataFrame({
        "issue_time": df_fixture["issue_time"],
        "horizon": df_fixture["horizon"],
        "target_time": df_fixture["target_time"],
        "actual": df_fixture["actual"],
        "forecast": df_fixture["candidate_pred"],
        "operator_forecast": df_fixture["neso_forecast"],
        "persistence_forecast": df_fixture["persistence_pred"],
        "seasonal_naive_forecast": df_fixture["seasonal_naive_pred"],
    })
    clean_results = BacktestResults(
        predictions=preds_clean,
        per_horizon_table=pd.DataFrame({"horizon": list(range(1, 97))}),
        gate_value=fixture_eval.gate_value,
        operator_gate_value=11.666,
        persistence_gate_value=53.741,
        seasonal_naive_gate_value=51.495,
        overall_skill_vs_operator=-2.34,
        skill_h2=-0.81,
        skill_h48=-2.73,
        skill_h96=-3.62,
        intersection_size=1440,
        excluded_counts={},
        origins_evaluated=list(df_fixture["issue_time"].unique()),
    )

    monkeypatch.setattr(
        "gridcast.backtest.run_rolling_origin_backtest",
        lambda **kwargs: clean_results,
    )

    # Verify clean run passes with tolerance 0.0
    res = reproduce_eval_fixture(
        parquet_path=DEFAULT_EVAL_FIXTURE_PARQUET,
        sidecar_path=DEFAULT_EVAL_FIXTURE_SIDECAR,
        max_pred_tolerance=0.0,
        gate_value_tolerance=0.0,
        verbose=False,
    )
    assert res["max_pred_delta"] == 0.0
    assert res["gate_value_delta"] == 0.0

    # Corrupted reproduction mock: predictions diverge
    preds_bad = preds_clean.copy()
    preds_bad.loc[0, "forecast"] = float(preds_bad.loc[0, "forecast"]) + 1.5
    # Precondition assertion
    assert abs(preds_bad.loc[0, "forecast"] - df_fixture.loc[0, "candidate_pred"]) == 1.5

    bad_results = BacktestResults(
        predictions=preds_bad,
        per_horizon_table=pd.DataFrame({"horizon": list(range(1, 97))}),
        gate_value=39.03247766,
        operator_gate_value=11.666,
        persistence_gate_value=53.741,
        seasonal_naive_gate_value=51.495,
        overall_skill_vs_operator=-2.34,
        skill_h2=-0.81,
        skill_h48=-2.73,
        skill_h96=-3.62,
        intersection_size=1440,
        excluded_counts={},
        origins_evaluated=list(df_fixture["issue_time"].unique()),
    )
    monkeypatch.setattr(
        "gridcast.backtest.run_rolling_origin_backtest",
        lambda **kwargs: bad_results,
    )

    with pytest.raises(RuntimeError, match="max per-row candidate_pred delta"):
        reproduce_eval_fixture(
            parquet_path=DEFAULT_EVAL_FIXTURE_PARQUET,
            sidecar_path=DEFAULT_EVAL_FIXTURE_SIDECAR,
            max_pred_tolerance=0.0,
            gate_value_tolerance=0.0,
            verbose=False,
        )


def test_cli_eval_reproduce_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Section 2: Test CLI eval-reproduce entry point."""
    monkeypatch.setattr(
        "gridcast.backtest.reproduce_eval_fixture",
        lambda **kwargs: {
            "max_pred_delta": 0.0,
            "gate_value_delta": 0.0,
            "retrained_gate_value": 39.032478,
            "fixture_gate_value": 39.032478,
            "elapsed_seconds": 19.0,
            "mismatches": [],
        },
    )

    rc = run_eval_reproduce_cli(verbose=False)
    assert rc == 0

    rc_main = main(["eval-reproduce", "--quiet"])
    assert rc_main == 0

    # Failure case
    def _raise(**kwargs: Any) -> None:
        raise RuntimeError("Delta exceeded tolerance")

    monkeypatch.setattr("gridcast.backtest.reproduce_eval_fixture", _raise)
    rc_fail = run_eval_reproduce_cli(verbose=False)
    assert rc_fail == 1
    captured = capsys.readouterr().err
    assert "Reproduction check failed: Delta exceeded tolerance" in captured


def test_v2_fixture_retains_v1_and_shares_truth_columns() -> None:
    """Verify v1 is retained on disk and v2 shares truth columns byte-for-byte with v1."""
    v1_parquet = Path("data/eval/frozen_split_v1.parquet")
    v1_sidecar = Path("data/eval/frozen_split_v1.json")
    assert v1_parquet.exists(), "v1 fixture must be retained on disk"
    assert v1_sidecar.exists(), "v1 sidecar must be retained on disk"

    v2_parquet = Path("data/eval/frozen_split_v2.parquet")
    v2_sidecar = Path("data/eval/frozen_split_v2.json")
    assert v2_parquet.exists(), "v2 fixture must exist on disk"
    assert v2_sidecar.exists(), "v2 sidecar must exist on disk"

    v1_df = pd.read_parquet(v1_parquet)
    v2_df = pd.read_parquet(v2_parquet)

    truth_columns = [
        "issue_time",
        "horizon",
        "target_time",
        "actual",
        "neso_forecast",
        "persistence_pred",
        "seasonal_naive_pred",
    ]
    for col in truth_columns:
        assert v1_df[col].equals(v2_df[col]), f"Truth column {col} mismatch between v1 and v2"
        if pd.api.types.is_datetime64_any_dtype(v1_df[col]):
            b1 = v1_df[col].astype("int64").to_numpy().tobytes()
            b2 = v2_df[col].astype("int64").to_numpy().tobytes()
        else:
            b1 = v1_df[col].to_numpy().tobytes()
            b2 = v2_df[col].to_numpy().tobytes()
        assert hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest(), (
            f"Truth column {col} byte hash mismatch between v1 and v2"
        )

    # Assert candidate_pred changed
    assert not v1_df["candidate_pred"].equals(v2_df["candidate_pred"])

    # Evaluate v2
    res = evaluate_from_fixture(parquet_path=v2_parquet, sidecar_path=v2_sidecar, verbose=False)
    assert math.isclose(res.gate_value, 39.118459, abs_tol=1e-5)


