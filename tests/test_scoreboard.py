"""Tests for persistence baseline forecast, scoreboard scoring, and docs generation."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from gridcast import cli
from gridcast.features import FEATURE_COLUMNS
from gridcast.scoreboard import (
    LGBM_MODEL_VERSION,
    MODEL_VERSION,
    SCORE_COLUMNS,
    generate_scoreboard_html,
    generate_scoreboard_markdown,
    get_operator_forecast_at_issue,
    issue_forecast,
    issue_lgbm_forecast,
    run_forecast_cli,
    score_forecasts,
)


def _make_curated_partition(
    target_date: date,
    actual_by_hour: dict[int, int | None] | None = None,
    forecast_by_hour: dict[int, int | None] | None = None,
    default_actual: int | None = 150,
    default_forecast: int | None = 160,
) -> pd.DataFrame:
    """Create a minimal curated day partition with 48 half-hourly periods."""
    start_utc = pd.Timestamp(f"{target_date} 00:00:00", tz="UTC")
    periods = [start_utc + pd.Timedelta(minutes=30 * i) for i in range(48)]

    actuals: list[object] = []
    forecasts: list[object] = []
    for p in periods:
        h = p.hour
        act = (
            actual_by_hour[h]
            if (actual_by_hour and h in actual_by_hour)
            else default_actual
        )
        fct = (
            forecast_by_hour[h]
            if (forecast_by_hour and h in forecast_by_hour)
            else default_forecast
        )
        actuals.append(act if act is not None else pd.NA)
        forecasts.append(fct if fct is not None else pd.NA)

    return pd.DataFrame(
        {
            "period_start_utc": periods,
            "period_end_utc": [p + pd.Timedelta(minutes=30) for p in periods],
            "utc_date": [p.date() for p in periods],
            "local_date": [target_date] * 48,
            "forecast": pd.Series(forecasts, dtype="Int32"),
            "actual": pd.Series(actuals, dtype="Int32"),
            "index": ["moderate"] * 48,
            "ingested_at": [start_utc] * 48,
            "first_seen_at": [start_utc] * 48,
        }
    )


def test_forecast_issues_96_points(tmp_path: Path) -> None:
    """One issue yields 96 rows, absolute half-hour offsets, spanning 48 hours."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=175)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "scoreboard" / "forecasts.parquet"
    issue_time = "2026-11-02 00:00:00+00:00"

    df, new_count = issue_forecast(
        issue_time=issue_time,
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    assert len(df) == 96
    assert new_count == 96
    assert (df["model_version"] == "persistence-v0").all()
    assert list(df["horizon"]) == list(range(1, 97))

    expected_issue = pd.Timestamp(issue_time)
    assert (df["issue_time"] == expected_issue).all()
    assert (df["value"] == 175.0).all()

    expected_targets = [
        expected_issue + pd.Timedelta(minutes=30 * h) for h in range(1, 97)
    ]
    assert list(df["target_time"]) == expected_targets
    assert df["target_time"].iloc[0] == expected_issue + pd.Timedelta(minutes=30)
    assert df["target_time"].iloc[-1] == expected_issue + pd.Timedelta(hours=48)

    assert forecasts_path.exists()
    loaded = pd.read_parquet(forecasts_path)
    assert len(loaded) == 96
    assert (loaded["value"] == 175.0).all()


def test_forecast_uses_availability_allowance(tmp_path: Path) -> None:
    """The value equals stored actual at issue_time - 3h, and raises if missing."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()

    # issue_time: 2026-11-02 00:00:00+00:00; issue_time - 3h is 2026-11-01 21:00:00+00:00
    p_prev = _make_curated_partition(
        date(2026, 11, 1),
        actual_by_hour={21: 182, 22: 190, 20: 140},
        default_actual=100,
    )
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    df, _ = issue_forecast(
        issue_time="2026-11-02 00:00:00+00:00",
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )
    assert (df["value"] == 182.0).all()

    # Null actual at 21:00 must raise rather than invent a number
    p_null = _make_curated_partition(
        date(2026, 11, 1),
        actual_by_hour={21: None},
        default_actual=100,
    )
    p_null.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    with pytest.raises(ValueError, match="is null"):
        issue_forecast(
            issue_time="2026-11-02 00:00:00+00:00",
            curated_dir=curated_dir,
            output_path=tmp_path / "f2.parquet",
        )

    # Missing partition / row must raise rather than invent a number
    empty_dir = tmp_path / "empty_curated"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="not found"):
        issue_forecast(
            issue_time="2026-11-02 00:00:00+00:00",
            curated_dir=empty_dir,
            output_path=tmp_path / "f3.parquet",
        )


def test_score_skips_unsettled_targets(tmp_path: Path) -> None:
    """A forecast whose target actual is null produces no score row."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    issue_forecast(
        issue_time="2026-11-02 00:00:00+00:00",
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    # Target partition 2026-11-02 has null actuals
    p_today = _make_curated_partition(date(2026, 11, 2), default_actual=None)
    p_today.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    summary = score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )

    assert summary.points_scored == 0
    assert summary.total_points == 0
    assert scores_path.exists()
    scores_df = pd.read_csv(scores_path)
    assert len(scores_df) == 0
    assert list(scores_df.columns) == SCORE_COLUMNS


def test_score_is_idempotent(tmp_path: Path) -> None:
    """Scoring twice leaves one row per (issue_time, horizon, model_version)."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    issue_forecast(
        issue_time="2026-11-02 00:00:00+00:00",
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    # Target partition 2026-11-02 has settled actuals for first 48 horizons
    p_today = _make_curated_partition(
        date(2026, 11, 2),
        default_actual=165,
        default_forecast=170,
    )
    p_today.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    summary1 = score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )
    assert summary1.points_scored == 47
    assert summary1.total_points == 47

    df1 = pd.read_csv(scores_path)
    assert len(df1) == 47

    # Second run must not duplicate any scored rows
    summary2 = score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )
    assert summary2.points_scored == 0
    assert summary2.total_points == 47

    df2 = pd.read_csv(scores_path)
    assert len(df2) == 47
    keys = list(zip(df2["issue_time"], df2["horizon"], df2["model_version"], strict=True))
    assert len(keys) == len(set(keys))


def test_score_records_null_operator_forecast_as_null(tmp_path: Path) -> None:
    """Operator forecast is left null where null upstream and never imputed."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    issue_forecast(
        issue_time="2026-11-02 00:00:00+00:00",
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    # 2026-11-02 has settled actuals, but hour 1 has null operator forecast
    p_today = _make_curated_partition(
        date(2026, 11, 2),
        actual_by_hour={1: 155},
        forecast_by_hour={1: None},
        default_actual=150,
        default_forecast=160,
    )
    p_today.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )

    df = pd.read_csv(scores_path)
    # Horizon 2 corresponds to 01:00:00
    row_h2 = df[df["horizon"] == 2]
    assert len(row_h2) == 1
    assert pd.isna(row_h2["neso_forecast"].iloc[0])
    assert pd.isna(row_h2["neso_abs_error"].iloc[0])
    assert row_h2["actual"].iloc[0] == 155.0
    assert row_h2["abs_error"].iloc[0] == abs(160.0 - 155.0)


def test_scoreboard_markdown_regenerates(tmp_path: Path) -> None:
    """Running twice produces the same file for same scores, reflecting distinct dates present."""
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    # Distinct dates: 2 consecutive issue dates
    rows = [
        {
            "issue_time": "2026-11-02 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-02 00:30:00+00:00",
            "model_version": MODEL_VERSION,
            "forecast": 150.0,
            "actual": 155.0,
            "neso_forecast": 160.0,
            "abs_error": 5.0,
            "neso_abs_error": 5.0,
        },
        {
            "issue_time": "2026-11-03 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-03 00:30:00+00:00",
            "model_version": MODEL_VERSION,
            "forecast": 150.0,
            "actual": 140.0,
            "neso_forecast": 145.0,
            "abs_error": 10.0,
            "neso_abs_error": 5.0,
        },
    ]
    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(scores_path, index=False)

    content1 = generate_scoreboard_markdown(scores_df, output_path=md_path)
    content2 = generate_scoreboard_markdown(scores_df, output_path=md_path)

    assert content1 == content2
    assert "scoreboard started 2026-11-02; consecutive days recorded: 2" in content1
    assert "0\u201312h" in content1
    assert "12\u201324h" in content1
    assert "24\u201336h" in content1
    assert "36\u201348h" in content1
    assert "Scored points:** 2" in content1

    # Empty scores dataframe check
    empty_md_path = tmp_path / "empty_scoreboard.md"
    empty_content = generate_scoreboard_markdown(
        pd.DataFrame(columns=SCORE_COLUMNS),
        output_path=empty_md_path,
        start_date=date(2026, 11, 2),
    )
    assert "scoreboard started 2026-11-02; consecutive days recorded: 0" in empty_content
    assert "Scored points:** 0" in empty_content
    assert "Date range covered:** None" in empty_content


def test_cli_forecast_and_score(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI subcommands execute and skip lgbm-v2 only when model artifact is absent."""
    monkeypatch.setattr("gridcast.scoreboard.DEFAULT_MODEL_PATH", tmp_path / "no_lgbm_model.txt")
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    code_fc = cli.main(
        [
            "forecast",
            "--date",
            "2026-11-02",
            "--curated-dir",
            str(curated_dir),
            "--output",
            str(forecasts_path),
        ]
    )
    assert code_fc == 0
    cap_fc = capsys.readouterr()
    assert "Issued 96 forecast points" in cap_fc.out
    assert "persistence-v0" in cap_fc.out

    # Re-issue (idempotent)
    code_fc2 = cli.main(
        [
            "forecast",
            "--date",
            "2026-11-02",
            "--curated-dir",
            str(curated_dir),
            "--output",
            str(forecasts_path),
        ]
    )
    assert code_fc2 == 0
    cap_fc2 = capsys.readouterr()
    assert "New points written: 0" in cap_fc2.out

    code_sc = cli.main(
        [
            "score",
            "--curated-dir",
            str(curated_dir),
            "--forecasts-path",
            str(forecasts_path),
            "--scores-path",
            str(scores_path),
            "--scoreboard-md-path",
            str(md_path),
        ]
    )
    assert code_sc == 0
    cap_sc = capsys.readouterr()
    assert "points scored: 0" in cap_sc.out

    # Error handling in CLI
    code_err = cli.main(
        [
            "forecast",
            "--date",
            "2026-11-02",
            "--curated-dir",
            str(tmp_path / "nonexistent"),
            "--output",
            str(forecasts_path),
        ]
    )
    assert code_err == 1


def test_forecast_snapshots_operator_published_at_issue(tmp_path: Path) -> None:
    """Snapshot reads operator forecast published as of T; null where unpublished at T."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    issue_ts = pd.Timestamp("2026-11-02 00:00:00+00:00")
    forecasts_path = tmp_path / "forecasts.parquet"

    # Create target partition 2026-11-02 with varying publication timestamps
    p_target = _make_curated_partition(date(2026, 11, 2), default_actual=150, default_forecast=160)
    # Add forecast_first_seen_at column with explicit timestamps
    # Horizon 1 (00:30, index 1): published before T
    # Horizon 2 (01:00, index 2): published after T
    # Horizon 3 (01:30, index 3): unverified vintage (null forecast_first_seen_at)
    # Horizon 4 (02:00, index 4): null operator forecast
    seen_col: list[object] = [pd.NaT] * 48
    seen_col[1] = issue_ts - pd.Timedelta(hours=1)  # published before T
    seen_col[2] = issue_ts + pd.Timedelta(hours=2)  # published after T
    seen_col[3] = pd.NaT  # unverified vintage
    seen_col[4] = issue_ts - pd.Timedelta(hours=1)

    p_target["forecast_first_seen_at"] = pd.to_datetime(seen_col, utc=True)
    p_target.loc[4, "forecast"] = pd.NA
    p_target.loc[4, "forecast_first_seen_at"] = pd.NaT
    p_target.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    df, count = issue_forecast(
        issue_time=issue_ts,
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    assert count == 96
    # Horizon 1: published before T -> snapshot captured
    assert df.loc[df["horizon"] == 1, "neso_forecast_at_issue"].iloc[0] == 160.0
    # Horizon 2: published after T -> null snapshot
    assert pd.isna(df.loc[df["horizon"] == 2, "neso_forecast_at_issue"].iloc[0])
    # Horizon 3: unverified vintage -> null snapshot
    assert pd.isna(df.loc[df["horizon"] == 3, "neso_forecast_at_issue"].iloc[0])
    # Horizon 4: null forecast upstream -> null snapshot
    assert pd.isna(df.loc[df["horizon"] == 4, "neso_forecast_at_issue"].iloc[0])


def test_score_records_snapshot_source_and_null_where_unpublished(tmp_path: Path) -> None:
    """Scoring marks neso_source=snapshot_at_issue or none; does not fill from later publication."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    p_prev = _make_curated_partition(date(2026, 11, 1), default_actual=160)
    p_prev.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)

    issue_ts = pd.Timestamp("2026-11-02 00:00:00+00:00")
    forecasts_path = tmp_path / "forecasts.parquet"
    scores_path = tmp_path / "scores.csv"
    md_path = tmp_path / "scoreboard.md"

    p_target = _make_curated_partition(date(2026, 11, 2), default_actual=150, default_forecast=170)
    seen_col: list[object] = [pd.NaT] * 48
    seen_col[1] = issue_ts - pd.Timedelta(hours=1)  # horizon 1 (index 1: 00:30) published before T
    p_target["forecast_first_seen_at"] = pd.to_datetime(seen_col, utc=True)
    p_target.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    issue_forecast(
        issue_time=issue_ts,
        curated_dir=curated_dir,
        output_path=forecasts_path,
    )

    score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
    )

    scores_df = pd.read_csv(scores_path)
    # Horizon 1 has snapshot
    h1 = scores_df[scores_df["horizon"] == 1].iloc[0]
    assert h1["neso_source"] == "snapshot_at_issue"
    assert h1["neso_forecast"] == 170.0
    assert h1["neso_abs_error"] == abs(170.0 - 150.0)

    # Horizon 2 has no snapshot published at T -> null and neso_source=none
    h2 = scores_df[scores_df["horizon"] == 2].iloc[0]
    assert h2["neso_source"] == "none"
    assert pd.isna(h2["neso_forecast"])
    assert pd.isna(h2["neso_abs_error"])


def test_scoreboard_markdown_separates_sources_and_refuses_to_aggregate(tmp_path: Path) -> None:
    """Scoreboard markdown reports snapshot and archive separately without aggregating."""
    md_path = tmp_path / "scoreboard.md"
    rows = [
        # Snapshot row
        {
            "issue_time": "2026-11-02 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-02 00:30:00+00:00",
            "model_version": MODEL_VERSION,
            "forecast": 150.0,
            "actual": 160.0,
            "neso_forecast": 170.0,
            "abs_error": 10.0,
            "neso_abs_error": 10.0,
            "neso_source": "snapshot_at_issue",
        },
        # Archive row
        {
            "issue_time": "2026-11-01 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-01 00:30:00+00:00",
            "model_version": MODEL_VERSION,
            "forecast": 150.0,
            "actual": 140.0,
            "neso_forecast": 145.0,
            "abs_error": 10.0,
            "neso_abs_error": 5.0,
            "neso_source": "archive_at_scoring",
        },
    ]
    df = pd.DataFrame(rows)
    content = generate_scoreboard_markdown(df, output_path=md_path)

    # Must contain both distinct sections
    assert "## Snapshot at Issue (Verified Vintage)" in content
    assert "## Archive at Scoring (Vintage Unverified)" in content
    # Refusal callout in archive section
    assert "Refusing to aggregate across sources" in content
    # Skill score only on snapshot section
    assert "Skill vs Operator:" in content
    assert "Published NESO forecast for the same periods, vintage unverified" in content


def _write_tiny_lgbm_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    """Train a tiny LightGBM booster and interval config for scoreboard unit tests."""
    rng = np.random.default_rng(7)
    x_train = pd.DataFrame(
        rng.normal(size=(64, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    y_train = np.full(64, 150.0, dtype=float)
    dtrain = lgb.Dataset(x_train, label=y_train)
    booster = lgb.train(
        {"objective": "regression_l1", "verbosity": -1, "num_leaves": 4},
        dtrain,
        num_boost_round=2,
    )
    model_path = tmp_path / "lgbm_model.txt"
    booster.save_model(str(model_path))

    interval_path = tmp_path / "interval_v2.json"
    interval_path.write_text(
        json.dumps(
            {
                "version": "interval-v2",
                "quantiles": {str(h): {"q10": -20.0, "q90": 20.0} for h in range(1, 97)},
            }
        ),
        encoding="utf-8",
    )
    return model_path, interval_path


def test_issue_lgbm_forecast_and_dual_cli_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """issue_lgbm_forecast and run_forecast_cli emit lgbm-v2 alongside persistence-v0."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    _make_curated_partition(date(2026, 11, 1)).to_parquet(
        curated_dir / "date=2026-11-01.parquet", index=False
    )
    model_path, interval_path = _write_tiny_lgbm_artifacts(tmp_path)
    forecasts_path = tmp_path / "forecasts.parquet"

    with pytest.raises(FileNotFoundError, match="missing_model.txt"):
        issue_lgbm_forecast(
            date_val="2026-11-02",
            curated_dir=curated_dir,
            model_path=tmp_path / "missing_model.txt",
            interval_path=interval_path,
            output_path=forecasts_path,
        )

    feat_rows = pd.DataFrame(
        {col: np.zeros(96, dtype=float) for col in FEATURE_COLUMNS}
    )
    if "horizon" in feat_rows.columns:
        feat_rows["horizon"] = np.arange(1, 97, dtype=float)

    monkeypatch.setattr(
        "gridcast.serve.resolve_features_for_issue",
        lambda **_kwargs: feat_rows.copy(),
    )

    written, added = issue_lgbm_forecast(
        date_val="2026-11-02",
        curated_dir=curated_dir,
        model_path=model_path,
        interval_path=interval_path,
        output_path=forecasts_path,
    )
    assert added == 96
    assert (written["model_version"] == LGBM_MODEL_VERSION).all()
    assert len(written) == 96
    assert np.isfinite(written["value"].astype(float)).all()

    _, added_again = issue_lgbm_forecast(
        date_val="2026-11-02",
        curated_dir=curated_dir,
        model_path=model_path,
        interval_path=interval_path,
        output_path=forecasts_path,
    )
    assert added_again == 0

    dual_path = tmp_path / "dual_forecasts.parquet"
    rc = run_forecast_cli(
        date_str="2026-11-02",
        curated_dir=curated_dir,
        output_path=dual_path,
        mix_dir=tmp_path / "mix",
        model_path=model_path,
        interval_path=interval_path,
    )
    assert rc == 0
    dual_df = pd.read_parquet(dual_path)
    assert len(dual_df) == 192
    assert set(dual_df["model_version"]) == {MODEL_VERSION, LGBM_MODEL_VERSION}


def test_scoreboard_markdown_includes_none_source_per_model(tmp_path: Path) -> None:
    """Live Model Performance includes neso_source == 'none' rows per model version."""
    base = pd.Timestamp("2026-11-02 00:00:00", tz="UTC")
    rows: list[dict[str, object]] = []
    for mv, err in [(MODEL_VERSION, 30.0), (LGBM_MODEL_VERSION, 15.0)]:
        for i in range(48):
            h = i + 1
            src = "archive_at_scoring" if (mv == MODEL_VERSION and h <= 10) else "none"
            neso_fct = 110.0 if src == "archive_at_scoring" else None
            rows.append(
                {
                    "issue_time": base.isoformat(),
                    "horizon": h,
                    "target_time": (base + pd.Timedelta(minutes=30 * i)).isoformat(),
                    "model_version": mv,
                    "forecast": 100.0 + err,
                    "actual": 100.0,
                    "neso_forecast": neso_fct,
                    "abs_error": err,
                    "neso_abs_error": 10.0 if neso_fct is not None else None,
                    "neso_source": src,
                }
            )
    df = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    md_path = tmp_path / "scoreboard.md"
    content = generate_scoreboard_markdown(df, output_path=md_path)

    assert "## Live Model Performance (All Settled Points)" in content
    assert f"### {LGBM_MODEL_VERSION}" in content
    assert f"### {MODEL_VERSION}" in content
    assert "- **Scored points:** 48" in content
    assert "snapshot_at_issue: 0, archive_at_scoring: 0, none: 48" in content
    assert "snapshot_at_issue: 0, archive_at_scoring: 10, none: 38" in content
    assert "Refusing to aggregate across sources" in content


def test_score_forecasts_generates_deterministic_static_html(tmp_path: Path) -> None:
    """score_forecasts emits deterministic self-contained docs/index.html alongside markdown."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    d1 = _make_curated_partition(date(2026, 11, 1), default_actual=140, default_forecast=145)
    d2 = _make_curated_partition(date(2026, 11, 2), default_actual=160, default_forecast=155)
    d1.to_parquet(curated_dir / "date=2026-11-01.parquet", index=False)
    d2.to_parquet(curated_dir / "date=2026-11-02.parquet", index=False)

    forecasts_path = tmp_path / "forecasts.parquet"
    issue_forecast(date_val="2026-11-02", curated_dir=curated_dir, output_path=forecasts_path)

    md_path = tmp_path / "docs" / "scoreboard.md"
    scores_path = tmp_path / "scores.csv"
    eval_fixture = Path("data/eval/frozen_split_v2.parquet")
    summary = score_forecasts(
        curated_dir=curated_dir,
        forecasts_path=forecasts_path,
        scores_path=scores_path,
        scoreboard_md_path=md_path,
        eval_fixture_path=eval_fixture,
    )
    assert summary.points_scored == 47

    html_path = tmp_path / "docs" / "index.html"
    assert html_path.exists()
    html_one = html_path.read_text(encoding="utf-8")
    scores_df = pd.read_csv(scores_path)
    html_two = generate_scoreboard_html(
        scores_df,
        output_path=html_path,
        forecasts_path=forecasts_path,
        eval_fixture_path=eval_fixture,
    )
    assert html_one == html_two
    assert "<script" not in html_one.lower()
    assert "https://cdn" not in html_one.lower()
    assert "<svg" in html_one
    assert "Live Settled Scoreboard" in html_one
    assert "Frozen 15-Origin Benchmark" in html_one
    assert "window 2026-09-05 to 2026-11-01" in html_one
    assert "Error Decomposition" in html_one
    assert "Latest Issued 48h Forecast Curve" in html_one


def test_run_forecast_cli_fails_loudly_when_model_exists_and_lgbm_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When lgbm_model.txt exists, unexpected lgbm-v2 failures return exit code 1."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    _make_curated_partition(date(2026, 11, 1)).to_parquet(
        curated_dir / "date=2026-11-01.parquet", index=False
    )
    model_path, interval_path = _write_tiny_lgbm_artifacts(tmp_path)
    out_path = tmp_path / "forecasts.parquet"

    rc = run_forecast_cli(
        date_str="2026-11-02",
        curated_dir=curated_dir,
        output_path=out_path,
        mix_dir=tmp_path / "missing_mix",
        model_path=model_path,
        interval_path=interval_path,
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert f"gridcast forecast error ({LGBM_MODEL_VERSION}):" in captured.err


def test_scoreboard_separates_snapshot_and_archive_per_model_version(tmp_path: Path) -> None:
    """Snapshot and Archive sections stay separated per model_version with per-slice ranges."""
    rows = [
        # persistence-v0 snapshot on 2026-11-02 (MAE 20, op 10, skill -100%)
        {
            "issue_time": "2026-11-02 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-02 00:30:00+00:00",
            "model_version": MODEL_VERSION,
            "forecast": 120.0,
            "actual": 100.0,
            "neso_forecast": 110.0,
            "abs_error": 20.0,
            "neso_abs_error": 10.0,
            "neso_source": "snapshot_at_issue",
        },
        # lgbm-v2 snapshot on 2026-11-03 (MAE 5, op 10, skill +50%)
        {
            "issue_time": "2026-11-03 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-03 00:30:00+00:00",
            "model_version": LGBM_MODEL_VERSION,
            "forecast": 105.0,
            "actual": 100.0,
            "neso_forecast": 110.0,
            "abs_error": 5.0,
            "neso_abs_error": 10.0,
            "neso_source": "snapshot_at_issue",
        },
        # persistence-v0 archive on 2026-11-01 (MAE 30, op 6)
        {
            "issue_time": "2026-11-01 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-01 00:30:00+00:00",
            "model_version": MODEL_VERSION,
            "forecast": 130.0,
            "actual": 100.0,
            "neso_forecast": 106.0,
            "abs_error": 30.0,
            "neso_abs_error": 6.0,
            "neso_source": "archive_at_scoring",
        },
        # lgbm-v2 archive on 2026-11-04 (MAE 12, op 8)
        {
            "issue_time": "2026-11-04 00:00:00+00:00",
            "horizon": 1,
            "target_time": "2026-11-04 00:30:00+00:00",
            "model_version": LGBM_MODEL_VERSION,
            "forecast": 112.0,
            "actual": 100.0,
            "neso_forecast": 108.0,
            "abs_error": 12.0,
            "neso_abs_error": 8.0,
            "neso_source": "archive_at_scoring",
        },
    ]
    df = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    md = generate_scoreboard_markdown(df, output_path=tmp_path / "scoreboard.md")
    # Snapshot per-model ranges and unpooled skill
    assert "- **Date range covered:** 2026-11-02 to 2026-11-02" in md
    assert "- **Date range covered:** 2026-11-03 to 2026-11-03" in md
    assert "- **Skill vs Operator:** +50.00%" in md
    assert "- **Skill vs Operator:** -100.00%" in md
    # Archive per-model ranges and unpooled MAEs
    assert "- **Date range covered:** 2026-11-01 to 2026-11-01" in md
    assert "- **Date range covered:** 2026-11-04 to 2026-11-04" in md
    assert "- **Model MAE:** 30.00" in md
    assert "- **Model MAE:** 12.00" in md
    # Pooled MAEs (12.50 or 21.00) must never appear
    assert "12.50" not in md
    assert "21.00" not in md

    html_out = generate_scoreboard_html(
        df,
        output_path=tmp_path / "index.html",
        forecasts_path=tmp_path / "missing_forecasts.parquet",
        eval_fixture_path=tmp_path / "missing_eval.parquet",
    )
    assert "<code>lgbm-v2</code>: <strong>1</strong> pts" in html_out
    assert "Skill: +50.00%" in html_out
    assert "Skill: -100.00%" in html_out
    assert "12.50" not in html_out
    assert "21.00" not in html_out


def test_get_operator_forecast_at_issue_partitioned_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partitioned curated_dir skips scanning existing files when target partition is absent."""
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    _make_curated_partition(date(2026, 11, 1)).to_parquet(
        curated_dir / "date=2026-11-01.parquet", index=False
    )

    read_calls = 0
    real_read_parquet = pd.read_parquet

    def _counting_read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal read_calls
        read_calls += 1
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr("gridcast.scoreboard.pd.read_parquet", _counting_read_parquet)

    res = get_operator_forecast_at_issue(
        curated_dir=curated_dir,
        target_ts=pd.Timestamp("2026-11-03 00:30:00+00:00"),
        issue_ts=pd.Timestamp("2026-11-02 00:00:00+00:00"),
    )
    assert res is None
    assert read_calls == 0
