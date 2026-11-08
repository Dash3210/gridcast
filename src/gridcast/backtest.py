"""Rolling-origin backtesting engine, evaluation metrics, and accuracy gate for gridcast.

Enforces five evaluation-integrity requirements:
1. Issue-anchored persistence verified by provenance guard.
2. Baselines and candidates scored on strictly identical period intersections.
3. Single scalar accuracy gate: GATE_VALUE = (1/96) * sum(MAE_h for h=1..96).
4. Training data ends strictly before the issue instant: max(train['target_time']) < T.
5. Skill scores computed on identical period sets at the point of division.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.features import (
    FEATURE_COLUMNS,
    build_example_frame,
    default_issue_schedule,
    issue_time_for_date,
    load_feature_sources,
    select_training_arrays,
)
from gridcast.models import (
    DEFAULT_LGBM_PARAMS,
    assert_baseline_provenance,
    get_model_artifact_hash,
    predict_lgbm,
    predict_persistence,
    predict_seasonal_naive,
    train_lgbm,
)

__all__: list[str] = [
    "DEFAULT_EVAL_FIXTURE_PARQUET",
    "DEFAULT_EVAL_FIXTURE_SIDECAR",
    "PINNED_EVAL_FIXTURE_SHA256",
    "REQUIRED_FIXTURE_COLUMNS",
    "BacktestResults",
    "EMBARGO_END_DATE",
    "EMBARGO_START_DATE",
    "FROZEN_TEST_ORIGINS",
    "TEST_SPLIT_END_DATE",
    "TEST_SPLIT_START_DATE",
    "TEST_SPLIT_STEP_DAYS",
    "WeatherAblationResults",
    "assert_fold_training_integrity",
    "assert_refresh_window_eligibility",
    "check_sidecar_library_versions",
    "compute_gap_gate_threshold",
    "compute_gate_value",
    "compute_per_horizon_metrics",
    "compute_skill_score",
    "derive_test_split_origins",
    "emit_weather_ablation_table",
    "evaluate_from_fixture",
    "evaluate_gap_gate",
    "evaluate_gate",
    "filter_evaluation_intersection",
    "refresh_eval_fixture",
    "reproduce_eval_fixture",
    "run_rolling_origin_backtest",
    "run_weather_ablation",
    "verify_fixture_integrity",
]

DEFAULT_EVAL_FIXTURE_PARQUET: Final[Path] = Path("data/eval/frozen_split_v1.parquet")
DEFAULT_EVAL_FIXTURE_SIDECAR: Final[Path] = Path("data/eval/frozen_split_v1.json")

# Byte hash (SHA-256 over raw file bytes, not git's object id).
PINNED_EVAL_FIXTURE_SHA256: Final[str] = (
    "205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4"
)

REQUIRED_FIXTURE_COLUMNS: Final[list[str]] = [
    "issue_time",
    "horizon",
    "target_time",
    "actual",
    "neso_forecast",
    "persistence_pred",
    "seasonal_naive_pred",
    "candidate_pred",
]

EMBARGO_START_DATE: Final[str] = "2026-08-31"
EMBARGO_END_DATE: Final[str] = "2026-09-04"
TEST_SPLIT_START_DATE: Final[str] = "2026-09-05"
TEST_SPLIT_END_DATE: Final[str] = "2026-11-01"
TEST_SPLIT_STEP_DAYS: Final[int] = 4


def derive_test_split_origins(
    start_date: str = TEST_SPLIT_START_DATE,
    end_date: str = TEST_SPLIT_END_DATE,
    step_days: int = TEST_SPLIT_STEP_DAYS,
) -> list[pd.Timestamp]:
    """Derive the fixed, committed test split origins (15 rolling origins).

    Derives issue timestamps at 00:00 Europe/London from start_date to end_date
    with a fixed step of step_days.
    Asserts that the first origin is at or after TEST_SPLIT_START_DATE and never
    inside the 4-day embargo.
    """
    schedule = default_issue_schedule(start_date, end_date)
    origins = [pd.Timestamp(ts).tz_convert("UTC") for ts in schedule[::step_days]]

    embargo_start_ts = issue_time_for_date(EMBARGO_START_DATE)
    embargo_end_ts = issue_time_for_date(EMBARGO_END_DATE)
    test_start_ts = issue_time_for_date(TEST_SPLIT_START_DATE)

    if not origins:
        raise AssertionError("No origins derived for test split")

    if origins[0] < test_start_ts:
        raise AssertionError(
            f"First origin {origins[0]} is before test split start {test_start_ts} (inside embargo)"
        )

    for o in origins:
        if embargo_start_ts <= o <= embargo_end_ts:
            raise AssertionError(
                f"Origin {o} falls inside the embargo ({EMBARGO_START_DATE} to {EMBARGO_END_DATE})"
            )

    return origins


FROZEN_TEST_ORIGINS: Final[list[pd.Timestamp]] = derive_test_split_origins()


@dataclass(frozen=True)
class BacktestResults:
    """Container for backtest execution results and metrics."""

    predictions: pd.DataFrame
    per_horizon_table: pd.DataFrame
    gate_value: float
    operator_gate_value: float
    persistence_gate_value: float
    seasonal_naive_gate_value: float
    overall_skill_vs_operator: float
    skill_h2: float
    skill_h48: float
    skill_h96: float
    intersection_size: int
    excluded_counts: dict[str, int]
    origins_evaluated: list[pd.Timestamp]


def assert_fold_training_integrity(train_df: pd.DataFrame, origin: pd.Timestamp) -> None:
    """Guard — temporal boundary (requirement 2.4).

    Asserts that for backtest issue at origin T, every training row's target_time is strictly < T.
    """
    if "target_time" not in train_df.columns:
        raise AssertionError("assert_fold_training_integrity: missing 'target_time' in train_df")

    train_tt = pd.to_datetime(train_df["target_time"], utc=True)
    origin_utc = pd.Timestamp(origin)
    if origin_utc.tzinfo is None:
        origin_utc = origin_utc.tz_localize("UTC")
    else:
        origin_utc = origin_utc.tz_convert("UTC")

    violating = train_tt >= origin_utc
    if violating.any():
        max_tt = train_tt.max()
        violating_count = int(violating.sum())
        raise AssertionError(
            f"Fold training leak: {violating_count} row(s) violate target_time < T. "
            f"max(train['target_time'])={max_tt} >= origin_T={origin_utc}"
        )


def compute_skill_score(
    model_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> float:
    """Compute forecast skill score (requirement 2.5): 1 - MAE_model / MAE_baseline.

    Enforces that both candidate and baseline MAEs are computed strictly over identical period sets.
    """
    if "target_time" not in model_df.columns or "target_time" not in baseline_df.columns:
        raise AssertionError("Both frames must contain 'target_time'")
    if "forecast" not in model_df.columns or "forecast" not in baseline_df.columns:
        raise AssertionError("Both frames must contain 'forecast'")
    if "actual" not in model_df.columns or "actual" not in baseline_df.columns:
        raise AssertionError("Both frames must contain 'actual'")

    model_periods = set(pd.to_datetime(model_df["target_time"], utc=True))
    baseline_periods = set(pd.to_datetime(baseline_df["target_time"], utc=True))

    # Assert equality of the period sets used for numerator and denominator at point of division
    if model_periods != baseline_periods:
        raise AssertionError(
            f"Skill score guard: identical period sets required at point of division. "
            f"Model periods: {len(model_periods)}, Baseline periods: {len(baseline_periods)}"
        )

    if len(model_df) != len(baseline_df):
        raise AssertionError(
            f"Skill score guard: row counts must match at point of division "
            f"({len(model_df)} vs {len(baseline_df)})"
        )

    join_cols = [
        c
        for c in ("issue_time", "horizon", "target_time")
        if c in model_df.columns and c in baseline_df.columns
    ]
    if not join_cols:
        join_cols = ["target_time"]

    merged = pd.merge(
        model_df[[*join_cols, "forecast", "actual"]].rename(columns={"forecast": "f_model"}),
        baseline_df[[*join_cols, "forecast"]].rename(columns={"forecast": "f_base"}),
        on=join_cols,
    )

    mae_model = float(np.mean(np.abs(merged["f_model"] - merged["actual"])))
    mae_base = float(np.mean(np.abs(merged["f_base"] - merged["actual"])))

    if mae_base == 0.0:
        return 0.0
    return 1.0 - (mae_model / mae_base)


def filter_evaluation_intersection(
    raw_results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Requirement 2.2: Compute intersection where actual, operator, and models are present.

    Returns the filtered DataFrame and a dictionary detailing excluded period counts per reason.
    """
    total_raw = len(raw_results)

    # Count exclusions
    missing_actual = int(raw_results["actual"].isna().sum()) if "actual" in raw_results else 0
    missing_operator = (
        int(raw_results["operator_forecast"].isna().sum())
        if "operator_forecast" in raw_results
        else 0
    )
    missing_model = int(raw_results["forecast"].isna().sum()) if "forecast" in raw_results else 0

    valid_mask = raw_results["actual"].notna()
    if "operator_forecast" in raw_results:
        valid_mask = valid_mask & raw_results["operator_forecast"].notna()
    if "forecast" in raw_results:
        valid_mask = valid_mask & raw_results["forecast"].notna()

    scored = raw_results[valid_mask].copy()

    # Enforce period set equality across baselines if present
    for baseline_col in ("persistence_forecast", "seasonal_naive_forecast"):
        if baseline_col in scored.columns:
            b_mask = scored[baseline_col].notna()
            scored = scored[b_mask]

    intersection_size = len(scored)
    total_excluded = total_raw - intersection_size

    exclusions = {
        "total_periods": total_raw,
        "intersection_size": intersection_size,
        "total_excluded": total_excluded,
        "missing_actual": missing_actual,
        "missing_operator_forecast": missing_operator,
        "missing_model_forecast": missing_model,
    }
    return scored, exclusions


def compute_gate_value(per_horizon_table: pd.DataFrame, mae_col: str = "model_mae") -> float:
    """Requirement 2.3: Compute the single scalar GATE_VALUE.

    GATE_VALUE = (1 / 96) * sum_{h=1}^{96} MAE_h

    Requires all 96 horizons to have at least one scored period and non-null MAE.
    Raises ValueError if any horizon is missing, has count == 0, or has null MAE.
    The gate scalar is always sum / 96.
    """
    if len(per_horizon_table) != 96:
        msg = f"per_horizon_table must have exactly 96 horizons, got {len(per_horizon_table)}"
        raise ValueError(msg)

    if "count" in per_horizon_table.columns:
        zero_counts = per_horizon_table[per_horizon_table["count"] == 0]
        if not zero_counts.empty:
            missing_h = zero_counts["horizon"].tolist()
            raise ValueError(
                "Silent denominator defect guard: all 96 horizons must have at least one "
                f"scored period. {len(missing_h)} horizon(s) have zero scored periods: {missing_h}"
            )

    if per_horizon_table[mae_col].isna().any():
        nan_h = per_horizon_table[per_horizon_table[mae_col].isna()]["horizon"].tolist()
        raise ValueError(
            f"Silent denominator defect guard: all 96 horizons must have non-null {mae_col}. "
            f"{len(nan_h)} horizon(s) are null: {nan_h}"
        )

    total_mae = float(per_horizon_table[mae_col].sum())
    return total_mae / 96.0


def compute_per_horizon_metrics(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-horizon MAE and skill score table for all 96 horizons."""
    records: list[dict[str, object]] = []

    for h in range(1, 97):
        h_df = scored_df[scored_df["horizon"] == h]
        n_periods = len(h_df)
        if n_periods == 0:
            records.append({
                "horizon": h,
                "count": 0,
                "model_mae": np.nan,
                "operator_mae": np.nan,
                "persistence_mae": np.nan,
                "seasonal_naive_mae": np.nan,
                "skill_vs_operator": np.nan,
            })
            continue

        actuals = h_df["actual"].to_numpy(dtype=float)
        m_preds = h_df["forecast"].to_numpy(dtype=float)
        m_mae = float(np.mean(np.abs(m_preds - actuals)))

        oper_mae = np.nan
        skill_oper = np.nan
        if "operator_forecast" in h_df and h_df["operator_forecast"].notna().any():
            o_preds = h_df["operator_forecast"].to_numpy(dtype=float)
            oper_mae = float(np.mean(np.abs(o_preds - actuals)))
            m_h = h_df[["target_time", "forecast", "actual"]]
            o_h = h_df[["target_time", "operator_forecast", "actual"]].rename(
                columns={"operator_forecast": "forecast"}
            )
            skill_oper = compute_skill_score(m_h, o_h)

        pers_mae = np.nan
        if "persistence_forecast" in h_df and h_df["persistence_forecast"].notna().any():
            p_preds = h_df["persistence_forecast"].to_numpy(dtype=float)
            pers_mae = float(np.mean(np.abs(p_preds - actuals)))

        sn_mae = np.nan
        if "seasonal_naive_forecast" in h_df and h_df["seasonal_naive_forecast"].notna().any():
            s_preds = h_df["seasonal_naive_forecast"].to_numpy(dtype=float)
            sn_mae = float(np.mean(np.abs(s_preds - actuals)))

        records.append({
            "horizon": h,
            "count": n_periods,
            "model_mae": m_mae,
            "operator_mae": oper_mae,
            "persistence_mae": pers_mae,
            "seasonal_naive_mae": sn_mae,
            "skill_vs_operator": skill_oper,
        })

    return pd.DataFrame(records)


def run_rolling_origin_backtest(
    frame: pd.DataFrame,
    origins: Sequence[pd.Timestamp | str] | None = None,
    operator_forecasts: pd.Series | dict[pd.Timestamp, float] | None = None,
    history_actuals: pd.Series | dict[pd.Timestamp, float] | None = None,
    lgbm_params: dict[str, object] | None = None,
    num_boost_round: int = 100,
    verbose: bool = True,
) -> BacktestResults:
    """Execute rolling-origin backtest over the specified origins.

    Requirement 4.1: Takes the assembled production feature frame as an argument.
    Does not import stores, slice raw partitions, or compute features independently.
    """
    for req_col in ("issue_time", "horizon", "target_time", "y", "actual_recent_3h"):
        if req_col not in frame.columns:
            raise ValueError(f"frame missing required column: {req_col}")

    # Build historical actuals lookup from frame's targets if not supplied
    if history_actuals is None:
        valid_targets = frame.loc[
            frame["y"].notna(), ["target_time", "y"]
        ].drop_duplicates("target_time")
        act_map = dict(
            zip(
                pd.to_datetime(valid_targets["target_time"], utc=True),
                valid_targets["y"].astype(float),
                strict=True,
            )
        )
    elif isinstance(history_actuals, pd.Series):
        act_map = dict(
            zip(
                pd.to_datetime(history_actuals.index, utc=True),
                history_actuals.astype(float),
                strict=True,
            )
        )
    else:
        act_map = history_actuals

    # Build operator forecasts lookup if supplied
    if operator_forecasts is not None:
        if isinstance(operator_forecasts, pd.Series):
            oper_map = dict(
                zip(
                    pd.to_datetime(operator_forecasts.index, utc=True),
                    operator_forecasts.astype(float),
                    strict=True,
                )
            )
        else:
            oper_map = operator_forecasts
    else:
        oper_map = {}

    if origins is None:
        resolved_origins = derive_test_split_origins()
    else:
        resolved_origins = []
        for o in origins:
            if isinstance(o, str) and len(o) == 10:
                resolved_origins.append(issue_time_for_date(o))
            else:
                ts = pd.Timestamp(o)
                if ts.tzinfo is None:
                    resolved_origins.append(ts.tz_localize("UTC"))
                else:
                    resolved_origins.append(ts.tz_convert("UTC"))

        embargo_start_ts = issue_time_for_date(EMBARGO_START_DATE)
        embargo_end_ts = issue_time_for_date(EMBARGO_END_DATE)
        for o in resolved_origins:
            if embargo_start_ts <= o <= embargo_end_ts:
                raise AssertionError(
                    f"Origin {o} falls inside the embargo "
                    f"({EMBARGO_START_DATE} to {EMBARGO_END_DATE})"
                )

    from pathlib import Path
    raw_curated_store: pd.DataFrame | None = None
    if Path("data/curated").exists():
        raw_curated_store = pd.read_parquet("data/curated")

    fold_predictions: list[pd.DataFrame] = []

    if verbose:
        print(f"Starting rolling-origin backtest across {len(resolved_origins)} origins...")

    for i, origin_T in enumerate(resolved_origins):
        # 1. Slice training rows strictly before origin_T (Requirement 2.4)
        train_slice = frame[frame["target_time"] < origin_T]
        train_slice = train_slice[train_slice["y"].notna()]

        if len(train_slice) == 0:
            raise ValueError(f"No training rows available before origin {origin_T}")

        # Assert fold integrity: max(train["target_time"]) < T
        assert_fold_training_integrity(train_slice, origin_T)
        max_train_tt = train_slice["target_time"].max()

        if verbose:
            print(f"  Fold {i+1:02d} | Origin T: {origin_T.strftime('%Y-%m-%d %H:%M')} | "
                  f"train rows: {len(train_slice):5d} | max(train target_time): {max_train_tt} < T")

        # 2. Extract X, y positively using production selector
        X_train, y_train = select_training_arrays(train_slice)

        # 3. Train LightGBM model
        booster = train_lgbm(X_train, y_train, params=lgbm_params, num_boost_round=num_boost_round)

        # 4. Slice test fold at origin_T
        test_fold = frame[frame["issue_time"] == origin_T].copy()
        if len(test_fold) != 96:
            raise ValueError(f"Origin {origin_T} has {len(test_fold)} horizons (expected 96)")

        # 5. Predict model
        m_preds = predict_lgbm(booster, test_fold)

        # 6. Predict baselines
        pers_df = predict_persistence(test_fold, raw_store=raw_curated_store)
        assert_baseline_provenance(pers_df, raw_store=raw_curated_store)

        sn_df = predict_seasonal_naive(test_fold, history_actuals=act_map)

        # 7. Collect results
        target_times_list = pd.to_datetime(test_fold["target_time"], utc=True).tolist()
        oper_vals = [oper_map.get(tt, np.nan) for tt in target_times_list]

        fold_res = pd.DataFrame({
            "issue_time": origin_T,
            "horizon": test_fold["horizon"].to_numpy(dtype=int),
            "target_time": pd.to_datetime(test_fold["target_time"], utc=True),
            "actual": test_fold["y"].to_numpy(dtype=float),
            "forecast": m_preds,
            "persistence_forecast": pers_df["forecast"].to_numpy(dtype=float),
            "persistence_source_ts": pers_df["source_period_start_utc"],
            "seasonal_naive_forecast": sn_df["forecast"].to_numpy(dtype=float),
            "seasonal_naive_source_ts": sn_df["source_period_start_utc"],
            "operator_forecast": np.array(oper_vals, dtype=float),
        })
        fold_predictions.append(fold_res)

    all_preds = pd.concat(fold_predictions, ignore_index=True)

    # Filter evaluation intersection (Requirement 2.2)
    scored_preds, exclusions = filter_evaluation_intersection(all_preds)

    # Compute per-horizon metrics
    per_h = compute_per_horizon_metrics(scored_preds)

    # Compute GATE_VALUE (Requirement 2.3)
    gate_val = compute_gate_value(per_h, "model_mae")
    oper_gate_val = compute_gate_value(per_h, "operator_mae")
    pers_gate_val = compute_gate_value(per_h, "persistence_mae")
    sn_gate_val = compute_gate_value(per_h, "seasonal_naive_mae")

    # Compute overall skill through compute_skill_score (Requirement 2.5)
    m_all = scored_preds[["issue_time", "horizon", "target_time", "forecast", "actual"]]
    oper_cols = ["issue_time", "horizon", "target_time", "operator_forecast", "actual"]
    o_all = scored_preds[oper_cols].rename(columns={"operator_forecast": "forecast"})
    overall_skill = compute_skill_score(m_all, o_all)

    # Skill scores at specific horizons h=2, h=48, h=96
    def _skill_at_h(h_target: int) -> float:
        row = per_h[per_h["horizon"] == h_target]
        if row.empty or pd.isna(row["skill_vs_operator"].iloc[0]):
            return np.nan
        return float(row["skill_vs_operator"].iloc[0])

    return BacktestResults(
        predictions=scored_preds,
        per_horizon_table=per_h,
        gate_value=gate_val,
        operator_gate_value=oper_gate_val,
        persistence_gate_value=pers_gate_val,
        seasonal_naive_gate_value=sn_gate_val,
        overall_skill_vs_operator=overall_skill,
        skill_h2=_skill_at_h(2),
        skill_h48=_skill_at_h(48),
        skill_h96=_skill_at_h(96),
        intersection_size=exclusions["intersection_size"],
        excluded_counts=exclusions,
        origins_evaluated=resolved_origins,
    )


def compute_gap_gate_threshold(
    persistence_mae: float,
    operator_mae: float,
    fraction: float = 0.30,
) -> float:
    """Compute required MAE threshold under ADR-021 gate rule.

    gap = MAE_persistence - MAE_operator
    required = MAE_persistence - fraction * gap
    """
    gap = persistence_mae - operator_mae
    return float(persistence_mae - fraction * gap)


def evaluate_gap_gate(
    candidate_mae: float,
    persistence_mae: float,
    operator_mae: float,
    fraction: float = 0.30,
) -> bool:
    """Evaluate whether candidate closes at least `fraction` (30%) of gap."""
    required = compute_gap_gate_threshold(persistence_mae, operator_mae, fraction=fraction)
    return bool(candidate_mae <= required)


def evaluate_gate(
    results: BacktestResults,
    threshold: float | None = None,
    verbose: bool = True,
) -> None:
    """Print backtest evaluation metrics and prominent no-verdict line (ADR-020).

    Phase P3 closes with neither a pass nor a fail claimed because the 35.0 threshold
    was derived from the candidate's own validation score.
    """
    if verbose:
        print("\n" + "=" * 78)
        print("GRIDCAST ACCURACY GATE EVALUATION")
        print("=" * 78)
        if results.origins_evaluated:
            first_origin = results.origins_evaluated[0].strftime("%Y-%m-%d %H:%M UTC")
            last_origin = results.origins_evaluated[-1].strftime("%Y-%m-%d %H:%M UTC")
            print(f"Evaluated origin window:     {first_origin} to {last_origin}")
            print(f"Origins evaluated ({len(results.origins_evaluated)}):")
            for o in results.origins_evaluated:
                print(f"  {o.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"Contributing horizons:       {len(results.per_horizon_table)} / 96")
        print(f"Intersection periods scored: {results.intersection_size} "
              f"(excluded {results.excluded_counts['total_excluded']} periods: "
              f"null actuals={results.excluded_counts['missing_actual']}, "
              f"null operator={results.excluded_counts['missing_operator_forecast']})")
        print(f"Gate Scalar (GATE_VALUE):    {results.gate_value:.3f} gCO2/kWh")
        print(f"Operator GATE_VALUE (NESO):  {results.operator_gate_value:.3f} gCO2/kWh")
        print(f"Persistence GATE_VALUE:      {results.persistence_gate_value:.3f} gCO2/kWh")
        print(f"Seasonal-Naive GATE_VALUE:   {results.seasonal_naive_gate_value:.3f} gCO2/kWh")
        print(f"Overall Skill vs Operator:   {results.overall_skill_vs_operator:+.2%}")
        print(f"Skill vs Operator @ h=2:     {results.skill_h2:+.2%}")
        print(f"Skill vs Operator @ h=48:    {results.skill_h48:+.2%}")
        print(f"Skill vs Operator @ h=96:    {results.skill_h96:+.2%}")
        print("-" * 78)
        print("GATE VERDICT: NONE - threshold invalid (ADR-020); no pass or fail is claimed")
        print("=" * 78 + "\n")


def assert_refresh_window_eligibility(
    newest_archive_period: pd.Timestamp | str | datetime,
    window_last_target: pd.Timestamp | str | datetime,
    reconcile_window_days: int = 3,
) -> None:
    """Refuse unless window's last target is at least reconcile_window_days behind newest period.

    Names both dates and the shortfall in the error message.
    """
    newest_ts = pd.Timestamp(newest_archive_period)
    if newest_ts.tzinfo is None:
        newest_ts = newest_ts.tz_localize("UTC")
    else:
        newest_ts = newest_ts.tz_convert("UTC")

    target_ts = pd.Timestamp(window_last_target)
    if target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize("UTC")
    else:
        target_ts = target_ts.tz_convert("UTC")

    diff = newest_ts - target_ts
    required_diff = pd.Timedelta(days=reconcile_window_days)

    if diff < required_diff:
        shortfall = required_diff - diff
        shortfall_days = shortfall.total_seconds() / 86400.0
        newest_str = newest_ts.strftime("%Y-%m-%d %H:%M UTC")
        target_str = target_ts.strftime("%Y-%m-%d %H:%M UTC")
        raise ValueError(
            f"Cannot refresh evaluation fixture: window's last target ({target_str}) "
            f"is not at least {reconcile_window_days} days behind newest archive period "
            f"({newest_str}). Shortfall: {shortfall_days:.2f} days."
        )


def verify_fixture_integrity(
    parquet_path: Path | str = DEFAULT_EVAL_FIXTURE_PARQUET,
    sidecar_path: Path | str = DEFAULT_EVAL_FIXTURE_SIDECAR,
    pinned_sha256: str | None = PINNED_EVAL_FIXTURE_SHA256,
) -> dict[str, Any]:
    """Verify SHA-256 of evaluation fixture against sidecar metadata and pinned constant.

    Hard-fails on missing sidecar, missing parquet, or SHA-256 mismatch.
    """
    p_sidecar = Path(sidecar_path)
    if not p_sidecar.exists():
        raise FileNotFoundError(f"Sidecar fixture not found at {p_sidecar}")

    p_parquet = Path(parquet_path)
    if not p_parquet.exists():
        raise FileNotFoundError(f"Parquet fixture not found at {p_parquet}")

    with open(p_sidecar, encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)

    expected_sha = meta.get("sha256") or meta.get("parquet_sha256")
    if not expected_sha:
        raise ValueError(f"Sidecar at {p_sidecar} does not contain 'sha256' field")

    # Byte hash: compute SHA-256 over raw file bytes (not git's object id)
    with open(p_parquet, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()

    if pinned_sha256 is not None and actual_sha != pinned_sha256:
        msg = (
            f"Evaluation fixture SHA-256 mismatch against pinned constant.\n"
            f"Expected: {pinned_sha256}\n"
            f"Actual:   {actual_sha}"
        )
        print(msg, file=sys.stderr)
        raise ValueError(msg)

    if actual_sha != expected_sha:
        raise ValueError(
            f"SHA-256 mismatch for fixture parquet {p_parquet}. "
            f"Expected: {expected_sha}, Got: {actual_sha}"
        )

    return meta


def evaluate_from_fixture(
    parquet_path: Path | str = DEFAULT_EVAL_FIXTURE_PARQUET,
    sidecar_path: Path | str = DEFAULT_EVAL_FIXTURE_SIDECAR,
    verbose: bool = True,
    pinned_sha256: str | None = PINNED_EVAL_FIXTURE_SHA256,
) -> BacktestResults:
    """Read frozen evaluation fixture and compute all benchmark metrics without curated data.

    Verifies sha256 against sidecar and pin, computing GATE_VALUE and baselines directly.
    """
    _ = verify_fixture_integrity(
        parquet_path=parquet_path,
        sidecar_path=sidecar_path,
        pinned_sha256=pinned_sha256,
    )
    df = pd.read_parquet(parquet_path)

    missing_cols = [c for c in REQUIRED_FIXTURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Fixture {parquet_path} missing required columns: {missing_cols}")

    if len(df) != 1440:
        raise ValueError(f"Expected fixture to have 1440 rows, got {len(df)}")

    # Compute per-horizon metrics table
    records: list[dict[str, object]] = []
    for h in range(1, 97):
        h_df = df[df["horizon"] == h]
        n_periods = len(h_df)
        if n_periods == 0:
            records.append({
                "horizon": h,
                "count": 0,
                "model_mae": np.nan,
                "operator_mae": np.nan,
                "persistence_mae": np.nan,
                "seasonal_naive_mae": np.nan,
                "skill_vs_operator": np.nan,
            })
            continue

        actuals = h_df["actual"].to_numpy(dtype=float)
        m_preds = h_df["candidate_pred"].to_numpy(dtype=float)
        m_mae = float(np.mean(np.abs(m_preds - actuals)))

        o_preds = h_df["neso_forecast"].to_numpy(dtype=float)
        oper_mae = float(np.mean(np.abs(o_preds - actuals)))

        p_preds = h_df["persistence_pred"].to_numpy(dtype=float)
        pers_mae = float(np.mean(np.abs(p_preds - actuals)))

        s_preds = h_df["seasonal_naive_pred"].to_numpy(dtype=float)
        sn_mae = float(np.mean(np.abs(s_preds - actuals)))

        m_h = pd.DataFrame({
            "target_time": h_df["target_time"],
            "forecast": h_df["candidate_pred"],
            "actual": h_df["actual"],
        })
        o_h = pd.DataFrame({
            "target_time": h_df["target_time"],
            "forecast": h_df["neso_forecast"],
            "actual": h_df["actual"],
        })
        skill_oper = compute_skill_score(m_h, o_h)

        records.append({
            "horizon": h,
            "count": n_periods,
            "model_mae": m_mae,
            "operator_mae": oper_mae,
            "persistence_mae": pers_mae,
            "seasonal_naive_mae": sn_mae,
            "skill_vs_operator": skill_oper,
        })

    per_h = pd.DataFrame(records)

    gate_val = compute_gate_value(per_h, "model_mae")
    oper_gate_val = compute_gate_value(per_h, "operator_mae")
    pers_gate_val = compute_gate_value(per_h, "persistence_mae")
    sn_gate_val = compute_gate_value(per_h, "seasonal_naive_mae")

    m_all = pd.DataFrame({
        "issue_time": df["issue_time"],
        "horizon": df["horizon"],
        "target_time": df["target_time"],
        "forecast": df["candidate_pred"],
        "actual": df["actual"],
    })
    o_all = pd.DataFrame({
        "issue_time": df["issue_time"],
        "horizon": df["horizon"],
        "target_time": df["target_time"],
        "forecast": df["neso_forecast"],
        "actual": df["actual"],
    })
    overall_skill = compute_skill_score(m_all, o_all)

    def _skill_at_h(h_target: int) -> float:
        row = per_h[per_h["horizon"] == h_target]
        if row.empty or pd.isna(row["skill_vs_operator"].iloc[0]):
            return np.nan
        return float(row["skill_vs_operator"].iloc[0])

    unique_origins = [pd.Timestamp(o) for o in df["issue_time"].drop_duplicates().tolist()]

    results = BacktestResults(
        predictions=df,
        per_horizon_table=per_h,
        gate_value=gate_val,
        operator_gate_value=oper_gate_val,
        persistence_gate_value=pers_gate_val,
        seasonal_naive_gate_value=sn_gate_val,
        overall_skill_vs_operator=overall_skill,
        skill_h2=_skill_at_h(2),
        skill_h48=_skill_at_h(48),
        skill_h96=_skill_at_h(96),
        intersection_size=len(df),
        excluded_counts={
            "total_periods": len(df),
            "intersection_size": len(df),
            "total_excluded": 0,
            "missing_actual": 0,
            "missing_operator_forecast": 0,
            "missing_model_forecast": 0,
        },
        origins_evaluated=unique_origins,
    )

    if verbose:
        print("\n" + "=" * 94)
        print("PER-HORIZON PERFORMANCE (h=1..96)")
        print("=" * 94)
        print(
            f"{'Horizon':<8}{'Periods':<10}{'Model MAE':<14}{'Operator MAE':<16}"
            f"{'Persistence':<14}{'Seasonal-Naive':<16}{'Skill vs NESO':<14}"
        )
        print("-" * 94)
        for _, r in results.per_horizon_table.iterrows():
            h_int = int(r["horizon"])
            cnt = int(r["count"])
            m_str = f"{r['model_mae']:.2f}" if pd.notna(r["model_mae"]) else "N/A"
            o_str = f"{r['operator_mae']:.2f}" if pd.notna(r["operator_mae"]) else "N/A"
            p_str = f"{r['persistence_mae']:.2f}" if pd.notna(r["persistence_mae"]) else "N/A"
            s_str = f"{r['seasonal_naive_mae']:.2f}" if pd.notna(r["seasonal_naive_mae"]) else "N/A"
            sk_str = f"{r['skill_vs_operator']:+.2%}" if pd.notna(r["skill_vs_operator"]) else "N/A"
            print(f"{h_int:<8}{cnt:<10}{m_str:<14}{o_str:<16}{p_str:<14}{s_str:<16}{sk_str:<14}")
        print("=" * 94)

        evaluate_gate(results, verbose=True)

    return results


def refresh_eval_fixture(
    version: str = "v1",
    output_dir: Path | str = "data/eval",
    start_date: str = TEST_SPLIT_START_DATE,
    end_date: str = TEST_SPLIT_END_DATE,
    num_boost_round: int = 100,
    reconcile_window_days: int = 3,
    verbose: bool = True,
    force: bool = False,
) -> tuple[Path, Path, str, float]:
    """Rebuild evaluation frame, execute rolling-origin backtest, and atomically write fixture.

    Enforces that window's last target is >= reconcile_window_days behind newest period.
    Refuses to edit v1 if model or features change; versioning must increment.
    """
    out_p = Path(output_dir)
    parquet_path = out_p / f"frozen_split_{version}.parquet"
    sidecar_path = out_p / f"frozen_split_{version}.json"

    # Governance check on v1 immutability
    if version == "v1" and parquet_path.exists() and not force:
        # Check if model artifact changed
        current_model_hash = get_model_artifact_hash()
        with open(sidecar_path, encoding="utf-8") as f:
            v1_meta = json.load(f)
        recorded_model_hash = v1_meta.get("model_artifact_sha256")
        if recorded_model_hash != current_model_hash:
            raise RuntimeError(
                f"Model artifact hash changed ({current_model_hash} != {recorded_model_hash}). "
                "Per governance rules, frozen_split_v1 is immutable. "
                "Write a new version file (e.g. v2) instead of editing v1."
            )

    if verbose:
        print(f"Refreshing evaluation fixture {version}...")
        print("Loading feature sources (intensity, weather, mix)...")

    intensity_df, weather_df, mix_df = load_feature_sources()
    newest_period = pd.to_datetime(intensity_df["period_start_utc"], utc=True).max()

    origins = derive_test_split_origins(start_date, end_date)
    window_last_target = origins[-1] + pd.Timedelta(minutes=30 * 96)

    # Enforce window eligibility rule
    assert_refresh_window_eligibility(
        newest_archive_period=newest_period,
        window_last_target=window_last_target,
        reconcile_window_days=reconcile_window_days,
    )

    frame_start = "2024-11-08"
    schedule = default_issue_schedule(frame_start, end_date)
    if verbose:
        print(
            f"Building example frame across {len(schedule)} issues "
            f"({frame_start} to {end_date})..."
        )

    frame = build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=schedule,
        require_aux=True,
    )

    operator_forecasts = pd.Series(
        intensity_df["forecast"].values,
        index=pd.to_datetime(intensity_df["period_start_utc"], utc=True),
    )

    results = run_rolling_origin_backtest(
        frame=frame,
        origins=origins,
        operator_forecasts=operator_forecasts,
        num_boost_round=num_boost_round,
        verbose=verbose,
    )

    scored_df = results.predictions.copy()
    fixture_df = pd.DataFrame({
        "issue_time": pd.to_datetime(scored_df["issue_time"], utc=True),
        "horizon": scored_df["horizon"].astype(int),
        "target_time": pd.to_datetime(scored_df["target_time"], utc=True),
        "actual": scored_df["actual"].astype(float),
        "neso_forecast": scored_df["operator_forecast"].astype(float),
        "persistence_pred": scored_df["persistence_forecast"].astype(float),
        "seasonal_naive_pred": scored_df["seasonal_naive_forecast"].astype(float),
        "candidate_pred": scored_df["forecast"].astype(float),
    })
    fixture_df = fixture_df.sort_values(["issue_time", "horizon"]).reset_index(drop=True)

    if len(fixture_df) != 1440:
        raise ValueError(f"Expected 1440 scored rows, got {len(fixture_df)}")

    out_p.mkdir(parents=True, exist_ok=True)
    tmp_parquet = out_p / f"frozen_split_{version}.parquet.tmp"
    tmp_sidecar = out_p / f"frozen_split_{version}.json.tmp"

    fixture_df.to_parquet(tmp_parquet, index=False)
    with open(tmp_parquet, "rb") as f:
        parquet_sha = hashlib.sha256(f.read()).hexdigest()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "3617143"

    model_hash = get_model_artifact_hash()

    sidecar_data: dict[str, Any] = {
        "schema_version": version,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "archive_commit": commit,
        "archive_newest_period": newest_period.strftime("%Y-%m-%d %H:%M UTC"),
        "window_start": start_date,
        "window_end": end_date,
        "origin_window": (
            f"frozen fixture {version}, sha256 {parquet_sha}, "
            f"archive commit {commit}, window {start_date} to {end_date}, "
            f"{len(origins)} origins"
        ),
        "origins": [o.strftime("%Y-%m-%d %H:%M UTC") for o in origins],
        "row_count": len(fixture_df),
        "sha256": parquet_sha,
        "parquet_sha256": parquet_sha,
        "model_artifact_sha256": model_hash,
        "training_window": f"{schedule[0]} to {schedule[-1]}",
        "lgbm_params": DEFAULT_LGBM_PARAMS,
        "num_boost_round": num_boost_round,
        "versions": {
            "python": sys.version.split()[0],
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    with open(tmp_sidecar, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)
        f.write("\n")

    # Atomic rename
    tmp_parquet.replace(parquet_path)
    tmp_sidecar.replace(sidecar_path)

    if verbose:
        print(f"Wrote fixture parquet: {parquet_path}")
        print(f"Parquet SHA-256:       {parquet_sha}")
        print(f"Wrote sidecar JSON:    {sidecar_path}")
        print(f"GATE_VALUE:            {results.gate_value:.6f} gCO2/kWh")

    return parquet_path, sidecar_path, parquet_sha, results.gate_value


def check_sidecar_library_versions(
    sidecar_meta: dict[str, Any],
) -> list[str]:
    """Check runtime environment versions against sidecar metadata; return mismatches."""
    current_versions = {
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    mismatches: list[str] = []
    recorded = sidecar_meta.get("versions", {})
    for pkg, exp_ver in recorded.items():
        curr_ver = current_versions.get(pkg)
        if curr_ver is None:
            mismatches.append(f"{pkg}: recorded {exp_ver}, but package is not installed/detected")
        elif curr_ver != exp_ver:
            mismatches.append(f"{pkg}: recorded {exp_ver}, current environment has {curr_ver}")
    return mismatches


def reproduce_eval_fixture(
    parquet_path: Path | str = DEFAULT_EVAL_FIXTURE_PARQUET,
    sidecar_path: Path | str = DEFAULT_EVAL_FIXTURE_SIDECAR,
    max_pred_tolerance: float = 0.0,
    gate_value_tolerance: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Retrain 15 folds on frozen origins and verify candidate predictions against fixture.

    Slow, manual reproduction check verifying:
    1. Per-row candidate_pred max absolute difference <= max_pred_tolerance.
    2. GATE_VALUE delta <= gate_value_tolerance.
    3. Library versions in sidecar reported as a mismatch rather than ignored.
    """
    import time

    t0 = time.perf_counter()

    meta = verify_fixture_integrity(parquet_path=parquet_path, sidecar_path=sidecar_path)
    mismatches = check_sidecar_library_versions(meta)
    if mismatches:
        print("WARNING: Sidecar library version mismatch(es) detected:")
        for m in mismatches:
            print(f"  - {m}")
    elif verbose:
        print("Library versions match sidecar (python, lightgbm, pandas, numpy).")

    if verbose:
        print("Loading feature sources for model reproduction...")
    intensity_df, weather_df, mix_df = load_feature_sources()

    start_date = str(meta.get("window_start", TEST_SPLIT_START_DATE))
    end_date = str(meta.get("window_end", TEST_SPLIT_END_DATE))
    origins = derive_test_split_origins(start_date, end_date)
    schedule = default_issue_schedule("2024-11-08", end_date)

    if verbose:
        print(f"Building example frame across {len(schedule)} issues...")
    frame = build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=schedule,
        require_aux=True,
    )

    operator_forecasts = pd.Series(
        intensity_df["forecast"].values,
        index=pd.to_datetime(intensity_df["period_start_utc"], utc=True),
    )

    num_boost_round = int(meta.get("num_boost_round", 100))
    lgbm_params = meta.get("lgbm_params")

    if verbose:
        print(f"Retraining 15 folds with {num_boost_round} boost rounds...")
    results = run_rolling_origin_backtest(
        frame=frame,
        origins=origins,
        operator_forecasts=operator_forecasts,
        lgbm_params=lgbm_params,
        num_boost_round=num_boost_round,
        verbose=verbose,
    )

    fixture_res = evaluate_from_fixture(
        parquet_path=parquet_path,
        sidecar_path=sidecar_path,
        verbose=False,
    )
    fixture_df = pd.read_parquet(parquet_path)

    scored_df = results.predictions.copy()
    scored_df["issue_time"] = pd.to_datetime(scored_df["issue_time"], utc=True)
    fixture_df["issue_time"] = pd.to_datetime(fixture_df["issue_time"], utc=True)

    merged = pd.merge(
        fixture_df,
        scored_df,
        on=["issue_time", "horizon"],
        suffixes=("_fixture", "_retrained"),
    )
    if len(merged) != len(fixture_df):
        raise ValueError(f"Matched {len(merged)} rows against fixture of {len(fixture_df)} rows")

    pred_diff = np.abs(
        merged["candidate_pred"].to_numpy(dtype=float) - merged["forecast"].to_numpy(dtype=float)
    )
    max_pred_delta = float(np.max(pred_diff))
    gate_value_delta = float(abs(results.gate_value - fixture_res.gate_value))
    elapsed = time.perf_counter() - t0

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL REPRODUCTION RESULTS")
        print("=" * 60)
        print(f"Runtime:                      {elapsed:.2f}s")
        print(f"Origins evaluated:            {len(origins)}")
        print(f"Scored periods:               {len(merged)}")
        print(f"Fixture GATE_VALUE:           {fixture_res.gate_value:.6f}")
        print(f"Retrained GATE_VALUE:         {results.gate_value:.6f}")
        print(
            f"GATE_VALUE delta:             {gate_value_delta:.8f} "
            f"(tolerance: {gate_value_tolerance})"
        )
        print(
            f"Max per-row candidate delta:  {max_pred_delta:.8f} "
            f"(tolerance: {max_pred_tolerance})"
        )
        if mismatches:
            print(f"Version mismatches:           {len(mismatches)}")
        else:
            print("Version check:                CLEAN")
        print("=" * 60)

    if max_pred_delta > max_pred_tolerance:
        raise RuntimeError(
            f"Model reproduction failed: max per-row candidate_pred delta "
            f"({max_pred_delta:.8f}) exceeds tolerance ({max_pred_tolerance})"
        )

    if gate_value_delta > gate_value_tolerance and not math.isclose(
        gate_value_delta, gate_value_tolerance, abs_tol=1e-8
    ):
        raise RuntimeError(
            f"Model reproduction failed: GATE_VALUE delta "
            f"({gate_value_delta:.8f}) exceeds tolerance ({gate_value_tolerance})"
        )

    return {
        "max_pred_delta": max_pred_delta,
        "gate_value_delta": gate_value_delta,
        "retrained_gate_value": results.gate_value,
        "fixture_gate_value": fixture_res.gate_value,
        "elapsed_seconds": elapsed,
        "mismatches": mismatches,
    }


def emit_weather_ablation_table(
    per_h_df: pd.DataFrame,
    doc_path: Path | str,
) -> Path:
    """Write generated per-horizon table directly into markdown document between markers.

    Prose outside the markers remains unchanged.
    """
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found at {path}")

    content = path.read_text(encoding="utf-8")
    begin_marker = "<!-- BEGIN GENERATED TABLE -->"
    end_marker = "<!-- END GENERATED TABLE -->"

    if begin_marker not in content or end_marker not in content:
        raise ValueError(
            f"Document {path} must contain '{begin_marker}' and '{end_marker}' markers."
        )

    header = (
        "| Horizon | Control MAE ($W$) | Treatment MAE ($N$) | Delta ($N - W$) | "
        "Delta (%) | NESO MAE | Persistence MAE | Seasonal-Naive MAE |"
    )
    lines: list[str] = [
        header,
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, r in per_h_df.sort_values("horizon").iterrows():
        h = int(r["horizon"])
        w = float(r["control_mae"])
        n = float(r["treatment_mae"])
        d = float(r["delta"])
        pct = float(r["delta_pct"])
        o = float(r["operator_mae"])
        p = float(r["persistence_mae"])
        s = float(r["seasonal_naive_mae"])
        lines.append(
            f"| {h} | {w:.4f} | {n:.4f} | {d:+.4f} | {pct:+.2f}% | {o:.4f} | {p:.4f} | {s:.4f} |"
        )

    table_block = f"{begin_marker}\n" + "\n".join(lines) + f"\n{end_marker}"

    pattern = re.compile(
        re.escape(begin_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )
    new_content = pattern.sub(table_block, content, count=1)
    path.write_text(new_content, encoding="utf-8")
    return path


@dataclass(frozen=True)
class WeatherAblationResults:
    """Results container for pre-registered weather removal ablation (Phase P4-c)."""

    control_gate_value: float
    treatment_gate_value: float
    delta_gate_value: float
    delta_percent: float
    decision: str
    max_control_delta: float
    decision_threshold_percent: float
    per_horizon_table: pd.DataFrame
    operator_gate_value: float
    persistence_gate_value: float
    seasonal_naive_gate_value: float
    key_horizons: dict[int, dict[str, float]]
    elapsed_seconds: float
    control_long_gate_value: float = 0.0
    treatment_long_gate_value: float = 0.0
    delta_long_gate_value: float = 0.0
    delta_long_percent: float = 0.0


def run_weather_ablation(
    parquet_path: Path | str = DEFAULT_EVAL_FIXTURE_PARQUET,
    sidecar_path: Path | str = DEFAULT_EVAL_FIXTURE_SIDECAR,
    output_doc: Path | str | None = None,
    max_control_tolerance: float = 0.0,
    decision_threshold_percent: float = 2.0,
    verbose: bool = True,
) -> WeatherAblationResults:
    """Execute pre-registered weather removal ablation benchmark (Phase P4-c).

    Arms:
    1. Control arm (W): LightGBM model trained with weather features. Precondition:
       reproduces frozen fixture bitwise (max candidate delta <= max_control_tolerance).
    2. Treatment arm (N): LightGBM model trained without weather features.
       Method: Drops weather columns ('temperature', 'temperature_24h_mean', 'temperature_24h_min')
       before training and prediction, because build_example_frame requires auxiliary weather
       inputs and populates NaNs when absent.

    Decision Rule (fixed before run):
        DROP weather if N <= W * 1.02  (no-weather is equal, better, or worse by < 2.0%)
        KEEP weather if N >  W * 1.02
    """
    import time

    t0 = time.perf_counter()

    meta = verify_fixture_integrity(parquet_path=parquet_path, sidecar_path=sidecar_path)
    fixture_res = evaluate_from_fixture(
        parquet_path=parquet_path,
        sidecar_path=sidecar_path,
        verbose=False,
    )
    fixture_df = pd.read_parquet(parquet_path)
    fixture_df["issue_time"] = pd.to_datetime(fixture_df["issue_time"], utc=True)

    if verbose:
        print("Loading feature sources for weather removal ablation...")
    intensity_df, weather_df, mix_df = load_feature_sources()

    start_date = str(meta.get("window_start", TEST_SPLIT_START_DATE))
    end_date = str(meta.get("window_end", TEST_SPLIT_END_DATE))
    origins = derive_test_split_origins(start_date, end_date)
    schedule = default_issue_schedule("2024-11-08", end_date)

    if verbose:
        print(f"Building example frame across {len(schedule)} issues...")
    frame = build_example_frame(
        intensity_df=intensity_df,
        weather_df=weather_df,
        mix_df=mix_df,
        issue_times=schedule,
        require_aux=True,
    )

    control_feature_cols = list(FEATURE_COLUMNS)
    weather_cols = ["temperature", "temperature_24h_mean", "temperature_24h_min"]
    treatment_feature_cols = [c for c in FEATURE_COLUMNS if c not in weather_cols]

    num_boost_round = int(meta.get("num_boost_round", 100))
    lgbm_params = meta.get("lgbm_params")
    merged_params = dict(DEFAULT_LGBM_PARAMS)
    if lgbm_params:
        merged_params.update(lgbm_params)

    # 1. Train and evaluate Control Arm (with weather)
    if verbose:
        print(f"\n--- Training Control Arm (with weather, 15 folds, {num_boost_round} rounds) ---")
    control_preds_list: list[pd.DataFrame] = []
    for origin_T in origins:
        train_slice = frame[frame["target_time"] < origin_T]
        train_slice = train_slice[train_slice["y"].notna()]
        assert_fold_training_integrity(train_slice, origin_T)

        X_train = train_slice[control_feature_cols]
        y_train = train_slice["y"].to_numpy(dtype=float)
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        booster = lgb.train(
            params=merged_params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
        )

        test_fold = frame[frame["issue_time"] == origin_T].copy()
        X_test = test_fold[control_feature_cols]
        preds = np.asarray(booster.predict(X_test), dtype=float)
        test_fold["pred"] = preds
        control_preds_list.append(test_fold[["issue_time", "horizon", "pred"]])

    control_df = pd.concat(control_preds_list, ignore_index=True)
    control_df["issue_time"] = pd.to_datetime(control_df["issue_time"], utc=True)

    merged_ctrl = pd.merge(fixture_df, control_df, on=["issue_time", "horizon"])
    if len(merged_ctrl) != len(fixture_df):
        raise ValueError(
            f"Control merge row count mismatch: {len(merged_ctrl)} vs {len(fixture_df)}"
        )

    ctrl_diff = np.abs(
        merged_ctrl["candidate_pred"].to_numpy(dtype=float)
        - merged_ctrl["pred"].to_numpy(dtype=float)
    )
    max_ctrl_delta = float(np.max(ctrl_diff))

    if verbose:
        print(f"Control arm max per-row candidate delta vs fixture: {max_ctrl_delta:.8f}")

    if max_ctrl_delta > max_control_tolerance:
        raise RuntimeError(
            f"Control arm failed to reproduce fixture: max per-row candidate_pred delta "
            f"({max_ctrl_delta:.8f}) exceeds tolerance ({max_control_tolerance}). "
            "Harness is wrong: stopping ablation; treatment number cannot be presented as usable."
        )

    # 2. Train and evaluate Treatment Arm (without weather)
    if verbose:
        print(f"\n--- Training Treatment Arm (no weather, 15 folds, {num_boost_round} rounds) ---")
        print(
            "Method: weather columns dropped before training/prediction "
            "(frame builder requires aux inputs)."
        )
    treatment_preds_list: list[pd.DataFrame] = []
    for origin_T in origins:
        train_slice = frame[frame["target_time"] < origin_T]
        train_slice = train_slice[train_slice["y"].notna()]
        assert_fold_training_integrity(train_slice, origin_T)

        X_train = train_slice[treatment_feature_cols]
        y_train = train_slice["y"].to_numpy(dtype=float)
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        booster = lgb.train(
            params=merged_params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
        )

        test_fold = frame[frame["issue_time"] == origin_T].copy()
        X_test = test_fold[treatment_feature_cols]
        preds = np.asarray(booster.predict(X_test), dtype=float)
        test_fold["pred"] = preds
        treatment_preds_list.append(test_fold[["issue_time", "horizon", "pred"]])

    treatment_df = pd.concat(treatment_preds_list, ignore_index=True)
    treatment_df["issue_time"] = pd.to_datetime(treatment_df["issue_time"], utc=True)

    merged_treat = pd.merge(fixture_df, treatment_df, on=["issue_time", "horizon"])
    if len(merged_treat) != len(fixture_df):
        raise ValueError(
            f"Treatment merge row count mismatch: {len(merged_treat)} vs {len(fixture_df)}"
        )

    # 3. Score both arms against fixture actuals and baselines
    records: list[dict[str, Any]] = []
    for h in range(1, 97):
        c_sub = merged_ctrl[merged_ctrl["horizon"] == h]
        t_sub = merged_treat[merged_treat["horizon"] == h]
        act = c_sub["actual"].to_numpy(dtype=float)
        w_p = c_sub["pred"].to_numpy(dtype=float)
        n_p = t_sub["pred"].to_numpy(dtype=float)
        o_p = c_sub["neso_forecast"].to_numpy(dtype=float)
        p_p = c_sub["persistence_pred"].to_numpy(dtype=float)
        s_p = c_sub["seasonal_naive_pred"].to_numpy(dtype=float)

        w_mae = float(np.mean(np.abs(w_p - act)))
        n_mae = float(np.mean(np.abs(n_p - act)))
        o_mae = float(np.mean(np.abs(o_p - act)))
        pers_mae = float(np.mean(np.abs(p_p - act)))
        sn_mae = float(np.mean(np.abs(s_p - act)))
        d_h = n_mae - w_mae
        pct_h = (d_h / w_mae) * 100.0

        records.append({
            "horizon": h,
            "periods": len(c_sub),
            "control_mae": w_mae,
            "treatment_mae": n_mae,
            "delta": d_h,
            "delta_pct": pct_h,
            "operator_mae": o_mae,
            "persistence_mae": pers_mae,
            "seasonal_naive_mae": sn_mae,
        })

    per_h_df = pd.DataFrame(records)
    w_gv = float(per_h_df["control_mae"].sum()) / 96.0
    n_gv = float(per_h_df["treatment_mae"].sum()) / 96.0
    oper_gv = float(per_h_df["operator_mae"].sum()) / 96.0
    pers_gv = float(per_h_df["persistence_mae"].sum()) / 96.0
    sn_gv = float(per_h_df["seasonal_naive_mae"].sum()) / 96.0

    if not math.isclose(w_gv, fixture_res.gate_value, abs_tol=1e-6):
        raise RuntimeError(
            f"Control arm GATE_VALUE ({w_gv:.6f}) does not match fixture "
            f"GATE_VALUE ({fixture_res.gate_value:.6f}). Harness is wrong: "
            "stopping ablation; treatment number cannot be presented as usable."
        )

    delta_gv = n_gv - w_gv
    pct_delta = (delta_gv / w_gv) * 100.0

    decision_limit = w_gv * (1.0 + decision_threshold_percent / 100.0)
    decision = "DROP" if n_gv <= decision_limit else "KEEP"

    h48_df = per_h_df[per_h_df["horizon"] >= 48]
    w_long = float(h48_df["control_mae"].mean())
    n_long = float(h48_df["treatment_mae"].mean())
    delta_long = n_long - w_long
    pct_long = (delta_long / w_long) * 100.0

    if output_doc is not None:
        emit_weather_ablation_table(per_h_df, output_doc)
        if verbose:
            print(f"Generated per-horizon table written to: {output_doc}")

    elapsed = time.perf_counter() - t0

    key_horizons: dict[int, dict[str, float]] = {}
    for h in (1, 2, 48, 96):
        row = per_h_df[per_h_df["horizon"] == h].iloc[0]
        key_horizons[h] = {
            "control_mae": float(row["control_mae"]),
            "treatment_mae": float(row["treatment_mae"]),
            "delta": float(row["delta"]),
            "delta_pct": float(row["delta_pct"]),
            "operator_mae": float(row["operator_mae"]),
            "persistence_mae": float(row["persistence_mae"]),
            "seasonal_naive_mae": float(row["seasonal_naive_mae"]),
        }

    if verbose:
        print("\n" + "=" * 90)
        print("PER-HORIZON WEATHER ABLATION PERFORMANCE (h=1..96)")
        print("=" * 90)
        header = (
            f"{'Horizon':<8}{'Periods':<9}{'Control (W)':<14}{'Treatment (N)':<15}"
            f"{'Delta (N-W)':<14}{'Delta %':>8}   {'NESO MAE':<11}{'Persistence':<13}"
            f"{'Seasonal-Naive'}"
        )
        print(header)
        print("-" * 90)
        for _, r in per_h_df.iterrows():
            print(
                f"{int(r['horizon']):<8}{int(r['periods']):<9}{r['control_mae']:<14.4f}"
                f"{r['treatment_mae']:<15.4f}{r['delta']:<+14.4f}{r['delta_pct']:>+7.2f}%   "
                f"{r['operator_mae']:<11.4f}{r['persistence_mae']:<13.4f}{r['seasonal_naive_mae']:<13.4f}"
            )
        print("=" * 90)
        print("WEATHER REMOVAL ABLATION RESULTS (Phase P4-c)")
        print("=" * 90)
        print(f"Runtime:                      {elapsed:.2f}s")
        print(f"Control (W) GATE_VALUE:       {w_gv:.6f} gCO2/kWh (table mean: {w_gv:.6f})")
        print(f"Treatment (N) GATE_VALUE:     {n_gv:.6f} gCO2/kWh (table mean: {n_gv:.6f})")
        print(f"Delta (N - W):                {delta_gv:+.6f} gCO2/kWh ({pct_delta:+.2f}%)")
        print(
            f"h >= 48:                      W {w_long:.4f}  N {n_long:.4f}  "
            f"delta {delta_long:+.3f} ({pct_long:+.2f}%)"
        )
        print(
            f"Threshold rule:               DROP if N <= W * "
            f"{1.0 + decision_threshold_percent / 100.0:.2f} "
            f"(+{decision_threshold_percent:.1f}%), else KEEP"
        )
        print(f"N <= W * 1.02 limit:          {decision_limit:.6f} gCO2/kWh")
        print(
            f"Precondition (control delta): {max_ctrl_delta:.8f} "
            f"(tolerance: {max_control_tolerance})"
        )
        print(f"DECISION:                     {decision}")
        print("-" * 90)
        print("KEY HORIZONS DETAIL:")
        for h in (1, 2, 48, 96):
            kh = key_horizons[h]
            print(
                f"  h={h:02d} | W: {kh['control_mae']:.4f} | N: {kh['treatment_mae']:.4f} | "
                f"delta: {kh['delta']:+.4f} ({kh['delta_pct']:+.2f}%) | "
                f"NESO: {kh['operator_mae']:.4f} | Pers: {kh['persistence_mae']:.4f} | "
                f"SN: {kh['seasonal_naive_mae']:.4f}"
            )
        print("=" * 90)

    return WeatherAblationResults(
        control_gate_value=w_gv,
        treatment_gate_value=n_gv,
        delta_gate_value=delta_gv,
        delta_percent=pct_delta,
        decision=decision,
        max_control_delta=max_ctrl_delta,
        decision_threshold_percent=decision_threshold_percent,
        per_horizon_table=per_h_df,
        operator_gate_value=oper_gv,
        persistence_gate_value=pers_gv,
        seasonal_naive_gate_value=sn_gv,
        key_horizons=key_horizons,
        elapsed_seconds=elapsed,
        control_long_gate_value=w_long,
        treatment_long_gate_value=n_long,
        delta_long_gate_value=delta_long,
        delta_long_percent=pct_long,
    )



