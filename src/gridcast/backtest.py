"""Rolling-origin backtesting engine, evaluation metrics, and accuracy gate for gridcast.

Enforces five evaluation-integrity requirements:
1. Issue-anchored persistence verified by provenance guard.
2. Baselines and candidates scored on strictly identical period intersections.
3. Single scalar accuracy gate: GATE_VALUE = (1/96) * sum(MAE_h for h=1..96).
4. Training data ends strictly before the issue instant: max(train['target_time']) < T.
5. Skill scores computed on identical period sets at the point of division.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from gridcast.features import (
    default_issue_schedule,
    issue_time_for_date,
    select_training_arrays,
)
from gridcast.models import (
    assert_baseline_provenance,
    predict_lgbm,
    predict_persistence,
    predict_seasonal_naive,
    train_lgbm,
)

__all__: list[str] = [
    "BacktestResults",
    "EMBARGO_END_DATE",
    "EMBARGO_START_DATE",
    "FROZEN_TEST_ORIGINS",
    "TEST_SPLIT_END_DATE",
    "TEST_SPLIT_START_DATE",
    "TEST_SPLIT_STEP_DAYS",
    "assert_fold_training_integrity",
    "compute_gap_gate_threshold",
    "compute_gate_value",
    "compute_per_horizon_metrics",
    "compute_skill_score",
    "derive_test_split_origins",
    "evaluate_gap_gate",
    "evaluate_gate",
    "filter_evaluation_intersection",
    "run_rolling_origin_backtest",
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
