"""Forecast models and accuracy gate baselines for gridcast.

Provides:
- Issue-anchored persistence baseline with provenance tracking (guard — provenance).
- Seasonal-naive baseline anchored to available history at issue time.
- Deterministic native LightGBM regression model and training pipeline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.features import (
    ACTUAL_AVAILABILITY_LAG,
    select_model_inputs,
)

__all__: list[str] = [
    "DEFAULT_LGBM_PARAMS",
    "EXPECTED_INTERVAL_SHA256",
    "EXPECTED_LIGHTGBM_VERSION",
    "EXPECTED_MODEL_SHA256",
    "MODEL_VERSION",
    "assert_baseline_provenance",
    "get_model_artifact_hash",
    "get_model_version",
    "load_model_booster",
    "model_artifact_hash",
    "predict_lgbm",
    "predict_persistence",
    "predict_seasonal_naive",
    "train_lgbm",
]

DEFAULT_LGBM_PARAMS: Final[dict[str, object]] = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "seed": 42,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
    "num_threads": 1,
}


def assert_baseline_provenance(
    df: pd.DataFrame,
    raw_store: pd.DataFrame | None = None,
    curated_dir: Path | str = "data/curated",
) -> None:
    """Guard — provenance: verify persistence baseline against raw store and timestamp identity.

    Enforces that persistence predictions:
    1. Read exact timestamp: source_period_start_utc == issue_time - ACTUAL_AVAILABILITY_LAG.
    2. Read from raw store: row exists and actual is non-null at that instant.
    3. Match store value: forecast equals raw store's actual value at source_period_start_utc.
    """
    if "source_period_start_utc" not in df.columns:
        raise AssertionError("guard — provenance: missing 'source_period_start_utc' column")
    if "issue_time" not in df.columns:
        raise AssertionError("guard — provenance: missing 'issue_time' column")
    if "forecast" not in df.columns:
        raise AssertionError("guard — provenance: missing 'forecast' column")

    actual_source_ts = pd.to_datetime(df["source_period_start_utc"], utc=True)
    expected_source_ts = pd.to_datetime(df["issue_time"], utc=True) - ACTUAL_AVAILABILITY_LAG

    mismatches = actual_source_ts != expected_source_ts
    if mismatches.any():
        num_mismatches = int(mismatches.sum())
        first_bad = df.loc[mismatches].iloc[0]
        raise AssertionError(
            f"guard — provenance violation: {num_mismatches} row(s) read invalid source timestamp. "
            f"Expected {first_bad['issue_time'] - ACTUAL_AVAILABILITY_LAG}, "
            f"got {first_bad['source_period_start_utc']}"
        )

    # 2 & 3. Verify against the raw store directly
    if raw_store is None:
        store_path = Path(curated_dir)
        if not store_path.exists():
            raise AssertionError(f"guard — provenance: raw store path '{curated_dir}' not found")
        raw_store = pd.read_parquet(store_path)

    if "period_start_utc" not in raw_store.columns or "actual" not in raw_store.columns:
        raise AssertionError("guard — provenance: raw store missing 'period_start_utc' or 'actual'")

    # Lookup by source timestamp
    unique_issues = df.drop_duplicates(subset=["issue_time"])
    for _, row in unique_issues.iterrows():
        it = pd.Timestamp(row["issue_time"])
        it_utc = it.tz_localize("UTC") if it.tzinfo is None else it.tz_convert("UTC")
        src_ts = it_utc - ACTUAL_AVAILABILITY_LAG

        store_rows = raw_store[pd.to_datetime(raw_store["period_start_utc"], utc=True) == src_ts]
        if store_rows.empty:
            raise AssertionError(
                f"guard — provenance: store row does not exist at instant {src_ts}"
            )
        store_actual = store_rows["actual"].iloc[0]
        if pd.isna(store_actual):
            raise AssertionError(
                f"guard — provenance: store row at {src_ts} has null actual"
            )

        expected_val = float(store_actual)
        matching_preds = (
            df[pd.to_datetime(df["issue_time"], utc=True) == it_utc]["forecast"]
            .to_numpy(dtype=float)
        )
        val_mismatches = ~np.isclose(matching_preds, expected_val, rtol=1e-5, atol=1e-7)
        if val_mismatches.any():
            bad_pred = matching_preds[val_mismatches][0]
            raise AssertionError(
                "guard — provenance: prediction value mismatch. "
                f"Prediction {bad_pred} != raw store actual {expected_val} "
                f"at source instant {src_ts}"
            )


def predict_persistence(
    frame: pd.DataFrame,
    raw_store: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Issue-anchored persistence baseline (requirement 2.1).

    Forecasts the carbon intensity actual available at issue_time - ACTUAL_AVAILABILITY_LAG,
    held flat across all 96 horizons.

    Args:
        frame: DataFrame containing 'issue_time', 'horizon', 'target_time', and
               optionally 'actual_recent_3h'.
        raw_store: Optional curated store DataFrame to verify baseline provenance.

    Returns:
        DataFrame with issue_time, horizon, target_time, forecast, and source_period_start_utc.
    """
    issue_times = pd.to_datetime(frame["issue_time"], utc=True)
    source_period_starts = issue_times - ACTUAL_AVAILABILITY_LAG

    if "actual_recent_3h" in frame.columns:
        values = frame["actual_recent_3h"].to_numpy(dtype=float)
    else:
        raise ValueError("frame must contain 'actual_recent_3h' to construct persistence forecast")

    res = pd.DataFrame({
        "issue_time": issue_times,
        "horizon": frame["horizon"].to_numpy(dtype=int),
        "target_time": pd.to_datetime(frame["target_time"], utc=True),
        "forecast": values,
        "source_period_start_utc": source_period_starts,
    })

    assert_baseline_provenance(res, raw_store=raw_store)
    return res


def predict_seasonal_naive(
    frame: pd.DataFrame,
    history_actuals: pd.Series | dict[pd.Timestamp, float],
) -> pd.DataFrame:
    """Seasonal-naive baseline anchored to history available at issue (requirement 3).

    For each issue_time and horizon h (with target_time = issue_time + h * 30min), finds
    the smallest k >= 1 such that:
    source_ts = target_time - k * 24h <= issue_time - ACTUAL_AVAILABILITY_LAG,
    matching the target half-hour's diurnal period from verified historical actuals.

    Args:
        frame: DataFrame containing 'issue_time', 'horizon', 'target_time'.
        history_actuals: Mapping or Series of historical actuals indexed by UTC period_start_utc.

    Returns:
        DataFrame with issue_time, horizon, target_time, forecast, and source_period_start_utc.
    """
    issue_times = pd.to_datetime(frame["issue_time"], utc=True)
    target_times = pd.to_datetime(frame["target_time"], utc=True)
    horizons = frame["horizon"].to_numpy(dtype=int)

    forecasts: list[float] = []
    source_ts_list: list[pd.Timestamp] = []

    for it, tt in zip(issue_times, target_times, strict=True):
        cutoff = it - ACTUAL_AVAILABILITY_LAG
        # Step back in 24h increments until source timestamp is <= cutoff
        source_ts = tt - pd.Timedelta(hours=24)
        while source_ts > cutoff:
            source_ts -= pd.Timedelta(hours=24)

        if isinstance(history_actuals, pd.Series):
            val = history_actuals.get(source_ts, np.nan)
        else:
            val = history_actuals.get(source_ts, np.nan)

        forecasts.append(float(val) if val is not None else np.nan)
        source_ts_list.append(source_ts)

    res = pd.DataFrame({
        "issue_time": issue_times,
        "horizon": horizons,
        "target_time": target_times,
        "forecast": np.array(forecasts, dtype=float),
        "source_period_start_utc": pd.to_datetime(source_ts_list, utc=True),
    })

    # Provenance invariant: every source timestamp must precede or equal issue cutoff
    max_allowed = issue_times - ACTUAL_AVAILABILITY_LAG
    violating = res["source_period_start_utc"] > max_allowed
    if violating.any():
        raise AssertionError("Seasonal-naive read source timestamp after issue availability cutoff")

    return res


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    params: dict[str, object] | None = None,
    num_boost_round: int = 100,
    feature_columns: Sequence[str] | None = None,
) -> lgb.Booster:
    """Train native LightGBM regressor on feature matrix X and target y.

    Args:
        X_train: Training features DataFrame containing FEATURE_COLUMNS.
        y_train: Non-null target series of carbon intensity actuals.
        params: LightGBM training hyperparameters (defaults to DEFAULT_LGBM_PARAMS).
        num_boost_round: Number of boosting iterations.
        feature_columns: Optional explicit feature columns sequence (defaults to FEATURE_COLUMNS).

    Returns:
        Trained LightGBM Booster artifact.
    """
    X_mat = select_model_inputs(X_train, feature_columns=feature_columns)
    y_vec = np.asarray(y_train, dtype=float)

    if np.isnan(y_vec).any():
        raise ValueError("train_lgbm: y_train must not contain null values")

    merged_params = dict(DEFAULT_LGBM_PARAMS)
    if params is not None:
        merged_params.update(params)

    dtrain = lgb.Dataset(X_mat, label=y_vec, free_raw_data=False)
    booster = lgb.train(
        params=merged_params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
    )
    return booster


def predict_lgbm(
    booster: lgb.Booster,
    X_test: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Generate predictions from native LightGBM booster.

    Args:
        booster: Trained LightGBM Booster artifact.
        X_test: Test features DataFrame containing FEATURE_COLUMNS.
        feature_columns: Optional explicit feature columns sequence (defaults to FEATURE_COLUMNS).

    Returns:
        Array of predicted carbon intensity values.
    """
    X_mat = select_model_inputs(X_test, feature_columns=feature_columns)
    preds: np.ndarray = np.asarray(booster.predict(X_mat), dtype=float)
    return preds


def model_artifact_hash(booster: lgb.Booster) -> str:
    """Compute SHA-256 hash of LightGBM model artifact text representation.

    Guarantees reproducible verification that the same training frame produces the same
    model artifact hash.
    """
    model_str = booster.model_to_string()
    return hashlib.sha256(model_str.encode("utf-8")).hexdigest()


MODEL_VERSION: Final[str] = "lgbm-v2"
EXPECTED_MODEL_SHA256: Final[str] = (
    "1cd13471d793735748f81d17fb195e34f6f784e8f642ac60a16f016280c45310"
)
EXPECTED_INTERVAL_SHA256: Final[str] = (
    "ab94aa7135bc797e4c81993a65c96702088dc02d845d0c4262aaf133e46b8965"
)
EXPECTED_LIGHTGBM_VERSION: Final[str] = "4.7.0"


def get_model_version() -> str:
    """Return model version identifier."""
    return MODEL_VERSION


def get_model_artifact_hash(artifact_path: Path | str = "data/models/lgbm_model.txt") -> str:
    """Compute SHA-256 hash of the committed LightGBM model artifact text representation.

    Reads model text using universal newlines to guarantee identical SHA-256 hashes
    across Windows and POSIX platforms regardless of checkout line endings.
    """
    p = Path(artifact_path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found at {p}")
    text = p.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_model_booster(artifact_path: Path | str = "data/models/lgbm_model.txt") -> lgb.Booster:
    """Load LightGBM Booster from artifact file, CRLF-safe across platforms."""
    p = Path(artifact_path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found at {p}")
    text = p.read_text(encoding="utf-8")
    return lgb.Booster(model_str=text)
