"""Tests for service startup servability verification and feature resolution."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from gridcast.models import load_model_booster
from gridcast.serve import check_startup_servability, resolve_features_for_issue

SAMPLE_ISSUE_DATE = "2026-11-01"


def test_resolved_feature_set_equals_booster_declared_set_exactly() -> None:
    """Verify that resolved features exactly match the booster artifact's declared features."""
    booster = load_model_booster()
    declared_features = booster.feature_name()
    assert len(declared_features) > 0, "Booster declares no features"

    resolved_df = resolve_features_for_issue(SAMPLE_ISSUE_DATE, booster=booster)

    assert isinstance(resolved_df, pd.DataFrame)
    assert len(resolved_df) == 96
    # Assert feature set equals booster declared set exactly
    assert set(resolved_df.columns) == set(declared_features)
    assert list(resolved_df.columns) == declared_features
    # Check that no feature column is entirely null
    for col in declared_features:
        assert not resolved_df[col].isna().all(), f"Feature '{col}' is entirely null"


def test_startup_fails_naming_missing_feature_when_weather_source_removed(
    tmp_path: Path,
) -> None:
    """Assert startup refuses to start naming 'temperature' when weather is removed."""
    booster = load_model_booster()
    assert "temperature" in booster.feature_name(), (
        "Precondition: booster must declare 'temperature'"
    )

    empty_weather_dir = tmp_path / "empty_weather"
    empty_weather_dir.mkdir(parents=True)
    # Assert injection took effect: empty directory has no partitions
    assert not list(empty_weather_dir.glob("date=*.parquet"))

    with pytest.raises(RuntimeError) as excinfo:
        check_startup_servability(
            issue_time=SAMPLE_ISSUE_DATE,
            booster=booster,
            weather_dir=empty_weather_dir,
            weather_df=None,
        )

    msg = str(excinfo.value)
    assert "temperature" in msg, f"Startup error message must name 'temperature', got: {msg}"


def test_startup_fails_naming_missing_feature_when_mix_source_removed(
    tmp_path: Path,
) -> None:
    """Assert startup refuses to start naming 'mix_share_wind' when mix is removed."""
    booster = load_model_booster()
    assert "mix_share_wind" in booster.feature_name(), (
        "Precondition: booster must declare 'mix_share_wind'"
    )

    empty_mix_dir = tmp_path / "empty_mix"
    empty_mix_dir.mkdir(parents=True)
    assert not list(empty_mix_dir.glob("date=*.parquet"))

    with pytest.raises(RuntimeError) as excinfo:
        check_startup_servability(
            issue_time=SAMPLE_ISSUE_DATE,
            booster=booster,
            mix_dir=empty_mix_dir,
            mix_df=None,
        )

    msg = str(excinfo.value)
    assert "mix_share_wind" in msg, f"Startup error message must name 'mix_share_wind', got: {msg}"


def test_startup_fails_naming_missing_feature_when_intensity_source_removed(
    tmp_path: Path,
) -> None:
    """Assert startup refuses to start naming actual features when intensity is removed."""
    booster = load_model_booster()
    assert "actual_recent_3h" in booster.feature_name(), (
        "Precondition: booster must declare 'actual_recent_3h'"
    )

    empty_curated_dir = tmp_path / "empty_curated"
    empty_curated_dir.mkdir(parents=True)
    assert not list(empty_curated_dir.glob("date=*.parquet"))

    with pytest.raises(RuntimeError) as excinfo:
        check_startup_servability(
            issue_time=SAMPLE_ISSUE_DATE,
            booster=booster,
            curated_dir=empty_curated_dir,
            intensity_df=None,
        )

    msg = str(excinfo.value)
    assert "actual_recent_3h" in msg, (
        f"Startup error message must name 'actual_recent_3h', got: {msg}"
    )


def test_feature_list_comes_from_booster_artifact_not_constant(tmp_path: Path) -> None:
    """Verify that feature requirements are inspected dynamically from booster artifact."""
    # Create a dummy booster that declares only a subset of features
    X = pd.DataFrame({
        "horizon": [1.0, 2.0],
        "hour_of_day": [0.0, 1.0],
    })
    y = np.array([100.0, 120.0])
    ds = lgb.Dataset(X, label=y)
    custom_booster = lgb.train(
        {"objective": "regression_l1", "verbose": -1, "min_data_in_bin": 1, "min_data_in_leaf": 1},
        ds,
        num_boost_round=1,
    )
    assert custom_booster.feature_name() == ["horizon", "hour_of_day"]

    # Even with empty weather and mix, resolving features for this custom booster succeeds
    # because it does NOT declare weather or mix features!
    empty_weather_dir = tmp_path / "empty_weather"
    empty_weather_dir.mkdir()
    empty_mix_dir = tmp_path / "empty_mix"
    empty_mix_dir.mkdir()

    resolved = resolve_features_for_issue(
        issue_time=SAMPLE_ISSUE_DATE,
        booster=custom_booster,
        weather_dir=empty_weather_dir,
        mix_dir=empty_mix_dir,
    )
    assert list(resolved.columns) == ["horizon", "hour_of_day"]
