"""Contract tests for gridcast Phase P4 forecast API service.

Asserts contracts for the five endpoints:
- GET /health: 200; model version, training date/window, data_freshness_at;
  non-200 when older than 48h
- GET /forecast?hours=48: exactly 96 points, timestamps strictly increasing at 30 min,
  lower <= value <= upper
- GET /metrics: parses as Prometheus exposition, request count, latency histogram,
  last measured MAE
- GET /model: matches MODEL_CARD (version, content sha256, training window, feature count,
  interval sha256)
- POST /reload: 401 without/wrong token, 503 unconfigured token, 200 with right token,
  500 when artifact missing
"""

from __future__ import annotations

import _socket
import hashlib
import json
import math
import socket
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from gridcast.features import (
    CURATED_DIR_DEFAULT,
    LONDON_TZ,
    build_example_frame,
    issue_time_for_date,
)
from gridcast.models import get_model_artifact_hash
from gridcast.serve import (
    FeatureSourceGapError,
    app,
    create_app,
    resolve_features_for_issue,
)

SAMPLE_ISSUE_DATE = "2026-11-01"


@pytest.fixture(autouse=True)
def allow_loopback_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow in-process loopback socket connections required by asyncio event loop on Windows.

    External network access remains strictly forbidden.
    """

    def selective_connect(self: socket.socket, *args: Any, **kwargs: Any) -> None:
        addr = args[0] if args else kwargs.get("address")
        if isinstance(addr, tuple) and addr[0] in ("127.0.0.1", "localhost", "::1"):
            _socket.socket.connect(self, addr)
            return
        raise RuntimeError("Network access is strictly forbidden in the test suite.")

    monkeypatch.setattr(socket.socket, "connect", selective_connect)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Synchronous test client using TestClient."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_healthy(client: TestClient) -> None:
    """GET /health: 200; model version, training date, data_freshness_at."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data.get("status") == "healthy"
    assert data.get("model_version") == "lgbm-v2"
    assert "2026-11-07" in str(data.get("training_date", "")) or "2026-11-07" in str(
        data.get("training_window", "")
    )
    assert "data_freshness_at" in data
    freshness_dt = datetime.fromisoformat(data["data_freshness_at"].replace("Z", "+00:00"))
    assert freshness_dt.tzinfo is not None

    # Check sha256 and feature count in /health
    model_sha = get_model_artifact_hash("data/models/lgbm_model.txt")
    assert (
        data.get("content_sha256") == model_sha
        or data.get("sha256") == model_sha
        or data.get("artifact_sha256") == model_sha
    )
    assert data.get("feature_count") == 26
    assert "interval_sha256" in data
    assert "served_issue_time" in data
    assert "is_current_issue" in data
    served = pd.Timestamp(data["served_issue_time"])
    assert served == issue_time_for_date(served.tz_convert(LONDON_TZ).date())
    assert data.get("is_current_issue") == (
        served == issue_time_for_date(datetime.now(LONDON_TZ).date())
    )


def test_health_endpoint_stays_200_when_data_older_than_48h(tmp_path: Path) -> None:
    """GET /health stays 200 as a liveness probe while the process is up and model is loaded."""
    # Create a mock curated directory with partition older than 48h
    stale_curated = tmp_path / "stale_curated"
    stale_curated.mkdir()
    old_ts = datetime.now(timezone.utc) - timedelta(hours=72)
    old_df = pd.DataFrame({
        "period_start_utc": [old_ts],
        "period_end_utc": [old_ts + timedelta(minutes=30)],
        "actual": [150.0],
    })
    old_df.to_parquet(stale_curated / "date=2020-01-01.parquet")

    stale_app = create_app(curated_dir=stale_curated)
    with TestClient(stale_app) as c:
        response = c.get("/health")
        assert response.status_code == 200, "Liveness /health must stay 200 while process is up"
        data = response.json()
        assert data.get("status") == "healthy"
        assert "data_freshness_at" in data
        assert "served_issue_time" in data
        assert "is_current_issue" in data


def test_forecast_hours_validation(client: TestClient) -> None:
    """hours must be validated: 1..48 integer; anything else is 422 with the accepted range."""
    # Out of range low
    r0 = client.get("/forecast?hours=0")
    assert r0.status_code == 422
    assert "1" in r0.text and "48" in r0.text

    # Out of range high
    r49 = client.get("/forecast?hours=49")
    assert r49.status_code == 422
    assert "1" in r49.text and "48" in r49.text

    # Negative
    r_neg = client.get("/forecast?hours=-5")
    assert r_neg.status_code == 422
    assert "1" in r_neg.text and "48" in r_neg.text

    # Non-integer
    r_str = client.get("/forecast?hours=invalid")
    assert r_str.status_code == 422


def assert_forecast_points_contract(data: dict[str, Any]) -> None:
    """Validate 48-hour 96-point forecast contract invariants."""
    points = data.get("points", [])
    assert len(points) == 96, f"Expected exactly 96 points, got {len(points)}"

    issue_ts = datetime.fromisoformat(str(data["issue_time"]).replace("Z", "+00:00"))
    values: list[float] = []
    required_fields = {"horizon", "timestamp", "target_time", "value", "lower", "upper"}

    timestamps: list[datetime] = []
    for i, pt in enumerate(points):
        # All six point fields present
        assert required_fields.issubset(pt.keys()), (
            f"Point {i} missing required fields: {required_fields - pt.keys()}"
        )

        # horizon == index + 1
        assert pt["horizon"] == i + 1, f"Point {i} horizon={pt['horizon']}, expected {i + 1}"

        # target_time == issue_time + 30 * (index + 1) minutes
        expected_target = issue_ts + timedelta(minutes=30 * (i + 1))
        target_ts = datetime.fromisoformat(str(pt["target_time"]).replace("Z", "+00:00"))
        assert target_ts == expected_target, (
            f"Point {i} target_time {target_ts} != expected {expected_target}"
        )

        # timestamp matches target_time
        assert pt["timestamp"] == pt["target_time"]
        timestamps.append(target_ts)

        # value, lower, upper finite
        val = float(pt["value"])
        lower = float(pt["lower"])
        upper = float(pt["upper"])
        assert math.isfinite(val), f"Point {i} value is not finite: {val}"
        assert math.isfinite(lower), f"Point {i} lower is not finite: {lower}"
        assert math.isfinite(upper), f"Point {i} upper is not finite: {upper}"

        # lower <= value <= upper
        assert lower <= val <= upper, (
            f"Point {i} bounds violated: lower={lower} <= value={val} <= upper={upper}"
        )

        values.append(val)

    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        assert delta == timedelta(minutes=30), (
            f"Step between point {i-1} and {i} is {delta}, expected exactly 30 min"
        )

    # Values must not be constant
    assert len(set(values)) > 1, (
        "Handler returned 96 constant points; expected realistic forecast curve"
    )


def test_forecast_48_hours_yields_exactly_96_points(client: TestClient) -> None:
    """GET /forecast?hours=48: exactly 96 points, 30 min step, lower <= value <= upper."""
    response = client.get(f"/forecast?hours=48&date={SAMPLE_ISSUE_DATE}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert isinstance(data, dict)
    assert "issue_time" in data
    assert "generated_at" in data
    assert "is_current_issue" in data
    assert data.get("hours") == 48

    assert_forecast_points_contract(data)


def test_forecast_contract_rejects_constant_points_mutation() -> None:
    """Mutation demonstration: a handler returning 96 constant points must fail the contract."""
    issue_time = "2026-11-01T00:00:00+00:00"
    issue_ts = datetime.fromisoformat(issue_time)
    constant_points = [
        {
            "horizon": i + 1,
            "timestamp": (issue_ts + timedelta(minutes=30 * (i + 1))).isoformat(),
            "target_time": (issue_ts + timedelta(minutes=30 * (i + 1))).isoformat(),
            "value": 150.0,
            "lower": 140.0,
            "upper": 160.0,
        }
        for i in range(96)
    ]
    mutation_payload = {
        "issue_time": issue_time,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "is_current_issue": True,
        "hours": 48,
        "points": constant_points,
    }

    with pytest.raises(AssertionError, match="Handler returned 96 constant points"):
        assert_forecast_points_contract(mutation_payload)


def test_metrics_endpoint_parses_prometheus_and_matches_scores_csv(
    client: TestClient,
) -> None:
    """GET /metrics: parses as Prometheus exposition; request count, latency, last MAE."""
    # Make a prior call to ensure request count > 0
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    # Must parse as Prometheus exposition
    families = list(text_string_to_metric_families(content))
    assert len(families) > 0, "Failed to parse any Prometheus metric families"

    family_names = {f.name for f in families}
    assert any("requests" in name for name in family_names), (
        f"Missing request counter in {family_names}"
    )
    assert any("duration" in name or "latency" in name for name in family_names), (
        f"Missing latency histogram in {family_names}"
    )
    assert any("mae" in name for name in family_names), f"Missing MAE metric in {family_names}"

    # Verify MAE values match scores.csv recomputation
    scores_path = Path("data/scoreboard/scores.csv")
    assert scores_path.exists(), "data/scoreboard/scores.csv not found"
    scores_df = pd.read_csv(scores_path)

    # Recompute macro mean (unweighted mean of per-horizon MAEs) directly in test
    pers_mask = scores_df["model_version"].str.startswith("persistence")
    pers_sub = scores_df.loc[pers_mask].dropna(subset=["abs_error", "horizon"]).copy()
    pers_sub["abs_error"] = pers_sub["abs_error"].astype(float)
    expected_pers_mae = float(pers_sub.groupby("horizon")["abs_error"].mean().mean())

    oper_mask = scores_df["neso_abs_error"].notna() & (scores_df["neso_abs_error"] != "")
    oper_sub = scores_df.loc[oper_mask].dropna(subset=["neso_abs_error", "horizon"]).copy()
    oper_sub["neso_abs_error"] = oper_sub["neso_abs_error"].astype(float)
    expected_oper_mae = float(oper_sub.groupby("horizon")["neso_abs_error"].mean().mean())

    cand_mask = scores_df["model_version"] == "lgbm-v2"
    has_candidate = cand_mask.any()

    # Find MAE metric family
    mae_family = next(f for f in families if "mae" in f.name)

    # Verify persistence sample labels and value
    pers_sample = next(
        s for s in mae_family.samples if s.labels.get("series") == "persistence"
    )
    assert pers_sample.labels.get("plane") == "live_scoreboard", (
        f"Expected plane='live_scoreboard', got {pers_sample.labels}"
    )
    assert pers_sample.labels.get("horizons") == "settled", (
        f"Expected horizons='settled' for persistence, got {pers_sample.labels}"
    )
    assert math.isclose(pers_sample.value, expected_pers_mae, rel_tol=1e-3), (
        f"Persistence MAE metric {pers_sample.value} != CSV {expected_pers_mae}"
    )

    # Verify operator sample labels and value
    oper_sample = next(s for s in mae_family.samples if s.labels.get("series") == "operator")
    assert oper_sample.labels.get("plane") == "live_scoreboard", (
        f"Expected plane='live_scoreboard', got {oper_sample.labels}"
    )
    assert oper_sample.labels.get("status") == "partial", (
        f"Expected status='partial' for operator, got {oper_sample.labels}"
    )
    assert oper_sample.labels.get("horizons") == "1-33", (
        f"Expected horizons='1-33' for operator, got {oper_sample.labels}"
    )
    assert math.isclose(oper_sample.value, expected_oper_mae, rel_tol=1e-3), (
        f"Operator MAE metric {oper_sample.value} != CSV {expected_oper_mae}"
    )

    # Verify companion gauge gridcast_last_measured_mae_samples
    samples_fam = next(f for f in families if f.name == "gridcast_last_measured_mae_samples")
    pers_cnt = next(s for s in samples_fam.samples if s.labels.get("series") == "persistence")
    oper_cnt = next(s for s in samples_fam.samples if s.labels.get("series") == "operator")
    assert pers_cnt.value == float(len(pers_sub))
    assert oper_cnt.value == float(len(oper_sub))

    # Verify gridcast_served_issue_age_seconds is exposed
    age_fam = next(f for f in families if f.name == "gridcast_served_issue_age_seconds")
    assert len(age_fam.samples) > 0
    assert age_fam.samples[0].value >= 0.0

    # Candidate has no settled rows in scores.csv -> must be exposed as absent with label
    if not has_candidate:
        cand_samples = [s for s in mae_family.samples if s.labels.get("series") == "candidate"]
        assert len(cand_samples) > 0, "Candidate series missing from MAE metric"
        cand_sample = cand_samples[0]
        assert cand_sample.labels.get("status") == "absent", (
            f"Candidate series should have status='absent', got {cand_sample.labels}"
        )
        assert cand_sample.labels.get("plane") == "live_scoreboard", (
            f"Candidate series should have plane='live_scoreboard', got {cand_sample.labels}"
        )
        assert cand_sample.labels.get("horizons") in ("settled", "partial"), (
            f"Candidate series should have valid horizons label, got {cand_sample.labels}"
        )
        assert math.isnan(cand_sample.value), (
            f"Candidate series without settled rows must be NaN, got {cand_sample.value}"
        )


def test_metrics_macro_mean_differs_from_row_mean(tmp_path: Path) -> None:
    """Test that /metrics reports the unweighted macro mean of per-horizon MAEs, not row mean."""
    # Write a synthetic scores.csv with a known difference:
    # horizon 1: 1 row with error 10.0
    # horizon 2: 3 rows with errors 30.0, 30.0, 30.0
    # Row mean = (10 + 30 + 30 + 30) / 4 = 25.0
    # Macro mean = (10.0 + 30.0) / 2 = 20.0
    scores_file = tmp_path / "scores.csv"
    rows = [
        "issue_time,horizon,target_time,model_version,forecast,actual,"
        "neso_forecast,abs_error,neso_abs_error,neso_source",
        "2026-11-02 00:00:00+00:00,1,2026-11-02 00:30:00+00:00,persistence-v0,150,140,,10,,none",
        "2026-11-02 00:00:00+00:00,2,2026-11-02 01:00:00+00:00,persistence-v0,150,120,,30,,none",
        "2026-11-03 00:00:00+00:00,2,2026-11-03 01:00:00+00:00,persistence-v0,150,120,,30,,none",
        "2026-11-04 00:00:00+00:00,2,2026-11-04 01:00:00+00:00,persistence-v0,150,120,,30,,none",
    ]
    scores_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    test_app = create_app(scores_path=scores_file)
    with TestClient(test_app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        families = list(text_string_to_metric_families(resp.text))
        mae_family = next(f for f in families if f.name == "gridcast_last_measured_mae")
        pers_sample = next(s for s in mae_family.samples if s.labels.get("series") == "persistence")

        # Assert metric is 20.0
        assert math.isclose(pers_sample.value, 20.0, rel_tol=1e-5), (
            f"Expected macro mean 20.0, got {pers_sample.value}"
        )
        # Assert metric is NOT 25.0, naming the bug it guards
        assert not math.isclose(pers_sample.value, 25.0, rel_tol=1e-5), (
            "Bug guarded: /metrics incorrectly reported row mean (25.0) instead of "
            "macro mean of per-horizon MAEs (20.0)."
        )

        # Companion gauge holds row count (4)
        samples_family = next(f for f in families if f.name == "gridcast_last_measured_mae_samples")
        pers_cnt = next(
            s for s in samples_family.samples if s.labels.get("series") == "persistence"
        )
        assert pers_cnt.value == 4.0


def test_model_endpoint_matches_model_card(client: TestClient) -> None:
    """GET /model: JSON matches MODEL_CARD metadata."""
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()

    # Check version
    assert data.get("version") == "lgbm-v2" or data.get("model_version") == "lgbm-v2"

    # Check sha256 equals data/models/lgbm_model.txt
    expected_sha = get_model_artifact_hash("data/models/lgbm_model.txt")
    actual_sha = (
        data.get("content_sha256") or data.get("sha256") or data.get("artifact_sha256")
    )
    assert actual_sha == expected_sha

    # Check training window
    assert "2024-11-08" in str(data.get("training_window", ""))
    assert "2026-11-07" in str(data.get("training_window", ""))

    # Check feature count
    assert data.get("feature_count") == 26

    # Check interval sha256
    interval_file = Path("data/models/interval_v2.json")
    assert interval_file.exists()
    expected_interval_sha = hashlib.sha256(
        interval_file.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    assert data.get("interval_sha256") == expected_interval_sha


def test_reload_endpoint_unconfigured_returns_503(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """POST /reload returns 503 when GRIDCAST_RELOAD_TOKEN is not configured."""
    monkeypatch.delenv("GRIDCAST_RELOAD_TOKEN", raising=False)
    response = client.post("/reload", headers={"Authorization": "Bearer some-token"})
    assert response.status_code == 503
    assert "not configured" in response.text.lower()


def test_reload_endpoint_authentication(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """POST /reload returns 401 without/with wrong token, 200 with right one."""
    token = "secret-reload-token-12345"
    monkeypatch.setenv("GRIDCAST_RELOAD_TOKEN", token)

    # Missing token
    r_no_token = client.post("/reload")
    assert r_no_token.status_code == 401

    # Wrong token
    r_wrong_token = client.post("/reload", headers={"Authorization": "Bearer wrong-token"})
    assert r_wrong_token.status_code == 401

    # Right token via Bearer header
    r_ok = client.post("/reload", headers={"Authorization": f"Bearer {token}"})
    assert r_ok.status_code == 200
    assert r_ok.json().get("status") in ("reloaded", "ok")


def test_reload_endpoint_missing_artifact_returns_500_and_keeps_booster(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """POST /reload: 500 when artifact missing; in-memory booster stays exactly as it was."""
    token = "test-token"
    monkeypatch.setenv("GRIDCAST_RELOAD_TOKEN", token)

    model_file = tmp_path / "model.txt"
    model_text = Path("data/models/lgbm_model.txt").read_text(encoding="utf-8")
    model_file.write_text(model_text, encoding="utf-8")
    interval_file = tmp_path / "interval.json"
    interval_text = Path("data/models/interval_v2.json").read_text(encoding="utf-8")
    interval_file.write_text(interval_text, encoding="utf-8")

    custom_app = create_app(model_path=model_file, interval_path=interval_file)
    with TestClient(custom_app) as c:
        # Check initial model
        r_init = c.get("/model")
        assert r_init.status_code == 200
        initial_sha = r_init.json().get("content_sha256") or r_init.json().get("sha256")

        # Delete artifact file to simulate missing artifact
        model_file.unlink()
        assert not model_file.exists()

        # Reload should fail with 500 without leaking absolute paths or filenames
        r_reload = c.post("/reload", headers={"Authorization": f"Bearer {token}"})
        assert r_reload.status_code == 500
        assert "model.txt" not in r_reload.text
        assert ":\\" not in r_reload.text and ":/" not in r_reload.text
        assert "lgbm_model" not in r_reload.text

        # Process did not crash and in-memory booster stays intact
        r_after = c.get("/model")
        assert r_after.status_code == 200
        after_sha = r_after.json().get("content_sha256") or r_after.json().get("sha256")
        assert after_sha == initial_sha, "In-memory booster was lost after failed reload"


def test_feature_path_row_for_row_equality() -> None:
    """The request path calls the same builder the backtest calls.

    Assert row-for-row equality for one sample issue time between the API's frame
    and build_example_frame.
    """
    sample_issue = SAMPLE_ISSUE_DATE
    api_frame = resolve_features_for_issue(sample_issue)

    int_df = pd.read_parquet(CURATED_DIR_DEFAULT)
    mix_df = pd.read_parquet("data/mix")
    sample_ts = issue_time_for_date(sample_issue)
    backtest_frame = build_example_frame(
        intensity_df=int_df,
        weather_df=None,
        mix_df=mix_df,
        issue_times=[sample_ts],
        include_weather=False,
        require_aux=False,
    )

    # Compare row-for-row equality on declared feature columns
    cols = list(api_frame.columns)
    assert len(api_frame) == 96
    assert len(backtest_frame) == 96

    for col in cols:
        assert col in backtest_frame.columns, f"Column {col} missing from backtest frame"
        v1 = api_frame[col].to_numpy()
        v2 = backtest_frame[col].to_numpy()
        np.testing.assert_array_equal(
            v1, v2, err_msg=f"Feature {col} differs between API and backtest"
        )


def test_source_gap_null_feature_refused_with_named_structured_error(
    tmp_path: Path,
) -> None:
    """No null reaches the model.

    If a required feature is null because of a source gap, refuse with a named structured
    error (FeatureSourceGapError / 503).
    """
    # Sabotage: create an intensity store where actuals around the issue instant are missing
    sabotaged_curated = tmp_path / "sabotaged_curated"
    sabotaged_curated.mkdir()

    # Read authentic partition but inject nulls into recent actuals
    real_p = sorted(Path(CURATED_DIR_DEFAULT).glob("date=*.parquet"))[-1]
    df = pd.read_parquet(real_p).copy()

    # Precondition assertion (brief section 5): assert injection took effect
    orig_null_count = int(df["actual"].isna().sum())
    df.loc[:, "actual"] = np.nan
    assert int(df["actual"].isna().sum()) > orig_null_count, "Injection precondition failed"

    df.to_parquet(sabotaged_curated / real_p.name)

    # Attempting to resolve features or forecast must raise FeatureSourceGapError or return 503
    with pytest.raises(FeatureSourceGapError) as excinfo:
        resolve_features_for_issue(
            issue_time=SAMPLE_ISSUE_DATE,
            curated_dir=sabotaged_curated,
        )

    err = excinfo.value
    assert "null" in str(err).lower() or "source gap" in str(err).lower()

    # Test via API endpoint
    sabotaged_app = create_app(curated_dir=sabotaged_curated)
    with TestClient(sabotaged_app, raise_server_exceptions=False) as c:
        resp = c.get(f"/forecast?hours=48&date={SAMPLE_ISSUE_DATE}")
        assert resp.status_code == 503
        assert "FeatureSourceGapError" in resp.text or "error" in resp.json()
        assert ":\\" not in resp.text and ":/" not in resp.text
        assert "sabotaged_curated" not in resp.text


def test_reload_endpoint_header_only_authentication(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """POST /reload requires token in header; query parameter and body are rejected."""
    token = "multi-auth-token"
    monkeypatch.setenv("GRIDCAST_RELOAD_TOKEN", token)

    # Via query parameter: must NOT authenticate (401)
    r_param = client.post(f"/reload?token={token}")
    assert r_param.status_code == 401

    # Via JSON body: must NOT authenticate (401)
    r_body = client.post("/reload", json={"token": token})
    assert r_body.status_code == 401

    # Via Authorization: Bearer header: authenticates successfully (200)
    r_bearer = client.post("/reload", headers={"Authorization": f"Bearer {token}"})
    assert r_bearer.status_code == 200

    # Via X-Reload-Token header: authenticates successfully (200)
    r_hdr = client.post("/reload", headers={"X-Reload-Token": token})
    assert r_hdr.status_code == 200


def test_get_data_freshness_error_paths(tmp_path: Path) -> None:
    """Test get_data_freshness_at error cases."""
    from gridcast.serve import _file_sha256, get_data_freshness_at

    # Non-existent dir
    with pytest.raises(FileNotFoundError):
        get_data_freshness_at(tmp_path / "does_not_exist")

    # Empty dir
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        get_data_freshness_at(empty_dir)

    # Empty partition without period_start_utc
    bad_df = pd.DataFrame({"dummy": [1]})
    bad_df.to_parquet(empty_dir / "date=2026-01-01.parquet")
    with pytest.raises(ValueError):
        get_data_freshness_at(empty_dir)

    # _file_sha256 on missing file
    with pytest.raises(FileNotFoundError):
        _file_sha256(tmp_path / "no_file.txt")


def test_run_serve_cli_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_serve_cli constructs app and calls uvicorn.run."""
    import uvicorn

    from gridcast.serve import run_serve_cli

    called: dict[str, Any] = {}

    def mock_run(server_app: object, host: str, port: int) -> None:
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr(uvicorn, "run", mock_run)
    res = run_serve_cli(host="127.0.0.1", port=9999)
    assert res == 0
    assert called.get("host") == "127.0.0.1"
    assert called.get("port") == 9999


def test_check_startup_servability_with_preloaded_df() -> None:
    """Test check_startup_servability with preloaded intensity_df."""
    from gridcast.serve import check_startup_servability

    int_df = pd.read_parquet(CURATED_DIR_DEFAULT)
    df = check_startup_servability(intensity_df=int_df)
    assert len(df) == 96
    assert len(df.columns) == 26


def test_sync_mae_metrics_with_candidate_data(tmp_path: Path) -> None:
    """Test metrics endpoint when candidate rows exist in scores.csv."""
    scores_file = tmp_path / "scores.csv"
    scores_data = pd.DataFrame({
        "model_version": ["persistence-v0", "lgbm-v2"],
        "abs_error": [25.0, 18.5],
        "neso_abs_error": [10.0, 10.0],
    })
    scores_data.to_csv(scores_file, index=False)

    test_app = create_app(scores_path=scores_file)
    with TestClient(test_app) as c:
        resp = c.get("/metrics")
        assert resp.status_code == 200
        families = list(text_string_to_metric_families(resp.text))
        mae_family = next(f for f in families if "mae" in f.name)
        cand_samples = [s for s in mae_family.samples if s.labels.get("series") == "candidate"]
        assert len(cand_samples) > 0
        assert cand_samples[0].labels.get("plane") == "live_scoreboard"
        assert math.isclose(cand_samples[0].value, 18.5)


def test_serve_additional_coverage(client: TestClient, tmp_path: Path) -> None:
    """Cover parameter branches in serve.py to achieve >85% coverage."""
    from datetime import date

    from gridcast.serve import (
        check_startup_servability,
        resolve_features_for_issue,
    )

    # 1. /forecast without explicit date/issue_time (default branch)
    r_def = client.get("/forecast?hours=2")
    assert r_def.status_code == 200
    data_def = r_def.json()
    assert isinstance(data_def, dict)
    assert len(data_def.get("points", [])) == 4
    assert data_def.get("hours") == 2

    # 2. resolve_features_for_issue with date, datetime, pd.Timestamp, and invalid type
    d_res = resolve_features_for_issue(date(2026, 11, 1))
    assert len(d_res) == 96

    dt_res = resolve_features_for_issue(datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc))
    assert len(dt_res) == 96

    ts_res = resolve_features_for_issue(pd.Timestamp("2026-11-01 00:00:00+00:00"))
    assert len(ts_res) == 96

    with pytest.raises(TypeError):
        resolve_features_for_issue(12345)

    # 3. check_startup_servability with issue_time=None and curated_dir
    c_res = check_startup_servability(issue_time=None)
    assert len(c_res) == 96

    # 4. reload with invalid interval file (missing quantiles)
    token = "coverage-token"
    bad_interval = tmp_path / "bad_interval.json"
    bad_interval.write_text(json.dumps({"no_quantiles": True}), encoding="utf-8")
    cov_app = create_app(interval_path=bad_interval)
    with TestClient(cov_app, raise_server_exceptions=False) as cov_c:
        import os
        os.environ["GRIDCAST_RELOAD_TOKEN"] = token
        r_bad = cov_c.post("/reload", headers={"Authorization": f"Bearer {token}"})
        assert r_bad.status_code == 500


def test_forecast_default_serves_fallback_with_named_gap(client: TestClient) -> None:
    """With a future 'today', the default serves the newest issue and labels the fallback."""
    from unittest.mock import patch

    class FutureDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> FutureDateTime:
            return cls(2026, 12, 1, 0, 0, 0, tzinfo=tz or timezone.utc)

    with patch("gridcast.serve.datetime", FutureDateTime):
        response = client.get("/forecast")
    assert response.status_code == 200
    data = response.json()

    assert "issue_time" in data
    assert data["is_current_issue"] is False
    assert "fallback_reason" in data
    assert "actual_recent_3h" in data["fallback_reason"]
    assert "2026-12-01" in data["fallback_reason"]
    assert "retrospective_horizons" in data
    assert data["retrospective_horizons"] > 0
    gen_ts = pd.Timestamp(data["generated_at"])
    expected_retro = sum(1 for pt in data["points"] if pd.Timestamp(pt["target_time"]) < gen_ts)
    assert data["retrospective_horizons"] == expected_retro
    assert data["hours"] == 48
    assert len(data["points"]) == 96

    # Verify points structure and bounds
    for pt in data["points"]:
        assert pt["lower"] <= pt["value"] <= pt["upper"]


def test_forecast_explicit_today_unbuildable_returns_503(client: TestClient) -> None:
    """GET /forecast?date=<future> returns 503 FeatureSourceGapError without falling back."""
    response = client.get("/forecast?date=2027-01-01")
    assert response.status_code == 503
    data = response.json()
    assert data.get("error") == "FeatureSourceGapError"
    assert "actual_recent_3h" in str(data.get("detail"))
    assert "2027-01-01" in str(data.get("detail"))


def test_served_issue_anchor_always_local_midnight(client: TestClient) -> None:
    """Assert served issue is issue_time_for_date(<date>) and its local hour is 0."""
    # Check default /forecast
    r_forecast = client.get("/forecast")
    assert r_forecast.status_code == 200
    data_f = r_forecast.json()
    served_f = pd.Timestamp(data_f["issue_time"])
    local_f = served_f.tz_convert(LONDON_TZ)
    assert served_f == issue_time_for_date(local_f.date())
    assert local_f.hour == 0
    assert local_f.minute == 0
    assert local_f.second == 0

    # Check /health served_issue_time
    r_health = client.get("/health")
    assert r_health.status_code == 200
    data_h = r_health.json()
    served_h = pd.Timestamp(data_h["served_issue_time"])
    local_h = served_h.tz_convert(LONDON_TZ)
    assert served_h == issue_time_for_date(local_h.date())
    assert local_h.hour == 0
    assert local_h.minute == 0
    assert local_h.second == 0

    # Check explicit issue date
    r_explicit = client.get("/forecast?date=2026-11-01")
    assert r_explicit.status_code == 200
    data_e = r_explicit.json()
    served_e = pd.Timestamp(data_e["issue_time"])
    local_e = served_e.tz_convert(LONDON_TZ)
    assert served_e == issue_time_for_date("2026-11-01")
    assert local_e.hour == 0
    assert local_e.minute == 0
    assert local_e.second == 0


def test_forecast_current_issue_has_zero_retrospective_horizons() -> None:
    """Assert retrospective_horizons is 0 when the issue is current and generated at anchor."""
    from unittest.mock import patch

    class AnchorDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> AnchorDateTime:
            # 2026-11-01 00:00:00 Europe/London is 2026-10-31 23:00:00 UTC (BST)
            if tz is not None:
                return cls(2026, 11, 1, 0, 0, 0, tzinfo=tz)
            return cls(2026, 10, 31, 23, 0, 0, tzinfo=timezone.utc)

    test_app = create_app()
    with (
        patch("gridcast.serve.datetime", AnchorDateTime),
        TestClient(test_app) as c,
    ):
        r = c.get("/forecast")
        assert r.status_code == 200
        data = r.json()
        assert data["is_current_issue"] is True
        assert "fallback_reason" not in data
        assert data["retrospective_horizons"] == 0
        assert "points" in data
        assert len(data["points"]) == 96
        gen_ts = pd.Timestamp(data["generated_at"])
        for pt in data["points"]:
            assert pd.Timestamp(pt["target_time"]) >= gen_ts


def test_forecast_enforces_midnight_london_anchor(client: TestClient) -> None:
    """Enforce that anchors must convert to 00:00:00 Europe/London, whatever offset is used.

    Tests the six canonical cases (four 422s, two 200s):
    1. 2026-11-01T12:00:00Z -> 422 (noon UTC, non-midnight)
    2. 2026-07-01T00:00:00Z -> 422 (UTC midnight is 01:00 London in BST)
    3. 2026-07-01T00:00:00+01:00 -> 200 (BST midnight: 00:00 Europe/London)
    4. 2026-10-25T00:00:00+01:00 -> 200 (BST midnight on transition day)
    5. 2026-10-25T00:00:00Z -> 422 (UTC midnight on transition day is 01:00 local)
    6. 2026-11-01T00:00:00+01:00 -> 422 (GMT winter date with BST offset: 23:00)
    """
    from zoneinfo import ZoneInfo

    # 1. 2026-11-01T12:00:00Z -> 422
    r1 = client.get("/forecast?issue_time=2026-11-01T12:00:00Z")
    assert r1.status_code == 422, f"Case 1 failed: {r1.status_code}"
    assert "Anchors must be 00:00 Europe/London." in r1.text

    # 2. 2026-07-01T00:00:00Z -> 422
    r2 = client.get("/forecast?issue_time=2026-07-01T00:00:00Z")
    assert r2.status_code == 422, f"Case 2 failed: {r2.status_code}"
    assert "Anchors must be 00:00 Europe/London." in r2.text

    # 3. 2026-07-01T00:00:00+01:00 -> 200 (BST midnight)
    r3 = client.get("/forecast?issue_time=2026-07-01T00:00:00+01:00")
    assert r3.status_code == 200, f"Case 3 failed: {r3.status_code}: {r3.text}"
    data3 = r3.json()
    assert pd.Timestamp(data3["issue_time"]).tz_convert(LONDON_TZ).hour == 0

    # 4. 2026-10-25T00:00:00+01:00 -> 200 (BST midnight on transition day)
    r4 = client.get("/forecast?issue_time=2026-10-25T00:00:00+01:00")
    assert r4.status_code == 200, f"Case 4 failed: {r4.status_code}: {r4.text}"
    data4 = r4.json()
    assert pd.Timestamp(data4["issue_time"]).tz_convert(LONDON_TZ).hour == 0

    # 5. 2026-10-25T00:00:00Z -> 422
    r5 = client.get("/forecast?issue_time=2026-10-25T00:00:00Z")
    assert r5.status_code == 422, f"Case 5 failed: {r5.status_code}"
    assert "Anchors must be 00:00 Europe/London." in r5.text

    # 6. 2026-11-01T00:00:00+01:00 -> 422
    r6 = client.get("/forecast?issue_time=2026-11-01T00:00:00+01:00")
    assert r6.status_code == 422, f"Case 6 failed: {r6.status_code}"
    assert "Anchors must be 00:00 Europe/London." in r6.text

    # Assert 00:00 local is never in ambiguous (fold) or nonexistent window on DST transitions
    # Autumn back transition (2026-10-25): clock change happens at 02:00 -> 01:00
    dt_autumn_0 = datetime(2026, 10, 25, 0, 0, 0, tzinfo=ZoneInfo("Europe/London"))
    assert dt_autumn_0.utcoffset() == timedelta(hours=1)
    assert getattr(dt_autumn_0, "fold", 0) == 0

    # Spring forward transition (2026-03-29): clock change happens at 01:00 -> 02:00
    dt_spring_0 = datetime(2026, 3, 29, 0, 0, 0, tzinfo=ZoneInfo("Europe/London"))
    assert dt_spring_0.utcoffset() == timedelta(hours=0)
    assert getattr(dt_spring_0, "fold", 0) == 0


def test_forecast_invalid_dates_return_422_without_leaking_input_or_paths(
    client: TestClient,
) -> None:
    """Invalid dates and issue times return 422, not 500, without echoing input or paths."""
    invalid_cases = [
        "2026-02-30",
        "not-a-date",
        "../../etc/passwd",
        "2026-11-01%00",
        "a" * 5000,
    ]

    for inv in invalid_cases:
        # Test on date param
        r_date = client.get(f"/forecast?date={inv}")
        assert r_date.status_code == 422, (
            f"Expected 422 for date={inv[:20]}, got {r_date.status_code}"
        )
        assert inv not in r_date.text, f"Response echoed input verbatim: {r_date.text}"
        assert ":\\" not in r_date.text and "/etc/" not in r_date.text
        assert "Internal Server Error" not in r_date.text

        # Test on issue_time param
        r_time = client.get(f"/forecast?issue_time={inv}")
        assert r_time.status_code == 422, (
            f"Expected 422 for issue_time={inv[:20]}, got {r_time.status_code}"
        )
        assert inv not in r_time.text, f"Response echoed input verbatim: {r_time.text}"
        assert ":\\" not in r_time.text and "/etc/" not in r_time.text
        assert "Internal Server Error" not in r_time.text


def test_metric_cardinality_is_bounded_under_path_fuzzing() -> None:
    """Distinct unknown paths map to 'unmatched' and keep Prometheus metrics strictly bounded."""
    app_instance = create_app()
    with TestClient(app_instance) as c:
        # Hit /forecast, /health, /metrics
        r_f = c.get(f"/forecast?hours=48&date={SAMPLE_ISSUE_DATE}")
        assert r_f.status_code == 200
        r_h = c.get("/health")
        assert r_h.status_code == 200
        r_m = c.get("/metrics")
        assert r_m.status_code == 200

        # Hit an unknown path to produce 'unmatched'
        r_u = c.get("/unknown_random_path_init")
        assert r_u.status_code == 404

        # Read /metrics and assert endpoint label set is EXACTLY the declared endpoints
        m_resp = c.get("/metrics")
        assert m_resp.status_code == 200
        families = list(text_string_to_metric_families(m_resp.text))
        dur_fam = next(f for f in families if f.name == "gridcast_request_duration_seconds")
        endpoint_labels = {s.labels.get("endpoint") for s in dur_fam.samples}
        expected_labels = {"/forecast", "/health", "/metrics", "unmatched"}
        assert endpoint_labels == expected_labels, (
            f"Expected endpoint label set exactly {expected_labels}, got {endpoint_labels}"
        )

        # After 200 unknown requests, the label set must NOT grow
        for i in range(200):
            c.get(f"/unknown_random_path_{i}")

        m_resp_200 = c.get("/metrics")
        assert m_resp_200.status_code == 200
        families_200 = list(text_string_to_metric_families(m_resp_200.text))
        dur_fam_200 = next(f for f in families_200 if f.name == "gridcast_request_duration_seconds")
        endpoint_labels_200 = {s.labels.get("endpoint") for s in dur_fam_200.samples}
        assert endpoint_labels_200 == {"/forecast", "/health", "/metrics", "unmatched"}, (
            f"Endpoint label set grew after 200 paths: {endpoint_labels_200}"
        )


def test_atomic_swap_primitive_directly() -> None:
    """Test the atomic snapshot swap primitive directly."""
    from gridcast.models import load_model_booster
    from gridcast.serve import ModelSnapshot, ServiceState

    booster1 = load_model_booster()
    snap1 = ModelSnapshot(
        booster=booster1,
        interval_data={"quantiles": {"1": {"q10": -5.0, "q90": 5.0}}},
        content_sha256="hash1",
        interval_sha256="ihash1",
        generation=1,
        state_version=1,
    )
    state = ServiceState(
        snapshot=snap1,
        model_path=Path("data/models/lgbm_model.txt"),
        interval_path=Path("data/models/interval_v2.json"),
        curated_dir=Path("data/curated"),
        scores_path=Path("data/scoreboard/scores.csv"),
    )

    # Initial state
    assert state.snapshot.generation == 1
    assert state.snapshot.state_version == 1
    assert state.snapshot.content_sha256 == "hash1"
    assert state.content_sha256 == "hash1"

    # Swap to snapshot 2
    snap2 = ModelSnapshot(
        booster=booster1,
        interval_data={"quantiles": {"1": {"q10": -10.0, "q90": 10.0}}},
        content_sha256="hash2",
        interval_sha256="ihash2",
        generation=2,
        state_version=2,
    )
    state.swap_snapshot(snap2)

    assert state.snapshot.generation == 2
    assert state.snapshot.state_version == 2
    assert state.snapshot.content_sha256 == "hash2"
    assert state.snapshot.interval_sha256 == "ihash2"


def test_concurrent_forecast_and_reload_atomic_consistency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run concurrent /forecast and /reload asserting every response is internally consistent."""
    import concurrent.futures

    token = "atomic-test-token"
    monkeypatch.setenv("GRIDCAST_RELOAD_TOKEN", token)

    model_text_1 = Path("data/models/lgbm_model.txt").read_text(encoding="utf-8")
    lines_2 = model_text_1.splitlines(keepends=True)
    lines_2.insert(1, "# generation 2 model artifact\n")
    model_text_2 = "".join(lines_2)

    interval_text_1 = json.dumps({
        "quantiles": {str(h): {"q10": -1.0, "q90": 1.0} for h in range(1, 97)}
    })
    interval_text_2 = json.dumps({
        "quantiles": {str(h): {"q10": -2.0, "q90": 2.0} for h in range(1, 97)}
    })

    m_file = tmp_path / "model.txt"
    m_file.write_text(model_text_1, encoding="utf-8")
    i_file = tmp_path / "interval.json"
    i_file.write_text(interval_text_1, encoding="utf-8")

    app_instance = create_app(model_path=m_file, interval_path=i_file)
    init_snap = app_instance.state.service.snapshot
    hash_v1 = (init_snap.content_sha256, init_snap.interval_sha256)

    # Write version 2 artifacts to disk
    m_file.write_text(model_text_2, encoding="utf-8")
    i_file.write_text(interval_text_2, encoding="utf-8")
    expected_v2_mhash = hashlib.sha256(model_text_2.encode("utf-8")).hexdigest()
    expected_v2_ihash = hashlib.sha256(interval_text_2.encode("utf-8")).hexdigest()
    hash_v2 = (expected_v2_mhash, expected_v2_ihash)

    # Precondition assertion: both artifacts differ between generations
    assert hash_v1[0] != hash_v2[0], "Precondition: model hash must differ"
    assert hash_v1[1] != hash_v2[1], "Precondition: interval hash must differ"

    valid_pairs = {hash_v1, hash_v2}

    results: list[tuple[str, str, int]] = []
    errors: list[Exception] = []

    with TestClient(app_instance) as c:

        def run_forecast() -> None:
            try:
                resp = c.get(f"/forecast?hours=48&date={SAMPLE_ISSUE_DATE}")
                if resp.status_code == 200:
                    data = resp.json()
                    m_h = str(data.get("content_sha256") or data.get("model_hash"))
                    i_h = str(data.get("interval_sha256") or data.get("interval_hash"))
                    gen = int(data.get("generation", 1))
                    results.append((m_h, i_h, gen))
            except Exception as e:
                errors.append(e)

        def run_reload() -> None:
            try:
                c.post("/reload", headers={"Authorization": f"Bearer {token}"})
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(30):
                futures.append(executor.submit(run_forecast))
                if i % 5 == 0:
                    futures.append(executor.submit(run_reload))
            concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"
    assert len(results) > 0, "No forecast results collected"

    for m_h, i_h, _gen in results:
        assert (m_h, i_h) in valid_pairs, (
            f"Response combined mismatched model hash ({m_h}) and interval hash ({i_h})"
        )


def test_health_stays_200_as_liveness_probe(client: TestClient) -> None:
    """GET /health stays 200 while process is up and model is loaded."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "training_date" in data
    assert "data_freshness_at" in data
    assert "served_issue_time" in data
    assert "is_current_issue" in data
    assert "content_sha256" in data
    assert "generation" in data


def test_ready_endpoint_returns_200_and_503_based_on_served_issue_age() -> None:
    """GET /ready returns 200 when served issue <= 27h old; 503 when older than 27 hours."""
    from unittest.mock import patch

    app_instance = create_app()

    # Derive the clock from the issue the app actually serves, not from a fixed calendar date.
    with TestClient(app_instance) as c0:
        served = pd.Timestamp(c0.get("/ready").json()["served_issue_time"])

    def make_clock(offset_hours: int) -> type[datetime]:
        moment = served.to_pydatetime() + timedelta(hours=offset_hours)

        class Clock(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> Clock:
                target = moment.astimezone(tz) if tz is not None else moment
                return cls(
                    target.year,
                    target.month,
                    target.day,
                    target.hour,
                    target.minute,
                    target.second,
                    target.microsecond,
                    tzinfo=target.tzinfo,
                )

        return Clock

    # 1. Fresh issue: 20h after the served issue -> 200
    with (
        patch("gridcast.serve.datetime", make_clock(20)),
        TestClient(app_instance) as c,
    ):
        resp = c.get("/ready")
        data = resp.json()
        assert resp.status_code == 200
        assert data["status"] == "ready"
        assert "served_issue_age_seconds" in data
        assert data["served_issue_age_seconds"] <= 27 * 3600
        assert "generation" in data

        # Verify Prometheus metric exposure for age
        m_resp = c.get("/metrics")
        assert m_resp.status_code == 200
        families = list(text_string_to_metric_families(m_resp.text))
        age_fam = next(f for f in families if f.name == "gridcast_served_issue_age_seconds")
        assert math.isclose(
            age_fam.samples[0].value, data["served_issue_age_seconds"], rel_tol=1e-3
        )

    # 2. Stale issue: 72h after the served issue, so today's issue cannot be built -> 503
    with (
        patch("gridcast.serve.datetime", make_clock(72)),
        TestClient(app_instance) as c_stale,
    ):
        resp_stale = c_stale.get("/ready")
        assert resp_stale.status_code == 503
        data_stale = resp_stale.json()
        assert data_stale["status"] == "not_ready"
        assert data_stale["served_issue_age_seconds"] > 27 * 3600
        assert "exceeding readiness threshold of 27 hours" in data_stale.get("detail", "")


def test_forecast_hours_point_counts_and_contains_future_points(client: TestClient) -> None:
    """Assert len(points) == hours * 2 for hours in 1, 24, 47 and contains_future_points is bool."""
    for h in [1, 24, 47, 48]:
        resp = client.get(f"/forecast?hours={h}&date={SAMPLE_ISSUE_DATE}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["hours"] == h
        assert len(data["points"]) == h * 2, (
            f"Expected {h * 2} points for hours={h}, got {len(data['points'])}"
        )
        assert "contains_future_points" in data
        assert isinstance(data["contains_future_points"], bool)



