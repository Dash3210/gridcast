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
    assert data.get("served_issue_time") == "2026-11-07T00:00:00+00:00"
    assert data.get("is_current_issue") is False


def test_health_endpoint_unhealthy_when_data_older_than_48h(tmp_path: Path) -> None:
    """GET /health returns non-200 when curated data freshness is older than 48 hours."""
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
        assert response.status_code != 200, "Expected non-200 for stale data (>48h)"
        assert response.status_code in (503, 500)
        data = response.json()
        assert data.get("status") == "unhealthy"
        assert "data_freshness_at" in data
        assert "served_issue_time" in data
        assert "is_current_issue" in data
        assert data.get("is_current_issue") is False


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

    points = data.get("points", [])

    # Assert exactly 96 points (do not derive from parameter)
    assert len(points) == 96, f"Expected exactly 96 points, got {len(points)}"

    # Timestamps strictly increasing at exactly 30 minutes
    timestamps = []
    for i, pt in enumerate(points):
        ts_str = pt.get("timestamp") or pt.get("target_time")
        assert ts_str is not None, f"Point {i} missing timestamp"
        ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        timestamps.append(ts)

        # Check bounds: lower <= value <= upper
        val = float(pt["value"])
        lower = float(pt["lower"])
        upper = float(pt["upper"])
        assert lower <= val <= upper, (
            f"Point {i} bounds violated: lower={lower} <= value={val} <= upper={upper}"
        )

    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        assert delta == timedelta(minutes=30), (
            f"Step between point {i-1} and {i} is {delta}, expected exactly 30 min"
        )


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

    # Recompute from CSV
    pers_mask = scores_df["model_version"].str.startswith("persistence")
    expected_pers_mae = float(scores_df.loc[pers_mask, "abs_error"].astype(float).mean())

    oper_series = scores_df["neso_abs_error"].dropna()
    oper_series = oper_series[oper_series != ""]
    expected_oper_mae = float(oper_series.astype(float).mean())

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
    assert oper_sample.labels.get("horizons") == "partial", (
        f"Expected horizons='partial' for operator, got {oper_sample.labels}"
    )
    assert math.isclose(oper_sample.value, expected_oper_mae, rel_tol=1e-3), (
        f"Operator MAE metric {oper_sample.value} != CSV {expected_oper_mae}"
    )

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

        # Reload should fail with 500
        r_reload = c.post("/reload", headers={"Authorization": f"Bearer {token}"})
        assert r_reload.status_code == 500

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
        assert resp.status_code in (503, 500)
        assert "FeatureSourceGapError" in resp.text or "error" in resp.json()


def test_reload_endpoint_alternative_token_methods(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """POST /reload supports X-Reload-Token header, query param, and JSON body."""
    token = "multi-auth-token"
    monkeypatch.setenv("GRIDCAST_RELOAD_TOKEN", token)

    # Via X-Reload-Token header
    r_hdr = client.post("/reload", headers={"X-Reload-Token": token})
    assert r_hdr.status_code == 200

    # Via query parameter
    r_param = client.post(f"/reload?token={token}")
    assert r_param.status_code == 200

    # Via JSON body
    r_body = client.post("/reload", json={"token": token})
    assert r_body.status_code == 200


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
    """GET /forecast on default date serves newest buildable issue and labels fallback."""
    response = client.get("/forecast")
    assert response.status_code == 200
    data = response.json()

    assert "issue_time" in data
    assert data["issue_time"] == "2026-11-07T00:00:00+00:00"
    assert "generated_at" in data
    assert data["is_current_issue"] is False
    assert "fallback_reason" in data
    assert "actual_recent_3h" in data["fallback_reason"]
    assert "2026-11-08" in data["fallback_reason"]
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
    """GET /forecast?date=2026-11-08 returns 503 FeatureSourceGapError without falling back."""
    response = client.get("/forecast?date=2026-11-08")
    assert response.status_code == 503
    data = response.json()
    assert data.get("error") == "FeatureSourceGapError"
    assert "actual_recent_3h" in str(data.get("detail"))
    assert "2026-11-08" in str(data.get("detail"))


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



