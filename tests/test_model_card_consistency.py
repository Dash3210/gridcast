"""Consistency test asserting published model card metrics match the frozen split fixture."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path("tests/fixtures/frozen_split_metrics.json")
MODEL_CARD_PATH = Path("MODEL_CARD.md")
README_PATH = Path("README.md")
DECISIONS_PATH = Path("docs/decisions.md")

REQUIRED_WINDOW_TEXT = "frozen split, 2026-09-05 to 2026-11-01, 15 origins, 2026-11-02 run"


def load_fixture() -> dict[str, Any]:
    """Load the committed frozen split metrics fixture."""
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"Fixture not found at {FIXTURE_PATH}")
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def parse_model_card_gate_scalars(card_text: str) -> dict[str, float]:
    """Parse candidate and baseline gate values from the MODEL_CARD scalar table."""
    scalars: dict[str, float] = {}
    lines = card_text.splitlines()
    in_table = False
    for line in lines:
        stripped = line.strip()
        if "| candidate | GATE_VALUE" in stripped:
            in_table = True
            continue
        if in_table:
            if not stripped.startswith("|"):
                break
            if "|---" in stripped:
                continue
            cols = [c.strip() for c in stripped.split("|")[1:-1]]
            if len(cols) >= 2:
                name, val_str = cols[0], cols[1]
                val = float(val_str)
                if "LightGBM" in name:
                    scalars["model"] = val
                elif "persistence" in name:
                    scalars["persistence"] = val
                elif "seasonal-naive" in name:
                    scalars["seasonal_naive"] = val
                elif "operator" in name or "NESO" in name:
                    scalars["operator"] = val
    return scalars


def parse_model_card_key_horizons(card_text: str) -> dict[int, dict[str, float]]:
    """Parse horizon MAE rows from the MODEL_CARD Key Horizons Detail table."""
    horizons: dict[int, dict[str, float]] = {}
    lines = card_text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cols = [c.strip() for c in stripped.split("|")[1:-1]]
        if len(cols) >= 7 and cols[0].startswith("h ="):
            h_match = re.search(r"h\s*=\s*(\d+)", cols[0])
            if h_match:
                h_val = int(h_match.group(1))
                horizons[h_val] = {
                    "model": float(cols[2]),
                    "operator": float(cols[3]),
                    "persistence": float(cols[4]),
                    "seasonal_naive": float(cols[5]),
                    "periods": float(cols[6]),
                }
    return horizons


def parse_readme_scalars(readme_text: str) -> list[dict[str, float]]:
    """Extract quoted GATE_VALUE blocks from README.md."""
    # Pattern matching: GATE_VALUE = <m> ... persistence (<p>) ...
    # seasonal-naive (<sn>) ... NESO operator (<o>)
    pattern = re.compile(
        r"GATE_VALUE\s*=\s*`?([0-9]+\.[0-9]+)`?\s*gCO2/kWh.*?"
        r"persistence\s*\(?`?([0-9]+\.[0-9]+)`?\s*(?:gCO2/kWh)?\)?.*?"
        r"seasonal-naive\s*\(?`?([0-9]+\.[0-9]+)`?\s*(?:gCO2/kWh)?\)?.*?"
        r"NESO operator\s*\(?`?([0-9]+\.[0-9]+)`?\s*(?:gCO2/kWh)?\)?",
        re.DOTALL,
    )
    matches = []
    for m in pattern.finditer(readme_text):
        matches.append({
            "model": float(m.group(1)),
            "persistence": float(m.group(2)),
            "seasonal_naive": float(m.group(3)),
            "operator": float(m.group(4)),
        })
    return matches


def test_model_card_gate_scalars_match_fixture() -> None:
    """Verify that MODEL_CARD scalar GATE_VALUE numbers match fixture at 3 decimals."""
    fixture = load_fixture()
    card_text = MODEL_CARD_PATH.read_text(encoding="utf-8")
    parsed = parse_model_card_gate_scalars(card_text)

    expected = fixture["scalars"]
    assert "model" in parsed, "Model GATE_VALUE missing from MODEL_CARD table"
    assert "operator" in parsed, "Operator GATE_VALUE missing from MODEL_CARD table"
    assert "persistence" in parsed, "Persistence GATE_VALUE missing from MODEL_CARD table"
    assert "seasonal_naive" in parsed, "Seasonal-naive GATE_VALUE missing from MODEL_CARD table"

    assert f"{parsed['model']:.3f}" == f"{expected['model']:.3f}"
    assert f"{parsed['operator']:.3f}" == f"{expected['operator']:.3f}"
    assert f"{parsed['persistence']:.3f}" == f"{expected['persistence']:.3f}"
    assert f"{parsed['seasonal_naive']:.3f}" == f"{expected['seasonal_naive']:.3f}"


def test_model_card_key_horizons_match_fixture() -> None:
    """Verify that MODEL_CARD key horizon rows match fixture at 2 decimals."""
    fixture = load_fixture()
    card_text = MODEL_CARD_PATH.read_text(encoding="utf-8")
    parsed_horizons = parse_model_card_key_horizons(card_text)

    expected_horizons = fixture["key_horizons"]
    for h_str, exp in expected_horizons.items():
        h = int(h_str)
        assert h in parsed_horizons, f"Horizon h={h} row missing from MODEL_CARD"
        actual = parsed_horizons[h]
        assert f"{actual['model']:.2f}" == f"{exp['model']:.2f}", f"h={h} model mismatch"
        assert f"{actual['operator']:.2f}" == f"{exp['operator']:.2f}", f"h={h} operator mismatch"
        msg_pers = f"h={h} persistence mismatch"
        assert f"{actual['persistence']:.2f}" == f"{exp['persistence']:.2f}", msg_pers
        msg_sn = f"h={h} seasonal-naive mismatch"
        assert f"{actual['seasonal_naive']:.2f}" == f"{exp['seasonal_naive']:.2f}", msg_sn
        assert int(actual["periods"]) == int(exp["periods"]), f"h={h} periods count mismatch"


def test_readme_scalars_match_fixture() -> None:
    """Verify that any scalar figures quoted in README match fixture at 3 decimals."""
    fixture = load_fixture()
    readme_text = README_PATH.read_text(encoding="utf-8")
    parsed_entries = parse_readme_scalars(readme_text)

    assert len(parsed_entries) >= 2, "Expected at least 2 GATE_VALUE references in README.md"
    expected = fixture["scalars"]

    for entry in parsed_entries:
        assert f"{entry['model']:.3f}" == f"{expected['model']:.3f}"
        assert f"{entry['operator']:.3f}" == f"{expected['operator']:.3f}"
        assert f"{entry['persistence']:.3f}" == f"{expected['persistence']:.3f}"
        assert f"{entry['seasonal_naive']:.3f}" == f"{expected['seasonal_naive']:.3f}"


def test_published_figures_carry_window() -> None:
    """Assert that every published number carries its origin window description."""
    fixture = load_fixture()
    expected_window = fixture.get("origin_window", REQUIRED_WINDOW_TEXT)

    for doc_path in [MODEL_CARD_PATH, README_PATH, DECISIONS_PATH]:
        text = doc_path.read_text(encoding="utf-8")
        assert expected_window in text, (
            f"{doc_path} must carry the origin window specification: '{expected_window}'"
        )


def test_no_stale_metrics() -> None:
    """Assert that deprecated/stale numbers do not appear in any documentation."""
    stale_values = ["39.032", "31.992", "39.605", "10.611"]
    for doc_path in [MODEL_CARD_PATH, README_PATH, DECISIONS_PATH]:
        text = doc_path.read_text(encoding="utf-8")
        for stale in stale_values:
            assert stale not in text, f"Found stale value {stale} in {doc_path}"


def test_schema_label_not_day_ahead() -> None:
    """Verify that README column contract does not label operator forecast as day-ahead."""
    readme_text = README_PATH.read_text(encoding="utf-8")
    for line in readme_text.splitlines():
        if "`forecast`" in line and "(`Int32`" in line:
            assert "day-ahead" not in line.lower(), (
                f"Curated contract forecast line must not say 'day-ahead': {line}"
            )
            assert "vintage unverified" in line.lower(), (
                f"Curated contract forecast line must declare 'vintage unverified': {line}"
            )


def write_metrics_fixture(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    """Regenerate the frozen split metrics fixture from a real eval run."""
    import pandas as pd

    from gridcast.backtest import (
        TEST_SPLIT_END_DATE,
        TEST_SPLIT_START_DATE,
        derive_test_split_origins,
        run_rolling_origin_backtest,
    )
    from gridcast.cli import build_example_frame, default_issue_schedule, load_feature_sources

    intensity_df, weather_df, mix_df = load_feature_sources()
    schedule = default_issue_schedule("2024-11-08", TEST_SPLIT_END_DATE)
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
    origins = derive_test_split_origins(TEST_SPLIT_START_DATE, TEST_SPLIT_END_DATE)
    results = run_rolling_origin_backtest(
        frame=frame,
        origins=origins,
        operator_forecasts=operator_forecasts,
        verbose=False,
    )

    pht = results.per_horizon_table
    key_horizons = {}
    for h in [2, 48, 96]:
        row = pht[pht["horizon"] == h].iloc[0]
        key_horizons[str(h)] = {
            "model": round(float(row["model_mae"]), 2),
            "operator": round(float(row["operator_mae"]), 2),
            "persistence": round(float(row["persistence_mae"]), 2),
            "seasonal_naive": round(float(row["seasonal_naive_mae"]), 2),
            "periods": int(row["count"]),
        }

    origin_window = (
        f"frozen split, {TEST_SPLIT_START_DATE} to {TEST_SPLIT_END_DATE}, "
        f"{len(origins)} origins, 2026-11-02 run"
    )

    data: dict[str, Any] = {
        "run_date": "2026-11-02",
        "origin_window": origin_window,
        "start_date": TEST_SPLIT_START_DATE,
        "end_date": TEST_SPLIT_END_DATE,
        "origins_count": len(origins),
        "first_origin": str(origins[0].strftime("%Y-%m-%d %H:%M UTC")),
        "last_origin": str(origins[-1].strftime("%Y-%m-%d %H:%M UTC")),
        "scalars": {
            "model": round(float(results.gate_value), 3),
            "operator": round(float(results.operator_gate_value), 3),
            "persistence": round(float(results.persistence_gate_value), 3),
            "seasonal_naive": round(float(results.seasonal_naive_gate_value), 3),
        },
        "key_horizons": key_horizons,
    }

    out_path = Path(fixture_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2) + "\n"
    out_path.write_text(serialized, encoding="utf-8")
    print(f"Wrote {out_path}:")
    print(serialized)
    return data
