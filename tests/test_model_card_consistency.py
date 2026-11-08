"""Consistency test asserting published model card metrics match the frozen split fixture.

data/eval/ is the single authority for all published numbers.
Figures are computed from data/eval/frozen_split_v1.parquet and sidecar at test time.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridcast.models import get_model_artifact_hash

EVAL_PARQUET_PATH = Path("data/eval/frozen_split_v1.parquet")
EVAL_SIDECAR_PATH = Path("data/eval/frozen_split_v1.json")
HASHES_FIXTURE_PATH = Path("tests/fixtures/frozen_split_metrics.json")
MODEL_CARD_PATH = Path("MODEL_CARD.md")
README_PATH = Path("README.md")
DECISIONS_PATH = Path("docs/decisions.md")

REQUIRED_WINDOW_TEXT = (
    "frozen fixture v1, sha256 205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, "
    "archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins"
)


def compute_published_metrics() -> dict[str, Any]:
    """Compute published benchmark metrics directly from data/eval fixture at test time.

    data/eval/ is the single authority for all published figures.
    """
    if not EVAL_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Evaluation fixture not found at {EVAL_PARQUET_PATH}")
    if not EVAL_SIDECAR_PATH.exists():
        raise FileNotFoundError(f"Evaluation sidecar not found at {EVAL_SIDECAR_PATH}")

    with open(EVAL_SIDECAR_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    # Verify parquet sha256 matches sidecar
    with open(EVAL_PARQUET_PATH, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == meta.get("sha256"), "Parquet SHA-256 does not match sidecar"

    df = pd.read_parquet(EVAL_PARQUET_PATH)
    assert len(df) == 1440, f"Expected 1440 rows in fixture, got {len(df)}"

    cand_maes = []
    oper_maes = []
    pers_maes = []
    sn_maes = []
    key_horizons: dict[str, dict[str, Any]] = {}

    for h in range(1, 97):
        h_df = df[df["horizon"] == h]
        act = h_df["actual"].to_numpy(dtype=float)
        c_mae = float(np.mean(np.abs(h_df["candidate_pred"].to_numpy(dtype=float) - act)))
        o_mae = float(np.mean(np.abs(h_df["neso_forecast"].to_numpy(dtype=float) - act)))
        p_mae = float(np.mean(np.abs(h_df["persistence_pred"].to_numpy(dtype=float) - act)))
        s_mae = float(np.mean(np.abs(h_df["seasonal_naive_pred"].to_numpy(dtype=float) - act)))

        cand_maes.append(c_mae)
        oper_maes.append(o_mae)
        pers_maes.append(p_mae)
        sn_maes.append(s_mae)

        if h in (2, 48, 96):
            key_horizons[str(h)] = {
                "model": round(c_mae, 2),
                "operator": round(o_mae, 2),
                "persistence": round(p_mae, 2),
                "seasonal_naive": round(s_mae, 2),
                "periods": len(h_df),
            }

    gate_val = sum(cand_maes) / 96.0
    oper_gate_val = sum(oper_maes) / 96.0
    pers_gate_val = sum(pers_maes) / 96.0
    sn_gate_val = sum(sn_maes) / 96.0

    return {
        "archive_commit": meta.get("archive_commit", "3617143"),
        "origin_window": meta.get("origin_window", REQUIRED_WINDOW_TEXT),
        "parquet_sha256": actual_sha,
        "model_artifact_sha256": meta.get("model_artifact_sha256"),
        "scalars": {
            "model": round(gate_val, 3),
            "operator": round(oper_gate_val, 3),
            "persistence": round(pers_gate_val, 3),
            "seasonal_naive": round(sn_gate_val, 3),
        },
        "key_horizons": key_horizons,
    }


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
    computed = compute_published_metrics()
    card_text = MODEL_CARD_PATH.read_text(encoding="utf-8")
    parsed = parse_model_card_gate_scalars(card_text)

    expected = computed["scalars"]
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
    computed = compute_published_metrics()
    card_text = MODEL_CARD_PATH.read_text(encoding="utf-8")
    parsed_horizons = parse_model_card_key_horizons(card_text)

    expected_horizons = computed["key_horizons"]
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
    computed = compute_published_metrics()
    readme_text = README_PATH.read_text(encoding="utf-8")
    parsed_entries = parse_readme_scalars(readme_text)

    assert len(parsed_entries) >= 2, "Expected at least 2 GATE_VALUE references in README.md"
    expected = computed["scalars"]

    for entry in parsed_entries:
        assert f"{entry['model']:.3f}" == f"{expected['model']:.3f}"
        assert f"{entry['operator']:.3f}" == f"{expected['operator']:.3f}"
        assert f"{entry['persistence']:.3f}" == f"{expected['persistence']:.3f}"
        assert f"{entry['seasonal_naive']:.3f}" == f"{expected['seasonal_naive']:.3f}"


def test_published_figures_carry_window() -> None:
    """Assert that every published number carries its origin window description."""
    computed = compute_published_metrics()
    expected_window = computed.get("origin_window", REQUIRED_WINDOW_TEXT)

    for doc_path in [MODEL_CARD_PATH, README_PATH, DECISIONS_PATH]:
        text = doc_path.read_text(encoding="utf-8")
        assert expected_window in text, (
            f"{doc_path} must carry the origin window specification: '{expected_window}'"
        )


def test_no_stale_metrics() -> None:
    """Assert that deprecated/stale numbers do not appear in any documentation."""
    stale_values = ["31.992", "39.605", "10.611"]
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


def test_model_card_artifact_sha256_matches() -> None:
    """Assert that the sha256 computed from data/models/lgbm_model.txt equals the card."""
    card_text = MODEL_CARD_PATH.read_text(encoding="utf-8")
    expected_hash = get_model_artifact_hash("data/models/lgbm_model.txt")
    assert expected_hash in card_text, (
        f"Artifact SHA-256 {expected_hash} not found in {MODEL_CARD_PATH}"
    )


def test_archive_commit_consistency() -> None:
    """Assert that the fixture carries the archive commit and docs name it."""
    computed = compute_published_metrics()
    assert "archive_commit" in computed, "Computed metrics must carry 'archive_commit'"
    commit = computed["archive_commit"]
    for doc_path in [MODEL_CARD_PATH, README_PATH, DECISIONS_PATH]:
        text = doc_path.read_text(encoding="utf-8")
        assert commit in text, f"{doc_path} must name the archive commit '{commit}'"


def test_guard_blind_spot_documented() -> None:
    """Assert that the guard's blind spot against live runs is named in docs and tests."""
    computed = compute_published_metrics()
    assert computed is not None
    for doc_path in [MODEL_CARD_PATH, DECISIONS_PATH]:
        text = doc_path.read_text(encoding="utf-8")
        assert "blind spot" in text.lower(), f"{doc_path} must name the guard's blind spot"


def test_fixture_file_is_hashes_only() -> None:
    """Requirement 4: tests/fixtures/frozen_split_metrics.json has only recorded hashes.

    Asserts it no longer contains scalars or key_horizons, preventing it from being
    a second authority for published numbers.
    """
    assert HASHES_FIXTURE_PATH.exists(), f"Fixture not found at {HASHES_FIXTURE_PATH}"
    with open(HASHES_FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert "scalars" not in data, "Fixture file must not contain 'scalars'"
    assert "key_horizons" not in data, "Fixture file must not contain 'key_horizons'"
    assert "archive_commit" in data, "Fixture file must contain 'archive_commit'"
    assert "parquet_sha256" in data, "Fixture file must contain 'parquet_sha256'"
    assert "model_artifact_sha256" in data, "Fixture file must contain 'model_artifact_sha256'"


def _find_all_doc_paths() -> list[Path]:
    """Discover all documentation markdown files across the repository."""
    root = Path(".")
    docs: list[Path] = []
    for p in root.glob("*.md"):
        if p.is_file():
            docs.append(p)
    docs_dir = root / "docs"
    if docs_dir.exists():
        for p in docs_dir.rglob("*.md"):
            if p.is_file():
                docs.append(p)
    return sorted(docs)


def _is_superseded_with_commit(line: str, score: str) -> bool:
    """Check if a score is explicitly marked as superseded/restated with its commit hash."""
    esc_score = re.escape(score)
    pattern = (
        r"(?:superseded|restated)\s+(?:from\s+)?"
        + r"(?:`" + esc_score + r"`|" + esc_score + r")"
        + r"(?:\s*gCO2/kWh)?\s+(?:at|with|from)\s+(?:commit\s+)?([0-9a-f]{7,40})"
    )
    return bool(re.search(pattern, line, re.IGNORECASE))


def _assert_line_valid_for_fixture_hash(line: str, location: str) -> None:
    """Assert published number paired with fixture hash is fixture value or superseded.

    Rule: Every published number is either the fixture value, or explicitly marked
    as superseded, with the commit it came from.
    """
    if "205222ce" not in line:
        return

    # Check any 39.xxx score on this line
    for score in re.findall(r"\b39\.\d{3}\b", line):
        if score == "39.032":
            continue
        assert _is_superseded_with_commit(line, score), (
            f"{location} pairs fixture hash 205222ce with unsourced score {score} "
            f"not explicitly marked as superseded with its commit: {line}"
        )

    # Any GATE_VALUE match on this line must be 39.032 or explicitly marked as superseded
    for gv in re.findall(r"GATE_VALUE\s*=\s*`?([0-9]+\.[0-9]+)`?", line):
        if gv in ("39.032", "39.032478"):
            continue
        assert _is_superseded_with_commit(line, gv), (
            f"{location} pairs fixture hash 205222ce with unsourced GATE_VALUE {gv} "
            f"not explicitly marked as superseded with its commit: {line}"
        )


def test_no_line_pairs_fixture_hash_with_non_fixture_score() -> None:
    """Assert every published number across all docs is fixture value or superseded with commit.

    Reads every documentation markdown file across the repository.
    """
    all_docs = _find_all_doc_paths()
    assert len(all_docs) >= 3, f"Expected at least 3 doc files, found {len(all_docs)}"
    for doc_path in all_docs:
        text = doc_path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            _assert_line_valid_for_fixture_hash(line, f"{doc_path}:{line_no}")


@pytest.mark.parametrize(
    "restored_line",
    [
        (
            "Score the test split (frozen fixture v1, sha256 "
            "205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, "
            "archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins) "
            "against this threshold. The candidate achieved `GATE_VALUE = 39.043 gCO2/kWh`"
        ),
        (
            "whereas on the committed documented split (frozen fixture v1, sha256 "
            "205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, "
            "archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins) "
            "it scored 39.043"
        ),
        (
            "On the frozen test split (frozen fixture v1, sha256 "
            "205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4, "
            "archive commit 3617143, window 2026-09-05 to 2026-11-01, 15 origins), "
            "the candidate scores `GATE_VALUE = 39.043 gCO2/kWh`"
        ),
    ],
)
def test_guard_fails_if_stale_line_restored(restored_line: str) -> None:
    """Prove guard fails if any of the three pre-freeze lines pairing v1 with 39.043 is restored."""
    # Precondition assertion (brief template section 5): prove injected line has 205222ce and 39.043
    assert "205222ce" in restored_line, "Precondition failed: restored line must have 205222ce"
    assert "39.043" in restored_line, "Precondition failed: restored line must have 39.043"

    with pytest.raises(AssertionError) as exc_info:
        _assert_line_valid_for_fixture_hash(restored_line, "restored_test_line")
    assert "39.043" in str(exc_info.value)


def test_pinned_eval_fixture_sha256_matches_docs() -> None:
    """Assert PINNED_EVAL_FIXTURE_SHA256 equals the sha256 quoted in ADR and MODEL_CARD.md."""
    from gridcast.backtest import PINNED_EVAL_FIXTURE_SHA256

    assert len(PINNED_EVAL_FIXTURE_SHA256) == 64
    assert PINNED_EVAL_FIXTURE_SHA256 == (
        "205222ce9b109e9ff28f663bef979c9c6d6d6be7c5c8e6ea7086b6067e40e7d4"
    )

    model_card_text = MODEL_CARD_PATH.read_text(encoding="utf-8")
    decisions_text = DECISIONS_PATH.read_text(encoding="utf-8")

    assert PINNED_EVAL_FIXTURE_SHA256 in model_card_text, (
        f"PINNED_EVAL_FIXTURE_SHA256 not found in {MODEL_CARD_PATH}"
    )
    assert PINNED_EVAL_FIXTURE_SHA256 in decisions_text, (
        f"PINNED_EVAL_FIXTURE_SHA256 not found in {DECISIONS_PATH}"
    )


def test_model_artifact_hash_commands_and_object_id_documented() -> None:
    """Assert both hash commands (content sha256, git hash-object) are in MODEL_CARD.md."""
    card_text = MODEL_CARD_PATH.read_text(encoding="utf-8")

    # Both exact commands
    assert "git hash-object data/models/lgbm_model.txt" in card_text
    assert "python -c" in card_text and "read_text" in card_text and "sha256" in card_text

    # Both hash values
    assert "8f841a6f048242085a36c33ed8b92b7acdf6d8e7" in card_text
    assert "5ddfe624a05dee49fa2ec339f6d22ba4ad6783dfcb1ea30eb7bca4a501e19e15" in card_text

    # Statement of which one GET /model returns
    assert "GET /model" in card_text
    assert "object id" in card_text.lower() or "object database" in card_text.lower()


def write_metrics_fixture(fixture_path: Path | str = HASHES_FIXTURE_PATH) -> dict[str, Any]:
    """Regenerate the recorded hashes fixture from data/eval/."""
    meta = compute_published_metrics()
    data: dict[str, Any] = {
        "archive_commit": meta["archive_commit"],
        "parquet_sha256": meta["parquet_sha256"],
        "model_artifact_sha256": meta["model_artifact_sha256"],
    }
    out_path = Path(fixture_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2) + "\n"
    out_path.write_text(serialized, encoding="utf-8")
    print(f"Wrote {out_path}:")
    print(serialized)
    return data
