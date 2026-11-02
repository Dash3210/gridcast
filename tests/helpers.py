"""Test assertion and diagnostic helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _is_null_scalar(val: object) -> bool:
    if val is None or val is pd.NA or val is pd.NaT:
        return True
    return bool(isinstance(val, (float, np.floating)) and math.isnan(val))


def _scalars_equal(a: object, b: object) -> bool:
    if _is_null_scalar(a) and _is_null_scalar(b):
        return True
    if _is_null_scalar(a) != _is_null_scalar(b):
        return False
    if (type(a) is bool) != (type(b) is bool):
        return False
    if type(a) is not type(b):
        if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
            return bool(a == b)
        return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _array_all_null(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return True
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
        return bool(np.isnan(arr).all())
    if arr.dtype == object:
        return all(_is_null_scalar(x) for x in arr.flat)
    return False


def _short_repr(val: object, max_len: int = 300) -> str:
    r = repr(val)
    if len(r) <= max_len:
        return r
    return r[:max_len] + f"... (truncated, total {len(r)} chars)"


def assert_injection_changed(before: Any, after: Any, label: str) -> None:
    """Fail if the injection did not change the observable output.

    Call this before asserting that a guard fires. It exists because an injection that
    silently failed proves nothing in either direction, and twice in this project a
    failed injection was mistaken for a hole in the guard.
    """
    diff_count: int | None = None
    is_equal = False

    if isinstance(before, pd.DataFrame) and isinstance(after, pd.DataFrame):
        b_all_null = before.empty or bool(before.isna().all().all())
        a_all_null = after.empty or bool(after.isna().all().all())
        if b_all_null and a_all_null:
            is_equal = True
            diff_count = 0
        elif b_all_null != a_all_null:
            is_equal = False
            diff_count = (
                int((~after.isna()).sum().sum())
                if b_all_null
                else int((~before.isna()).sum().sum())
            )
        elif (
            before.shape != after.shape
            or list(before.columns) != list(after.columns)
            or not before.index.equals(after.index)
        ):
            is_equal = False
            common_cols = [c for c in before.columns if c in after.columns]
            common_idx = before.index.intersection(after.index)
            diff_count = (before.size - len(common_idx) * len(common_cols)) + (
                after.size - len(common_idx) * len(common_cols)
            )
            for col in common_cols:
                sb = before.loc[common_idx, col]
                sa = after.loc[common_idx, col]
                both_null = sb.isna() & sa.isna()
                diff_count += int(((sb != sa) & ~both_null).sum())
        else:
            diff_count = 0
            for col in before.columns:
                sb = before[col]
                sa = after[col]
                both_null = sb.isna() & sa.isna()
                diff_count += int(((sb != sa) & ~both_null).sum())
            is_equal = diff_count == 0

    elif isinstance(before, pd.Series) and isinstance(after, pd.Series):
        b_all_null = before.empty or bool(before.isna().all())
        a_all_null = after.empty or bool(after.isna().all())
        if b_all_null and a_all_null:
            is_equal = True
            diff_count = 0
        elif b_all_null != a_all_null:
            is_equal = False
            diff_count = int((~after.isna()).sum()) if b_all_null else int((~before.isna()).sum())
        elif len(before) != len(after) or not before.index.equals(after.index):
            is_equal = False
            common_idx = before.index.intersection(after.index)
            diff_count = (len(before) - len(common_idx)) + (len(after) - len(common_idx))
            if len(common_idx) > 0:
                sb = before.loc[common_idx]
                sa = after.loc[common_idx]
                both_null = sb.isna() & sa.isna()
                diff_count += int(((sb != sa) & ~both_null).sum())
        else:
            both_null = before.isna() & after.isna()
            diff_mask = (before != after) & ~both_null
            diff_count = int(diff_mask.sum())
            is_equal = diff_count == 0

    elif isinstance(before, (pd.DataFrame, pd.Series)) or isinstance(
        after, (pd.DataFrame, pd.Series)
    ):
        is_equal = False
        target = before if isinstance(before, (pd.DataFrame, pd.Series)) else after
        diff_count = target.size

    elif isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
        b_all_null = _array_all_null(before)
        a_all_null = _array_all_null(after)
        if b_all_null and a_all_null:
            is_equal = True
            diff_count = 0
        elif b_all_null != a_all_null:
            is_equal = False
            diff_count = before.size if not b_all_null else after.size
        elif before.shape != after.shape:
            is_equal = False
            diff_count = max(before.size, after.size)
        elif np.issubdtype(before.dtype, np.floating) and np.issubdtype(after.dtype, np.floating):
            both_nan = np.isnan(before) & np.isnan(after)
            diff_count = int(np.count_nonzero((before != after) & ~both_nan))
            is_equal = diff_count == 0
        else:
            diff_count = sum(
                1 for x, y in zip(before.flat, after.flat, strict=True) if not _scalars_equal(x, y)
            )
            is_equal = diff_count == 0

    elif isinstance(before, (list, tuple, pd.Index)) and isinstance(
        after, (list, tuple, pd.Index)
    ):
        b_list = list(before)
        a_list = list(after)
        if len(b_list) != len(a_list):
            is_equal = False
        else:
            is_equal = all(_scalars_equal(x, y) for x, y in zip(b_list, a_list, strict=True))

    elif isinstance(before, set) and isinstance(after, set):
        is_equal = before == after

    elif isinstance(before, dict) and isinstance(after, dict):
        if before.keys() != after.keys():
            is_equal = False
        else:
            is_equal = all(_scalars_equal(before[k], after[k]) for k in before)

    else:
        is_equal = _scalars_equal(before, after)

    if is_equal:
        lines = [
            f"Injection '{label}' did not change the observable output.",
            f"  before: {_short_repr(before)}",
            f"  after:  {_short_repr(after)}",
        ]
        if diff_count is not None:
            lines.append(f"  differing elements: {diff_count}")
        raise AssertionError("\n".join(lines))
