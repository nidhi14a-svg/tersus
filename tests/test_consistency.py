"""Unit tests for logical consistency checks in Tersus."""

import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, ConsistencyResult, ConsistencySuggestion
from tersus.consistency import (
    check_date_order,
    check_relationship,
    check_value_order,
    resolve_consistency,
)


def test_date_order_valid_cases():
    df = pd.DataFrame(
        {
            "start_date": ["2026-01-01", "2026-02-15", "2026-03-01", None],
            "end_date": ["2026-01-10", "2026-02-15", "2026-03-15", "2026-04-01"],
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_date_order("start_date", "end_date", allow_equal=True)

    assert res.is_consistent is True
    assert res.inconsistent_count == 0
    assert len(res) == 0
    assert not res.has_inconsistencies
    assert bool(res) is True
    assert len(res.suggestions) == 0
    assert "PASSED" in str(res)


def test_date_order_invalid_and_allow_equal_flag():
    df = pd.DataFrame(
        {
            "start_date": ["2026-01-01", "2026-05-15", "2026-06-01", "2026-07-01"],
            "end_date": ["2026-01-05", "2026-05-01", "2026-06-01", None],
        }
    )
    cleaner = Cleaner(df)

    # 1. With allow_equal=True (only row 1 is inverted: May 15 > May 1)
    res = cleaner.check_date_order("start_date", "end_date", allow_equal=True)
    assert res.is_consistent is False
    assert res.inconsistent_count == 1
    assert res.inconsistent_indices == [1]
    assert res.inconsistent_rows.loc[1, "start_date"] == "2026-05-15"
    assert res.inconsistent_rows.loc[1, "end_date"] == "2026-05-01"
    assert len(res.suggestions) == 4
    assert "FAILED (1 inconsistent row(s) found)" in str(res)

    # 2. With allow_equal=False (both row 1 and row 2 [equal dates] are invalid)
    res_strict = cleaner.check_date_order("start_date", "end_date", allow_equal=False)
    assert res_strict.is_consistent is False
    assert res_strict.inconsistent_count == 2
    assert res_strict.inconsistent_indices == [1, 2]


def test_value_order_min_max_valid_and_invalid():
    df = pd.DataFrame(
        {
            "min_salary": [50000, 75000, 100000, None, 40000],
            "max_salary": [70000, 60000, 100000, 90000, 40000],
        }
    )
    cleaner = Cleaner(df)

    # 1. Allow equal: row 1 is invalid (75000 > 60000)
    res = cleaner.check_value_order("min_salary", "max_salary", allow_equal=True)
    assert not res.is_consistent
    assert res.inconsistent_count == 1
    assert res.inconsistent_indices == [1]
    assert res.inconsistent_rows.loc[1, "min_salary"] == 75000
    assert res.inconsistent_rows.loc[1, "max_salary"] == 60000

    # 2. Alias check_min_max works identically
    res_alias = cleaner.check_min_max("min_salary", "max_salary", allow_equal=True)
    assert res_alias.inconsistent_count == 1

    # 3. Disallow equal: row 1 (75k > 60k), row 2 (100k == 100k), and row 4 (40k == 40k) fail
    res_strict = cleaner.check_value_order("min_salary", "max_salary", allow_equal=False)
    assert res_strict.inconsistent_count == 3
    assert res_strict.inconsistent_indices == [1, 2, 4]


def test_user_defined_relationship_callable():
    df = pd.DataFrame(
        {
            "subtotal": [100.0, 200.0, 50.0],
            "tax": [10.0, 20.0, 5.0],
            "total": [110.0, 250.0, 55.0],  # row 1: 200 + 20 != 250
        }
    )
    cleaner = Cleaner(df)

    res = cleaner.check_relationship(
        columns=["subtotal", "tax", "total"],
        rule=lambda r: np.isclose(r["total"], r["subtotal"] + r["tax"]),
        rule_name="total_must_equal_subtotal_plus_tax",
        suggestion="Recalculate total or adjust subtotal/tax",
    )

    assert not res.is_consistent
    assert res.inconsistent_count == 1
    assert res.inconsistent_indices == [1]
    assert any(s.strategy == "custom" for s in res.suggestions)
    assert any("Recalculate total" in s.description for s in res.suggestions)


def test_user_defined_relationship_expression():
    df = pd.DataFrame(
        {
            "price": [100, 200, 300],
            "discount": [10, 250, 50],  # row 1: discount > price (invalid)
        }
    )
    cleaner = Cleaner(df)

    res = cleaner.check_relationship(
        columns=["price", "discount"],
        rule="discount <= price",
        rule_name="discount_within_price",
    )

    assert not res.is_consistent
    assert res.inconsistent_count == 1
    assert res.inconsistent_indices == [1]


def test_separation_detection_does_not_modify_data():
    df = pd.DataFrame(
        {
            "start": ["2026-05-01", "2026-10-01"],
            "end": ["2026-01-01", "2026-12-01"],
        }
    )
    orig_copy = df.copy(deep=True)
    cleaner = Cleaner(df)

    # Perform detection
    res = cleaner.check_date_order("start", "end")
    assert not res.is_consistent

    # Verify original df and cleaner data are completely unmodified
    pd.testing.assert_frame_equal(cleaner.data, orig_copy)
    pd.testing.assert_frame_equal(df, orig_copy)
    assert len(cleaner.history) == 0


def test_suggestions_inspection():
    df = pd.DataFrame(
        {
            "low": [50, 100],
            "high": [40, 200],  # row 0 has low > high
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_value_order("low", "high")

    assert len(res.suggestions) == 4
    strategies = [s.strategy for s in res.suggestions]
    assert "swap" in strategies
    assert "align" in strategies
    assert "set_null" in strategies
    assert "drop" in strategies

    d = res.to_dict()
    assert d["inconsistent_count"] == 1
    assert len(d["suggestions"]) == 4


def test_correction_strategy_swap():
    df = pd.DataFrame(
        {
            "start_date": ["2026-05-15", "2026-01-01"],
            "end_date": ["2026-05-01", "2026-01-10"],  # row 0 is inverted
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_date_order("start_date", "end_date")
    assert res.inconsistent_count == 1

    # Explicitly swap
    cleaner.swap_inconsistent(res)

    # Verify swapped
    assert cleaner.data["start_date"].iloc[0] == "2026-05-01"
    assert cleaner.data["end_date"].iloc[0] == "2026-05-15"
    assert len(cleaner.history) == 1
    assert cleaner.history[0].operation == "resolve_consistency"
    assert cleaner.history[0].strategy == "swap"

    # Re-check passes
    res2 = cleaner.check_date_order("start_date", "end_date")
    assert res2.is_consistent is True


def test_correction_strategy_drop():
    df = pd.DataFrame(
        {
            "min_val": [10, 100, 30],
            "max_val": [20, 50, 40],  # row 1 is invalid (100 > 50)
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_value_order("min_val", "max_val")

    # Explicitly drop
    cleaner.drop_inconsistent(res)

    assert len(cleaner.data) == 2
    assert cleaner.data["min_val"].tolist() == [10, 30]
    assert cleaner.data["max_val"].tolist() == [20, 40]
    assert len(cleaner.history) == 1


def test_correction_strategy_set_null_and_align():
    df = pd.DataFrame(
        {
            "min_val": [100, 20],
            "max_val": [50, 40],  # row 0 is invalid
        }
    )
    # 1. set_null
    cleaner1 = Cleaner(df)
    res1 = cleaner1.check_value_order("min_val", "max_val")
    cleaner1.resolve_consistency(res1, strategy="set_null", target_column="max_val")
    assert pd.isna(cleaner1.data["max_val"].iloc[0])

    # 2. align
    cleaner2 = Cleaner(df)
    res2 = cleaner2.check_value_order("min_val", "max_val")
    cleaner2.resolve_consistency(
        res2, strategy="align", source_column="min_val", target_column="max_val"
    )
    assert cleaner2.data["max_val"].iloc[0] == 100


def test_correction_strategy_custom():
    df = pd.DataFrame(
        {
            "item": ["A", "B"],
            "subtotal": [100.0, 200.0],
            "tax": [10.0, 20.0],
            "total": [110.0, 999.0],  # row 1 invalid
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_relationship(
        columns=["subtotal", "tax", "total"],
        rule=lambda r: np.isclose(r["total"], r["subtotal"] + r["tax"]),
    )

    cleaner.resolve_consistency(
        res,
        strategy="custom",
        func=lambda r: {"total": r["subtotal"] + r["tax"]},
    )
    assert cleaner.data["total"].iloc[1] == 220.0


def test_edge_cases_and_error_handling():
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    cleaner = Cleaner(df)

    # Missing column
    with pytest.raises(KeyError, match="not found in dataset"):
        cleaner.check_date_order("col_a", "non_existent")

    with pytest.raises(KeyError, match="not found in dataset"):
        cleaner.check_value_order("non_existent", "col_b")

    # Unsupported strategy
    res = cleaner.check_value_order("col_a", "col_b")
    with pytest.raises(ValueError, match="Unsupported strategy"):
        cleaner.resolve_consistency(res, strategy="invalid_strategy")

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["start", "end"])
    empty_cleaner = Cleaner(empty_df)
    empty_res = empty_cleaner.check_date_order("start", "end")
    assert empty_res.is_consistent is True
    assert empty_res.inconsistent_count == 0
