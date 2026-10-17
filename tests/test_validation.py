import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, CleaningRecord, ValidationResult


def test_specification_example_check_range():
    df = pd.DataFrame(
        {
            "student": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "marks": [85, 105, -5, 92, 50],
        }
    )
    cleaner = Cleaner(df)

    result = cleaner.check_range(
        "marks",
        min_value=0,
        max_value=100,
    )

    assert isinstance(result, ValidationResult)
    assert result.is_valid is False
    assert result.has_violations is True
    assert result.invalid_count == 2
    assert result.invalid_indices == [1, 2]
    assert result.invalid_values == [105, -5]

    # Verify invalid_rows DataFrame
    assert isinstance(result.invalid_rows, pd.DataFrame)
    assert len(result.invalid_rows) == 2
    assert result.invalid_rows["student"].tolist() == ["Bob", "Charlie"]

    # CRITICAL: DataFrame must NOT be automatically modified
    assert len(cleaner.data) == 5
    assert cleaner.data["marks"].tolist() == [85, 105, -5, 92, 50]


def test_range_check_with_missing_and_non_numeric():
    df = pd.DataFrame(
        {
            "score": [50.0, np.nan, 120.0, -10.0, None, "bad_val"],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_range("score", min_value=0, max_value=100)

    # Missing values (index 1, 4) are NOT range violations
    # 120 (index 2), -10 (index 3), and "bad_val" (index 5) ARE violations
    assert result.invalid_count == 3
    assert result.invalid_indices == [2, 3, 5]
    assert 120.0 in result.invalid_values
    assert -10.0 in result.invalid_values
    assert "bad_val" in result.invalid_values


def test_open_ended_and_exclusive_ranges():
    df = pd.DataFrame({"val": [0, 50, 100]})
    cleaner = Cleaner(df)

    # Only min_value
    r_min = cleaner.check_range("val", min_value=50)
    assert r_min.invalid_count == 1
    assert r_min.invalid_values == [0]

    # Only max_value
    r_max = cleaner.check_range("val", max_value=50)
    assert r_max.invalid_count == 1
    assert r_max.invalid_values == [100]

    # Exclusive bounds
    r_excl = cleaner.check_range("val", min_value=0, max_value=100, inclusive=False)
    assert r_excl.invalid_count == 2
    assert r_excl.invalid_values == [0, 100]


def test_all_valid_range():
    df = pd.DataFrame({"marks": [75, 80, 95]})
    cleaner = Cleaner(df)
    result = cleaner.check_range("marks", min_value=0, max_value=100)

    assert result.is_valid is True
    assert result.has_violations is False
    assert result.invalid_count == 0
    assert result.invalid_indices == []
    assert result.invalid_values == []
    assert result.invalid_rows.empty


def test_check_values_allowed_set():
    df = pd.DataFrame(
        {
            "status": ["Active", "Alumni", "Suspended", "On leave", "Unknown"],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_values(
        "status",
        allowed_values=["Active", "Alumni", "On leave"],
    )

    assert result.is_valid is False
    assert result.invalid_count == 2
    assert result.invalid_indices == [2, 4]
    assert result.invalid_values == ["Suspended", "Unknown"]


def test_check_rule_custom_predicate():
    df = pd.DataFrame(
        {
            "code": ["TX-101", "TX-202", "INVALID", "TX-303"],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_rule(
        "code",
        rule=lambda s: str(s).startswith("TX-"),
        rule_name="must_start_with_TX-",
    )

    assert result.is_valid is False
    assert result.invalid_count == 1
    assert result.invalid_indices == [2]
    assert result.invalid_values == ["INVALID"]


def test_user_action_drop_invalid():
    df = pd.DataFrame(
        {
            "marks": [80, 150, 90],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_range("marks", min_value=0, max_value=100)

    # User explicitly decides to drop invalid rows
    cleaner.drop_invalid(result)

    assert len(cleaner.data) == 2
    assert cleaner.data["marks"].tolist() == [80, 90]
    assert len(cleaner.history) == 1
    assert cleaner.last_operation.operation == "drop_invalid"


def test_user_action_replace_invalid():
    df = pd.DataFrame(
        {
            "marks": [80, 150, 90],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_range("marks", min_value=0, max_value=100)

    # User explicitly decides to replace invalid value with NaN
    cleaner.replace_invalid(result, np.nan)

    assert cleaner.data.loc[0, "marks"] == 80
    assert pd.isna(cleaner.data.loc[1, "marks"])
    assert cleaner.data.loc[2, "marks"] == 90
    assert cleaner.last_operation.operation == "replace_invalid"


def test_user_action_clamp_range():
    df = pd.DataFrame(
        {
            "marks": [-10, 50, 120],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clamp_range("marks", min_value=0, max_value=100)

    assert cleaner.data["marks"].tolist() == [0.0, 50.0, 100.0]
    assert cleaner.last_operation.operation == "clamp_range"


def test_validation_result_dict_and_str():
    df = pd.DataFrame({"score": [50, 150]})
    cleaner = Cleaner(df)
    result = cleaner.check_range("score", min_value=0, max_value=100)

    d = result.to_dict()
    assert d["column"] == "score"
    assert d["invalid_count"] == 1
    assert d["is_valid"] is False

    rep = str(result)
    assert "FAILED" in rep
    assert "Row 1: 150" in rep


def test_invalid_parameters():
    df = pd.DataFrame({"score": [50]})
    cleaner = Cleaner(df)

    with pytest.raises(ValueError, match="At least one"):
        cleaner.check_range("score")

    with pytest.raises(ValueError, match="min_value .* cannot be greater than max_value"):
        cleaner.check_range("score", min_value=100, max_value=50)

    with pytest.raises(KeyError, match="not found"):
        cleaner.check_range("nonexistent", min_value=0)


def test_empty_dataframe():
    cleaner = Cleaner(pd.DataFrame(columns=["marks"]))
    result = cleaner.check_range("marks", min_value=0, max_value=100)

    assert result.is_valid is True
    assert result.invalid_count == 0
    assert result.invalid_indices == []
