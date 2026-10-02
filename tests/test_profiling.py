import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, ProfileResult
from tersus.profiling import profile_dataframe


def test_profile_normal_dataset():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "score": [85.0, 90.0, 75.0, 80.0],
        }
    )
    cleaner = Cleaner(df)
    profile = cleaner.profile()

    assert isinstance(profile, ProfileResult)
    assert profile.row_count == 4
    assert profile.n_rows == 4
    assert profile.column_count == 3
    assert profile.n_columns == 3
    assert profile.columns == ["id", "name", "score"]
    assert profile.duplicate_rows == 0
    assert profile.has_duplicates is False
    assert profile.has_missing_values is False

    # Missing counts and percentages
    assert profile.missing_counts == {"id": 0, "name": 0, "score": 0}
    assert profile.missing_percentages == {"id": 0.0, "name": 0.0, "score": 0.0}

    # Unique counts
    assert profile.unique_counts == {"id": 4, "name": 4, "score": 4}

    # Dict access and to_dict
    assert profile["row_count"] == 4
    assert profile["n_rows"] == 4
    assert "columns" in profile
    as_dict = profile.to_dict()
    assert as_dict["row_count"] == 4
    assert as_dict["columns"] == ["id", "name", "score"]


def test_profile_missing_values():
    df = pd.DataFrame(
        {
            "col_a": [1.0, np.nan, 3.0, np.nan, 5.0],  # 2 missing -> 40.0%
            "col_b": ["a", "b", np.nan, "d", "e"],  # 1 missing -> 20.0%
            "col_c": [10, 20, 30, 40, 50],  # 0 missing -> 0.0%
        }
    )
    cleaner = Cleaner(df)
    profile = cleaner.profile()

    assert profile.has_missing_values is True
    assert profile.missing_counts == {"col_a": 2, "col_b": 1, "col_c": 0}
    assert profile.missing_percentages == {
        "col_a": 40.0,
        "col_b": 20.0,
        "col_c": 0.0,
    }


def test_profile_duplicate_rows():
    df = pd.DataFrame(
        {
            "x": [1, 2, 2, 3, 2],
            "y": ["a", "b", "b", "c", "b"],
        }
    )
    cleaner = Cleaner(df)
    profile = cleaner.profile()

    # Rows 2 and 4 are duplicates of row 1
    assert profile.duplicate_rows == 2
    assert profile.duplicate_count == 2
    assert profile.has_duplicates is True


def test_profile_numeric_columns():
    df = pd.DataFrame(
        {
            "val": [10.0, 20.0, 30.0, 40.0, 50.0],
            "count": [1, 2, 3, 4, 5],
            "text": ["p", "q", "r", "s", "t"],
        }
    )
    cleaner = Cleaner(df)
    profile = cleaner.profile()

    assert set(profile.numeric_columns) == {"val", "count"}
    assert profile.categorical_columns == ["text"]

    stats_val = profile.numeric_stats["val"]
    assert stats_val["min"] == 10.0
    assert stats_val["max"] == 50.0
    assert stats_val["mean"] == 30.0
    assert stats_val["median"] == 30.0
    assert stats_val["q25"] == 20.0
    assert stats_val["q75"] == 40.0
    assert stats_val["std"] is not None

    stats_count = profile.numeric_stats["count"]
    assert stats_count["min"] == 1.0
    assert stats_count["max"] == 5.0
    assert stats_count["mean"] == 3.0


def test_profile_numeric_all_nan():
    df = pd.DataFrame({"all_nan": pd.Series([np.nan, np.nan], dtype=float)})
    profile = profile_dataframe(df)

    assert "all_nan" in profile.numeric_stats
    assert profile.numeric_stats["all_nan"]["min"] is None
    assert profile.numeric_stats["all_nan"]["mean"] is None


def test_profile_numeric_single_value():
    df = pd.DataFrame({"single": [42.0]})
    profile = profile_dataframe(df)

    assert "single" in profile.numeric_stats
    assert profile.numeric_stats["single"]["min"] == 42.0
    assert profile.numeric_stats["single"]["mean"] == 42.0
    assert profile.numeric_stats["single"]["std"] == 0.0


def test_profile_categorical_and_boolean_columns():
    df = pd.DataFrame(
        {
            "category": pd.Series(["cat", "dog", "cat", "bird"], dtype="category"),
            "string_col": ["alpha", "beta", "alpha", "gamma"],
            "flag": [True, False, True, True],
        }
    )
    cleaner = Cleaner(df)
    profile = cleaner.profile()

    # Boolean is not treated as numeric statistics
    assert "flag" not in profile.numeric_stats
    assert "flag" in profile.categorical_columns
    assert "category" in profile.categorical_columns
    assert "string_col" in profile.categorical_columns

    assert profile.unique_counts["category"] == 3
    assert profile.unique_counts["string_col"] == 3
    assert profile.unique_counts["flag"] == 2


def test_profile_empty_dataframe():
    # 0 rows, 0 columns
    empty_df = pd.DataFrame()
    profile_empty = profile_dataframe(empty_df)

    assert profile_empty.row_count == 0
    assert profile_empty.column_count == 0
    assert profile_empty.columns == []
    assert profile_empty.duplicate_rows == 0
    assert profile_empty.missing_counts == {}
    assert profile_empty.missing_percentages == {}
    assert profile_empty.numeric_stats == {}
    assert "No columns" in str(profile_empty)

    # 0 rows, 2 columns
    empty_with_cols = pd.DataFrame(columns=["a", "b"])
    profile_cols = profile_dataframe(empty_with_cols)

    assert profile_cols.row_count == 0
    assert profile_cols.column_count == 2
    assert profile_cols.columns == ["a", "b"]
    assert profile_cols.duplicate_rows == 0
    assert profile_cols.missing_counts == {"a": 0, "b": 0}
    assert profile_cols.missing_percentages == {"a": 0.0, "b": 0.0}
    assert profile_cols.unique_counts == {"a": 0, "b": 0}


def test_profile_does_not_modify_dataframe():
    original_df = pd.DataFrame(
        {
            "id": [1, 2, 2],
            "city": ["Pune", np.nan, "Pune"],
            "score": [80.0, 90.0, 80.0],
        }
    )
    cleaner = Cleaner(original_df)
    before_df = cleaner.data.copy()

    # Run profiling
    profile = cleaner.profile()
    assert profile.row_count == 3

    # Ensure cleaner.data and original_df were not modified
    pd.testing.assert_frame_equal(cleaner.data, before_df)
    pd.testing.assert_frame_equal(original_df, before_df)


def test_profile_deterministic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    cleaner = Cleaner(df)

    profile1 = cleaner.profile()
    profile2 = cleaner.profile()

    assert profile1.to_dict() == profile2.to_dict()
    assert str(profile1) == str(profile2)


def test_profile_key_error():
    df = pd.DataFrame({"a": [1]})
    cleaner = Cleaner(df)
    profile = cleaner.profile()

    with pytest.raises(KeyError, match="Key 'nonexistent' not found"):
        _ = profile["nonexistent"]
