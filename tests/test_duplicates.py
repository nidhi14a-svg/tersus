import pandas as pd
import pytest

from tersus import Cleaner, CleaningRecord


def test_exact_duplicate_rows():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 2, 4],
            "val": ["a", "b", "c", "b", "e"],
        }
    )
    cleaner = Cleaner(df)

    assert cleaner.has_duplicates() is True
    assert cleaner.duplicate_count() == 1

    # Return duplicate records
    dup_df = cleaner.duplicates()
    assert isinstance(dup_df, pd.DataFrame)
    assert len(dup_df) == 1
    assert dup_df.iloc[0]["id"] == 2
    assert dup_df.iloc[0]["val"] == "b"

    # Remove duplicates
    cleaner.drop_duplicates()
    assert cleaner.has_duplicates() is False
    assert cleaner.duplicate_count() == 0
    assert len(cleaner.data) == 4
    assert cleaner.data["id"].tolist() == [1, 2, 3, 4]

    # Verify history
    assert len(cleaner.history) == 1
    record = cleaner.last_operation
    assert isinstance(record, CleaningRecord)
    assert record.operation == "drop_duplicates"
    assert record.affected_count == 1
    assert "Removed 1 duplicate row" in str(record)


def test_duplicates_based_on_selected_columns():
    df = pd.DataFrame(
        {
            "user_id": [101, 102, 101, 103],
            "device": ["mobile", "desktop", "tablet", "mobile"],
            "score": [10, 20, 30, 40],
        }
    )
    cleaner = Cleaner(df)

    # Across all columns, no rows are identical
    assert cleaner.duplicate_count() == 0
    assert cleaner.has_duplicates() is False

    # Based on selected column "user_id"
    assert cleaner.duplicate_count(subset=["user_id"]) == 1
    assert cleaner.has_duplicates(subset=["user_id"]) is True

    # Single column passed as string
    assert cleaner.duplicate_count(subset="user_id") == 1

    dups = cleaner.duplicates(subset=["user_id"])
    assert len(dups) == 1
    assert dups.iloc[0]["user_id"] == 101
    assert dups.iloc[0]["device"] == "tablet"  # second occurrence

    # Drop based on subset
    cleaner.drop_duplicates(subset=["user_id"])
    assert len(cleaner.data) == 3
    assert cleaner.data["user_id"].tolist() == [101, 102, 103]
    # First occurrence was kept
    assert cleaner.data.loc[0, "device"] == "mobile"


def test_no_duplicates():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )
    cleaner = Cleaner(df)

    assert cleaner.has_duplicates() is False
    assert cleaner.duplicate_count() == 0
    assert cleaner.duplicates().empty

    cleaner.drop_duplicates()
    assert len(cleaner.data) == 3
    assert cleaner.last_operation.affected_count == 0


def test_empty_dataframe():
    empty_cleaner = Cleaner(pd.DataFrame())
    assert empty_cleaner.has_duplicates() is False
    assert empty_cleaner.duplicate_count() == 0
    assert empty_cleaner.duplicates().empty

    empty_cleaner.drop_duplicates()
    assert len(empty_cleaner.data) == 0

    cleaner_with_cols = Cleaner(pd.DataFrame(columns=["col1", "col2"]))
    assert cleaner_with_cols.has_duplicates() is False
    assert cleaner_with_cols.duplicate_count() == 0
    assert list(cleaner_with_cols.duplicates().columns) == ["col1", "col2"]

    cleaner_with_cols.remove_duplicates()
    assert len(cleaner_with_cols.data) == 0


def test_pandas_keep_semantics():
    df = pd.DataFrame(
        {
            "k": ["dup", "dup", "unique", "dup"],
            "v": [1, 2, 3, 4],
        }
    )

    # keep='first' (default)
    c1 = Cleaner(df)
    c1.drop_duplicates(subset=["k"], keep="first")
    assert c1.data["v"].tolist() == [1, 3]

    # keep='last'
    c2 = Cleaner(df)
    c2.drop_duplicates(subset=["k"], keep="last")
    assert c2.data["v"].tolist() == [3, 4]

    # keep=False (drop all duplicates)
    c3 = Cleaner(df)
    assert c3.duplicate_count(subset=["k"], keep=False) == 3
    all_dups = c3.duplicates(subset=["k"], keep=False)
    assert len(all_dups) == 3
    assert all_dups["v"].tolist() == [1, 2, 4]

    c3.drop_duplicates(subset=["k"], keep=False)
    assert c3.data["v"].tolist() == [3]


def test_original_data_not_mutated():
    original_df = pd.DataFrame({"x": [1, 1, 2]})
    original_copy = original_df.copy()

    cleaner = Cleaner(original_df)
    cleaner.drop_duplicates()

    assert len(cleaner.data) == 2
    pd.testing.assert_frame_equal(original_df, original_copy)


def test_alias_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1]})
    cleaner = Cleaner(df)
    cleaner.remove_duplicates()
    assert len(cleaner.data) == 1


def test_invalid_subset_and_keep():
    df = pd.DataFrame({"a": [1, 2]})
    cleaner = Cleaner(df)

    with pytest.raises(KeyError, match="not found"):
        cleaner.duplicate_count(subset=["missing_col"])

    with pytest.raises(KeyError, match="not found"):
        cleaner.drop_duplicates(subset=["missing_col"])

    with pytest.raises(ValueError, match="Invalid 'keep' parameter"):
        cleaner.drop_duplicates(keep="invalid_option")

    with pytest.raises(ValueError, match="cannot be empty"):
        cleaner.drop_duplicates(subset=[])
