import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, CleaningRecord
from tersus.text import normalize_text_value


def test_normalize_text_value_units():
    # Leading and trailing spaces
    assert normalize_text_value("  hello world  ") == "hello world"

    # Multiple spaces
    assert normalize_text_value("hello   beautiful   world") == "hello beautiful world"

    # Tabs and newlines
    assert normalize_text_value("hello\tworld\n\rhow are\nyou") == "hello world how are you"

    # Empty string
    assert normalize_text_value("") == ""
    assert normalize_text_value("   ") == ""

    # None and NaN preservation
    assert normalize_text_value(None) is None
    assert pd.isna(normalize_text_value(np.nan))

    # Normal text
    assert normalize_text_value("Normal Text") == "Normal Text"

    # Mixed-case text with casing options
    assert normalize_text_value("mIxEd CaSe", case="lower") == "mixed case"
    assert normalize_text_value("mIxEd CaSe", case="upper") == "MIXED CASE"
    assert normalize_text_value("mIxEd CaSe", case="title") == "Mixed Case"
    assert normalize_text_value("mIxEd CaSe", case=None) == "mIxEd CaSe"


def test_clean_text_spaces_tabs_newlines():
    df = pd.DataFrame(
        {
            "text": ["  leading", "trailing  ", "  both  ", "multiple   spaces", "tab\there", "new\nline\r\nhere"],
            "id": [1, 2, 3, 4, 5, 6],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_text("text")

    expected = [
        "leading",
        "trailing",
        "both",
        "multiple spaces",
        "tab here",
        "new line here",
    ]
    assert cleaner.data["text"].tolist() == expected
    # Numeric column 'id' is untouched
    assert cleaner.data["id"].tolist() == [1, 2, 3, 4, 5, 6]


def test_preserve_missing_values_nan_and_none():
    df = pd.DataFrame(
        {
            "name": [" Alice ", None, "Bob  ", np.nan],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_text("name")

    # Values stripped
    assert cleaner.data.loc[0, "name"] == "Alice"
    assert cleaner.data.loc[2, "name"] == "Bob"

    # Missing values must remain missing (pd.isna is True), never string 'None' or 'nan'
    assert pd.isna(cleaner.data.loc[1, "name"])
    assert pd.isna(cleaner.data.loc[3, "name"])
    assert cleaner.data.loc[1, "name"] != "None"
    assert cleaner.data.loc[3, "name"] != "nan"


def test_case_transformations():
    df = pd.DataFrame(
        {
            "city": ["pune", "MUMBAI", "new delhi"],
        }
    )

    # To upper
    c_upper = Cleaner(df)
    c_upper.to_upper("city")
    assert c_upper.data["city"].tolist() == ["PUNE", "MUMBAI", "NEW DELHI"]

    # To lower
    c_lower = Cleaner(df)
    c_lower.to_lower("city")
    assert c_lower.data["city"].tolist() == ["pune", "mumbai", "new delhi"]

    # To title
    c_title = Cleaner(df)
    c_title.to_title("city")
    assert c_title.data["city"].tolist() == ["Pune", "Mumbai", "New Delhi"]

    # Default clean_text preserves existing case
    c_preserve = Cleaner(df)
    c_preserve.clean_text("city")
    assert c_preserve.data["city"].tolist() == ["pune", "MUMBAI", "new delhi"]


def test_do_not_alter_numeric_columns():
    df = pd.DataFrame(
        {
            "num": [10.5, 20.0, 30.2],
            "text": ["  val1  ", "val2  ", " val3"],
        }
    )
    cleaner = Cleaner(df)

    # Explicitly targeting a numeric column raises TypeError
    with pytest.raises(TypeError, match="Cannot apply text cleaning to numeric column"):
        cleaner.clean_text("num")

    # Calling without columns only cleans text columns
    cleaner.clean_text()
    assert cleaner.data["text"].tolist() == ["val1", "val2", "val3"]
    assert cleaner.data["num"].tolist() == [10.5, 20.0, 30.2]


def test_empty_string_handling():
    df = pd.DataFrame(
        {
            "col": ["", "   ", "valid"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_text("col")

    assert cleaner.data["col"].tolist() == ["", "", "valid"]


def test_strip_whitespace_convenience_method():
    df = pd.DataFrame({"msg": ["  hello  ", "world  "]})
    cleaner = Cleaner(df)
    cleaner.strip_whitespace("msg")
    assert cleaner.data["msg"].tolist() == ["hello", "world"]


def test_original_dataframe_not_mutated():
    original_df = pd.DataFrame({"msg": ["  hello  "]})
    original_copy = original_df.copy()

    cleaner = Cleaner(original_df)
    cleaner.clean_text("msg")

    assert cleaner.data["msg"].iloc[0] == "hello"
    assert original_df["msg"].iloc[0] == "  hello  "
    pd.testing.assert_frame_equal(original_df, original_copy)


def test_history_recording():
    df = pd.DataFrame(
        {
            "a": ["  one  ", "two"],  # 1 modified
            "b": ["three", "four"],  # 0 modified
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_text()

    assert len(cleaner.history) == 1
    record = cleaner.last_operation
    assert isinstance(record, CleaningRecord)
    assert record.operation == "clean_text"
    assert record.affected_count == 1
    assert "1 value(s) modified" in str(record)


def test_empty_dataframe():
    empty_cleaner = Cleaner(pd.DataFrame())
    empty_cleaner.clean_text()
    assert len(empty_cleaner.data) == 0

    cleaner_with_cols = Cleaner(pd.DataFrame(columns=["a", "b"]))
    cleaner_with_cols.clean_text()
    assert len(cleaner_with_cols.data) == 0
