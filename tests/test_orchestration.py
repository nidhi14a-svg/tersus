"""Unit tests for Tersus cleaning orchestration layer (clean)."""

import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, clean
from tersus.orchestration import validate_clean_options


def test_clean_empty_options_does_not_modify_data():
    df = pd.DataFrame(
        {
            " Score ": [80.0, None, 80.0],
            "text": ["  hello  ", "world", "  hello  "],
        }
    )
    cleaner = Cleaner(df)
    orig_copy = cleaner.data.copy(deep=True)

    # Calling clean() with no arguments must not execute any destructive or automatic changes
    result = cleaner.clean()

    assert result is cleaner
    pd.testing.assert_frame_equal(cleaner.data, orig_copy)
    assert len(cleaner.history) == 0


def test_clean_individual_missing_median():
    df = pd.DataFrame(
        {
            "id": [1, 1],  # Duplicate
            "val": [10.0, None],  # Missing
            "name": [" Alice ", "Bob"],  # Whitespace
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(missing="median")

    # Missing is filled with median (10.0)
    assert cleaner.data["val"].tolist() == [10.0, 10.0]
    # Unrequested operations must NOT have run
    assert len(cleaner.data) == 2  # Duplicates NOT dropped
    assert cleaner.data["name"].iloc[0] == " Alice "  # Whitespace NOT stripped
    assert len(cleaner.history) == 1
    assert cleaner.history[0].operation == "fill_missing"
    assert cleaner.history[0].strategy == "median"
    assert cleaner.history[0].affected_count == 1


def test_clean_individual_missing_mean():
    df = pd.DataFrame({"score": [10.0, 20.0, None]})
    cleaner = Cleaner(df)
    cleaner.clean(missing="mean")

    assert cleaner.data["score"].iloc[2] == 15.0
    assert len(cleaner.history) == 1
    assert cleaner.history[0].strategy == "mean"


def test_clean_individual_missing_mode():
    df = pd.DataFrame({"category": ["A", "B", "A", None]})
    cleaner = Cleaner(df)
    cleaner.clean(missing="mode")

    assert cleaner.data["category"].iloc[3] == "A"
    assert len(cleaner.history) == 1
    assert cleaner.history[0].strategy == "mode"


def test_clean_individual_missing_drop():
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    cleaner = Cleaner(df)
    cleaner.clean(missing="drop")

    assert len(cleaner.data) == 2
    assert cleaner.data["a"].tolist() == [1.0, 2.0]
    assert len(cleaner.history) == 1
    assert cleaner.history[0].operation == "drop_missing"
    assert cleaner.history[0].affected_count == 1


def test_clean_individual_duplicates():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "val": [10, 10, 20],
            "score": [100.0, 100.0, 50.0],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(duplicates=True)

    assert len(cleaner.data) == 2
    assert cleaner.data["id"].tolist() == [1, 2]
    assert cleaner.data["score"].tolist() == [100.0, 50.0]
    assert len(cleaner.history) == 1
    assert cleaner.history[0].operation == "drop_duplicates"
    assert cleaner.history[0].affected_count == 1


def test_clean_duplicates_with_dict_options():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "val": [10, 999, 20],  # id is duplicate, val is different
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(duplicates={"subset": ["id"], "keep": "last"})

    assert len(cleaner.data) == 2
    assert cleaner.data["val"].iloc[0] == 999  # Kept last
    assert cleaner.history[0].affected_count == 1


def test_clean_individual_column_names():
    df = pd.DataFrame(
        {
            " Student Name ": ["Alice", "Bob"],
            "Total Marks (%)": [90, 85],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(column_names=True)

    assert list(cleaner.data.columns) == ["student_name", "total_marks"]
    assert len(cleaner.history) == 1
    assert cleaner.history[0].operation == "clean_column_names"


def test_clean_individual_text():
    df = pd.DataFrame(
        {
            "name": ["  Alice  ", "Bob   Smith", "Charlie\t\n"],
            "num": [1, 2, 3],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(text=True)

    assert cleaner.data["name"].iloc[0] == "Alice"
    assert cleaner.data["name"].iloc[1] == "Bob Smith"
    assert cleaner.data["name"].iloc[2] == "Charlie"
    # Number column is untouched
    assert cleaner.data["num"].tolist() == [1, 2, 3]
    assert len(cleaner.history) == 1
    assert cleaner.history[0].operation == "clean_text"


def test_clean_combined_example_concept():
    # User's exact requested example:
    # cleaner.clean(
    #     missing="median",
    #     duplicates=True,
    #     column_names=True,
    #     text=True
    # )
    df = pd.DataFrame(
        {
            " Student Name ": [" Alice ", " Alice ", "Bob", "Charlie"],
            " Final Score ": [80.0, 80.0, None, 100.0],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(
        missing="median",
        duplicates=True,
        column_names=True,
        text=True,
    )

    # 1. Column names standardized
    assert list(cleaner.data.columns) == ["student_name", "final_score"]

    # 2. Duplicates dropped (first row duplicated in row 1)
    assert len(cleaner.data) == 3

    # 3. Text cleaned (whitespace stripped)
    assert cleaner.data["student_name"].tolist() == ["Alice", "Bob", "Charlie"]

    # 4. Missing values filled with median
    # Remaining valid scores: 80.0, 100.0 -> median is 90.0
    assert cleaner.data["final_score"].iloc[1] == 90.0

    # 5. History recorded every operation with affected counts
    assert len(cleaner.history) == 4
    operations = [r.operation for r in cleaner.history]
    assert operations == [
        "clean_column_names",
        "clean_text",
        "drop_duplicates",
        "fill_missing",
    ]


def test_clean_predictable_execution_order():
    # Verify that duplicate removal happens BEFORE median calculation,
    # so duplicate records do not bias the median statistic
    df = pd.DataFrame(
        {
            "val": [10.0, 10.0, 10.0, 10.0, 100.0, None],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(
        missing="median",
        duplicates=True,
    )

    # If duplicates were dropped first: [10.0, 100.0, None] -> median is 55.0
    # If duplicates were NOT dropped first: [10.0, 10.0, 10.0, 10.0, 100.0] -> median is 10.0
    assert cleaner.data["val"].iloc[2] == 55.0
    assert cleaner.data["val"].tolist() == [10.0, 100.0, 55.0]


def test_clean_unrequested_operations_not_performed():
    df = pd.DataFrame(
        {
            " Bad Name ": [" dirty ", " dirty "],
            "score": [None, 50.0],
        }
    )
    cleaner = Cleaner(df)

    # Only request duplicates
    cleaner.clean(duplicates=True)

    # Column name NOT cleaned
    assert " Bad Name " in cleaner.data.columns
    # Text NOT cleaned
    assert cleaner.data[" Bad Name "].iloc[0] == " dirty "
    # Missing NOT filled
    assert pd.isna(cleaner.data["score"].iloc[0])
    # Only 1 record in history
    assert len(cleaner.history) == 1
    assert cleaner.history[0].operation == "drop_duplicates"


def test_clean_validation_invalid_options():
    df = pd.DataFrame({"col": [1, 2, 3]})
    cleaner = Cleaner(df)

    # Invalid missing strategy
    with pytest.raises(ValueError, match="Invalid missing cleaning strategy 'invalid_strategy'"):
        cleaner.clean(missing="invalid_strategy")

    # Invalid type for duplicates
    with pytest.raises(TypeError, match="'duplicates' option must be a bool or a dict"):
        cleaner.clean(duplicates="yes")

    # Invalid type for column_names
    with pytest.raises(TypeError, match="'column_names' option must be a bool or a dict"):
        cleaner.clean(column_names=123)

    # Invalid type for text
    with pytest.raises(TypeError, match="'text' option must be a bool or a dict"):
        cleaner.clean(text=[])

    # Invalid type for types
    with pytest.raises(TypeError, match="'types' option must be a dictionary"):
        cleaner.clean(types="int")

    # Unknown keyword argument
    with pytest.raises(ValueError, match="Unsupported cleaning option: 'unknown_option'"):
        cleaner.clean(unknown_option=True)


def test_clean_return_df_and_chaining():
    df = pd.DataFrame({"val": [10.0, None, 10.0]})

    # return_df=False (default): returns Cleaner for chaining
    cleaner = Cleaner(df)
    res_cleaner = cleaner.clean(missing="median", return_df=False)
    assert isinstance(res_cleaner, Cleaner)
    assert res_cleaner is cleaner

    # return_df=True: returns pd.DataFrame
    cleaner2 = Cleaner(df)
    res_df = cleaner2.clean(missing="median", return_df=True)
    assert isinstance(res_df, pd.DataFrame)
    assert res_df["val"].tolist() == [10.0, 10.0, 10.0]


def test_standalone_clean_function():
    df = pd.DataFrame(
        {
            " Name ": [" Alice ", "Bob"],
            "score": [10.0, None],
        }
    )

    # 1. Called on DataFrame directly -> returns cleaned pd.DataFrame
    cleaned_df = clean(df, column_names=True, missing="median")
    assert isinstance(cleaned_df, pd.DataFrame)
    assert list(cleaned_df.columns) == ["name", "score"]
    assert cleaned_df["score"].tolist() == [10.0, 10.0]

    # 2. Called on Cleaner instance -> returns Cleaner instance
    c = Cleaner(df)
    cleaned_c = clean(c, column_names=True, text=True)
    assert isinstance(cleaned_c, Cleaner)
    assert list(cleaned_c.data.columns) == ["name", "score"]
    assert cleaned_c.data["name"].iloc[0] == "Alice"
    assert len(cleaned_c.history) == 2


def test_clean_with_types_and_whitespace_and_categories():
    df = pd.DataFrame(
        {
            "id_str": ["1", "2", "3"],
            "text": ["  clean me  ", "also  me", "done"],
            "category": ["Male", "male", "Female"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(
        types={"id_str": "int"},
        whitespace=True,
        categories={"column": "category", "mapping": {"male": "Male"}},
    )

    assert cleaner.data["id_str"].dtype.name in ("int32", "int64", "Int64")
    assert cleaner.data["text"].iloc[0] == "clean me"
    assert cleaner.data["category"].tolist() == ["Male", "Male", "Female"]
    assert len(cleaner.history) == 3


def test_clean_with_extended_pipeline_options():
    df = pd.DataFrame(
        {
            "Full Name": ["  John Doe ", "John Doe", "Jane", "Alice", "Bob"],
            "Email": ["john@example.com", "john@example.com", "not-an-email", "alice@example.com", "bob@example.com"],
            "Phone": ["+91 9876543210", "98765-43210", "123", "+919876543211", "+919876543212"],
            "Score": [120.0, 120.0, -10.0, 85.0, 90.0],
            "Salary": [50000, 50000, 10000000, 52000, 49000],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(
        column_names=True,
        text=True,
        duplicates=True,
        emails=True,
        phones=True,
        ranges={"score": {"min_value": 0, "max_value": 100}},
        outliers="cap",
    )

    # 1. Names trimmed and duplicated John Doe removed
    assert len(cleaner.data) == 4
    assert cleaner.data["full_name"].tolist() == ["John Doe", "Jane", "Alice", "Bob"]

    # 2. Invalid email coerced to NaN
    assert pd.isna(cleaner.data["email"].iloc[1])

    # 3. Invalid phone coerced to NaN
    assert pd.isna(cleaner.data["phone"].iloc[1])

    # 4. Score clamped to 0..100
    assert cleaner.data["score"].iloc[0] == 100.0
    assert cleaner.data["score"].iloc[1] == 0.0

    # 5. Outlier salary capped
    assert cleaner.data["salary"].iloc[1] < 10000000


def test_cleaner_auto_clean():
    df = pd.DataFrame(
        {
            " Name ": ["  Alice  ", "Bob", "Alice"],
            "Age": ["twenty", None, "20"],
            "Email": ["alice@domain.com", "invalid-email", "alice@domain.com"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.auto_clean()

    # Column names standardized
    assert "name" in cleaner.data.columns
    assert "age" in cleaner.data.columns
    assert "email" in cleaner.data.columns

    # Whitespace stripped and duplicate dropped
    assert len(cleaner.data) == 2

    # Words converted and missing imputed
    assert pd.api.types.is_numeric_dtype(cleaner.data["age"])
    assert cleaner.data["age"].isna().sum() == 0

    # Invalid email coerced
    assert pd.isna(cleaner.data["email"].iloc[1])


