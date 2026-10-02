import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, CleaningRecord


def test_missing_detection_counts_and_percentages():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "city": ["Pune", None, "Mumbai", np.nan],
            "score": [80.0, np.nan, 90.0, 85.0],
            "dept": ["CS", "CS", "Math", "Physics"],
        }
    )
    cleaner = Cleaner(df)

    assert cleaner.has_missing() is True
    assert cleaner.has_missing("id") is False
    assert cleaner.has_missing("city") is True

    # Missing counts
    counts = cleaner.missing_counts()
    assert counts == {"id": 0, "city": 2, "score": 1, "dept": 0}
    assert cleaner.missing_count("city") == 2
    assert cleaner.missing_count("id") == 0

    # Missing percentages
    pcts = cleaner.missing_percentages()
    assert pcts == {"id": 0.0, "city": 50.0, "score": 25.0, "dept": 0.0}
    assert cleaner.missing_percentage("city") == 50.0

    # Nonexistent column raises KeyError
    with pytest.raises(KeyError, match="not found"):
        cleaner.missing_counts("nonexistent")


def test_fill_mean():
    df = pd.DataFrame(
        {
            "num1": [10.0, 20.0, np.nan, 30.0],  # Mean is 20.0
            "num2": [1.0, np.nan, 3.0, 4.0],  # Mean is 2.6667
        }
    )
    cleaner = Cleaner(df)
    cleaner.fill_mean("num1")

    assert cleaner.data["num1"].isna().sum() == 0
    assert cleaner.data.loc[2, "num1"] == 20.0
    assert cleaner.data["num2"].isna().sum() == 1  # Unaffected

    # Fill all remaining numeric
    cleaner.fill_mean()
    assert cleaner.data["num2"].isna().sum() == 0
    assert round(cleaner.data.loc[1, "num2"], 2) == 2.67

    # Verify history record
    assert len(cleaner.history) == 2
    record = cleaner.last_operation
    assert isinstance(record, CleaningRecord)
    assert record.operation == "fill_missing"
    assert record.strategy == "mean"
    assert record.affected_count == 1
    assert "num2" in record.details["fill_values"]


def test_fill_median():
    # [10, 20, 100, NaN] -> valid: 10, 20, 100 -> median is 20.0
    df = pd.DataFrame({"salary": [10.0, 20.0, 100.0, np.nan]})
    cleaner = Cleaner(df)
    cleaner.fill_median("salary")

    assert cleaner.data["salary"].isna().sum() == 0
    assert cleaner.data.loc[3, "salary"] == 20.0
    assert cleaner.last_operation.strategy == "median"


def test_fill_mode():
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "red", None, "green", np.nan],
        }
    )
    cleaner = Cleaner(df)
    cleaner.fill_mode("color")

    assert cleaner.data["color"].isna().sum() == 0
    # Mode is "red"
    assert cleaner.data.loc[3, "color"] == "red"
    assert cleaner.data.loc[5, "color"] == "red"
    assert cleaner.last_operation.strategy == "mode"
    assert cleaner.last_operation.affected_count == 2


def test_fill_constant_text_and_numeric():
    df = pd.DataFrame(
        {
            "category": ["A", None, "B", np.nan],
            "count": [10, np.nan, 20, 30],
        }
    )
    cleaner = Cleaner(df)

    cleaner.fill_constant("Missing", columns="category")
    assert cleaner.data["category"].tolist() == ["A", "Missing", "B", "Missing"]

    cleaner.fill_constant(0, columns="count")
    assert cleaner.data["count"].tolist() == [10.0, 0.0, 20.0, 30.0]

    assert len(cleaner.history) == 2


def test_drop_missing_any():
    df = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4],
            "b": ["x", None, "z", "w"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.drop_missing(how="any")

    # Rows 1 (index 1) and 2 (index 2) have missing values, so 2 rows dropped
    assert len(cleaner.data) == 2
    assert cleaner.data["a"].tolist() == [1.0, 4.0]
    assert cleaner.data["b"].tolist() == ["x", "w"]
    assert cleaner.last_operation.operation == "drop_missing"
    assert cleaner.last_operation.affected_count == 2


def test_drop_missing_subset():
    df = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4],
            "b": ["x", None, "z", "w"],
        }
    )
    cleaner = Cleaner(df)
    # Only drop rows where 'a' is missing
    cleaner.drop_missing(columns=["a"])

    assert len(cleaner.data) == 3
    assert cleaner.data["a"].tolist() == [1.0, 2.0, 4.0]
    # Row with None in 'b' remains because 'b' was not in subset
    assert cleaner.data["b"].isna().sum() == 1


def test_drop_missing_all():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.nan],
            "b": ["x", "y", np.nan],
        }
    )
    cleaner = Cleaner(df)
    cleaner.drop_missing(how="all")

    # Only row 2 (both a and b are NaN) is dropped
    assert len(cleaner.data) == 2
    assert cleaner.last_operation.affected_count == 1


def test_drop_missing_invalid_how():
    df = pd.DataFrame({"a": [1, np.nan]})
    cleaner = Cleaner(df)
    with pytest.raises(ValueError, match="Invalid 'how' parameter"):
        cleaner.drop_missing(how="invalid")


def test_type_safety_mean_on_text():
    df = pd.DataFrame({"name": ["Alice", None, "Charlie"]})
    cleaner = Cleaner(df)

    with pytest.raises(TypeError, match="non-numeric"):
        cleaner.fill_mean("name")

    with pytest.raises(TypeError, match="non-numeric"):
        cleaner.fill_median("name")


def test_type_safety_constant_incompatible():
    df = pd.DataFrame({"score": [90.5, np.nan, 85.0]})
    cleaner = Cleaner(df)

    # Filling a numeric column with a string constant should raise TypeError
    with pytest.raises(TypeError, match="Cannot fill numeric column"):
        cleaner.fill_constant("NOT_A_NUMBER", columns="score")


def test_all_null_column_safety():
    df = pd.DataFrame(
        {
            "all_null_num": [np.nan, np.nan],
            "all_null_cat": [None, None],
        }
    )
    cleaner = Cleaner(df)

    with pytest.raises(ValueError, match="contains only missing values"):
        cleaner.fill_mean("all_null_num")

    with pytest.raises(ValueError, match="contains only missing values"):
        cleaner.fill_median("all_null_num")

    with pytest.raises(ValueError, match="contains only missing values"):
        cleaner.fill_mode("all_null_cat")

    # Constant on all-null column should succeed
    cleaner.fill_constant(0.0, columns="all_null_num")
    assert cleaner.data["all_null_num"].tolist() == [0.0, 0.0]


def test_empty_dataframe_safety():
    empty_cleaner = Cleaner(pd.DataFrame())
    assert empty_cleaner.missing_counts() == {}
    assert empty_cleaner.has_missing() is False

    # Chained calls should succeed without error
    empty_cleaner.fill_mean().drop_missing()
    assert len(empty_cleaner.data) == 0

    cleaner_with_cols = Cleaner(pd.DataFrame(columns=["x", "y"]))
    assert cleaner_with_cols.missing_counts() == {"x": 0, "y": 0}
    cleaner_with_cols.fill_constant(0, columns="x").drop_missing()
    assert len(cleaner_with_cols.data) == 0


def test_strategy_validation():
    df = pd.DataFrame({"x": [1, np.nan]})
    cleaner = Cleaner(df)

    with pytest.raises(ValueError, match="Invalid strategy"):
        cleaner.fill_missing(strategy="magic")

    with pytest.raises(ValueError, match="A 'value' must be provided"):
        cleaner.fill_missing(strategy="constant")


def test_original_dataframe_not_mutated_by_cleaning():
    original = pd.DataFrame({"col": [1.0, np.nan, 3.0]})
    original_copy = original.copy()

    cleaner = Cleaner(original)
    cleaner.fill_mean("col")

    # cleaner working copy is cleaned
    assert cleaner.data["col"].isna().sum() == 0
    # user's original DataFrame is unchanged
    pd.testing.assert_frame_equal(original, original_copy)


def test_method_chaining_and_history():
    df = pd.DataFrame(
        {
            "num": [10.0, np.nan, 30.0, np.nan],
            "text": ["a", "b", None, "d"],
            "extra": [1, 2, 3, np.nan],
        }
    )
    cleaner = Cleaner(df)

    cleaner.fill_mean("num").fill_mode("text").drop_missing(columns=["extra"])

    assert cleaner.has_missing() is False
    assert len(cleaner.history) == 3

    assert cleaner.history[0].strategy == "mean"
    assert cleaner.history[1].strategy == "mode"
    assert cleaner.history[2].strategy == "drop"

    # Test record to_dict and str
    dict_record = cleaner.history[0].to_dict()
    assert dict_record["operation"] == "fill_missing"
    assert dict_record["strategy"] == "mean"
    assert "Filled" in str(cleaner.history[0])
