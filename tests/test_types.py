import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, CleaningRecord


def test_inspect_and_detect_dtypes():
    df = pd.DataFrame(
        {
            "age": [25, 30],
            "salary": [50000.5, 60000.0],
            "name": ["Alice", "Bob"],
        }
    )
    cleaner = Cleaner(df)

    dtypes_all = cleaner.dtypes()
    assert isinstance(dtypes_all, dict)
    assert dtypes_all["age"] == "int64"
    assert dtypes_all["salary"] == "float64"
    assert dtypes_all["name"] in ("object", "str", "string")

    # Single column inspect
    assert cleaner.dtypes("age") == "int64"
    assert cleaner.inspect_types("salary") == "float64"

    with pytest.raises(KeyError, match="not found"):
        cleaner.dtypes("nonexistent")


def test_example_api_and_conversions():
    df = pd.DataFrame(
        {
            "age": ["25", "30", "35"],
            "salary": ["50000.50", "60000.75", "75000.00"],
            "joining_date": ["2023-07-12", "2024-01-10", "2022-08-01"],
        }
    )
    cleaner = Cleaner(df)

    # Example API from user specification
    cleaner.convert_type("age", "int")
    cleaner.convert_type("salary", "float")
    cleaner.convert_type("joining_date", "datetime")

    assert pd.api.types.is_integer_dtype(cleaner.data["age"])
    assert pd.api.types.is_float_dtype(cleaner.data["salary"])
    assert pd.api.types.is_datetime64_any_dtype(cleaner.data["joining_date"])

    assert cleaner.data["age"].tolist() == [25, 30, 35]
    assert cleaner.data["salary"].tolist() == [50000.5, 60000.75, 75000.0]


def test_integer_conversion_preserving_missing_values():
    df = pd.DataFrame({"count": ["10", "20", np.nan, None]})
    cleaner = Cleaner(df)
    cleaner.convert_type("count", "int")

    assert pd.api.types.is_integer_dtype(cleaner.data["count"])
    assert cleaner.data.loc[0, "count"] == 10
    assert cleaner.data.loc[1, "count"] == 20
    assert pd.isna(cleaner.data.loc[2, "count"])
    assert pd.isna(cleaner.data.loc[3, "count"])


def test_float_conversion_preserving_missing_values():
    df = pd.DataFrame({"val": ["1.25", np.nan, "3.50"]})
    cleaner = Cleaner(df)
    cleaner.convert_type("val", "float")

    assert pd.api.types.is_float_dtype(cleaner.data["val"])
    assert cleaner.data["val"].iloc[0] == 1.25
    assert pd.isna(cleaner.data["val"].iloc[1])
    assert cleaner.data["val"].iloc[2] == 3.50


def test_string_conversion_preserving_missing_values():
    df = pd.DataFrame({"code": [101, 202, np.nan, None]})
    cleaner = Cleaner(df)
    cleaner.convert_type("code", "string")

    assert cleaner.data["code"].iloc[0] == "101"
    assert cleaner.data["code"].iloc[1] == "202"
    # Crucial: missing values must NOT be converted to string 'nan' or 'None'
    assert pd.isna(cleaner.data["code"].iloc[2])
    assert pd.isna(cleaner.data["code"].iloc[3])
    assert cleaner.data["code"].iloc[2] != "nan"
    assert cleaner.data["code"].iloc[3] != "None"


def test_boolean_conversion():
    df = pd.DataFrame(
        {
            "flag1": ["true", "False", "YES", "no", None],
            "flag2": [1, 0, 1, 0, np.nan],
        }
    )
    cleaner = Cleaner(df)
    cleaner.convert_type("flag1", "bool")
    cleaner.convert_type("flag2", "bool")

    assert pd.api.types.is_bool_dtype(cleaner.data["flag1"])
    assert pd.api.types.is_bool_dtype(cleaner.data["flag2"])

    assert cleaner.data["flag1"].iloc[0] == True
    assert cleaner.data["flag1"].iloc[1] == False
    assert cleaner.data["flag1"].iloc[2] == True
    assert cleaner.data["flag1"].iloc[3] == False
    assert pd.isna(cleaner.data["flag1"].iloc[4])


def test_datetime_conversion_with_format_and_missing():
    df = pd.DataFrame(
        {
            "d1": ["2023-01-15", None, "2024-05-20"],
            "d2": ["15/01/2023", "20/05/2024", np.nan],
        }
    )
    cleaner = Cleaner(df)
    cleaner.convert_type("d1", "datetime")
    cleaner.convert_type("d2", "datetime", format="%d/%m/%Y")

    assert pd.api.types.is_datetime64_any_dtype(cleaner.data["d1"])
    assert pd.api.types.is_datetime64_any_dtype(cleaner.data["d2"])
    assert pd.isna(cleaner.data["d1"].iloc[1])
    assert pd.isna(cleaner.data["d2"].iloc[2])


def test_conversion_failure_does_not_silently_destroy():
    df = pd.DataFrame({"age": ["25", "twenty", "30"]})
    cleaner = Cleaner(df)

    with pytest.raises(ValueError, match="Failed to convert column 'age' to integer"):
        cleaner.convert_type("age", "int")

    # Column must not be destroyed or converted to NaN
    assert cleaner.data["age"].tolist() == ["25", "twenty", "30"]


def test_conversion_errors_coerce_and_ignore():
    df = pd.DataFrame({"num": ["10", "bad", "30"]})

    # Coerce turns bad values into NaN
    c_coerce = Cleaner(df)
    c_coerce.convert_type("num", "int", errors="coerce")
    assert c_coerce.data["num"].iloc[0] == 10
    assert pd.isna(c_coerce.data["num"].iloc[1])
    assert c_coerce.data["num"].iloc[2] == 30

    # Ignore leaves column unchanged
    c_ignore = Cleaner(df)
    c_ignore.convert_type("num", "int", errors="ignore")
    assert c_ignore.data["num"].tolist() == ["10", "bad", "30"]


def test_batch_convert_types_dict():
    df = pd.DataFrame(
        {
            "a": ["1", "2"],
            "b": ["3.5", "4.5"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.convert_types({"a": "int", "b": "float"})

    assert pd.api.types.is_integer_dtype(cleaner.data["a"])
    assert pd.api.types.is_float_dtype(cleaner.data["b"])


def test_invalid_target_type():
    df = pd.DataFrame({"a": [1, 2]})
    cleaner = Cleaner(df)
    with pytest.raises(ValueError, match="Unsupported target type"):
        cleaner.convert_type("a", "unknown_type")


def test_original_dataframe_not_mutated():
    original = pd.DataFrame({"x": ["1", "2"]})
    original_copy = original.copy()

    cleaner = Cleaner(original)
    cleaner.convert_type("x", "int")

    assert pd.api.types.is_integer_dtype(cleaner.data["x"])
    assert not pd.api.types.is_integer_dtype(original["x"])
    pd.testing.assert_frame_equal(original, original_copy)


def test_history_recording():
    df = pd.DataFrame({"val": ["10", "20"]})
    cleaner = Cleaner(df)
    cleaner.convert_type("val", "int")

    assert len(cleaner.history) == 1
    record = cleaner.last_operation
    assert isinstance(record, CleaningRecord)
    assert record.operation == "convert_type"
    assert record.strategy == "integer"
    assert "Converted column 'val'" in str(record)
