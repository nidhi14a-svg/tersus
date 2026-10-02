from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner


def test_dataframe_input():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    cleaner = Cleaner(df)

    assert isinstance(cleaner.data, pd.DataFrame)
    assert cleaner.data.shape == (2, 2)
    pd.testing.assert_frame_equal(cleaner.data, df)


def test_dataframe_not_mutated():
    original_df = pd.DataFrame({"score": [10, 20, 30]})
    cleaner = Cleaner(original_df)

    # Working copy must be a separate object
    assert cleaner.data is not original_df

    # Modifying cleaner working copy does not affect original dataframe
    cleaner.data.loc[0, "score"] = 999
    assert original_df.loc[0, "score"] == 10


def test_csv_loading_str_path(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("id,val\n1,100\n2,200\n", encoding="utf-8")

    cleaner = Cleaner(str(csv_file))
    assert cleaner.data.shape == (2, 2)
    assert list(cleaner.data.columns) == ["id", "val"]
    assert cleaner.data["val"].tolist() == [100, 200]


def test_csv_loading_path_object(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("x,y\n1.5,2.5\n", encoding="utf-8")

    cleaner = Cleaner(csv_file)
    assert cleaner.data.shape == (1, 2)
    assert cleaner.data["x"].iloc[0] == 1.5


def test_empty_dataframe():
    empty_df = pd.DataFrame()
    cleaner = Cleaner(empty_df)

    assert cleaner.data.empty
    assert cleaner.data.shape == (0, 0)

    empty_with_cols = pd.DataFrame(columns=["alpha", "beta"])
    cleaner_with_cols = Cleaner(empty_with_cols)

    assert cleaner_with_cols.data.empty
    assert list(cleaner_with_cols.data.columns) == ["alpha", "beta"]


def test_empty_dataframe_save(tmp_path):
    empty_with_cols = pd.DataFrame(columns=["col1", "col2"])
    cleaner = Cleaner(empty_with_cols)
    out_path = tmp_path / "empty_out.csv"

    cleaner.save(out_path)
    assert out_path.exists()

    reloaded = pd.read_csv(out_path)
    assert list(reloaded.columns) == ["col1", "col2"]
    assert reloaded.empty


def test_invalid_input_none_and_empty():
    with pytest.raises(ValueError, match="A data source is required"):
        Cleaner(None)

    with pytest.raises(ValueError, match="A data source is required"):
        Cleaner()


def test_invalid_input_types():
    with pytest.raises(TypeError, match="Invalid input type"):
        Cleaner(12345)

    with pytest.raises(TypeError, match="Invalid input type"):
        Cleaner(["a", "b", "c"])

    with pytest.raises(TypeError, match="Invalid input type"):
        Cleaner({"key": "val"})


def test_invalid_csv_path(tmp_path):
    # Non-existent file
    with pytest.raises(FileNotFoundError, match="File not found"):
        Cleaner(tmp_path / "missing.csv")

    # Empty string path
    with pytest.raises(ValueError, match="CSV file path cannot be empty"):
        Cleaner("   ")

    # Directory instead of file
    with pytest.raises(ValueError, match="Expected a CSV file, but got a directory"):
        Cleaner(tmp_path)

    # Non-CSV extension
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("just some text", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        Cleaner(txt_file)


def test_save_to_csv(tmp_path):
    df = pd.DataFrame({"city": ["London", "Paris"], "code": ["LON", "PAR"]})
    cleaner = Cleaner(df)

    out_file = tmp_path / "cities.csv"
    cleaner.save(out_file)

    assert out_file.exists()
    reloaded = pd.read_csv(out_file)
    pd.testing.assert_frame_equal(reloaded, df)


def test_save_with_string_path_and_kwargs(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cleaner = Cleaner(df)

    out_file = tmp_path / "custom_delim.csv"
    cleaner.save(str(out_file), sep=";")

    reloaded = pd.read_csv(out_file, sep=";")
    pd.testing.assert_frame_equal(reloaded, df)


def test_save_creates_nested_directories(tmp_path):
    df = pd.DataFrame({"val": [1]})
    cleaner = Cleaner(df)

    nested_file = tmp_path / "deep" / "subfolder" / "result.csv"
    cleaner.save(nested_file)

    assert nested_file.exists()
    reloaded = pd.read_csv(nested_file)
    pd.testing.assert_frame_equal(reloaded, df)


def test_save_invalid_paths(tmp_path):
    df = pd.DataFrame({"x": [1]})
    cleaner = Cleaner(df)

    with pytest.raises(TypeError, match="Invalid path type"):
        cleaner.save(None)

    with pytest.raises(TypeError, match="Invalid path type"):
        cleaner.save(999)

    with pytest.raises(ValueError, match="Destination path cannot be empty"):
        cleaner.save("   ")

    with pytest.raises(ValueError, match="existing directory"):
        cleaner.save(tmp_path)


def test_cleaner_repr():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    cleaner = Cleaner(df)
    assert repr(cleaner) == "Cleaner(rows=2, columns=3)"
