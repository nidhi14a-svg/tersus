import pandas as pd
import pytest

from tersus import Cleaner, CleaningRecord
from tersus.columns import clean_single_column_name, make_unique_column_names


def test_clean_column_examples_from_specification():
    df = pd.DataFrame(
        columns=[" Student Name ", "Total Marks (%)"]
    )
    cleaner = Cleaner(df)
    cleaner.clean_column_names()

    assert list(cleaner.data.columns) == ["student_name", "total_marks"]


def test_clean_single_column_name_cases():
    assert clean_single_column_name("  First Name  ") == "first_name"
    assert clean_single_column_name("Student-ID") == "student_id"
    assert clean_single_column_name("Email.Address") == "email_address"
    assert clean_single_column_name("Salary ($/yr)") == "salary_yr"
    assert clean_single_column_name("Discount (10%)") == "discount_10"
    assert clean_single_column_name("User#") == "user"
    assert clean_single_column_name("Crème Brûlée") == "creme_brulee"


def test_case_options():
    df = pd.DataFrame(columns=[" Student Name "])

    # Lowercase (default)
    c1 = Cleaner(df)
    c1.clean_column_names(case="lower")
    assert list(c1.data.columns) == ["student_name"]

    # Uppercase
    c2 = Cleaner(df)
    c2.clean_column_names(case="upper")
    assert list(c2.data.columns) == ["STUDENT_NAME"]

    # Preserve case
    c3 = Cleaner(df)
    c3.clean_column_names(case="preserve")
    assert list(c3.data.columns) == ["Student_Name"]

    # Snake case with camelCase
    df_camel = pd.DataFrame(columns=["studentFirstName", "TotalScore"])
    c4 = Cleaner(df_camel)
    c4.clean_column_names(case="snake")
    assert list(c4.data.columns) == ["student_first_name", "total_score"]

    # Invalid case option
    c5 = Cleaner(df)
    with pytest.raises(ValueError, match="Invalid case option"):
        c5.clean_column_names(case="unsupported")


def test_custom_replacement_character():
    df = pd.DataFrame(columns=["User First Name", "Postal Code"])
    cleaner = Cleaner(df)
    cleaner.clean_column_names(replace_spaces="-")
    assert list(cleaner.data.columns) == ["user-first-name", "postal-code"]


def test_empty_column_names_safety():
    df = pd.DataFrame(columns=["", "   ", "###", "Valid"])
    cleaner = Cleaner(df)
    cleaner.clean_column_names()

    # Empty and special-only columns become unnamed, with deterministic suffixes
    cols = list(cleaner.data.columns)
    assert cols[0] == "unnamed"
    assert cols[1] == "unnamed_1"
    assert cols[2] == "unnamed_2"
    assert cols[3] == "valid"


def test_duplicate_column_names_handling():
    # Initial duplicates
    df = pd.DataFrame([[1, 2, 3]], columns=["id", "id", "id"])
    cleaner = Cleaner(df)
    cleaner.clean_column_names()

    assert list(cleaner.data.columns) == ["id", "id_1", "id_2"]

    # Duplicates arising after cleaning
    df_collision = pd.DataFrame([[1, 2]], columns=["First Name", "first_name"])
    cleaner_collision = Cleaner(df_collision)
    cleaner_collision.clean_column_names()

    assert list(cleaner_collision.data.columns) == ["first_name", "first_name_1"]


def test_duplicate_collision_with_existing_suffix():
    names = ["col", "col", "col_1"]
    unique = make_unique_column_names(names)
    assert len(unique) == 3
    assert len(set(unique)) == 3
    assert unique[0] == "col"
    assert unique[1] == "col_2"
    assert unique[2] == "col_1"


def test_rename_overrides():
    df = pd.DataFrame(columns=["fname", "lname", "Age"])
    cleaner = Cleaner(df)
    cleaner.clean_column_names(rename={"fname": "first_name", "lname": "last_name"})

    assert list(cleaner.data.columns) == ["first_name", "last_name", "age"]


def test_clean_columns_alias():
    df = pd.DataFrame(columns=[" Item Count "])
    cleaner = Cleaner(df)
    cleaner.clean_columns()
    assert list(cleaner.data.columns) == ["item_count"]


def test_original_dataframe_not_mutated():
    original_df = pd.DataFrame(columns=[" Old Column "])
    original_copy = original_df.copy()

    cleaner = Cleaner(original_df)
    cleaner.clean_column_names()

    assert list(cleaner.data.columns) == ["old_column"]
    assert list(original_df.columns) == [" Old Column "]
    pd.testing.assert_frame_equal(original_df, original_copy)


def test_history_recording():
    df = pd.DataFrame(columns=[" Student ID ", "Marks (%)", "grade"])
    cleaner = Cleaner(df)
    cleaner.clean_column_names()

    assert len(cleaner.history) == 1
    record = cleaner.last_operation
    assert isinstance(record, CleaningRecord)
    assert record.operation == "clean_column_names"
    assert record.affected_count == 2  # student_id and marks changed; grade did not
    assert record.details["renamed"] == {
        " Student ID ": "student_id",
        "Marks (%)": "marks",
    }
    assert record.details["before"] == [" Student ID ", "Marks (%)", "grade"]
    assert record.details["after"] == ["student_id", "marks", "grade"]


def test_empty_dataframe():
    cleaner = Cleaner(pd.DataFrame())
    cleaner.clean_column_names()
    assert list(cleaner.data.columns) == []
    assert cleaner.last_operation.affected_count == 0
