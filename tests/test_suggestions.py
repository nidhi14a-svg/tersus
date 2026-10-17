"""Unit tests for the Tersus suggestion engine."""

import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, Suggestion, SuggestionReport
from tersus.suggestions import generate_suggestions


def test_suggest_empty_dataframe():
    df = pd.DataFrame()
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    assert len(report) == 0
    assert not report.has_suggestions
    assert report.to_dataframe().empty
    assert report.to_dict() == []
    assert "No cleaning suggestions" in str(report)


def test_suggest_clean_dataset():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "score": [80.0, 85.0, 90.0, 95.0],
            "status": ["active", "inactive", "active", "active"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    # Clean dataset should have 0 suggestions
    assert len(report) == 0
    assert not report.has_suggestions


def test_suggest_missing_values_and_severities():
    df = pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, None, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],  # 1/10 = 10%
            "abandoned_score": [None] * 10,  # 100% missing
            "sparse_feature": [1.0, None, None, None, None, None, None, None, None, None],  # 90% missing
            "city": ["Mumbai", "Delhi", None, "Bangalore", "Chennai", "Kolkata", "Pune", "Hyderabad", "Goa", "Jaipur"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    # 1. Low missing numeric: fill with median (safe to automate: True)
    age_sugg = report.by_column("age")
    assert len(age_sugg) == 1
    assert age_sugg[0].issue == "missing_values"
    assert age_sugg[0].count == 1
    assert age_sugg[0].suggestion == "fill with median"
    assert age_sugg[0].safe_to_automate is True
    assert age_sugg[0].severity == "low"

    # 2. 100% missing: drop all-null column (safe to automate: True)
    abandoned_sugg = report.by_column("abandoned_score")
    assert len(abandoned_sugg) == 1
    assert abandoned_sugg[0].suggestion == "drop all-null column"
    assert abandoned_sugg[0].safe_to_automate is True
    assert abandoned_sugg[0].severity == "high"

    # 3. High missing rate: review (safe to automate: False)
    sparse_sugg = report.by_column("sparse_feature")
    assert len(sparse_sugg) == 1
    assert sparse_sugg[0].safe_to_automate is False
    assert sparse_sugg[0].severity == "high"

    # 4. Categorical missing: fill with mode or user constant (safe to automate: False)
    city_sugg = report.by_column("city")
    assert len(city_sugg) == 1
    assert "fill with mode" in city_sugg[0].suggestion
    assert city_sugg[0].safe_to_automate is False


def test_suggest_duplicate_rows():
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "val": ["A", "B", "B", "C"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    dup_suggs = report.by_issue("duplicate_rows")
    assert len(dup_suggs) == 1
    assert dup_suggs[0].count == 1
    assert dup_suggs[0].column is None
    assert dup_suggs[0].suggestion == "drop duplicate rows"
    assert dup_suggs[0].safe_to_automate is True


def test_suggest_inconsistent_column_names():
    df = pd.DataFrame(
        {
            " Student Name ": ["Alice", "Bob"],
            "Total Marks (%)": [90, 85],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    col_suggs = report.by_issue("inconsistent_column_name")
    assert len(col_suggs) == 2

    cleaned_names = [s.details["cleaned_name"] for s in col_suggs]
    assert "student_name" in cleaned_names
    assert "total_marks" in cleaned_names
    assert all(s.safe_to_automate is True for s in col_suggs)


def test_suggest_whitespace_problems():
    df = pd.DataFrame(
        {
            "item": [" Apple ", "Banana\u200B", "Cherry   Pie", "Date"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    ws_suggs = report.by_issue("whitespace_problems")
    assert len(ws_suggs) == 1
    assert ws_suggs[0].column == "item"
    assert ws_suggs[0].count == 3  # Apple, Banana, Cherry Pie
    assert ws_suggs[0].safe_to_automate is True


def test_suggest_suspicious_dtypes():
    df = pd.DataFrame(
        {
            "str_numbers": ["10", "20", "30", "40"],
            "str_dates": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    dtype_suggs = report.by_issue("suspicious_dtype")
    assert len(dtype_suggs) == 2

    cols = [s.column for s in dtype_suggs]
    assert "str_numbers" in cols
    assert "str_dates" in cols

    num_sugg = report.by_column("str_numbers")[0]
    assert "convert data type to int" in num_sugg.suggestion
    assert num_sugg.safe_to_automate is False

    date_sugg = report.by_column("str_dates")[0]
    assert "convert data type to datetime" in date_sugg.suggestion
    assert date_sugg.safe_to_automate is False


def test_suggest_potential_outliers():
    # Symmetric normal data with a huge outlier
    df = pd.DataFrame(
        {
            "salary": [50000, 52000, 49000, 51000, 50500, 49500, 50000, 1000000],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    outlier_suggs = report.by_issue("potential_outlier")
    assert len(outlier_suggs) == 1
    assert outlier_suggs[0].column == "salary"
    assert outlier_suggs[0].count == 1
    assert outlier_suggs[0].suggestion == "review manually"
    assert outlier_suggs[0].safe_to_automate is False


def test_suggest_invalid_dates():
    df = pd.DataFrame(
        {
            "event_date": ["2026-01-01", "2026-02-30", "not-a-date", "2026-03-15"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    date_suggs = report.by_issue("invalid_dates")
    assert len(date_suggs) == 1
    assert date_suggs[0].column == "event_date"
    assert date_suggs[0].count == 2
    assert date_suggs[0].severity == "high"


def test_suggest_invalid_email_and_phone():
    df = pd.DataFrame(
        {
            "user_email": ["alice@example.com", "bob@@invalid..com", "carol@domain.org"],
            "contact_phone": ["9876543210", "12345", "+919876543210"],  # "12345" invalid
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    email_suggs = report.by_issue("invalid_email")
    assert len(email_suggs) == 1
    assert email_suggs[0].column == "user_email"
    assert email_suggs[0].count == 1

    phone_suggs = report.by_issue("invalid_phone")
    assert len(phone_suggs) == 1
    assert phone_suggs[0].column == "contact_phone"
    assert phone_suggs[0].count == 1


def test_suggest_category_inconsistencies():
    df = pd.DataFrame(
        {
            "gender": ["Male", "male", "Female", "female", "MALE", "Female"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    cat_suggs = report.by_issue("category_inconsistencies")
    assert len(cat_suggs) == 1
    assert cat_suggs[0].column == "gender"
    assert cat_suggs[0].safe_to_automate is False


def test_suggest_user_rules_range_violations():
    df = pd.DataFrame(
        {
            "marks": [85, 92, 105, -5, 78],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest(rules={"marks": {"min_value": 0, "max_value": 100}})

    range_suggs = report.by_issue("range_violations")
    assert len(range_suggs) == 1
    assert range_suggs[0].column == "marks"
    assert range_suggs[0].count == 2  # 105 and -5


def test_suggest_consistency_results():
    df = pd.DataFrame(
        {
            "start": ["2026-05-01", "2026-01-01"],
            "end": ["2026-01-01", "2026-01-10"],  # row 0 inverted
        }
    )
    cleaner = Cleaner(df)
    c_res = cleaner.check_date_order("start", "end")

    report = cleaner.suggest(consistency_results=[c_res])
    cons_suggs = report.by_issue("logical_inconsistency")
    assert len(cons_suggs) == 1
    assert cons_suggs[0].count == 1


def test_suggestion_report_filtering_and_dataframe_and_dict():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],  # Duplicate rows 0 and 1
            "name": [" Alice ", " Alice ", "Bob"],  # Whitespace
            "val": [10.0, 10.0, None],  # Missing
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.suggest()

    assert len(report) >= 3
    # Filtering
    safe = report.safe_suggestions
    assert len(safe) >= 2
    assert all(s.safe_to_automate for s in safe)

    # DataFrame conversion
    report_df = report.to_dataframe()
    assert isinstance(report_df, pd.DataFrame)
    assert "issue" in report_df.columns
    assert "safe_to_automate" in report_df.columns

    # Dict conversion and access
    s0 = report[0]
    assert isinstance(s0.to_dict(), dict)
    assert s0["issue"] == s0.issue
    assert s0["count"] == s0.count

    # Summary
    assert report.summary["total"] == len(report)
    assert "safe_to_automate_count" in report.summary


def test_suggest_does_not_modify_dataframe():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "col name": [" A ", "B", "C"],
            "age": [20, None, 150],
        }
    )
    orig_copy = df.copy(deep=True)
    cleaner = Cleaner(df)

    # Run suggest
    report = cleaner.suggest()
    assert len(report) > 0

    # Ensure no mutation occurred
    pd.testing.assert_frame_equal(cleaner.data, orig_copy)
    pd.testing.assert_frame_equal(df, orig_copy)
    assert len(cleaner.history) == 0
