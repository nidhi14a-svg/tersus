"""Tests for the Tersus final reporting system."""

import numpy as np
import pandas as pd
import pytest

from tersus import (
    Cleaner,
    CleaningReport,
    DatasetInfo,
    OperationSummary,
    SessionReport,
    generate_session_report,
    report as top_level_report,
)


def test_reporting_on_clean_dataset():
    """Reporting on a clean dataset with no operations performed."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [10.0, 20.0, 30.0],
            "name": ["alice", "bob", "charlie"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.report()

    assert isinstance(report, SessionReport)
    assert isinstance(report, CleaningReport)

    # 1. Dataset information
    assert isinstance(report.dataset_info, DatasetInfo)
    assert report.dataset_info.initial_rows == 3
    assert report.dataset_info.current_rows == 3
    assert report.dataset_info.rows_change == 0
    assert report.dataset_info.initial_columns == 3
    assert report.dataset_info.current_columns == 3
    assert report.dataset_info.columns == ["id", "score", "name"]

    # 2. Issues detected before cleaning
    assert len(report.before_issues) == 0

    # 3. Operations performed
    assert len(report.operations_performed) == 0
    assert report.operation_count == 0

    # 4. Number of affected rows/values
    assert report.total_affected_count == 0

    # 5. Issues remaining
    assert len(report.remaining_issues) == 0
    assert not report.has_remaining_issues

    # 6. Validation status
    assert report.is_valid is True
    assert report.goals_satisfied is True

    # 7. Warnings requiring manual review
    assert len(report.manual_review_warnings) == 0
    assert not report.has_manual_review_warnings


def test_complete_cleaning_session_report():
    """Comprehensive test covering all 7 reporting requirements in a full cleaning workflow."""
    # Create dataset with multiple issues:
    # - 3 missing values in 'age'
    # - 2 duplicate rows (last 3 rows are identical)
    # - Inconsistent column names: ' Student Name '
    # - Whitespace in text
    # - Potential outliers in 'salary'
    # - Invalid emails in 'email'
    data = {
        " Student Name ": [" Alice ", " Bob ", " Charlie ", " David ", " Eve ", " Frank ", " Grace ", " Grace ", " Grace "],
        "age": [25.0, np.nan, 30.0, np.nan, 28.0, np.nan, 35.0, 35.0, 35.0],
        "salary": [50000, 52000, 51000, 49000, 50500, 51500, 999999, 999999, 999999],
        "email": [
            "alice@example.com",
            "bob@example.com",
            "charlie@example.com",
            "david@example.com",
            "eve@example.com",
            "frank@example.com",
            "not_an_email",
            "not_an_email",
            "not_an_email",
        ],
    }
    df = pd.DataFrame(data)
    cleaner = Cleaner(df)

    # Perform cleaning operations
    cleaner.clean(
        column_names=True,
        missing="median",
        duplicates=True,
        text=True,
    )

    report = cleaner.report()

    # Requirement 1: Dataset information
    info = report.dataset_info
    assert info.initial_rows == 9
    # Duplicate rows were dropped (2 duplicates of Grace)
    assert info.current_rows < info.initial_rows
    assert info.rows_change == info.current_rows - info.initial_rows
    assert "student_name" in info.columns  # Column names cleaned
    assert " Student Name " not in info.columns
    assert "age" in info.dtypes

    # Requirement 2: Issues detected before cleaning
    assert isinstance(report.before_issues, dict)
    assert "missing_values" in report.before_issues
    assert report.before_issues["missing_values"] == 3
    assert "duplicate_rows" in report.before_issues
    assert report.before_issues["duplicate_rows"] == 2
    assert "inconsistent_column_name" in report.before_issues

    # Requirement 3: Cleaning operations performed
    assert len(report.operations_performed) >= 3
    op_names = [op.operation for op in report.operations_performed]
    assert "clean_column_names" in op_names
    assert "fill_missing" in op_names
    assert "drop_duplicates" in op_names

    for op in report.operations_performed:
        assert isinstance(op, OperationSummary)
        assert op.step >= 1
        assert isinstance(op.message, str)
        assert len(op.message) > 0

    # Requirement 4: Number of affected rows/values
    assert report.total_affected_count > 0
    # Sum of individual operations matches total
    assert sum(op.affected_count for op in report.operations_performed) == report.total_affected_count

    # Requirement 5: Issues remaining after cleaning
    assert report.has_remaining_issues
    rem_issues = {r.issue for r in report.remaining_issues}
    # Targeted issues (missing, duplicates, column names) should NOT remain
    assert "missing_values" not in rem_issues
    assert "duplicate_rows" not in rem_issues
    assert "inconsistent_column_name" not in rem_issues

    # Untargeted manual review issues (outliers, invalid emails) SHOULD remain
    assert "potential_outlier" in rem_issues
    assert "invalid_email" in rem_issues

    # Requirement 6: Validation status
    assert report.goals_satisfied is True
    assert report.is_valid is True

    # Requirement 7: Warnings requiring manual review
    assert report.has_manual_review_warnings
    warning_issues = {w.issue for w in report.manual_review_warnings}
    assert "potential_outlier" in warning_issues
    assert "invalid_email" in warning_issues
    for warn in report.manual_review_warnings:
        assert warn.requires_manual_review is True


def test_report_human_readable_console_output():
    """Verify human-readable console string formatting for beginner friendliness."""
    df = pd.DataFrame(
        {
            "val": [10.0, np.nan, 30.0],
            "email": ["user@test.com", "bad_email", "admin@site.org"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(missing="median")
    report = cleaner.report()

    text = str(report)

    # All 7 sections must appear in the human-readable text
    assert "1. DATASET OVERVIEW" in text
    assert "2. ISSUES DETECTED BEFORE CLEANING" in text
    assert "3. CLEANING OPERATIONS PERFORMED" in text
    assert "4. ISSUES REMAINING AFTER CLEANING" in text
    assert "5. VALIDATION STATUS" in text
    assert "6. WARNINGS REQUIRING MANUAL REVIEW" in text

    # Friendly indicators
    assert "Missing values" in text
    assert "fill_missing" in text
    assert "Status: PASSED" in text
    assert "require manual human review" in text
    assert "Recommendation:" in text


def test_report_print_helper(capsys):
    """Test report.print() and print_report=True parameter."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    cleaner = Cleaner(df)

    # 1. report.print()
    report = cleaner.report()
    report.print()
    captured = capsys.readouterr()
    assert "TERSUS DATA CLEANING SESSION REPORT" in captured.out

    # 2. cleaner.report(print_report=True)
    cleaner.report(print_report=True)
    captured2 = capsys.readouterr()
    assert "TERSUS DATA CLEANING SESSION REPORT" in captured2.out


def test_report_structured_data_to_dict():
    """Verify that report.to_dict() provides a clean JSON-serializable structure."""
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 3],
            "b": [10.0, 10.0, np.nan, 30.0],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(missing="median", duplicates=True)
    report = cleaner.report()

    d = report.to_dict()
    assert isinstance(d, dict)
    assert "dataset_info" in d
    assert "before_issues" in d
    assert "operations_performed" in d
    assert "total_affected_count" in d
    assert "remaining_issues" in d
    assert "validation" in d
    assert "manual_review_warnings" in d
    assert "is_valid" in d
    assert "goals_satisfied" in d

    # Verify dataset_info dict
    ds_info = d["dataset_info"]
    assert ds_info["initial_rows"] == 4
    assert ds_info["current_rows"] == 3
    assert ds_info["rows_change"] == -1
    assert "a" in ds_info["columns"]
    assert "b" in ds_info["columns"]

    # Verify operations dicts
    assert len(d["operations_performed"]) == 2
    for op in d["operations_performed"]:
        assert "operation" in op
        assert "strategy" in op
        assert "affected_count" in op
        assert "message" in op


def test_report_with_explicit_goals_and_evaluations():
    """Verify reporting when explicit goals are provided and evaluated."""
    df = pd.DataFrame(
        {
            "score": [10, 11, 12, 1000],  # 1 potential outlier
            "code": ["a", "b", "c", "d"],
        }
    )
    cleaner = Cleaner(df)

    # Explicit goal requiring 0 outliers (which will fail)
    report_failed = cleaner.report(goals={"outliers": 0})
    assert report_failed.goals_satisfied is False
    assert report_failed.is_valid is False
    assert "ACTION REQUIRED" in str(report_failed)

    # Goal allowing 1 outlier (which will pass)
    report_passed = cleaner.report(goals={"outliers": 1})
    assert report_passed.goals_satisfied is True
    assert report_passed.is_valid is True
    assert "PASSED" in str(report_passed)


def test_standalone_generate_session_report_with_dataframes():
    """generate_session_report works directly with pandas DataFrames."""
    raw_df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    clean_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    report = generate_session_report(clean_df, initial_df=raw_df, goals=["missing"])
    assert isinstance(report, SessionReport)
    assert report.dataset_info.initial_rows == 3
    assert report.dataset_info.current_rows == 3
    assert report.goals_satisfied is True
    assert "missing_values" in report.before_issues


def test_generate_session_report_type_error():
    """generate_session_report raises TypeError on invalid input."""
    with pytest.raises(TypeError, match="Expected cleaner to be a Cleaner instance or pd.DataFrame"):
        generate_session_report(["invalid_input"])  # type: ignore


def test_top_level_report_export():
    """Verify top-level report function is accessible and works."""
    assert callable(top_level_report)
    df = pd.DataFrame({"x": [1, 2, 3]})
    cleaner = Cleaner(df)
    rep = top_level_report(cleaner)
    assert isinstance(rep, SessionReport)


def test_uncleaned_dataset_reporting():
    """Report on a dataset that has issues, but no cleaning operations were run yet."""
    df = pd.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [10.0, 10.0, np.nan],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.report()

    assert report.operation_count == 0
    assert report.total_affected_count == 0
    assert "missing_values" in report.before_issues
    assert "duplicate_rows" in report.before_issues
    # Both issues still remain
    rem_issue_names = {r.issue for r in report.remaining_issues}
    assert "missing_values" in rem_issue_names
    assert "duplicate_rows" in rem_issue_names
    assert "(No cleaning operations were performed during this session.)" in str(report)
