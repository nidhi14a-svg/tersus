"""Tests for Tersus post-cleaning validation and goal verification module."""

import numpy as np
import pandas as pd
import pytest

from tersus import (
    Cleaner,
    RemainingIssue,
    ResolvedIssue,
    ValidationReport,
    validate,
    validate_cleaning,
)


def test_clean_dataset_has_no_issues():
    """A clean dataset with no quality problems should immediately pass validation."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "score": [10.0, 20.0, 30.0, 40.0, 50.0],
            "name": ["alice", "bob", "charlie", "david", "eve"],
        }
    )
    cleaner = Cleaner(df)
    report = cleaner.validate()

    assert isinstance(report, ValidationReport)
    assert report.is_valid is True
    assert report.goals_satisfied is True
    assert bool(report) is True
    assert len(report.resolved_issues) == 0
    assert len(report.remaining_issues) == 0
    assert len(report.manual_review_issues) == 0
    assert not report.has_remaining_issues
    assert not report.has_manual_review_issues


def test_validation_does_not_modify_dataframe():
    """validate() must be strictly non-destructive and not modify DataFrame or history."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "age": [25.0, np.nan, np.nan, 30.0],
            "email": ["user@test.com", "not_an_email", "invalid@", "admin@site.org"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(missing="median", duplicates=True)

    df_before_val = cleaner.data.copy()
    history_len_before = len(cleaner.history)

    report = cleaner.validate()

    pd.testing.assert_frame_equal(cleaner.data, df_before_val)
    assert len(cleaner.history) == history_len_before
    assert isinstance(report, ValidationReport)


def test_validate_distinguishes_resolved_remaining_and_manual_review():
    """Test resolution of missing and duplicate rows while distinguishing manual review issues.

    Matches user requirement:
    Before: Missing values, Duplicates
    After: Missing values: 0, Duplicates: 0
    Remaining: Potential outliers, Invalid emails (manual review)
    """
    # Create dataset with:
    # - 4 missing values in age
    # - 2 duplicate rows
    # - potential outliers in salary
    # - 2 invalid emails
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12],
        "age": [20.0, 22.0, np.nan, 24.0, 25.0, np.nan, 27.0, 28.0, np.nan, 30.0, np.nan, 32.0, 32.0, 32.0],
        # Salary distribution with clear IQR outliers:
        # median around 50k, 10k-60k range, and extreme values 999999, 888888, 777777
        "salary": [50000, 52000, 51000, 53000, 49000, 50500, 51500, 49500, 52500, 999999, 888888, 777777, 777777, 777777],
        "email": [
            "user1@example.com",
            "user2@example.com",
            "user3@example.com",
            "user4@example.com",
            "user5@example.com",
            "user6@example.com",
            "user7@example.com",
            "user8@example.com",
            "user9@example.com",
            "bad-email-1",
            "bad-email-2",
            "user12@example.com",
            "user12@example.com",
            "user12@example.com",
        ],
    }
    df = pd.DataFrame(data)
    cleaner = Cleaner(df)

    # Clean only missing values and duplicate rows
    cleaner.clean(missing="median", duplicates=True)

    report = cleaner.validate()

    # 1. Before vs After summaries
    assert "missing_values" in report.before_summary
    assert report.before_summary["missing_values"] > 0
    assert report.after_summary["missing_values"] == 0

    assert "duplicate_rows" in report.before_summary
    assert report.before_summary["duplicate_rows"] > 0
    assert report.after_summary["duplicate_rows"] == 0

    # 2. Issues resolved
    assert report.has_resolved_issues
    resolved_issue_names = {r.issue for r in report.resolved_issues}
    assert "missing_values" in resolved_issue_names
    assert "duplicate_rows" in resolved_issue_names

    # Check specific resolved record
    missing_resolved = report.resolved_by_issue("missing_values")
    assert len(missing_resolved) > 0
    assert missing_resolved[0].resolved_count > 0
    assert missing_resolved[0].after_count == 0

    # 3. Issues remaining
    assert report.has_remaining_issues
    remaining_issue_names = {r.issue for r in report.remaining_issues}
    assert "potential_outlier" in remaining_issue_names
    assert "invalid_email" in remaining_issue_names

    # Missing and duplicates should NOT be in remaining
    assert "missing_values" not in remaining_issue_names
    assert "duplicate_rows" not in remaining_issue_names

    # 4. Issues requiring manual review
    assert report.has_manual_review_issues
    manual_issue_names = {m.issue for m in report.manual_review_issues}
    assert "potential_outlier" in manual_issue_names
    assert "invalid_email" in manual_issue_names

    for m in report.manual_review_issues:
        assert m.requires_manual_review is True

    # 5. Goals satisfied (inferred from clean() history: missing and duplicates)
    assert report.goals_satisfied is True
    assert report.is_valid is True


def test_explicit_goals_satisfied_and_failed():
    """Explicitly specified goals must be evaluated strictly."""
    df = pd.DataFrame(
        {
            "val": [10.0, np.nan, 20.0, np.nan, 30.0],
            "code": ["bad_email", "another_bad", "test@test.com", "user@test.com", "valid@test.com"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(missing="median")

    # Missing goal satisfied
    report_pass = cleaner.validate(goals=["missing"])
    assert report_pass.goals_satisfied is True
    assert report_pass.is_valid is True

    # Emails goal was not resolved, so goals_satisfied should be False
    report_fail = cleaner.validate(goals=["missing", "emails"])
    assert report_fail.goals_satisfied is False
    assert report_fail.is_valid is False
    assert report_fail.details["goal_evaluations"]["emails"]["passed"] is False


def test_dictionary_goals_with_thresholds():
    """Goals specified as dict with target thresholds."""
    df = pd.DataFrame(
        {
            "score": [10, 11, 12, 13, 14, 15, 1000],  # 1 potential outlier
            "name": ["A", "B", "C", "D", "E", "F", "G"],
        }
    )
    cleaner = Cleaner(df)

    # Threshold allowing up to 2 outliers
    report_allow = cleaner.validate(goals={"outliers": 2})
    assert report_allow.goals_satisfied is True

    # Threshold requiring 0 outliers
    report_strict = cleaner.validate(goals={"outliers": 0})
    assert report_strict.goals_satisfied is False


def test_column_rename_tracking_in_validation():
    """Renaming columns should not cause missing values to be falsely marked as resolved."""
    df = pd.DataFrame(
        {
            " Student Name ": [" Alice ", " Bob ", np.nan, " Charlie "],
            " Score ": [90.0, np.nan, 80.0, 70.0],
        }
    )
    cleaner = Cleaner(df)
    # Only clean column names, do NOT fill missing values
    cleaner.clean(column_names=True)

    report = cleaner.validate()

    # Column names issue was resolved
    resolved_names = {r.issue for r in report.resolved_issues}
    assert "inconsistent_column_name" in resolved_names

    # Missing values should NOT be marked resolved because they were not cleaned
    assert "missing_values" not in resolved_names
    remaining_names = {r.issue for r in report.remaining_issues}
    assert "missing_values" in remaining_names


def test_validation_report_summary_and_str():
    """Test string representations and summary output."""
    df = pd.DataFrame(
        {
            "age": [20.0, np.nan, 25.0],
            "email": ["not_an_email", "a@b.com", "c@d.com"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(missing="median")
    report = cleaner.validate()

    summary_str = report.summary()
    assert "Before:" in summary_str
    assert "After:" in summary_str
    assert "Remaining:" in summary_str
    assert "Missing values" in summary_str
    assert "Invalid email" in summary_str

    full_str = str(report)
    assert "Tersus Cleaning Validation Report:" in full_str
    assert "Issues Resolved" in full_str
    assert "Issues Requiring Manual Review" in full_str


def test_validation_report_to_dict():
    """ValidationReport.to_dict() returns structured dict."""
    df = pd.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [np.nan, 2.0, 3.0],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean(missing="median", duplicates=True)
    report = cleaner.validate()

    d = report.to_dict()
    assert isinstance(d, dict)
    assert "is_valid" in d
    assert "goals_satisfied" in d
    assert "before_summary" in d
    assert "after_summary" in d
    assert "resolved_issues" in d
    assert "remaining_issues" in d
    assert "manual_review_issues" in d

    for item in d["resolved_issues"]:
        assert "issue" in item
        assert "resolved_count" in item


def test_standalone_validate_cleaning_with_dataframes():
    """validate_cleaning should work directly with DataFrames."""
    raw_df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]})
    cleaned_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    report = validate_cleaning(cleaned_df, initial_df=raw_df, goals=["missing"])
    assert report.goals_satisfied is True
    assert report.is_valid is True
    assert len(report.resolved_issues) == 1
    assert report.resolved_issues[0].issue == "missing_values"
    assert report.resolved_issues[0].resolved_count == 1


def test_standalone_validate_type_error():
    """validate_cleaning raises TypeError for invalid input."""
    with pytest.raises(TypeError, match="Expected cleaner to be a Cleaner instance or pd.DataFrame"):
        validate_cleaning("not_a_cleaner_or_dataframe")  # type: ignore


def test_top_level_validate_import():
    """tersus.validate and tersus.validate_cleaning can be imported directly."""
    assert callable(validate)
    assert callable(validate_cleaning)


def test_validate_with_custom_rules():
    """validate() accepts custom validation rules for ranges."""
    df = pd.DataFrame({"score": [10, 50, 150]})  # 150 > 100
    cleaner = Cleaner(df)

    rules = {"score": {"min_value": 0, "max_value": 100}}
    report = cleaner.validate(rules=rules)

    remaining_issues = report.remaining_by_issue("range_violations")
    assert len(remaining_issues) == 1
    assert remaining_issues[0].count == 1
    assert remaining_issues[0].requires_manual_review is True
