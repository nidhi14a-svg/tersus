"""Comprehensive tests for Tersus date-related cleaning and validation."""

from datetime import date, datetime
import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, CleaningRecord, DateValidationResult, ValidationResult
from tersus.dates import (
    detect_datetime_columns,
    is_date_like_string,
    is_datetime_column,
    normalize_format_string,
    parse_date_element,
    validate_dates,
)


def test_detect_datetime_columns():
    df = pd.DataFrame(
        {
            "iso_dates": ["2026-10-01", "2026-10-02", "2026-10-03"],
            "slash_dates": ["01/10/2026", "02/10/2026", None],
            "text_dates": ["Oct 1, 2026", "Nov 2, 2026", "Dec 3, 2026"],
            "names": ["Alice", "Bob", "Charlie"],
            "scores": [95, 80, 88],
            "is_active": [True, False, True],
            "all_null": [None, np.nan, None],
        }
    )
    cleaner = Cleaner(df)

    detected = cleaner.detect_datetime_columns()
    assert "iso_dates" in detected
    assert "slash_dates" in detected
    assert "text_dates" in detected
    assert "names" not in detected
    assert "scores" not in detected
    assert "is_active" not in detected
    assert "all_null" not in detected

    assert cleaner.is_datetime_column("iso_dates") is True
    assert cleaner.is_datetime_column("slash_dates") is True
    assert cleaner.is_datetime_column("text_dates") is True
    assert cleaner.is_datetime_column("names") is False
    assert cleaner.is_datetime_column("scores") is False


def test_valid_dates_various_formats():
    df = pd.DataFrame(
        {
            "event": ["Conference", "Workshop", "Meetup", "Webinar", "Summit"],
            "date": [
                "2026-10-01",           # ISO
                "2026/10/02",           # ISO slash
                "Oct 3, 2026",          # Named month short
                "October 4, 2026",      # Named month full
                "15/10/2026",           # Unambiguous dayfirst (15 > 12)
            ],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_dates("date")

    assert isinstance(result, DateValidationResult)
    assert isinstance(result, ValidationResult)
    assert result.is_valid is True
    assert result.has_violations is False
    assert result.invalid_count == 0
    assert result.invalid_indices == []
    assert result.invalid_values == []
    assert len(result.invalid_rows) == 0


def test_specification_ambiguous_dates_flagged_without_configuration():
    # 01/10/2026 could be Oct 1 (DD/MM/YYYY) or Jan 10 (MM/DD/YYYY)
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date_col": ["01/10/2026", "05/06/2024", "11/12/2025"],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_dates("date_col")

    # Ambiguous dates must be flagged when user has not configured dayfirst or format
    assert result.is_valid is False
    assert result.has_violations is True
    assert result.invalid_count == 3
    assert result.invalid_indices == [0, 1, 2]
    assert result.ambiguous_indices == [0, 1, 2]
    assert result.ambiguous_values == ["01/10/2026", "05/06/2024", "11/12/2025"]
    assert "ambiguous" in result.reasons[0].lower()


def test_ambiguous_dates_resolved_with_dayfirst():
    df = pd.DataFrame(
        {
            "date_col": ["01/10/2026", "05/06/2024"],
        }
    )
    cleaner = Cleaner(df)

    # When dayfirst=True is configured by user:
    res_dayfirst = cleaner.check_dates("date_col", dayfirst=True)
    assert res_dayfirst.is_valid is True
    assert res_dayfirst.invalid_count == 0

    # Parse to datetime with dayfirst=True:
    cleaner.parse_dates("date_col", dayfirst=True)
    assert cleaner.data["date_col"].iloc[0] == pd.Timestamp("2026-10-01")
    assert cleaner.data["date_col"].iloc[1] == pd.Timestamp("2024-06-05")


def test_ambiguous_dates_resolved_with_monthfirst():
    df = pd.DataFrame(
        {
            "date_col": ["01/10/2026", "05/06/2024"],
        }
    )
    cleaner = Cleaner(df)

    # When dayfirst=False (month first) is configured:
    res_monthfirst = cleaner.check_dates("date_col", dayfirst=False)
    assert res_monthfirst.is_valid is True
    assert res_monthfirst.invalid_count == 0

    cleaner.parse_dates("date_col", dayfirst=False)
    assert cleaner.data["date_col"].iloc[0] == pd.Timestamp("2026-01-10")
    assert cleaner.data["date_col"].iloc[1] == pd.Timestamp("2024-05-06")


def test_specification_mixed_formats_example():
    # Prompt example:
    # 01/10/2026
    # 2026-10-01
    # Oct 1, 2026
    df = pd.DataFrame(
        {
            "record": [1, 2, 3],
            "joining_date": [
                "01/10/2026",
                "2026-10-01",
                "Oct 1, 2026",
            ],
        }
    )
    cleaner = Cleaner(df)

    # Without dayfirst: row 0 is flagged as ambiguous, but rows 1 and 2 are valid
    unconfigured_res = cleaner.check_dates("joining_date")
    assert unconfigured_res.is_valid is False
    assert unconfigured_res.invalid_count == 1
    assert unconfigured_res.invalid_indices == [0]
    assert unconfigured_res.invalid_values == ["01/10/2026"]
    assert unconfigured_res.ambiguous_values == ["01/10/2026"]

    # With dayfirst=True: all 3 resolve to the same date: 2026-10-01
    configured_res = cleaner.check_dates("joining_date", dayfirst=True)
    assert configured_res.is_valid is True
    assert configured_res.invalid_count == 0

    # Standardize all 3 to ISO 'YYYY-MM-DD'
    cleaner.standardize_dates("joining_date", output_format="YYYY-MM-DD", dayfirst=True)
    expected = ["2026-10-01", "2026-10-01", "2026-10-01"]
    assert cleaner.data["joining_date"].tolist() == expected

    # Convert to datetime
    cleaner.to_datetime("joining_date")
    assert pd.api.types.is_datetime64_any_dtype(cleaner.data["joining_date"])
    for val in cleaner.data["joining_date"]:
        assert val == pd.Timestamp("2026-10-01")


def test_invalid_dates_detection_and_reporting():
    df = pd.DataFrame(
        {
            "user": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "date": [
                "2026-10-01",       # Valid
                "invalid_date",     # Unparseable text
                "31/02/2026",       # February 31st does not exist
                "2026-13-01",       # Month 13 does not exist
                "2026-10-05",       # Valid
            ],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_dates("date", dayfirst=True)

    assert result.is_valid is False
    assert result.has_violations is True
    assert result.invalid_count == 3
    assert result.invalid_indices == [1, 2, 3]
    assert result.invalid_values == ["invalid_date", "31/02/2026", "2026-13-01"]
    assert result.unparseable_indices == [1, 2, 3]

    # Verify invalid_rows DataFrame
    assert len(result.invalid_rows) == 3
    assert result.invalid_rows["user"].tolist() == ["Bob", "Charlie", "David"]

    # Verify original DataFrame was NOT modified
    assert len(cleaner.data) == 5
    assert cleaner.data["date"].iloc[1] == "invalid_date"


def test_missing_dates_handled_safely():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "date": [
                "2026-10-01",
                None,
                np.nan,
                pd.NA,
                "",
                "   ",
                "nat",
            ],
        }
    )
    cleaner = Cleaner(df)

    # Missing dates should NOT be flagged as format violations when allow_na=True (default)
    result = cleaner.check_dates("date")
    assert result.is_valid is True
    assert result.invalid_count == 0

    # When allow_na=False, missing values are flagged
    strict_res = cleaner.check_dates("date", allow_na=False)
    assert strict_res.is_valid is False
    assert strict_res.invalid_count == 6
    assert strict_res.invalid_indices == [1, 2, 3, 4, 5, 6]

    # Standardize dates: missing dates remain missing safely
    cleaner.standardize_dates("date", output_format="YYYY-MM-DD")
    assert cleaner.data["date"].iloc[0] == "2026-10-01"
    for i in range(1, 7):
        assert pd.isna(cleaner.data["date"].iloc[i])

    # Convert to datetime: missing dates become NaT
    cleaner.to_datetime("date")
    assert cleaner.data["date"].iloc[0] == pd.Timestamp("2026-10-01")
    for i in range(1, 7):
        assert pd.isna(cleaner.data["date"].iloc[i])


def test_user_choice_drop_invalid_dates():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "date": ["2026-10-01", "not_a_date", None, "2026-10-04"],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_dates("date")

    assert result.invalid_count == 1
    assert result.invalid_indices == [1]

    # User chooses to drop invalid rows
    cleaner.drop_invalid(result)
    assert len(cleaner.data) == 3
    assert cleaner.data["id"].tolist() == [1, 3, 4]
    assert cleaner.data["date"].iloc[0] == "2026-10-01"
    assert pd.isna(cleaner.data["date"].iloc[1])
    assert cleaner.data["date"].iloc[2] == "2026-10-04"


def test_user_choice_replace_invalid_dates():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2026-10-01", "bad_date", "2026-10-03"],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_dates("date")

    cleaner.replace_invalid(result, "2026-01-01")
    assert len(cleaner.data) == 3
    assert cleaner.data["date"].tolist() == ["2026-10-01", "2026-01-01", "2026-10-03"]

    # Now all dates are valid
    assert cleaner.check_dates("date").is_valid is True


def test_parse_dates_error_modes():
    df = pd.DataFrame(
        {
            "date": ["2026-10-01", "invalid_date", "2026-10-03"],
        }
    )

    # 1. errors="raise" (default) raises ValueError
    cleaner1 = Cleaner(df)
    with pytest.raises(ValueError, match="Invalid date detected at row 1"):
        cleaner1.parse_dates("date", errors="raise")

    # 2. errors="coerce" converts invalid to NaT
    cleaner2 = Cleaner(df)
    cleaner2.parse_dates("date", errors="coerce")
    assert cleaner2.data["date"].iloc[0] == pd.Timestamp("2026-10-01")
    assert pd.isna(cleaner2.data["date"].iloc[1])
    assert cleaner2.data["date"].iloc[2] == pd.Timestamp("2026-10-03")

    # 3. errors="ignore" leaves column unchanged
    cleaner3 = Cleaner(df)
    cleaner3.parse_dates("date", errors="ignore")
    assert cleaner3.data["date"].iloc[1] == "invalid_date"


def test_parse_dates_ambiguous_error_mode():
    df = pd.DataFrame({"date": ["01/10/2026"]})
    cleaner = Cleaner(df)

    # Unconfigured ambiguous date with errors="raise" must raise ValueError
    with pytest.raises(ValueError, match="Ambiguous date detected at row 0"):
        cleaner.parse_dates("date", errors="raise")

    # With errors="coerce": turns into NaT
    cleaner_coerce = Cleaner(df)
    cleaner_coerce.parse_dates("date", errors="coerce")
    assert pd.isna(cleaner_coerce.data["date"].iloc[0])


def test_standardize_dates_formats_and_aliases():
    df = pd.DataFrame(
        {
            "date": ["2026-10-01", "2026-10-02"],
        }
    )
    cleaner = Cleaner(df)

    # Shorthand alias DD/MM/YYYY
    cleaner.standardize_dates("date", output_format="DD/MM/YYYY")
    assert cleaner.data["date"].tolist() == ["01/10/2026", "02/10/2026"]

    # Shorthand alias MM-DD-YYYY
    cleaner.standardize_dates("date", output_format="MM-DD-YYYY", input_format="%d/%m/%Y")
    assert cleaner.data["date"].tolist() == ["10-01-2026", "10-02-2026"]

    # Standard strftime format '%B %d, %Y'
    cleaner.standardize_dates("date", output_format="%B %d, %Y", input_format="%m-%d-%Y")
    assert cleaner.data["date"].tolist() == ["October 01, 2026", "October 02, 2026"]


def test_standardize_dates_with_multiple_columns():
    df = pd.DataFrame(
        {
            "start": ["2026-01-01", "2026-02-01"],
            "end": ["2026-01-15", "2026-02-15"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.standardize_dates(["start", "end"], output_format="DD/MM/YYYY")

    assert cleaner.data["start"].tolist() == ["01/01/2026", "01/02/2026"]
    assert cleaner.data["end"].tolist() == ["15/01/2026", "15/02/2026"]
    assert len(cleaner.history) == 2


def test_existing_datetime_objects_handling():
    df = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2026-10-01 10:30:00"), None],
            "d": [date(2026, 10, 1), None],
            "dt": [datetime(2026, 10, 1, 12, 0), None],
        }
    )
    cleaner = Cleaner(df)

    # Check dates should pass cleanly for native datetime objects
    assert cleaner.check_dates("ts").is_valid is True
    assert cleaner.check_dates("d").is_valid is True
    assert cleaner.check_dates("dt").is_valid is True

    # Standardize to format
    cleaner.standardize_dates("ts", output_format="YYYY-MM-DD")
    assert cleaner.data["ts"].iloc[0] == "2026-10-01"
    assert pd.isna(cleaner.data["ts"].iloc[1])


def test_original_dataframe_not_mutated():
    original_df = pd.DataFrame(
        {
            "date": ["01/10/2026", "2026-10-01"],
        }
    )
    original_copy = original_df.copy(deep=True)
    cleaner = Cleaner(original_df)

    # Check dates does not mutate
    cleaner.check_dates("date")
    pd.testing.assert_frame_equal(original_df, original_copy)

    # Standardize dates does not mutate original input df
    cleaner.standardize_dates("date", output_format="YYYY-MM-DD", dayfirst=True)
    pd.testing.assert_frame_equal(original_df, original_copy)


def test_history_recording():
    df = pd.DataFrame({"date": ["2026-10-01", "2026-10-02"]})
    cleaner = Cleaner(df)

    cleaner.standardize_dates("date", output_format="DD/MM/YYYY")
    assert len(cleaner.history) == 1
    rec1 = cleaner.last_operation
    assert isinstance(rec1, CleaningRecord)
    assert rec1.operation == "standardize_dates"
    assert rec1.columns == ["date"]
    assert rec1.affected_count == 2

    cleaner.parse_dates("date", format="%d/%m/%Y")
    assert len(cleaner.history) == 2
    rec2 = cleaner.last_operation
    assert rec2.operation == "parse_dates"
    assert rec2.strategy == "datetime_conversion"


def test_edge_cases_empty_dataframe_and_missing_column():
    empty_df = pd.DataFrame({"date": []})
    cleaner = Cleaner(empty_df)

    # Empty column validation
    res = cleaner.check_dates("date")
    assert res.is_valid is True
    assert res.invalid_count == 0

    cleaner.standardize_dates("date")
    assert len(cleaner.data) == 0

    cleaner.parse_dates("date")
    assert len(cleaner.data) == 0

    # Missing column
    with pytest.raises(KeyError, match="not found in dataset"):
        cleaner.check_dates("non_existent_col")

    with pytest.raises(KeyError, match="not found in dataset"):
        cleaner.is_datetime_column("non_existent_col")


def test_validation_result_details_and_str():
    df = pd.DataFrame(
        {
            "date": ["01/10/2026", "invalid_val"],
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_dates("date")

    d = res.to_dict()
    assert d["column"] == "date"
    assert d["is_valid"] is False
    assert d["invalid_count"] == 2
    assert d["ambiguous_indices"] == [0]
    assert d["unparseable_indices"] == [1]

    s = str(res)
    assert "FAILED" in s
    assert "Ambiguous dates" in s
    assert "Unparseable" in s
