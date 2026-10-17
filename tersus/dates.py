"""Date-related cleaning and validation module for Tersus."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from dateutil import parser
import numpy as np
import pandas as pd

from tersus.missing import CleaningRecord
from tersus.validation import ValidationResult

NAMED_MONTHS = {
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

FORMAT_ALIASES = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYY/MM/DD": "%Y/%m/%d",
    "DD/MM/YYYY": "%d/%m/%Y",
    "MM/DD/YYYY": "%m/%d/%Y",
    "DD-MM-YYYY": "%d-%m-%Y",
    "MM-DD-YYYY": "%m-%d-%Y",
    "DD.MM.YYYY": "%d.%m.%Y",
    "MM.DD.YYYY": "%m.%d.%Y",
    "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
    "YYYYMMDD": "%Y%m%d",
}


def normalize_format_string(fmt: Optional[str]) -> Optional[str]:
    """Convert common date format shorthand (e.g. 'YYYY-MM-DD') to strftime format.

    Args:
        fmt: Date format string or None.

    Returns:
        Standard strftime format string, or None if input was None.
    """
    if fmt is None:
        return None
    upper_fmt = fmt.strip().upper()
    return FORMAT_ALIASES.get(upper_fmt, fmt)


@dataclass
class DateValidationResult(ValidationResult):
    """Structured result of date validation check."""

    ambiguous_indices: List[Any] = field(default_factory=list)
    ambiguous_values: List[Any] = field(default_factory=list)
    unparseable_indices: List[Any] = field(default_factory=list)
    unparseable_values: List[Any] = field(default_factory=list)
    reasons: Dict[Any, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        base = super().to_dict()
        base.update(
            {
                "ambiguous_indices": list(self.ambiguous_indices),
                "ambiguous_values": list(self.ambiguous_values),
                "unparseable_indices": list(self.unparseable_indices),
                "unparseable_values": list(self.unparseable_values),
                "reasons": dict(self.reasons),
            }
        )
        return base

    def __str__(self) -> str:
        status = (
            "PASSED"
            if self.is_valid
            else f"FAILED ({self.invalid_count} invalid date(s) found)"
        )
        lines = [
            f"Date Validation Check on '{self.column}':",
            f"  Rule: {self.rule}",
            f"  Status: {status}",
        ]
        if self.ambiguous_indices:
            lines.append(f"  Ambiguous dates ({len(self.ambiguous_indices)}):")
            for idx, val in zip(
                self.ambiguous_indices[:5], self.ambiguous_values[:5]
            ):
                lines.append(f"    - Row {idx}: {val!r}")
            if len(self.ambiguous_indices) > 5:
                lines.append(f"    ... and {len(self.ambiguous_indices) - 5} more")

        if self.unparseable_indices:
            lines.append(f"  Unparseable/Invalid dates ({len(self.unparseable_indices)}):")
            for idx, val in zip(
                self.unparseable_indices[:5], self.unparseable_values[:5]
            ):
                reason = self.reasons.get(idx, "")
                lines.append(f"    - Row {idx}: {val!r} ({reason})")
            if len(self.unparseable_indices) > 5:
                lines.append(
                    f"    ... and {len(self.unparseable_indices) - 5} more"
                )

        return "\n".join(lines)


def is_date_like_string(val: Any) -> bool:
    """Check whether a single value has common date-like structure.

    Args:
        val: Any value to inspect.

    Returns:
        bool: True if the value appears to represent a date, False otherwise.
    """
    if val is None or pd.isna(val):
        return False
    if isinstance(val, (datetime, pd.Timestamp, np.datetime64)):
        return True

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat", "null", "<na>"):
        return False

    # ISO pattern: YYYY-MM-DD or YYYY/MM/DD
    if re.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", s):
        return True

    # Delimited: DD/MM/YYYY, MM/DD/YYYY, etc.
    if re.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", s):
        return True

    # Named month with digits: "Oct 1, 2026", "1 October 2026", etc.
    words = re.findall(r"[a-zA-Z]+", s.lower())
    has_month = any(
        w in NAMED_MONTHS or any(w.startswith(m) for m in NAMED_MONTHS)
        for w in words
    )
    has_digits = bool(re.search(r"\d+", s))
    if has_month and has_digits:
        return True

    # 8-digit compact: YYYYMMDD
    if re.match(r"^\d{8}$", s):
        year = int(s[:4])
        month = int(s[4:6])
        day = int(s[6:8])
        if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            return True

    return False


def is_datetime_column(series: pd.Series, threshold: float = 0.5) -> bool:
    """Determine if a Pandas Series contains datetime or date-like values.

    Args:
        series: Pandas Series to evaluate.
        threshold: Minimum fraction of non-empty values that must look like dates
            (default 0.5).

    Returns:
        bool: True if the series is considered datetime-like.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Pure numeric or boolean columns should not be detected as datetime
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False

    non_null = series.dropna()
    clean_values = [v for v in non_null if str(v).strip() not in ("", "nan", "none", "nat", "null", "<na>")]
    if not clean_values:
        return False

    date_like_count = sum(1 for v in clean_values if is_date_like_string(v))
    ratio = date_like_count / len(clean_values)
    return ratio >= threshold


def detect_datetime_columns(
    df: pd.DataFrame, threshold: float = 0.5
) -> List[str]:
    """Identify all columns in a DataFrame that represent datetime-like data.

    Args:
        df: Pandas DataFrame to inspect.
        threshold: Proportion threshold for detecting date strings (default 0.5).

    Returns:
        List[str]: Column names identified as datetime-like.
    """
    return [col for col in df.columns if is_datetime_column(df[col], threshold=threshold)]


def parse_date_element(
    val: Any,
    format: Optional[str] = None,
    dayfirst: Optional[bool] = None,
    allow_na: bool = True,
) -> Tuple[Optional[pd.Timestamp], Optional[str], bool]:
    """Parse a single value into a pandas Timestamp, detecting ambiguity and errors.

    Args:
        val: The value to parse.
        format: Optional explicit strftime format or shorthand alias.
        dayfirst: Optional boolean indicating whether day precedes month.
            If None, ambiguous date formats (e.g. 01/10/2026) are flagged.
        allow_na: Whether missing values (None, NaN, empty strings) are allowed.

    Returns:
        Tuple of (timestamp, error_message, is_ambiguous):
            - timestamp: pd.Timestamp if successfully parsed, None otherwise.
            - error_message: Description of error or ambiguity, or None if valid.
            - is_ambiguous: True if the failure was specifically due to date ambiguity.
    """
    if val is None or val is pd.NA:
        return (None, None, False) if allow_na else (None, "missing date value", False)
    if pd.isna(val):
        return (None, None, False) if allow_na else (None, "missing date value", False)

    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.Timestamp(val), None, False

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat", "null", "<na>"):
        return (None, None, False) if allow_na else (None, "missing date value", False)

    norm_format = normalize_format_string(format)

    # 1. User provided an explicit format
    if norm_format is not None:
        try:
            dt = datetime.strptime(s, norm_format)
            return pd.Timestamp(dt), None, False
        except Exception as exc:
            return None, f"does not match format '{format}': {exc}", False

    # 2. ISO 8601: YYYY-MM-DD (or YYYY/MM/DD) with optional time
    iso_match = re.match(
        r"^\s*(\d{4})[-/. ](\d{1,2})[-/. ](\d{1,2})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?\s*$",
        s,
    )
    if iso_match:
        y = int(iso_match.group(1))
        m = int(iso_match.group(2))
        d = int(iso_match.group(3))
        h = int(iso_match.group(4) or 0)
        mi = int(iso_match.group(5) or 0)
        sec = int(iso_match.group(6) or 0)
        try:
            return pd.Timestamp(datetime(y, m, d, h, mi, sec)), None, False
        except ValueError as exc:
            return None, f"invalid calendar date: {exc}", False

    # 3. Named month (e.g. "Oct 1, 2026", "October 1, 2026", "1 Oct 2026")
    words = re.findall(r"[a-zA-Z]+", s.lower())
    has_month = any(
        w in NAMED_MONTHS or any(w.startswith(m) for m in NAMED_MONTHS)
        for w in words
    )
    has_digits = bool(re.search(r"\d+", s))
    if has_month and has_digits:
        try:
            dt = parser.parse(s, dayfirst=dayfirst if dayfirst is not None else False)
            return pd.Timestamp(dt), None, False
        except Exception as exc:
            return None, f"invalid date: {exc}", False

    # 4. Delimited numerical date: DD/MM/YYYY or MM/DD/YYYY
    delim_match = re.match(
        r"^\s*(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?\s*$",
        s,
    )
    if delim_match:
        p1 = int(delim_match.group(1))
        p2 = int(delim_match.group(2))
        p3 = int(delim_match.group(3))
        h = int(delim_match.group(4) or 0)
        mi = int(delim_match.group(5) or 0)
        sec = int(delim_match.group(6) or 0)
        year = p3 + 2000 if p3 < 100 else p3

        if p1 <= 12 and p2 <= 12 and p1 != p2:
            if dayfirst is True:
                day, month = p1, p2
            elif dayfirst is False:
                month, day = p1, p2
            else:
                return (
                    None,
                    (
                        f"ambiguous date format '{s}': cannot determine whether {p1} or {p2} "
                        f"is day/month without user configuration (specify dayfirst=True/False or format)"
                    ),
                    True,
                )
        elif p1 > 12 and p2 <= 12:
            day, month = p1, p2
        elif p2 > 12 and p1 <= 12:
            month, day = p1, p2
        elif p1 == p2 and 1 <= p1 <= 12:
            day, month = p1, p2
        else:
            return None, f"invalid calendar values in date '{s}': ({p1}, {p2})", False

        try:
            return pd.Timestamp(datetime(year, month, day, h, mi, sec)), None, False
        except ValueError as exc:
            return None, f"invalid calendar date: {exc}", False

    # 5. 8-digit compact: YYYYMMDD
    compact_match = re.match(r"^\s*(\d{4})(\d{2})(\d{2})\s*$", s)
    if compact_match:
        y = int(compact_match.group(1))
        m = int(compact_match.group(2))
        d = int(compact_match.group(3))
        try:
            return pd.Timestamp(datetime(y, m, d)), None, False
        except ValueError as exc:
            return None, f"invalid calendar date: {exc}", False

    # 6. Fallback parser when dayfirst is explicitly configured
    if dayfirst is not None:
        try:
            dt = parser.parse(s, dayfirst=dayfirst)
            return pd.Timestamp(dt), None, False
        except Exception as exc:
            return None, f"invalid date: {exc}", False

    return None, f"unrecognized or unparseable date format: '{s}'", False


def validate_dates(
    df: pd.DataFrame,
    column: str,
    format: Optional[str] = None,
    dayfirst: Optional[bool] = None,
    allow_na: bool = True,
) -> DateValidationResult:
    """Validate dates in a DataFrame column, identifying invalid and ambiguous values.

    Args:
        df: Pandas DataFrame.
        column: Column name to validate.
        format: Optional explicit strftime format or alias (e.g. 'YYYY-MM-DD').
        dayfirst: Optional boolean indicating whether day precedes month.
            When None, ambiguous dates such as '01/10/2026' are flagged.
        allow_na: Whether missing values are considered valid (default True).

    Returns:
        DateValidationResult: Structured validation result.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    series = df[column]
    invalid_indices: List[Any] = []
    invalid_values: List[Any] = []
    ambiguous_indices: List[Any] = []
    ambiguous_values: List[Any] = []
    unparseable_indices: List[Any] = []
    unparseable_values: List[Any] = []
    reasons: Dict[Any, str] = {}

    rule_parts = ["valid_dates"]
    if format:
        rule_parts.append(f"format='{format}'")
    if dayfirst is not None:
        rule_parts.append(f"dayfirst={dayfirst}")
    rule_name = ", ".join(rule_parts)

    for idx, val in series.items():
        _, err_msg, is_ambiguous = parse_date_element(
            val, format=format, dayfirst=dayfirst, allow_na=allow_na
        )
        if err_msg is not None:
            invalid_indices.append(idx)
            invalid_values.append(val)
            reasons[idx] = err_msg
            if is_ambiguous:
                ambiguous_indices.append(idx)
                ambiguous_values.append(val)
            else:
                unparseable_indices.append(idx)
                unparseable_values.append(val)

    invalid_count = len(invalid_indices)
    invalid_rows = df.loc[invalid_indices].copy()

    return DateValidationResult(
        column=column,
        rule=rule_name,
        is_valid=(invalid_count == 0),
        invalid_count=invalid_count,
        invalid_indices=invalid_indices,
        invalid_values=invalid_values,
        invalid_rows=invalid_rows,
        ambiguous_indices=ambiguous_indices,
        ambiguous_values=ambiguous_values,
        unparseable_indices=unparseable_indices,
        unparseable_values=unparseable_values,
        reasons=reasons,
    )


def parse_column_dates(
    df: pd.DataFrame,
    column: str,
    format: Optional[str] = None,
    dayfirst: Optional[bool] = None,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, CleaningRecord]:
    """Parse dates in a column and convert them into datetime values.

    Args:
        df: Pandas DataFrame.
        column: Column to parse.
        format: Optional explicit strftime format or shorthand.
        dayfirst: Optional boolean indicating day-first disambiguation.
        errors: 'raise', 'coerce', or 'ignore'.

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )
    if errors not in ("raise", "coerce", "ignore"):
        raise ValueError(f"errors must be 'raise', 'coerce', or 'ignore', got '{errors}'.")

    series = df[column]
    parsed_values: List[Any] = []
    affected_count = 0

    for idx, val in series.items():
        ts, err_msg, is_ambiguous = parse_date_element(
            val, format=format, dayfirst=dayfirst, allow_na=True
        )
        if err_msg is not None:
            if errors == "raise":
                kind = "Ambiguous date" if is_ambiguous else "Invalid date"
                raise ValueError(
                    f"{kind} detected at row {idx} in column '{column}': {err_msg}"
                )
            elif errors == "coerce":
                parsed_values.append(pd.NaT)
            else:  # errors == "ignore"
                # Keep original column untouched on ignore
                return df.copy(), CleaningRecord(
                    operation="parse_dates",
                    strategy="ignore",
                    columns=[column],
                    affected_count=0,
                    details={"error": err_msg, "row": idx},
                    message=f"Ignored date parsing failure on column '{column}'.",
                )
        else:
            if ts is None:
                parsed_values.append(pd.NaT)
            else:
                parsed_values.append(ts)
                affected_count += 1

    df_clean = df.copy()
    df_clean[column] = pd.to_datetime(pd.Series(parsed_values, index=series.index))

    record = CleaningRecord(
        operation="parse_dates",
        strategy="datetime_conversion",
        columns=[column],
        affected_count=affected_count,
        details={
            "format": format,
            "dayfirst": dayfirst,
            "errors": errors,
        },
        message=f"Parsed {affected_count} date value(s) in column '{column}' to datetime.",
    )
    return df_clean, record


def standardize_column_dates(
    df: pd.DataFrame,
    column: str,
    output_format: str = "%Y-%m-%d",
    input_format: Optional[str] = None,
    dayfirst: Optional[bool] = None,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, CleaningRecord]:
    """Standardize dates in a column into a uniform user-selected string format.

    Args:
        df: Pandas DataFrame.
        column: Column to standardize.
        output_format: Target format (e.g. '%Y-%m-%d', 'YYYY-MM-DD', 'DD/MM/YYYY').
        input_format: Optional explicit input format.
        dayfirst: Optional boolean for ambiguous date disambiguation.
        errors: 'raise', 'coerce', or 'ignore'.

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )
    if errors not in ("raise", "coerce", "ignore"):
        raise ValueError(f"errors must be 'raise', 'coerce', or 'ignore', got '{errors}'.")

    target_fmt = normalize_format_string(output_format)
    if target_fmt is None:
        target_fmt = "%Y-%m-%d"

    series = df[column]
    formatted_values: List[Any] = []
    affected_count = 0

    for idx, val in series.items():
        ts, err_msg, is_ambiguous = parse_date_element(
            val, format=input_format, dayfirst=dayfirst, allow_na=True
        )
        if err_msg is not None:
            if errors == "raise":
                kind = "Ambiguous date" if is_ambiguous else "Invalid date"
                raise ValueError(
                    f"{kind} detected at row {idx} in column '{column}': {err_msg}"
                )
            elif errors == "coerce":
                formatted_values.append(pd.NA)
            else:  # errors == "ignore"
                return df.copy(), CleaningRecord(
                    operation="standardize_dates",
                    strategy="ignore",
                    columns=[column],
                    affected_count=0,
                    details={"error": err_msg, "row": idx},
                    message=f"Ignored date standardization failure on column '{column}'.",
                )
        else:
            if ts is None:
                formatted_values.append(pd.NA)
            else:
                formatted_values.append(ts.strftime(target_fmt))
                affected_count += 1

    df_clean = df.copy()
    df_clean[column] = pd.Series(formatted_values, index=series.index, dtype="object")

    record = CleaningRecord(
        operation="standardize_dates",
        strategy="format_conversion",
        columns=[column],
        affected_count=affected_count,
        details={
            "output_format": output_format,
            "target_format": target_fmt,
            "input_format": input_format,
            "dayfirst": dayfirst,
            "errors": errors,
        },
        message=(
            f"Standardized {affected_count} date value(s) in column '{column}' "
            f"to format '{output_format}'."
        ),
    )
    return df_clean, record
