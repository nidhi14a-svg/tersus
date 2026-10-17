"""Email validation and normalization module for Tersus.

Features:
- Syntactic validation of email address formats according to RFC 5322 specifications.
- Safe normalization of whitespace and casing.
- Safe preservation of missing/null values.

Important Disclaimer:
---------------------
Tersus verifies the syntactic structure of an email address.
It does NOT check DNS MX records, perform SMTP handshakes, or claim
that an email address actually exists or is currently deliverable.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Union

import pandas as pd

from tersus.missing import CleaningRecord
from tersus.validation import ValidationResult

# RFC 5322 practical regex pattern
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9_.+-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$"
)


def is_valid_email(val: Any) -> bool:
    """Check whether a single value has a syntactically valid email format.

    Note: This validates format syntax only. It does not verify that the
    email address actually exists or is active.

    Args:
        val: Any value to check.

    Returns:
        bool: True if val is a valid email string, False otherwise.
    """
    if not isinstance(val, str):
        return False
    s = val.strip()
    if not s or "@" not in s:
        return False

    if not EMAIL_REGEX.match(s):
        return False

    local_part, domain = s.split("@", 1)
    if local_part.startswith(".") or local_part.endswith("."):
        return False
    if ".." in local_part or ".." in domain:
        return False
    if domain.startswith("-") or domain.endswith("-"):
        return False

    labels = domain.split(".")
    for label in labels:
        if not label or label.startswith("-") or label.endswith("-"):
            return False

    return True


def normalize_email(
    val: Any, lowercase: bool = True, strip_whitespace: bool = True
) -> Any:
    """Safely normalize an email string, preserving missing values.

    Args:
        val: Email value to normalize.
        lowercase: Whether to convert email to lowercase (default True).
        strip_whitespace: Whether to strip leading/trailing spaces (default True).

    Returns:
        Normalized email string, or original value if missing/non-string.
    """
    if val is None or pd.isna(val):
        return pd.NA

    s = str(val)
    if strip_whitespace:
        s = s.strip()

    if not s or s.lower() in ("nan", "none", "null", "<na>"):
        return pd.NA

    if lowercase:
        s = s.lower()

    return s


def validate_email(
    df: pd.DataFrame, column: str, allow_na: bool = True
) -> ValidationResult:
    """Validate email formats in a DataFrame column, identifying invalid entries.

    Note: Validating format does not verify that the email address actually exists.

    Args:
        df: Pandas DataFrame.
        column: Column containing email addresses.
        allow_na: Whether missing values are allowed without failing validation (default True).

    Returns:
        ValidationResult: Structured validation report with invalid rows and indices.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    series = df[column]
    invalid_indices: List[Any] = []
    invalid_values: List[Any] = []

    for idx, val in series.items():
        if val is None or pd.isna(val) or str(val).strip() in ("", "nan", "none", "<na>"):
            if not allow_na:
                invalid_indices.append(idx)
                invalid_values.append(val)
        else:
            if not is_valid_email(val):
                invalid_indices.append(idx)
                invalid_values.append(val)

    invalid_count = len(invalid_indices)
    invalid_rows = df.loc[invalid_indices].copy()

    return ValidationResult(
        column=column,
        rule="valid_email",
        is_valid=(invalid_count == 0),
        invalid_count=invalid_count,
        invalid_indices=invalid_indices,
        invalid_values=invalid_values,
        invalid_rows=invalid_rows,
    )


def clean_email_column(
    df: pd.DataFrame,
    column: str,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    errors: str = "ignore",
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Normalize email addresses in a column (whitespace, casing).

    Args:
        df: Pandas DataFrame.
        column: Column to clean.
        lowercase: Whether to convert to lowercase (default True).
        strip_whitespace: Whether to strip surrounding whitespace (default True).
        errors: 'ignore' (keep invalid emails normalized), 'coerce' (turn invalid emails into pd.NA),
            or 'raise' (raise ValueError if invalid email found).

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")
    if errors not in ("ignore", "coerce", "raise"):
        raise ValueError(f"errors must be 'ignore', 'coerce', or 'raise', got '{errors}'.")

    series = df[column]
    cleaned_values: List[Any] = []
    affected_count = 0

    for idx, val in series.items():
        if val is None or pd.isna(val):
            cleaned_values.append(pd.NA)
            continue

        raw_str = str(val).strip()
        if not raw_str or raw_str.lower() in ("nan", "none", "null", "<na>"):
            cleaned_values.append(pd.NA)
            continue

        norm = normalize_email(val, lowercase=lowercase, strip_whitespace=strip_whitespace)
        valid = is_valid_email(norm)

        if not valid:
            if errors == "raise":
                raise ValueError(
                    f"Invalid email format '{val}' detected at row {idx} in column '{column}'."
                )
            elif errors == "coerce":
                cleaned_values.append(pd.NA)
                affected_count += 1
                continue

        cleaned_values.append(norm)
        if norm != val:
            affected_count += 1

    df_clean = df.copy()
    df_clean[column] = pd.Series(cleaned_values, index=series.index, dtype="object")

    record = CleaningRecord(
        operation="clean_email",
        strategy="normalize",
        columns=[column],
        affected_count=affected_count,
        details={
            "lowercase": lowercase,
            "strip_whitespace": strip_whitespace,
            "errors": errors,
        },
        message=f"Cleaned {affected_count} email value(s) in column '{column}'.",
    )
    return df_clean, record
