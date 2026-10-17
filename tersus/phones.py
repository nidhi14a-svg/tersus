"""Phone-number validation and normalization module for Tersus.

Features:
- Normalization and validation for Indian mobile phone numbers (V1 support).
- Supports parsing varying representations (+91, 91, 0-prefix, spaces, dashes, brackets).
- Safe formatting into 10-digit national or E.164 (+91) representations.
- Safe handling and preservation of missing values.

Important Disclaimer:
---------------------
Tersus verifies the syntactic and structural validity of phone numbers according
to regional numbering plans (such as the Indian National Numbering Plan).
It does NOT check subscriber network registries, perform HLR lookups, or claim
that a phone number is currently active, connected, or in service.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple, Union

import pandas as pd

from tersus.missing import CleaningRecord
from tersus.validation import ValidationResult


def parse_indian_phone(val: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse an Indian phone number value into standard 10-digit core numbers.

    Rules for Indian mobile numbers (National Numbering Plan):
    - Standard mobile numbers are 10 digits in length.
    - First digit must be 6, 7, 8, or 9.
    - Can be prefixed with +91, 91, or domestic trunk prefix 0.

    Args:
        val: Any phone number value (str, int, float).

    Returns:
        Tuple of (core_10_digits, error_reason):
            - core_10_digits: Exactly 10 digits as string if valid, None if invalid or missing.
            - error_reason: Error message string if invalid, None if valid or missing.
    """
    if val is None or pd.isna(val):
        return None, None

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>"):
        return None, None

    # Check for invalid non-numeric alphabetic characters
    if re.search(r"[a-zA-Z]", s):
        return None, f"Phone number contains letters: '{s}'"

    # Extract all digits
    digits = re.sub(r"\D", "", s)

    core: Optional[str] = None
    if len(digits) == 10:
        core = digits
    elif len(digits) == 11 and digits.startswith("0"):
        core = digits[1:]
    elif len(digits) == 12 and digits.startswith("91"):
        core = digits[2:]
    elif len(digits) == 13 and digits.startswith("091"):
        core = digits[3:]
    else:
        return (
            None,
            f"Invalid length ({len(digits)} digits) for Indian phone number: '{s}'",
        )

    # In Indian telecom numbering, valid mobile numbers begin with 6, 7, 8, or 9
    if core[0] not in "6789":
        return (
            None,
            f"Indian mobile numbers must start with 6, 7, 8, or 9 (got '{core[0]}' in '{core}')",
        )

    # Check for dummy numbers (e.g. 0000000000)
    if len(set(core)) == 1:
        return None, f"Repeating dummy number: '{core}'"

    return core, None


def format_indian_phone(core_digits: str, output_format: str = "10-digit") -> str:
    """Format 10-digit Indian phone number into user-selected representation.

    Args:
        core_digits: 10-digit clean string (e.g. '9876543210').
        output_format:
            - '10-digit' or 'national': '9876543210'
            - '+91' or 'e164' or 'international': '+919876543210'
            - 'spaced': '+91 98765 43210'
            - 'hyphenated': '+91-98765-43210'
            - 'national-spaced': '98765 43210'

    Returns:
        Formatted phone number string.
    """
    clean_fmt = output_format.strip().lower()
    if clean_fmt in ("10-digit", "national", "digits"):
        return core_digits
    elif clean_fmt in ("+91", "e164", "international"):
        return f"+91{core_digits}"
    elif clean_fmt in ("spaced", "+91-spaced"):
        return f"+91 {core_digits[:5]} {core_digits[5:]}"
    elif clean_fmt in ("hyphenated", "+91-hyphenated"):
        return f"+91-{core_digits[:5]}-{core_digits[5:]}"
    elif clean_fmt == "national-spaced":
        return f"{core_digits[:5]} {core_digits[5:]}"
    else:
        return core_digits


def is_valid_phone(val: Any, country: str = "IN") -> bool:
    """Check whether a phone number is valid for the specified country.

    Note: This validates format syntax only. It does not verify that the
    phone number is currently active or in service.

    Args:
        val: Value to check.
        country: Country code ('IN' supported for V1).

    Returns:
        bool: True if valid, False otherwise.
    """
    if country.upper() == "IN":
        core, err = parse_indian_phone(val)
        return core is not None and err is None
    else:
        raise NotImplementedError(
            f"Country '{country}' is not supported yet in V1. Supported: 'IN'."
        )


def validate_phone(
    df: pd.DataFrame,
    column: str,
    country: str = "IN",
    allow_na: bool = True,
) -> ValidationResult:
    """Validate phone numbers in a column, identifying invalid entries.

    Note: Validating format does not verify that the phone number is active.

    Args:
        df: Pandas DataFrame.
        column: Column containing phone numbers.
        country: Country code (default 'IN').
        allow_na: Whether missing values are allowed without failing validation (default True).

    Returns:
        ValidationResult: Structured validation report.
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
            if not is_valid_phone(val, country=country):
                invalid_indices.append(idx)
                invalid_values.append(val)

    invalid_count = len(invalid_indices)
    invalid_rows = df.loc[invalid_indices].copy()

    return ValidationResult(
        column=column,
        rule=f"valid_phone(country='{country}')",
        is_valid=(invalid_count == 0),
        invalid_count=invalid_count,
        invalid_indices=invalid_indices,
        invalid_values=invalid_values,
        invalid_rows=invalid_rows,
    )


def clean_phone_column(
    df: pd.DataFrame,
    column: str,
    country: str = "IN",
    output_format: str = "10-digit",
    errors: str = "raise",
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Normalize phone numbers in a column into standard representation.

    Args:
        df: Pandas DataFrame.
        column: Column to clean.
        country: Country code (default 'IN').
        output_format: Desired format ('10-digit', '+91', 'spaced', 'hyphenated').
        errors: 'raise' (default), 'coerce' (turn invalid to pd.NA), or 'ignore'.

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")
    if errors not in ("raise", "coerce", "ignore"):
        raise ValueError(f"errors must be 'raise', 'coerce', or 'ignore', got '{errors}'.")

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

        core, err = parse_indian_phone(val)
        if err is not None or core is None:
            if errors == "raise":
                raise ValueError(
                    f"Invalid phone number '{val}' at row {idx} in column '{column}': {err}"
                )
            elif errors == "coerce":
                cleaned_values.append(pd.NA)
                affected_count += 1
                continue
            else:  # errors == "ignore"
                cleaned_values.append(val)
                continue

        formatted = format_indian_phone(core, output_format=output_format)
        cleaned_values.append(formatted)
        if formatted != str(val):
            affected_count += 1

    df_clean = df.copy()
    df_clean[column] = pd.Series(cleaned_values, index=series.index, dtype="object")

    record = CleaningRecord(
        operation="clean_phone",
        strategy="normalize",
        columns=[column],
        affected_count=affected_count,
        details={
            "country": country,
            "output_format": output_format,
            "errors": errors,
        },
        message=f"Cleaned {affected_count} phone number(s) in column '{column}'.",
    )
    return df_clean, record
