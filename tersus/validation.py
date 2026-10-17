"""Value validation module for Tersus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tersus.missing import CleaningRecord


@dataclass
class ValidationResult:
    """Result of a data validation check."""

    column: str
    rule: str
    is_valid: bool
    invalid_count: int
    invalid_indices: List[Any]
    invalid_values: List[Any]
    invalid_rows: pd.DataFrame

    @property
    def has_violations(self) -> bool:
        """Whether any invalid values were detected."""
        return not self.is_valid

    def __bool__(self) -> bool:
        """Truthiness evaluates to True if valid (no violations), False if invalid."""
        return self.is_valid

    def __len__(self) -> int:
        """Number of invalid values found."""
        return self.invalid_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "column": self.column,
            "rule": self.rule,
            "is_valid": self.is_valid,
            "invalid_count": self.invalid_count,
            "invalid_indices": list(self.invalid_indices),
            "invalid_values": list(self.invalid_values),
        }

    def __repr__(self) -> str:
        return (
            f"ValidationResult(column='{self.column}', rule='{self.rule}', "
            f"is_valid={self.is_valid}, invalid_count={self.invalid_count})"
        )

    def __str__(self) -> str:
        status = (
            "PASSED"
            if self.is_valid
            else f"FAILED ({self.invalid_count} invalid value(s) found)"
        )
        lines = [
            f"Validation Check on '{self.column}':",
            f"  Rule: {self.rule}",
            f"  Status: {status}",
        ]
        if self.invalid_count > 0:
            lines.append("  Invalid items:")
            for idx, val in zip(self.invalid_indices[:10], self.invalid_values[:10]):
                lines.append(f"    - Row {idx}: {val}")
            if self.invalid_count > 10:
                lines.append(f"    ... and {self.invalid_count - 10} more")
        return "\n".join(lines)


def validate_range(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    inclusive: bool = True,
) -> ValidationResult:
    """Find values in a column outside the specified numerical range.

    Args:
        df: The pandas DataFrame.
        column: Column name to validate.
        min_value: Minimum allowed value (or None for no lower bound).
        max_value: Maximum allowed value (or None for no upper bound).
        inclusive: Whether bounds are inclusive (default True).

    Returns:
        ValidationResult: The structured validation report.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    if min_value is None and max_value is None:
        raise ValueError("At least one of min_value or max_value must be provided.")

    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(
            f"min_value ({min_value}) cannot be greater than max_value ({max_value})."
        )

    series = df[column]

    # Convert to numeric to compare, but keep track of non-numeric values
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Missing values in the original data are excluded from range violations
    not_na = ~series.isna()

    # Identify out of range values
    invalid_mask = pd.Series(False, index=series.index)

    # Any non-missing value that failed numeric conversion is out of range
    non_numeric_mask = not_na & numeric_series.isna()
    invalid_mask |= non_numeric_mask

    if min_value is not None:
        if inclusive:
            invalid_mask |= not_na & ~non_numeric_mask & (numeric_series < min_value)
        else:
            invalid_mask |= not_na & ~non_numeric_mask & (numeric_series <= min_value)

    if max_value is not None:
        if inclusive:
            invalid_mask |= not_na & ~non_numeric_mask & (numeric_series > max_value)
        else:
            invalid_mask |= not_na & ~non_numeric_mask & (numeric_series >= max_value)

    invalid_indices = list(df.index[invalid_mask])
    invalid_values = list(series.loc[invalid_indices])
    invalid_rows = df.loc[invalid_indices].copy()
    invalid_count = len(invalid_indices)

    # Build human-readable rule description
    lower_op = "<=" if inclusive else "<"
    upper_op = "<=" if inclusive else "<"
    if min_value is not None and max_value is not None:
        rule_desc = f"{min_value} {lower_op} value {upper_op} {max_value}"
    elif min_value is not None:
        rule_desc = f"value >= {min_value}" if inclusive else f"value > {min_value}"
    else:
        rule_desc = f"value <= {max_value}" if inclusive else f"value < {max_value}"

    return ValidationResult(
        column=column,
        rule=rule_desc,
        is_valid=(invalid_count == 0),
        invalid_count=invalid_count,
        invalid_indices=invalid_indices,
        invalid_values=invalid_values,
        invalid_rows=invalid_rows,
    )


def validate_values(
    df: pd.DataFrame,
    column: str,
    allowed_values: Sequence[Any],
) -> ValidationResult:
    """Validate that values in a column belong to an allowed set of values.

    Args:
        df: The pandas DataFrame.
        column: Column name to validate.
        allowed_values: Sequence of allowed values.

    Returns:
        ValidationResult: The structured validation report.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    series = df[column]
    allowed_set = set(allowed_values)

    # Missing values are ignored by value set check
    not_na = ~series.isna()
    invalid_mask = not_na & ~series.isin(allowed_set)

    invalid_indices = list(df.index[invalid_mask])
    invalid_values = list(series.loc[invalid_indices])
    invalid_rows = df.loc[invalid_indices].copy()
    invalid_count = len(invalid_indices)

    rule_desc = f"value in {list(allowed_values)}"

    return ValidationResult(
        column=column,
        rule=rule_desc,
        is_valid=(invalid_count == 0),
        invalid_count=invalid_count,
        invalid_indices=invalid_indices,
        invalid_values=invalid_values,
        invalid_rows=invalid_rows,
    )


def validate_rule(
    df: pd.DataFrame,
    column: str,
    rule: Callable[[Any], bool],
    rule_name: str = "custom_rule",
) -> ValidationResult:
    """Validate values in a column using a custom predicate function.

    Args:
        df: The pandas DataFrame.
        column: Column name to validate.
        rule: Callable taking a value and returning True if valid, False if invalid.
        rule_name: Name or description of the rule.

    Returns:
        ValidationResult: The structured validation report.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    series = df[column]
    invalid_indices: List[Any] = []
    invalid_values: List[Any] = []

    for idx, val in series.items():
        if pd.isna(val):
            continue
        try:
            if not rule(val):
                invalid_indices.append(idx)
                invalid_values.append(val)
        except Exception:
            invalid_indices.append(idx)
            invalid_values.append(val)

    invalid_count = len(invalid_indices)
    invalid_rows = df.loc[invalid_indices].copy()

    return ValidationResult(
        column=column,
        rule=rule_name,
        is_valid=(invalid_count == 0),
        invalid_count=invalid_count,
        invalid_indices=invalid_indices,
        invalid_values=invalid_values,
        invalid_rows=invalid_rows,
    )


def drop_invalid_rows(
    df: pd.DataFrame, validation_result: ValidationResult
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Drop rows identified as invalid by a ValidationResult."""
    df_clean = df.drop(index=validation_result.invalid_indices).reset_index(drop=True)
    dropped_count = validation_result.invalid_count

    msg = (
        f"Dropped {dropped_count} row(s) failing validation rule "
        f"'{validation_result.rule}' on column '{validation_result.column}'."
    )
    record = CleaningRecord(
        operation="drop_invalid",
        strategy="drop",
        columns=[validation_result.column],
        affected_count=dropped_count,
        details={
            "rule": validation_result.rule,
            "dropped_indices": validation_result.invalid_indices,
        },
        message=msg,
    )
    return df_clean, record


def replace_invalid_values(
    df: pd.DataFrame, validation_result: ValidationResult, value: Any
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Replace invalid values identified by a ValidationResult with a user-provided value."""
    df_clean = df.copy()
    col = validation_result.column
    df_clean.loc[validation_result.invalid_indices, col] = value
    replaced_count = validation_result.invalid_count

    msg = (
        f"Replaced {replaced_count} invalid value(s) in column '{col}' "
        f"with '{value}' (rule: '{validation_result.rule}')."
    )
    record = CleaningRecord(
        operation="replace_invalid",
        strategy="replace",
        columns=[col],
        affected_count=replaced_count,
        details={
            "rule": validation_result.rule,
            "replacement_value": value,
            "replaced_indices": validation_result.invalid_indices,
        },
        message=msg,
    )
    return df_clean, record


def clamp_column_range(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Clamp numeric values in a column to the specified range."""
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    df_clean = df.copy()
    series = df_clean[column]

    numeric_series = pd.to_numeric(series, errors="coerce")
    clamped = numeric_series.clip(lower=min_value, upper=max_value)

    changed_mask = (numeric_series != clamped) & ~numeric_series.isna()
    affected_count = int(changed_mask.sum())

    df_clean[column] = clamped

    msg = (
        f"Clamped {affected_count} value(s) in column '{column}' to range "
        f"[{min_value}, {max_value}]."
    )
    record = CleaningRecord(
        operation="clamp_range",
        strategy="clamp",
        columns=[column],
        affected_count=affected_count,
        details={
            "min_value": min_value,
            "max_value": max_value,
        },
        message=msg,
    )
    return df_clean, record
