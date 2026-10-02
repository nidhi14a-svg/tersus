"""Missing-value detection and cleaning module for Tersus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningRecord:
    """Record of a cleaning action performed on the dataset."""

    operation: str
    strategy: str
    columns: List[str]
    affected_count: int
    details: Dict[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a dictionary."""
        return {
            "operation": self.operation,
            "strategy": self.strategy,
            "columns": list(self.columns),
            "affected_count": self.affected_count,
            "details": dict(self.details),
            "message": self.message,
        }

    def __repr__(self) -> str:
        """Return concise string representation."""
        return (
            f"CleaningRecord(operation='{self.operation}', strategy='{self.strategy}', "
            f"affected_count={self.affected_count})"
        )

    def __str__(self) -> str:
        """Return human-readable message."""
        return self.message


def _resolve_columns(
    df: pd.DataFrame, columns: Optional[Union[str, Sequence[str]]]
) -> List[str]:
    """Validate and resolve column list against DataFrame columns."""
    if columns is None:
        return [str(c) for c in df.columns]
    if isinstance(columns, str):
        col_list = [columns]
    else:
        col_list = [str(c) for c in columns]

    for col in col_list:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in dataset. Available columns: {list(df.columns)}"
            )
    return col_list


def get_missing_counts(
    df: pd.DataFrame, columns: Optional[Union[str, Sequence[str]]] = None
) -> Dict[str, int]:
    """Return missing-value count per column."""
    cols = _resolve_columns(df, columns)
    return {str(col): int(df[col].isna().sum()) for col in cols}


def get_missing_percentages(
    df: pd.DataFrame, columns: Optional[Union[str, Sequence[str]]] = None
) -> Dict[str, float]:
    """Return missing-value percentage per column."""
    cols = _resolve_columns(df, columns)
    total_rows = len(df)
    if total_rows == 0:
        return {str(col): 0.0 for col in cols}
    return {
        str(col): round((int(df[col].isna().sum()) / total_rows) * 100.0, 2)
        for col in cols
    }


def fill_missing(
    df: pd.DataFrame,
    strategy: str,
    columns: Optional[Union[str, Sequence[str]]] = None,
    value: Any = None,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Fill missing values in the DataFrame according to the chosen strategy.

    Args:
        df: The pandas DataFrame to clean.
        strategy: Strategy to use ('mean', 'median', 'mode', 'constant').
        columns: Target column or list of columns. If None, targets columns with missing values.
        value: Constant value to use when strategy='constant'.

    Returns:
        tuple[pd.DataFrame, CleaningRecord]: The cleaned DataFrame and operation record.

    Raises:
        ValueError: If strategy is invalid, constant value is missing, or statistical column is all-null.
        TypeError: If numerical strategy is applied to non-numeric column, or constant value type is incompatible.
    """
    valid_strategies = ("mean", "median", "mode", "constant")
    if not isinstance(strategy, str) or strategy.lower() not in valid_strategies:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Supported strategies: {', '.join(valid_strategies)}"
        )

    strategy = strategy.lower()

    if strategy == "constant" and value is None:
        raise ValueError(
            "A 'value' must be provided when strategy is 'constant'."
        )

    # Determine target columns
    if columns is None:
        if strategy in ("mean", "median"):
            target_cols = [
                str(c)
                for c in df.columns
                if df[c].isna().any()
                and pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])
            ]
        else:
            target_cols = [str(c) for c in df.columns if df[c].isna().any()]
    else:
        target_cols = _resolve_columns(df, columns)

    df_clean = df.copy()
    fill_values: Dict[str, Any] = {}
    total_filled = 0

    for col in target_cols:
        series = df_clean[col]
        missing_mask = series.isna()
        missing_in_col = int(missing_mask.sum())

        if missing_in_col == 0:
            continue

        if strategy in ("mean", "median"):
            if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                raise TypeError(
                    f"Cannot apply '{strategy}' strategy to non-numeric column '{col}' "
                    f"(dtype: {series.dtype})."
                )

            valid_values = series.dropna()
            if valid_values.empty:
                raise ValueError(
                    f"Cannot calculate {strategy} for column '{col}': column contains only missing values."
                )

            if strategy == "mean":
                fill_val = round(float(valid_values.mean()), 4)
            else:
                fill_val = round(float(valid_values.median()), 4)

            df_clean[col] = series.fillna(fill_val)
            fill_values[col] = fill_val
            total_filled += missing_in_col

        elif strategy == "mode":
            valid_values = series.dropna()
            if valid_values.empty:
                raise ValueError(
                    f"Cannot calculate mode for column '{col}': column contains only missing values."
                )

            modes = valid_values.mode()
            fill_val = modes.iloc[0]

            df_clean[col] = series.fillna(fill_val)
            fill_values[col] = fill_val
            total_filled += missing_in_col

        elif strategy == "constant":
            # Prevent silent type conversion
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                if not isinstance(value, (int, float, np.number)):
                    raise TypeError(
                        f"Cannot fill numeric column '{col}' (dtype: {series.dtype}) "
                        f"with non-numeric value '{value}' (type: {type(value).__name__})."
                    )
            elif pd.api.types.is_bool_dtype(series):
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError(
                        f"Cannot fill boolean column '{col}' with non-boolean value '{value}'."
                    )

            df_clean[col] = series.fillna(value)
            fill_values[col] = value
            total_filled += missing_in_col

    if total_filled > 0:
        col_summary = ", ".join(f"'{c}'" for c in fill_values.keys())
        msg = f"Filled {total_filled} missing value(s) in {col_summary} using {strategy} strategy."
    else:
        msg = "No missing values were filled (none found in selected columns)."

    record = CleaningRecord(
        operation="fill_missing",
        strategy=strategy,
        columns=target_cols,
        affected_count=total_filled,
        details={"fill_values": fill_values},
        message=msg,
    )

    return df_clean, record


def drop_missing(
    df: pd.DataFrame,
    columns: Optional[Union[str, Sequence[str]]] = None,
    how: str = "any",
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Drop rows containing missing values from the DataFrame.

    Args:
        df: The pandas DataFrame.
        columns: Specific column or list of columns to consider. If None, considers all columns.
        how: 'any' to drop if any missing value exists, 'all' if all considered values are missing.

    Returns:
        tuple[pd.DataFrame, CleaningRecord]: The cleaned DataFrame and operation record.

    Raises:
        ValueError: If 'how' is not 'any' or 'all'.
        KeyError: If a specified column does not exist.
    """
    if how not in ("any", "all"):
        raise ValueError(f"Invalid 'how' parameter: '{how}'. Expected 'any' or 'all'.")

    subset = _resolve_columns(df, columns) if columns is not None else None

    before_count = len(df)
    df_clean = df.dropna(subset=subset, how=how).reset_index(drop=True)
    after_count = len(df_clean)
    dropped_count = before_count - after_count

    cols_desc = ", ".join(f"'{c}'" for c in subset) if subset else "all columns"
    msg = f"Dropped {dropped_count} row(s) containing missing values (considered: {cols_desc}, how='{how}')."

    record = CleaningRecord(
        operation="drop_missing",
        strategy="drop",
        columns=subset if subset else [str(c) for c in df.columns],
        affected_count=dropped_count,
        details={"how": how, "rows_before": before_count, "rows_after": after_count},
        message=msg,
    )

    return df_clean, record
