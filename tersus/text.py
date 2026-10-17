"""Text normalization module for Tersus."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Union

import pandas as pd

from tersus.missing import CleaningRecord


def normalize_text_value(
    val: Any,
    strip: bool = True,
    collapse_spaces: bool = True,
    remove_invisible: bool = True,
    case: Optional[str] = None,
) -> Any:
    """Normalize a single text value while safely preserving missing values.

    Args:
        val: The value to normalize.
        strip: Whether to strip leading and trailing whitespace.
        collapse_spaces: Whether to collapse multiple consecutive spaces into a single space.
        remove_invisible: Whether to replace invisible characters (tabs, newlines, etc.) with space.
        case: Case transformation ('lower', 'upper', 'title', or None/'preserve').

    Returns:
        Any: Normalized string or original missing/non-string value.
    """
    if val is None or pd.isna(val):
        return val

    if not isinstance(val, str):
        return val

    result = val

    if remove_invisible:
        result = re.sub(r"[\t\n\r\f\v]+", " ", result)

    if collapse_spaces:
        result = re.sub(r"[ ]{2,}", " ", result)

    if strip:
        result = result.strip()

    if case == "lower":
        result = result.lower()
    elif case == "upper":
        result = result.upper()
    elif case == "title":
        result = result.title()

    return result


def _resolve_text_columns(
    df: pd.DataFrame, columns: Optional[Union[str, Sequence[str]]]
) -> List[str]:
    """Validate and resolve target text columns."""
    if columns is None:
        # Default to all non-numeric columns
        return [
            str(c)
            for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]

    if isinstance(columns, str):
        col_list = [columns]
    else:
        col_list = [str(c) for c in columns]

    for col in col_list:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in dataset. Available columns: {list(df.columns)}"
            )
        if pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Cannot apply text cleaning to numeric column '{col}' (dtype: {df[col].dtype})."
            )

    return col_list


def clean_dataframe_text(
    df: pd.DataFrame,
    columns: Optional[Union[str, Sequence[str]]] = None,
    strip: bool = True,
    collapse_spaces: bool = True,
    remove_invisible: bool = True,
    case: Optional[str] = None,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Normalize text in the specified DataFrame columns.

    Args:
        df: The pandas DataFrame.
        columns: Target column name or sequence of column names. If None, targets all non-numeric columns.
        strip: Whether to strip leading and trailing whitespace.
        collapse_spaces: Whether to collapse multiple spaces into one.
        remove_invisible: Whether to replace tabs, newlines, etc. with space.
        case: Case transformation ('lower', 'upper', 'title', or None/'preserve').

    Returns:
        tuple[pd.DataFrame, CleaningRecord]: The cleaned DataFrame and operation record.
    """
    if case not in ("lower", "upper", "title", None, "preserve"):
        raise ValueError(
            f"Invalid case option '{case}'. Expected 'lower', 'upper', 'title', or None/'preserve'."
        )

    target_cols = _resolve_text_columns(df, columns)
    df_clean = df.copy()

    total_modified_cells = 0
    modified_columns: List[str] = []

    for col in target_cols:
        series = df_clean[col]

        # Convert categorical to object to allow free string transformations
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)

        cleaned_series = series.apply(
            lambda v: normalize_text_value(
                v,
                strip=strip,
                collapse_spaces=collapse_spaces,
                remove_invisible=remove_invisible,
                case=case if case != "preserve" else None,
            )
        )

        # Count modified cells, accurately handling NaN != NaN
        changed = (series != cleaned_series) & ~(series.isna() & cleaned_series.isna())
        col_modified = int(changed.sum())

        if col_modified > 0:
            total_modified_cells += col_modified
            modified_columns.append(col)

        df_clean[col] = cleaned_series

    col_desc = ", ".join(f"'{c}'" for c in target_cols) if target_cols else "none"
    case_desc = f", case='{case}'" if case and case != "preserve" else ""
    msg = (
        f"Normalized text in {len(target_cols)} column(s) ({col_desc}): "
        f"{total_modified_cells} value(s) modified{case_desc}."
    )

    record = CleaningRecord(
        operation="clean_text",
        strategy=str(case) if case else "normalize",
        columns=target_cols,
        affected_count=total_modified_cells,
        details={
            "columns": target_cols,
            "modified_columns": modified_columns,
            "case": case,
            "strip": strip,
            "collapse_spaces": collapse_spaces,
            "remove_invisible": remove_invisible,
        },
        message=msg,
    )

    return df_clean, record
