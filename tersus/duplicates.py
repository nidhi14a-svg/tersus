"""Duplicate detection and removal module for Tersus."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import pandas as pd

from tersus.missing import CleaningRecord


def _resolve_subset(
    df: pd.DataFrame, subset: Optional[Union[str, Sequence[str]]]
) -> Optional[List[str]]:
    """Validate and resolve column subset for duplicate checking."""
    if subset is None:
        return None
    if isinstance(subset, str):
        col_list = [subset]
    else:
        col_list = [str(c) for c in subset]

    if len(col_list) == 0:
        raise ValueError("Column subset cannot be empty.")

    for col in col_list:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in dataset. Available columns: {list(df.columns)}"
            )
    return col_list


def _validate_keep(keep: Union[str, bool]) -> Union[str, bool]:
    """Validate the 'keep' parameter against allowed pandas options.

    Accepts 'first', 'last', or False (to drop all duplicate occurrences).
    As a convenience, the string "false" (any case) is also accepted as an
    alias for the boolean False, since callers passing options from text
    sources (e.g. CLI args, CSV/JSON config) may not have a native bool.
    """
    if keep in ("first", "last", False):
        return keep
    if isinstance(keep, str) and keep.lower() == "false":
        return False
    raise ValueError(
        f"Invalid 'keep' parameter: '{keep}'. Expected 'first', 'last', or False."
    )


def get_duplicate_count(
    df: pd.DataFrame,
    subset: Optional[Union[str, Sequence[str]]] = None,
    keep: Union[str, bool] = "first",
) -> int:
    """Return the count of duplicate rows in the DataFrame.

    Args:
        df: The pandas DataFrame.
        subset: Column or list of columns to consider. If None, considers all columns.
        keep: 'first', 'last', or False.

    Returns:
        int: Number of duplicate rows.
    """
    if len(df) == 0:
        return 0

    validated_keep = _validate_keep(keep)
    cols = _resolve_subset(df, subset)
    return int(df.duplicated(subset=cols, keep=validated_keep).sum())


def get_duplicates(
    df: pd.DataFrame,
    subset: Optional[Union[str, Sequence[str]]] = None,
    keep: Union[str, bool] = "first",
) -> pd.DataFrame:
    """Return duplicate rows from the DataFrame.

    Args:
        df: The pandas DataFrame.
        subset: Column or list of columns to consider. If None, considers all columns.
        keep: 'first', 'last', or False.

    Returns:
        pd.DataFrame: A copy of the duplicate rows.
    """
    if len(df) == 0:
        return df.copy()

    validated_keep = _validate_keep(keep)
    cols = _resolve_subset(df, subset)
    mask = df.duplicated(subset=cols, keep=validated_keep)
    return df[mask].copy().reset_index(drop=True)


def drop_duplicates(
    df: pd.DataFrame,
    subset: Optional[Union[str, Sequence[str]]] = None,
    keep: Union[str, bool] = "first",
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Drop duplicate rows from the DataFrame.

    Args:
        df: The pandas DataFrame.
        subset: Column or list of columns to consider. If None, considers all columns.
        keep: 'first', 'last', or False.

    Returns:
        tuple[pd.DataFrame, CleaningRecord]: The cleaned DataFrame and operation record.
    """
    validated_keep = _validate_keep(keep)
    cols = _resolve_subset(df, subset)

    before_count = len(df)
    df_clean = df.drop_duplicates(subset=cols, keep=validated_keep).reset_index(
        drop=True
    )
    after_count = len(df_clean)
    dropped_count = before_count - after_count

    subset_desc = ", ".join(f"'{c}'" for c in cols) if cols else "all columns"
    msg = (
        f"Removed {dropped_count} duplicate row(s) "
        f"(subset: {subset_desc}, keep='{validated_keep}')."
    )

    record = CleaningRecord(
        operation="drop_duplicates",
        strategy=f"keep_{validated_keep}",
        columns=cols if cols else [str(c) for c in df.columns],
        affected_count=dropped_count,
        details={
            "subset": cols,
            "keep": validated_keep,
            "rows_before": before_count,
            "rows_after": after_count,
        },
        message=msg,
    )

    return df_clean, record
