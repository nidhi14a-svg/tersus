"""Column-name cleaning and standardization module for Tersus."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from tersus.missing import CleaningRecord


def _normalize_string(text: str) -> str:
    """Normalize unicode characters to ASCII where possible (e.g., accents)."""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ASCII", "ignore")
        .decode("utf-8")
    )


def clean_single_column_name(
    name: Any,
    case: Optional[str] = "lower",
    replace_spaces: str = "_",
    remove_special: bool = True,
    strip: bool = True,
    normalize_accents: bool = True,
    default_name: str = "unnamed",
) -> str:
    """Clean and standardize a single column name.

    Args:
        name: Original column name (any type, converted to string).
        case: Case transformation ('lower', 'upper', 'snake', or None/'preserve').
        replace_spaces: Replacement string for whitespace and separators.
        remove_special: Whether to remove unwanted special characters.
        strip: Whether to strip leading and trailing whitespace and separators.
        normalize_accents: Whether to convert accented characters to ASCII equivalents.
        default_name: Fallback name if the cleaned column name is empty.

    Returns:
        str: Standardized column name.
    """
    if name is None or pd.isna(name):
        return default_name

    col_str = str(name)

    if strip:
        col_str = col_str.strip()

    if not col_str:
        return default_name

    if normalize_accents:
        col_str = _normalize_string(col_str)

    # Convert camelCase / PascalCase to snake_case if case='snake'
    if case == "snake":
        col_str = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", col_str)
        col_str = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", col_str)

    if remove_special:
        # Replace non-alphanumeric and non-space characters with space
        col_str = re.sub(r"[^\w\s]", " ", col_str)

    # Collapse any sequences of whitespace and existing separators into replace_spaces
    sep_pattern = (
        r"[\s" + re.escape(replace_spaces) + r"]+"
        if replace_spaces
        else r"\s+"
    )
    col_str = re.sub(sep_pattern, replace_spaces, col_str)

    if strip and replace_spaces:
        col_str = col_str.strip(replace_spaces).strip()

    if not col_str:
        return default_name

    # Apply case transformation
    if case in ("lower", "snake"):
        col_str = col_str.lower()
    elif case == "upper":
        col_str = col_str.upper()

    return col_str if col_str else default_name


def make_unique_column_names(names: Sequence[str]) -> List[str]:
    """Ensure all column names in the sequence are unique and deterministic.

    Args:
        names: Sequence of column names.

    Returns:
        List[str]: List of unique column names with '_1', '_2' suffixes where necessary.
    """
    seen: Dict[str, int] = {}
    unique_names: List[str] = []

    for name in names:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
            while candidate in seen or candidate in names:
                seen[name] += 1
                candidate = f"{name}_{seen[name]}"
            seen[candidate] = 0
            unique_names.append(candidate)

    return unique_names


def clean_dataframe_columns(
    df: pd.DataFrame,
    case: Optional[str] = "lower",
    replace_spaces: str = "_",
    remove_special: bool = True,
    strip: bool = True,
    normalize_accents: bool = True,
    rename: Optional[Dict[str, str]] = None,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Clean all column names of a DataFrame in a configurable manner.

    Args:
        df: The pandas DataFrame.
        case: Case transformation ('lower', 'upper', 'snake', or None/'preserve').
        replace_spaces: String to replace whitespace and separators with.
        remove_special: Whether to remove unwanted special characters.
        strip: Whether to strip leading/trailing whitespace and separators.
        normalize_accents: Whether to convert accented characters to ASCII.
        rename: Optional dictionary mapping existing names directly to new names.

    Returns:
        tuple[pd.DataFrame, CleaningRecord]: The DataFrame with cleaned columns and the operation record.
    """
    if case not in ("lower", "upper", "snake", None, "preserve"):
        raise ValueError(
            f"Invalid case option '{case}'. Expected 'lower', 'upper', 'snake', or None/'preserve'."
        )

    original_columns = [str(c) for c in df.columns]
    rename_overrides = rename or {}

    cleaned_raw: List[str] = []
    for col in original_columns:
        if col in rename_overrides:
            cleaned_raw.append(rename_overrides[col])
        else:
            cleaned_name = clean_single_column_name(
                col,
                case=case,
                replace_spaces=replace_spaces,
                remove_special=remove_special,
                strip=strip,
                normalize_accents=normalize_accents,
            )
            cleaned_raw.append(cleaned_name)

    final_columns = make_unique_column_names(cleaned_raw)

    df_clean = df.copy()
    df_clean.columns = final_columns

    renamed_map = {
        orig: new
        for orig, new in zip(original_columns, final_columns)
        if orig != new
    }

    record = CleaningRecord(
        operation="clean_column_names",
        strategy=str(case) if case else "preserve",
        columns=final_columns,
        affected_count=len(renamed_map),
        details={
            "renamed": renamed_map,
            "before": original_columns,
            "after": final_columns,
        },
        message=(
            f"Cleaned {len(renamed_map)} column name(s): {renamed_map}"
            if renamed_map
            else "No column names were changed."
        ),
    )

    return df_clean, record
