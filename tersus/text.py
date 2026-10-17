"""Text normalization module for Tersus."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Union
import unicodedata

import numpy as np
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


# Regex constants for invisible and control characters
INVISIBLE_REGEX = re.compile(
    r"[\u200B-\u200F\uFEFF\u2060-\u206F\u180E\u00AD\u202A-\u202E\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]"
)
NBSP_REGEX = re.compile(r"[\u00A0\u202F]")
CURRENCY_SYMBOLS_REGEX = re.compile(
    r"[$€£¥₹]|\b(?:USD|EUR|GBP|INR|JPY|CAD|AUD)\b", re.IGNORECASE
)


def clean_invisible_characters(val: Any) -> Any:
    """Remove zero-width spaces, control characters, and normalize non-breaking spaces.

    Args:
        val: Input text value.

    Returns:
        Cleaned text string, or original value if missing/non-string.
    """
    if val is None or pd.isna(val) or not isinstance(val, str):
        return val

    # 1. Normalize non-breaking spaces to standard space
    res = NBSP_REGEX.sub(" ", val)
    # 2. Strip zero-width and non-printable control characters
    res = INVISIBLE_REGEX.sub("", res)
    # 3. Replace tabs, newlines, carriage returns with standard space
    res = re.sub(r"[\t\n\r]+", " ", res)
    # 4. Collapse consecutive spaces
    res = re.sub(r"[ ]{2,}", " ", res)
    return res.strip()


def clean_invisible_chars(
    df: pd.DataFrame, columns: Optional[Union[str, Sequence[str]]] = None
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Clean invisible characters, zero-width spaces, and control codes across DataFrame columns."""
    target_cols = _resolve_text_columns(df, columns)
    df_clean = df.copy()
    total_modified = 0

    for col in target_cols:
        series = df_clean[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)

        cleaned = series.apply(clean_invisible_characters)
        changed = (series != cleaned) & ~(series.isna() & cleaned.isna())
        total_modified += int(changed.sum())
        df_clean[col] = cleaned

    record = CleaningRecord(
        operation="clean_invisible_chars",
        strategy="remove_invisible",
        columns=target_cols,
        affected_count=total_modified,
        details={"columns": target_cols},
        message=f"Cleaned invisible characters in {len(target_cols)} column(s) ({total_modified} value(s) modified).",
    )
    return df_clean, record


def normalize_unicode_text(
    val: Any, form: str = "NFC", strip_accents: bool = False
) -> Any:
    """Normalize Unicode representation and optionally strip diacritical accents.

    Args:
        val: Input text value.
        form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD'). Default 'NFC'.
        strip_accents: Whether to remove accents (e.g. 'Café' -> 'Cafe'). Default False.

    Returns:
        Normalized text string, or original value if missing/non-string.
    """
    if val is None or pd.isna(val) or not isinstance(val, str):
        return val

    norm = unicodedata.normalize(form, val)
    if strip_accents:
        decomposed = unicodedata.normalize("NFKD", norm)
        norm = "".join(c for c in decomposed if not unicodedata.combining(c))

    return norm


def normalize_unicode(
    df: pd.DataFrame,
    columns: Optional[Union[str, Sequence[str]]] = None,
    form: str = "NFC",
    strip_accents: bool = False,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Apply safe Unicode normalization and optional accent stripping to DataFrame columns."""
    target_cols = _resolve_text_columns(df, columns)
    df_clean = df.copy()
    total_modified = 0

    for col in target_cols:
        series = df_clean[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)

        cleaned = series.apply(
            lambda v: normalize_unicode_text(v, form=form, strip_accents=strip_accents)
        )
        changed = (series != cleaned) & ~(series.isna() & cleaned.isna())
        total_modified += int(changed.sum())
        df_clean[col] = cleaned

    record = CleaningRecord(
        operation="normalize_unicode",
        strategy=f"unicode_{form}" + ("_strip_accents" if strip_accents else ""),
        columns=target_cols,
        affected_count=total_modified,
        details={"columns": target_cols, "form": form, "strip_accents": strip_accents},
        message=f"Normalized Unicode in {len(target_cols)} column(s) ({total_modified} value(s) modified).",
    )
    return df_clean, record


def clean_special_characters(
    val: Any,
    allowed_chars: str = "",
    replace_with: str = "",
    keep_alphanumeric: bool = True,
    keep_spaces: bool = True,
) -> Any:
    """Remove or replace unwanted special characters and symbols from text.

    Args:
        val: Input text value.
        allowed_chars: Permitted non-alphanumeric characters (e.g. '-_.@').
        replace_with: Replacement string for disallowed symbols (default '').
        keep_alphanumeric: Whether to keep standard letters and numbers (default True).
        keep_spaces: Whether to preserve whitespace (default True).

    Returns:
        Cleaned text string, or original value if missing/non-string.
    """
    if val is None or pd.isna(val) or not isinstance(val, str):
        return val

    pattern_parts = []
    if keep_alphanumeric:
        pattern_parts.append(r"a-zA-Z0-9")
    if keep_spaces:
        pattern_parts.append(r"\s")
    if allowed_chars:
        pattern_parts.append(re.escape(allowed_chars))

    char_class = "".join(pattern_parts)
    regex = rf"[^{char_class}]+"
    res = re.sub(regex, replace_with, val)

    if keep_spaces:
        res = re.sub(r"\s+", " ", res).strip()

    return res


def clean_special_chars(
    df: pd.DataFrame,
    columns: Optional[Union[str, Sequence[str]]] = None,
    allowed_chars: str = "",
    replace_with: str = "",
    keep_alphanumeric: bool = True,
    keep_spaces: bool = True,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Remove or replace special characters across DataFrame columns."""
    target_cols = _resolve_text_columns(df, columns)
    df_clean = df.copy()
    total_modified = 0

    for col in target_cols:
        series = df_clean[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)

        cleaned = series.apply(
            lambda v: clean_special_characters(
                v,
                allowed_chars=allowed_chars,
                replace_with=replace_with,
                keep_alphanumeric=keep_alphanumeric,
                keep_spaces=keep_spaces,
            )
        )
        changed = (series != cleaned) & ~(series.isna() & cleaned.isna())
        total_modified += int(changed.sum())
        df_clean[col] = cleaned

    record = CleaningRecord(
        operation="clean_special_chars",
        strategy="filter_symbols",
        columns=target_cols,
        affected_count=total_modified,
        details={
            "columns": target_cols,
            "allowed_chars": allowed_chars,
            "replace_with": replace_with,
        },
        message=f"Cleaned special characters in {len(target_cols)} column(s) ({total_modified} value(s) modified).",
    )
    return df_clean, record


def extract_numeric_value(
    val: Any,
    as_type: str = "float",
    handle_commas: bool = True,
    which: str = "first",
    errors: str = "coerce",
) -> Any:
    """Extract numeric values from messy text strings.

    Examples:
        '120 km/h' -> 120.0
        'Score: 95.5%' -> 95.5
        '$4,500.50' -> 4500.5

    Args:
        val: Input text or numeric value.
        as_type: 'float' (default) or 'int'.
        handle_commas: Whether to remove thousand commas between digits (default True).
        which: Which number to extract if multiple exist ('first' or 'last'). Default 'first'.
        errors: 'coerce' (return pd.NA if no number found), 'raise', or 'ignore'.

    Returns:
        Extracted float/int, or pd.NA if missing/unparseable.
    """
    if val is None or pd.isna(val):
        return pd.NA if as_type == "int" else np.nan
    if isinstance(val, (int, float)):
        return int(val) if as_type == "int" else float(val)

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>"):
        return pd.NA if as_type == "int" else np.nan

    if handle_commas:
        s = re.sub(r"(?<=\d),(?=\d)", "", s)

    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    valid_nums = [m for m in matches if m not in ("", ".", "-", "+")]

    if not valid_nums:
        if errors == "raise":
            raise ValueError(f"No numeric value found in text: {val!r}")
        elif errors == "coerce":
            return pd.NA if as_type == "int" else np.nan
        else:
            return val

    num_str = valid_nums[0] if which == "first" else valid_nums[-1]
    return int(float(num_str)) if as_type == "int" else float(num_str)


def extract_numeric_column(
    df: pd.DataFrame,
    column: str,
    as_type: str = "float",
    handle_commas: bool = True,
    which: str = "first",
    errors: str = "coerce",
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Extract numeric values from a text column and convert to numeric dtype."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")

    series = df[column]
    extracted: List[Any] = []
    affected_count = 0

    for idx, val in series.items():
        if val is None or pd.isna(val):
            extracted.append(pd.NA if as_type == "int" else np.nan)
            continue

        res = extract_numeric_value(
            val,
            as_type=as_type,
            handle_commas=handle_commas,
            which=which,
            errors=errors,
        )
        if pd.isna(res):
            extracted.append(pd.NA if as_type == "int" else np.nan)
        else:
            extracted.append(res)
        if res != val and not pd.isna(res):
            affected_count += 1

    df_clean = df.copy()
    s = pd.Series(extracted, index=series.index)
    target_dtype = "Int64" if as_type == "int" else "float64"
    if errors != "ignore":
        s = s.astype(target_dtype)
    else:
        try:
            s = s.astype(target_dtype)
        except (ValueError, TypeError):
            s = s.astype("object")
    df_clean[column] = s

    record = CleaningRecord(
        operation="extract_numeric",
        strategy=f"extract_{as_type}",
        columns=[column],
        affected_count=affected_count,
        details={
            "as_type": as_type,
            "handle_commas": handle_commas,
            "which": which,
            "errors": errors,
        },
        message=f"Extracted numeric ({as_type}) values from column '{column}' ({affected_count} modified).",
    )
    return df_clean, record


def clean_currency_value(
    val: Any, as_numeric: bool = True, errors: str = "coerce"
) -> Any:
    """Clean currency string, handling symbols ($€£¥₹), accounting negatives, and commas.

    Examples:
        '$1,250.00' -> 1250.0
        '($50.00)' -> -50.0
        '₹ 45,000' -> 45000.0

    Args:
        val: Input currency value.
        as_numeric: Return float if True, or formatted string if False (default True).
        errors: 'coerce' (default), 'raise', or 'ignore'.

    Returns:
        Float value, formatted string, or pd.NA if missing/unparseable.
    """
    if val is None or pd.isna(val):
        return np.nan if as_numeric else pd.NA
    if isinstance(val, (int, float)):
        return float(val) if as_numeric else str(val)

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>"):
        return np.nan if as_numeric else pd.NA

    # Check for accounting negative format: ($50.00) or (50.00)
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()
    elif s.startswith("-") or s.endswith("-"):
        is_negative = True
        s = s.replace("-", "").strip()

    # Remove currency symbols and codes
    s = CURRENCY_SYMBOLS_REGEX.sub("", s).strip()

    # Remove thousand commas
    s = re.sub(r"(?<=\d),(?=\d)", "", s).strip()

    try:
        num = float(s)
        if is_negative:
            num = -num
        return num if as_numeric else str(num)
    except ValueError as exc:
        if errors == "raise":
            raise ValueError(f"Could not parse currency value from: {val!r}") from exc
        elif errors == "coerce":
            return np.nan if as_numeric else pd.NA
        else:
            return val


def clean_currency_column(
    df: pd.DataFrame,
    column: str,
    as_numeric: bool = True,
    errors: str = "coerce",
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Clean currency strings in a DataFrame column."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")

    series = df[column]
    cleaned: List[Any] = []
    affected_count = 0

    for idx, val in series.items():
        if val is None or pd.isna(val):
            cleaned.append(np.nan if as_numeric else pd.NA)
            continue

        res = clean_currency_value(val, as_numeric=as_numeric, errors=errors)
        if pd.isna(res):
            cleaned.append(np.nan if as_numeric else pd.NA)
        else:
            cleaned.append(res)
        if res != val and not pd.isna(res):
            affected_count += 1

    df_clean = df.copy()
    s = pd.Series(cleaned, index=series.index)
    if as_numeric:
        if errors != "ignore":
            s = s.astype("float64")
        else:
            try:
                s = s.astype("float64")
            except (ValueError, TypeError):
                s = s.astype("object")
    else:
        s = s.astype("object")
    df_clean[column] = s

    record = CleaningRecord(
        operation="clean_currency",
        strategy="currency_parse",
        columns=[column],
        affected_count=affected_count,
        details={"as_numeric": as_numeric, "errors": errors},
        message=f"Cleaned currency values in column '{column}' ({affected_count} modified).",
    )
    return df_clean, record
