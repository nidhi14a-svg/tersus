"""Data type inspection and controlled conversion module for Tersus."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tersus.missing import CleaningRecord

TRUE_VALUES = {"true", "1", "1.0", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "0.0", "no", "n", "f"}

CANONICAL_TYPES = {
    "int": "integer",
    "integer": "integer",
    "int64": "integer",
    "int32": "integer",
    "float": "float",
    "float64": "float",
    "float32": "float",
    "double": "float",
    "str": "string",
    "string": "string",
    "text": "string",
    "object": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "datetime": "datetime",
    "date": "datetime",
    "datetime64": "datetime",
    "timestamp": "datetime",
}


def get_column_dtypes(
    df: pd.DataFrame, column: Optional[str] = None
) -> Union[Dict[str, str], str]:
    """Inspect and return the Pandas data types of columns in a DataFrame.

    Args:
        df: The pandas DataFrame.
        column: Optional specific column name to inspect.

    Returns:
        Dict[str, str] mapping column names to dtype strings, or str for a single column.
    """
    if column is not None:
        if column not in df.columns:
            raise KeyError(
                f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
            )
        return str(df[column].dtype)

    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


WORD_TO_NUMBER = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}


def _parse_word_numbers(series: pd.Series) -> pd.Series:
    """Map common English word numbers to digits in a series."""
    def _map_val(v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in WORD_TO_NUMBER:
                return WORD_TO_NUMBER[s]
        return v

    return series.map(_map_val)


def _to_integer(
    series: pd.Series,
    col_name: str,
    errors: str = "raise",
    convert_words: bool = False,
) -> pd.Series:
    """Convert a Series to integer type safely preserving missing values."""
    if convert_words:
        series = _parse_word_numbers(series)
    try:
        num = pd.to_numeric(series, errors=errors)
    except Exception as exc:
        raise ValueError(
            f"Failed to convert column '{col_name}' to integer: {exc}"
        ) from exc

    # If there are missing values, use nullable Int64; otherwise int64
    has_na = bool(num.isna().any())
    target_dtype = "Int64" if has_na else "int64"

    try:
        return num.astype(target_dtype)
    except Exception as exc:
        raise ValueError(
            f"Failed to cast column '{col_name}' to {target_dtype}: {exc}"
        ) from exc


def _to_float(
    series: pd.Series,
    col_name: str,
    errors: str = "raise",
    convert_words: bool = False,
) -> pd.Series:
    """Convert a Series to float type safely preserving missing values."""
    if convert_words:
        series = _parse_word_numbers(series)
    try:
        num = pd.to_numeric(series, errors=errors)
        return num.astype("float64")
    except Exception as exc:
        raise ValueError(
            f"Failed to convert column '{col_name}' to float: {exc}"
        ) from exc


def _to_string(series: pd.Series) -> pd.Series:
    """Convert a Series to string type safely preserving missing values."""

    def _convert(v: Any) -> Any:
        if v is None or pd.isna(v):
            return v
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    return series.apply(_convert).astype("object")


def _to_boolean(series: pd.Series, col_name: str, errors: str = "raise") -> pd.Series:
    """Convert a Series to boolean type safely preserving missing values."""
    converted: List[Any] = []
    has_na = False

    for idx, val in series.items():
        if val is None or pd.isna(val):
            has_na = True
            converted.append(pd.NA)
        elif isinstance(val, (bool, np.bool_)):
            converted.append(bool(val))
        elif isinstance(val, (int, float, np.number)):
            if val == 1:
                converted.append(True)
            elif val == 0:
                converted.append(False)
            else:
                if errors == "raise":
                    raise ValueError(
                        f"Failed to convert column '{col_name}' to boolean: "
                        f"value '{val}' at row '{idx}' is not a valid boolean indicator."
                    )
                has_na = True
                converted.append(pd.NA)
        elif isinstance(val, str):
            clean_str = val.strip().lower()
            if clean_str in TRUE_VALUES:
                converted.append(True)
            elif clean_str in FALSE_VALUES:
                converted.append(False)
            elif clean_str == "":
                has_na = True
                converted.append(pd.NA)
            else:
                if errors == "raise":
                    raise ValueError(
                        f"Failed to convert column '{col_name}' to boolean: "
                        f"value '{val}' at row '{idx}' cannot be parsed as boolean."
                    )
                has_na = True
                converted.append(pd.NA)
        else:
            if errors == "raise":
                raise ValueError(
                    f"Failed to convert column '{col_name}' to boolean: "
                    f"unsupported type '{type(val).__name__}' at row '{idx}'."
                )
            has_na = True
            converted.append(pd.NA)

    target_dtype = "boolean" if has_na else "bool"
    return pd.Series(converted, index=series.index, dtype=target_dtype)


def _to_datetime(
    series: pd.Series, col_name: str, errors: str = "raise", **kwargs: Any
) -> pd.Series:
    """Convert a Series to datetime type safely preserving missing values."""
    try:
        return pd.to_datetime(series, errors=errors, **kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to convert column '{col_name}' to datetime: {exc}"
        ) from exc


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    errors: str = "raise",
    **kwargs: Any,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Convert a single DataFrame column to the specified target type.

    Args:
        df: The pandas DataFrame.
        column: Column name to convert.
        target_type: Target type ('int', 'float', 'string', 'boolean', 'datetime').
        errors: 'raise', 'coerce', or 'ignore'. Defaults to 'raise'.
        **kwargs: Additional parameters passed to conversion handlers (e.g. format for datetime).

    Returns:
        tuple[pd.DataFrame, CleaningRecord]: The updated DataFrame and operation record.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    if errors not in ("raise", "coerce", "ignore"):
        raise ValueError(
            f"Invalid errors parameter '{errors}'. Expected 'raise', 'coerce', or 'ignore'."
        )

    norm_type = CANONICAL_TYPES.get(target_type.lower())
    if not norm_type:
        supported = ", ".join(sorted(set(CANONICAL_TYPES.values())))
        raise ValueError(
            f"Unsupported target type '{target_type}'. Supported types: {supported}."
        )

    series = df[column]
    orig_dtype = str(series.dtype)

    try:
        convert_words = kwargs.get("convert_words", False)
        if norm_type == "integer":
            converted = _to_integer(series, column, errors=errors, convert_words=convert_words)
        elif norm_type == "float":
            converted = _to_float(series, column, errors=errors, convert_words=convert_words)
        elif norm_type == "string":
            converted = _to_string(series)
        elif norm_type == "boolean":
            converted = _to_boolean(series, column, errors=errors)
        elif norm_type == "datetime":
            converted = _to_datetime(series, column, errors=errors, **kwargs)
        else:
            raise ValueError(f"Unknown conversion handler for '{norm_type}'.")
    except Exception:
        if errors == "ignore":
            converted = series
        else:
            raise

    df_clean = df.copy()
    df_clean[column] = converted
    new_dtype = str(converted.dtype)

    msg = f"Converted column '{column}' from {orig_dtype} to {new_dtype}."
    record = CleaningRecord(
        operation="convert_type",
        strategy=norm_type,
        columns=[column],
        affected_count=len(df),
        details={
            "column": column,
            "target_type": target_type,
            "canonical_type": norm_type,
            "from_dtype": orig_dtype,
            "to_dtype": new_dtype,
        },
        message=msg,
    )

    return df_clean, record
