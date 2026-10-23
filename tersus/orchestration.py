"""Orchestration layer for Tersus.

Provides a unified clean() interface to execute selected cleaning operations
in a predictable, validated sequence while maintaining full auditability and
avoiding unrequested destructive changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tersus.cleaner import Cleaner

VALID_CLEANING_OPTIONS = {
    "missing",
    "duplicates",
    "column_names",
    "text",
    "whitespace",
    "invisible_chars",
    "types",
    "categories",
    "dates",
    "emails",
    "phones",
    "ranges",
    "outliers",
    "auto",
    "return_df",
}

VALID_MISSING_STRATEGIES = {"mean", "median", "mode", "constant", "drop"}


def validate_clean_options(options: Dict[str, Any]) -> None:
    """Validate requested cleaning options before execution.

    Args:
        options: Dictionary of cleaning keyword arguments.

    Raises:
        ValueError: If an unknown cleaning option or invalid strategy is supplied.
        TypeError: If an option has an incorrect type.
    """
    for key in options:
        if key not in VALID_CLEANING_OPTIONS:
            raise ValueError(
                f"Unsupported cleaning option: '{key}'. Supported options: "
                f"{sorted(list(VALID_CLEANING_OPTIONS))}"
            )

    # 1. Validate 'missing'
    missing = options.get("missing")
    if missing is not None and missing is not False:
        if isinstance(missing, str):
            if missing.lower() not in VALID_MISSING_STRATEGIES:
                raise ValueError(
                    f"Invalid missing cleaning strategy '{missing}'. "
                    f"Supported strategies: {sorted(list(VALID_MISSING_STRATEGIES))}"
                )
        elif isinstance(missing, bool):
            # True means drop missing or use default
            pass
        elif isinstance(missing, dict):
            if "strategy" in missing:
                strat = str(missing["strategy"]).lower()
                if strat not in VALID_MISSING_STRATEGIES:
                    raise ValueError(
                        f"Invalid missing cleaning strategy '{strat}' in configuration dict. "
                        f"Supported strategies: {sorted(list(VALID_MISSING_STRATEGIES))}"
                    )
        else:
            raise TypeError(
                f"'missing' option must be a strategy string, boolean, or configuration dict, "
                f"got {type(missing).__name__}."
            )

    # 2. Validate 'duplicates'
    duplicates = options.get("duplicates")
    if duplicates is not None and not isinstance(duplicates, (bool, dict)):
        raise TypeError(
            f"'duplicates' option must be a bool or a dict of options, got {type(duplicates).__name__}."
        )

    # 3. Validate 'column_names'
    column_names = options.get("column_names")
    if column_names is not None and not isinstance(column_names, (bool, dict)):
        raise TypeError(
            f"'column_names' option must be a bool or a dict of options, got {type(column_names).__name__}."
        )

    # 4. Validate 'text'
    text = options.get("text")
    if text is not None and not isinstance(text, (bool, dict)):
        raise TypeError(
            f"'text' option must be a bool or a dict of options, got {type(text).__name__}."
        )

    # 5. Validate 'whitespace'
    whitespace = options.get("whitespace")
    if whitespace is not None and not isinstance(whitespace, (bool, dict)):
        raise TypeError(
            f"'whitespace' option must be a bool or a dict of options, got {type(whitespace).__name__}."
        )

    # 6. Validate 'invisible_chars'
    invisible_chars = options.get("invisible_chars")
    if invisible_chars is not None and not isinstance(invisible_chars, (bool, dict)):
        raise TypeError(
            f"'invisible_chars' option must be a bool or a dict of options, got {type(invisible_chars).__name__}."
        )

    # 7. Validate 'types'
    types_opt = options.get("types")
    if types_opt is not None and not isinstance(types_opt, dict):
        raise TypeError(
            f"'types' option must be a dictionary mapping column names to target types, "
            f"got {type(types_opt).__name__}."
        )

    # 8. Validate 'categories'
    categories = options.get("categories")
    if categories is not None and not isinstance(categories, (bool, dict, list, tuple, str)):
        raise TypeError(
            f"'categories' option must be a boolean, dictionary, list, or column string, "
            f"got {type(categories).__name__}."
        )

    # 9. Validate 'dates'
    dates = options.get("dates")
    if dates is not None and not isinstance(dates, (bool, dict)):
        raise TypeError(
            f"'dates' option must be a boolean or dictionary of date standardization options, "
            f"got {type(dates).__name__}."
        )

    # 10. Validate 'emails'
    emails = options.get("emails")
    if emails is not None and not isinstance(emails, (bool, dict, list, tuple, str)):
        raise TypeError(
            f"'emails' option must be a boolean, dictionary, list, or column string, "
            f"got {type(emails).__name__}."
        )

    # 11. Validate 'phones'
    phones = options.get("phones")
    if phones is not None and not isinstance(phones, (bool, dict, list, tuple, str)):
        raise TypeError(
            f"'phones' option must be a boolean, dictionary, list, or column string, "
            f"got {type(phones).__name__}."
        )

    # 12. Validate 'ranges'
    ranges = options.get("ranges")
    if ranges is not None and not isinstance(ranges, (bool, dict)):
        raise TypeError(
            f"'ranges' option must be a boolean or a dictionary of column ranges, "
            f"got {type(ranges).__name__}."
        )

    # 13. Validate 'outliers'
    outliers = options.get("outliers")
    if outliers is not None and not isinstance(outliers, (bool, str, dict)):
        raise TypeError(
            f"'outliers' option must be a boolean, strategy string ('cap' or 'drop'), or dictionary, "
            f"got {type(outliers).__name__}."
        )

    # 14. Validate 'auto'
    auto = options.get("auto")
    if auto is not None and not isinstance(auto, bool):
        raise TypeError(
            f"'auto' option must be a boolean, got {type(auto).__name__}."
        )

    # 15. Validate 'return_df'
    return_df = options.get("return_df")
    if return_df is not None and not isinstance(return_df, bool):
        raise TypeError(
            f"'return_df' option must be a boolean, got {type(return_df).__name__}."
        )


def _resolve_col_name(df: pd.DataFrame, col: str) -> str:
    """Find matching column name in df, supporting snake_case or case-insensitive matching."""
    if col in df.columns:
        return col
    from tersus.columns import clean_single_column_name
    clean_c = clean_single_column_name(col)
    if clean_c in df.columns:
        return clean_c
    lower_map = {str(c).lower().strip(): str(c) for c in df.columns}
    if str(col).lower().strip() in lower_map:
        return lower_map[str(col).lower().strip()]
    return col


def execute_cleaning_pipeline(
    cleaner: Cleaner,
    missing: Any = None,
    duplicates: Any = None,
    column_names: Any = None,
    text: Any = None,
    whitespace: Any = None,
    invisible_chars: Any = None,
    types: Optional[Dict[str, str]] = None,
    categories: Optional[Any] = None,
    dates: Optional[Any] = None,
    emails: Optional[Any] = None,
    phones: Optional[Any] = None,
    ranges: Optional[Any] = None,
    outliers: Optional[Any] = None,
    auto: bool = False,
) -> Cleaner:
    """Execute selected cleaning operations on a Cleaner instance in predictable order.

    Execution Sequence:
    -------------------
    1. column_names: Standardize column headers first so downstream operations target clean names.
    2. invisible_chars / whitespace / text: Normalize text and strip invisible codes.
    3. duplicates: Drop redundant rows (after text cleaning so whitespace differences are resolved).
    4. types: Convert explicit or detected data types before value imputation.
    5. missing: Impute or drop missing values using clean statistics.
    6. categories: Normalize categorical variations and mappings.
    7. dates: Standardize date formats.
    8. emails: Clean email addresses and coerce invalid formats.
    9. phones: Standardize phone numbers and coerce invalid formats.
    10. ranges: Clamp numerical values to defined bounds.
    11. outliers: Cap or drop statistical anomalies.

    Args:
        cleaner: Cleaner instance to operate on.
        missing: Strategy ('median', 'mean', 'mode', 'drop') or config dict.
        duplicates: bool or config dict.
        column_names: bool or config dict.
        text: bool or config dict.
        whitespace: bool or config dict.
        invisible_chars: bool or config dict.
        types: Dict mapping column names to target types.
        categories: bool, config dict, or mapping.
        dates: bool or config dict for date standardization.
        emails: bool, config dict, column name, or list of columns.
        phones: bool, config dict, column name, or list of columns.
        ranges: bool or dict mapping columns to (min, max).
        outliers: bool, str ('cap', 'drop'), or config dict.
        auto: bool enabling full automated quality correction.

    Returns:
        Cleaner: The cleaner instance with updated data and history.
    """
    if auto:
        if column_names is None:
            column_names = True
        if invisible_chars is None:
            invisible_chars = True
        if whitespace is None:
            whitespace = True
        if text is None:
            text = True
        if duplicates is None:
            duplicates = True
        if missing is None:
            missing = "median"
        if categories is None:
            categories = True
        if dates is None:
            dates = True
        if emails is None:
            emails = True
        if phones is None:
            phones = True
        if ranges is None:
            ranges = True
        if outliers is None:
            outliers = "cap"

    # 1. Column names
    if column_names:
        opts = column_names if isinstance(column_names, dict) else {}
        cleaner.clean_column_names(**opts)

    # 2. Invisible characters
    if invisible_chars:
        opts = invisible_chars if isinstance(invisible_chars, dict) else {}
        cleaner.clean_invisible_chars(**opts)

    # 3. Whitespace
    if whitespace:
        opts = whitespace if isinstance(whitespace, dict) else {}
        cleaner.strip_whitespace(**opts)

    # 4. Text normalization
    if text:
        opts = text if isinstance(text, dict) else {}
        cleaner.clean_text(**opts)

    # 5. Categories
    if categories is True:
        from tersus.categories import find_category_variations
        profile = cleaner.profile()
        for col in profile.categorical_columns:
            cleaner.normalize_categories(column=col, case="title")
            try:
                res = find_category_variations(cleaner.data, col)
                if res.suggested_mapping:
                    cleaner.map_categories(column=col, mapping=res.suggested_mapping)
            except Exception:
                pass
    elif categories:
        if isinstance(categories, (list, tuple)):
            for c in categories:
                cleaner.normalize_categories(column=c, case="title")
        elif isinstance(categories, str):
            cleaner.normalize_categories(column=categories, case="title")
        elif isinstance(categories, dict):
            if "mapping" in categories:
                col = categories.get("column")
                mapping = categories["mapping"]
                if col:
                    cleaner.map_categories(
                        column=col,
                        mapping=mapping,
                        case_sensitive=categories.get("case_sensitive", False),
                        default=categories.get("default", None),
                    )
                else:
                    for c, m in mapping.items():
                        cleaner.map_categories(column=c, mapping=m)
            elif "column" in categories:
                cleaner.normalize_categories(**categories)
            else:
                for c, m in categories.items():
                    if isinstance(m, dict):
                        cleaner.map_categories(column=c, mapping=m)
                    else:
                        cleaner.normalize_categories(column=c)

    # 6. Dates
    if dates is True:
        from tersus.dates import detect_datetime_columns
        dt_cols = detect_datetime_columns(cleaner.data)
        for col in dt_cols:
            cleaner.standardize_dates(col, output_format="%Y-%m-%d", errors="coerce", dayfirst=True)
    elif dates:
        opts = dates if isinstance(dates, dict) else {}
        if "column" in opts:
            col = opts.pop("column")
            cleaner.standardize_dates(col, **opts)
        else:
            from tersus.dates import detect_datetime_columns
            dt_cols = detect_datetime_columns(cleaner.data)
            for col in dt_cols:
                cleaner.standardize_dates(col, **opts)

    # 7. Emails
    if emails:
        opts = emails if isinstance(emails, dict) else {"errors": "coerce"}
        target_cols = []
        if isinstance(emails, str):
            target_cols = [emails]
        elif isinstance(emails, (list, tuple)):
            target_cols = list(emails)
        else:
            for c in cleaner.data.columns:
                if "email" in c.lower():
                    target_cols.append(c)
        for c in target_cols:
            if c in cleaner.data.columns:
                cleaner.clean_email(c, **opts)

    # 8. Phones
    if phones:
        opts = phones if isinstance(phones, dict) else {"errors": "coerce", "output_format": "+91"}
        target_cols = []
        if isinstance(phones, str):
            target_cols = [phones]
        elif isinstance(phones, (list, tuple)):
            target_cols = list(phones)
        else:
            for c in cleaner.data.columns:
                if any(k in c.lower() for k in ("phone", "mobile", "contact", "cell")):
                    target_cols.append(c)
        for c in target_cols:
            if c in cleaner.data.columns:
                cleaner.clean_phone(c, **opts)

    # 9. Types conversion
    if auto and types is None:
        # Detect suspicious object columns that are mostly numeric
        from tersus.types import _parse_word_numbers
        auto_types: Dict[str, str] = {}
        for col in cleaner.data.columns:
            series = cleaner.data[col]
            if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    parsed = pd.to_numeric(_parse_word_numbers(non_null), errors="coerce")
                    if (parsed.notna().sum() / len(non_null)) >= 0.7:
                        is_int = bool((parsed.dropna() % 1 == 0).all()) if not parsed.dropna().empty else False
                        auto_types[col] = "int" if is_int else "float"
        for c, t in auto_types.items():
            cleaner.convert_type(c, target_type=t, errors="coerce", convert_words=True)
    elif types:
        resolved_types = {_resolve_col_name(cleaner.data, c): t for c, t in types.items()}
        for c, t in resolved_types.items():
            cleaner.convert_type(c, target_type=t, errors="coerce", convert_words=True)

    # 10. Duplicate rows (executed after text, category, date, email, phone, and type normalizations)
    if duplicates:
        opts = duplicates if isinstance(duplicates, dict) else {}
        cleaner.drop_duplicates(**opts)

    # 11. Missing values
    if missing is not None and missing is not False:
        if isinstance(missing, str):
            strat = missing.lower()
            if strat == "drop":
                cleaner.drop_missing(how="any")
            else:
                cleaner.fill_missing(strategy=strat)
        elif missing is True:
            cleaner.fill_missing(strategy="median")
        elif isinstance(missing, dict):
            if "strategy" in missing:
                strat = str(missing["strategy"]).lower()
                subset_cols = missing.get("subset", missing.get("columns"))
                if subset_cols:
                    if isinstance(subset_cols, (list, tuple)):
                        subset_cols = [_resolve_col_name(cleaner.data, c) for c in subset_cols]
                    else:
                        subset_cols = _resolve_col_name(cleaner.data, subset_cols)
                if strat == "drop":
                    cleaner.drop_missing(
                        how=missing.get("how", "any"),
                        subset=subset_cols,
                    )
                else:
                    cleaner.fill_missing(
                        strategy=strat,
                        columns=subset_cols,
                        value=missing.get("value"),
                    )
            else:
                # Dict mapping column name -> strategy or constant value
                for raw_col, spec in missing.items():
                    col = _resolve_col_name(cleaner.data, raw_col)
                    if isinstance(spec, str) and spec.lower() in ("median", "mean", "mode"):
                        cleaner.fill_missing(strategy=spec.lower(), columns=[col])
                    elif spec == "drop":
                        cleaner.drop_missing(subset=[col])
                    else:
                        cleaner.fill_missing(strategy="constant", columns=[col], value=spec)

    # 12. Ranges
    if ranges:
        if ranges is True:
            cleaner.clamp_ranges()
        elif isinstance(ranges, dict):
            cleaner.clamp_ranges(ranges)

    # 13. Outliers
    if outliers:
        method = "iqr"
        action = "cap"
        factor = 1.5
        target_cols = []
        if isinstance(outliers, str):
            action = outliers.lower()
        elif isinstance(outliers, dict):
            action = outliers.get("action", "cap").lower()
            method = outliers.get("method", "iqr")
            factor = outliers.get("factor", 1.5)
            if "column" in outliers:
                target_cols = [outliers["column"]]
            elif "columns" in outliers:
                target_cols = list(outliers["columns"])

        if not target_cols:
            profile = cleaner.profile()
            from tersus.outliers import detect_outliers
            range_cols = set()
            if ranges:
                if isinstance(ranges, dict):
                    range_cols.update(_resolve_col_name(cleaner.data, c) for c in ranges.keys())
                if hasattr(cleaner, "_range_rules"):
                    range_cols.update(_resolve_col_name(cleaner.data, c) for c in cleaner._range_rules.keys())
            for col in profile.numeric_columns:
                if col in range_cols:
                    continue
                try:
                    res = detect_outliers(cleaner.data, col, method=method, factor=factor)
                    if res.outlier_count > 0:
                        target_cols.append(col)
                except Exception:
                    pass

        for c in target_cols:
            if c in cleaner.data.columns:
                if action == "drop":
                    cleaner.drop_outliers(c, method=method, factor=factor)
                else:
                    cleaner.cap_outliers(c, method=method, factor=factor)

    return cleaner


def clean(
    data: Union[pd.DataFrame, Cleaner],
    missing: Any = None,
    duplicates: Any = None,
    column_names: Any = None,
    text: Any = None,
    whitespace: Any = None,
    invisible_chars: Any = None,
    types: Optional[Dict[str, str]] = None,
    categories: Optional[Any] = None,
    dates: Optional[Any] = None,
    emails: Optional[Any] = None,
    phones: Optional[Any] = None,
    ranges: Optional[Any] = None,
    outliers: Optional[Any] = None,
    auto: bool = False,
    return_df: Optional[bool] = None,
    **kwargs: Any,
) -> Union[Cleaner, pd.DataFrame]:
    """Top-level functional cleaning interface.

    Accepts either a pandas DataFrame or a Cleaner instance, validates options,
    and executes the selected cleaning operations in predictable order.

    Args:
        data: A pandas DataFrame or a Cleaner object.
        missing: 'median', 'mean', 'mode', 'drop', or config dict.
        duplicates: bool or config dict.
        column_names: bool or config dict.
        text: bool or config dict.
        whitespace: bool or config dict.
        invisible_chars: bool or config dict.
        types: Dict mapping column names to target types.
        categories: bool, config dict, or mapping.
        dates: bool or config dict for date standardization.
        emails: bool, config dict, or column name(s).
        phones: bool, config dict, or column name(s).
        ranges: bool or dict mapping columns to (min, max).
        outliers: bool, str ('cap', 'drop'), or config dict.
        auto: bool enabling full automated quality cleaning. When True and
            `types` is not explicitly provided, object columns are
            auto-detected as int/float conversion candidates using the same
            70%-parse-ratio heuristic documented in
            `tersus.missing.fill_missing`.
        return_df: Whether to return a DataFrame instead of Cleaner.
            Defaults to True if data is a DataFrame, or False if data is a Cleaner.
        **kwargs: Extra parameters (validated against supported options).

    Returns:
        pd.DataFrame or Cleaner depending on return_df setting.
    """
    from tersus.cleaner import Cleaner

    all_options = {
        "missing": missing,
        "duplicates": duplicates,
        "column_names": column_names,
        "text": text,
        "whitespace": whitespace,
        "invisible_chars": invisible_chars,
        "types": types,
        "categories": categories,
        "dates": dates,
        "emails": emails,
        "phones": phones,
        "ranges": ranges,
        "outliers": outliers,
        "auto": auto,
        **kwargs,
    }
    if return_df is not None:
        all_options["return_df"] = return_df

    validate_clean_options(all_options)

    if isinstance(data, Cleaner):
        cleaner_obj = data
        should_return_df = return_df if return_df is not None else False
    elif isinstance(data, pd.DataFrame):
        cleaner_obj = Cleaner(data)
        should_return_df = return_df if return_df is not None else True
    else:
        raise TypeError(
            f"Expected data to be a pandas DataFrame or Cleaner instance, "
            f"got {type(data).__name__}."
        )

    execute_cleaning_pipeline(
        cleaner=cleaner_obj,
        missing=missing,
        duplicates=duplicates,
        column_names=column_names,
        text=text,
        whitespace=whitespace,
        invisible_chars=invisible_chars,
        types=types,
        categories=categories,
        dates=dates,
        emails=emails,
        phones=phones,
        ranges=ranges,
        outliers=outliers,
        auto=auto,
    )

    return cleaner_obj.data if should_return_df else cleaner_obj
