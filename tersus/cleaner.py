"""Cleaner module for Tersus."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd

from tersus.categories import (
    CategoryVariationsResult,
    find_category_variations,
    map_category_values,
    normalize_category_strings,
)
from tersus.columns import clean_dataframe_columns, make_unique_column_names
from tersus.consistency import (
    ConsistencyResult,
    ConsistencySuggestion,
    check_date_order,
    check_relationship,
    check_value_order,
    resolve_consistency,
)
from tersus.dates import (
    DateValidationResult,
    detect_datetime_columns,
    is_datetime_column,
    parse_column_dates,
    standardize_column_dates,
    validate_dates,
)
from tersus.emails import (
    clean_email_column,
    is_valid_email,
    normalize_email,
    validate_email,
)
from tersus.phones import (
    clean_phone_column,
    format_indian_phone,
    is_valid_phone,
    parse_indian_phone,
    validate_phone,
)
from tersus.duplicates import (
    drop_duplicates as duplicates_drop,
    get_duplicate_count,
    get_duplicates,
)
from tersus.missing import (
    CleaningRecord,
    drop_missing as missing_drop,
    fill_missing as missing_fill,
    get_missing_counts,
    get_missing_percentages,
)
from tersus.orchestration import (
    execute_cleaning_pipeline,
    validate_clean_options,
)
from tersus.outliers import (
    OutlierResult,
    cap_outliers as outliers_cap,
    detect_outliers,
    detect_outliers_iqr,
    detect_outliers_zscore,
    drop_outliers as outliers_drop,
)
from tersus.profiling import ProfileResult, profile_dataframe
from tersus.suggestions import (
    Suggestion,
    SuggestionReport,
    generate_suggestions,
)
from tersus.text import (
    clean_currency_column,
    clean_dataframe_text,
    clean_invisible_chars,
    clean_special_chars,
    extract_numeric_column,
    normalize_unicode,
)
from tersus.types import convert_column_type, get_column_dtypes
from tersus.units import convert_column_units
from tersus.validation import (
    ValidationResult,
    clamp_column_range,
    drop_invalid_rows,
    replace_invalid_values,
    validate_range,
    validate_rule,
    validate_values,
)
from tersus.reporting import (
    CleaningReport,
    DatasetInfo,
    OperationSummary,
    SessionReport,
    generate_session_report,
)
from tersus.verification import (
    RemainingIssue,
    ResolvedIssue,
    ValidationReport,
    validate_cleaning,
)


class Cleaner:
    """Core data cleaning and quality orchestrator for Tersus.

    Accepts either a CSV file path or a pandas DataFrame, stores an internal
    working copy of the data, and provides accessors, profiling, missing-value
    cleaning, and export functionality.
    """

    def __init__(self, data: Union[str, Path, pd.DataFrame, None] = None) -> None:
        """Initialize the Cleaner with a CSV file path or pandas DataFrame.

        Args:
            data: A file path to a CSV file (str or Path) or an existing pandas DataFrame.

        Raises:
            ValueError: If data is None, empty, points to a directory, or has an unsupported format.
            FileNotFoundError: If the specified CSV file does not exist.
            TypeError: If data is neither a valid file path nor a pandas DataFrame.
        """
        if data is None:
            raise ValueError(
                "A data source is required. Expected a CSV file path (str or Path) or a pandas DataFrame."
            )

        if isinstance(data, pd.DataFrame):
            self._df: pd.DataFrame = data.copy()
        elif isinstance(data, (str, Path)):
            path_str = str(data).strip()
            if not path_str:
                raise ValueError("CSV file path cannot be empty.")

            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"File not found: '{data}'")
            if path.is_dir():
                raise ValueError(f"Expected a CSV file, but got a directory: '{data}'")
            if path.suffix.lower() != ".csv":
                raise ValueError(
                    f"Unsupported file format '{path.suffix}'. Tersus Cleaner currently only supports CSV files."
                )

            try:
                self._df = pd.read_csv(path)
            except Exception as exc:
                raise ValueError(f"Failed to read CSV file '{data}': {exc}") from exc
        else:
            raise TypeError(
                f"Invalid input type: '{type(data).__name__}'. Expected a pandas DataFrame or a CSV file path (str or Path)."
            )

        if self._df.columns.duplicated().any():
            duplicated_names = sorted(set(self._df.columns[self._df.columns.duplicated()]))
            self._df.columns = make_unique_column_names(list(self._df.columns))
            warnings.warn(
                "Input data contains duplicate column name(s): "
                f"{', '.join(str(name) for name in duplicated_names)}. "
                "They have been automatically renamed with numeric suffixes "
                "(e.g. 'col', 'col_1') so that every column can be addressed "
                "unambiguously.",
                UserWarning,
                stacklevel=2,
            )

        self._initial_df: pd.DataFrame = self._df.copy()
        self._history: List[CleaningRecord] = []
        self._range_rules: Dict[str, Dict[str, Any]] = {}

    @property
    def initial_data(self) -> pd.DataFrame:
        """Access a copy of the initial DataFrame before any cleaning was applied.

        Returns:
            pd.DataFrame: Copy of the original DataFrame.
        """
        return self._initial_df.copy()

    @property
    def data(self) -> pd.DataFrame:
        """Access the current working DataFrame.

        Returns:
            pd.DataFrame: The current working DataFrame.
        """
        return self._df

    @property
    def history(self) -> List[CleaningRecord]:
        """Return the chronological history of cleaning actions performed."""
        return list(self._history)

    @property
    def last_operation(self) -> Optional[CleaningRecord]:
        """Return the most recent cleaning operation record, if any."""
        return self._history[-1] if self._history else None

    def profile(self) -> ProfileResult:
        """Inspect the dataset and identify its basic quality characteristics.

        Returns:
            ProfileResult: A structured object containing dataset quality statistics.
        """
        return profile_dataframe(self._df)

    def missing_counts(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Union[Dict[str, int], int]:
        """Return the missing-value count per column, or for a single column.

        Args:
            columns: Optional column name or sequence of column names.

        Returns:
            Dict[str, int] or int if a single column name is provided.
        """
        counts = get_missing_counts(self._df, columns)
        if isinstance(columns, str):
            return counts[columns]
        return counts

    def missing_count(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Union[Dict[str, int], int]:
        """Alias for missing_counts."""
        return self.missing_counts(columns)

    def missing_percentages(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Union[Dict[str, float], float]:
        """Return the missing-value percentage per column, or for a single column.

        Args:
            columns: Optional column name or sequence of column names.

        Returns:
            Dict[str, float] or float if a single column name is provided.
        """
        percentages = get_missing_percentages(self._df, columns)
        if isinstance(columns, str):
            return percentages[columns]
        return percentages

    def missing_percentage(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Union[Dict[str, float], float]:
        """Alias for missing_percentages."""
        return self.missing_percentages(columns)

    def has_missing(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> bool:
        """Return True if the dataset (or specified columns) contains any missing values."""
        counts = get_missing_counts(self._df, columns)
        return any(c > 0 for c in counts.values())

    def fill_missing(
        self,
        strategy: str,
        columns: Optional[Union[str, Sequence[str]]] = None,
        value: Any = None,
    ) -> Cleaner:
        """Fill missing values using an explicitly chosen strategy.

        Args:
            strategy: Cleaning strategy ('mean', 'median', 'mode', 'constant').
            columns: Target column or list of columns. If None, targets columns with missing values.
            value: Constant value when strategy='constant'.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = missing_fill(
            self._df, strategy=strategy, columns=columns, value=value
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def fill_mean(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Cleaner:
        """Fill missing values in numeric columns using their mean.

        Args:
            columns: Optional target column or sequence of columns.

        Returns:
            Cleaner: self for method chaining.
        """
        return self.fill_missing(strategy="mean", columns=columns)

    def fill_median(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Cleaner:
        """Fill missing values in numeric columns using their median.

        Args:
            columns: Optional target column or sequence of columns.

        Returns:
            Cleaner: self for method chaining.
        """
        return self.fill_missing(strategy="median", columns=columns)

    def fill_mode(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Cleaner:
        """Fill missing values in columns using their mode (most frequent value).

        Args:
            columns: Optional target column or sequence of columns.

        Returns:
            Cleaner: self for method chaining.
        """
        return self.fill_missing(strategy="mode", columns=columns)

    def fill_constant(
        self, value: Any, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Cleaner:
        """Fill missing values in columns using a user-provided constant value.

        Args:
            value: The constant value to fill missing entries with.
            columns: Optional target column or sequence of columns.

        Returns:
            Cleaner: self for method chaining.
        """
        return self.fill_missing(strategy="constant", columns=columns, value=value)

    def drop_missing(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        how: str = "any",
    ) -> Cleaner:
        """Drop rows containing missing values.

        Args:
            columns: Optional column name or sequence of columns to consider.
            how: 'any' to drop rows with at least one missing value, 'all' to drop if all are missing.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = missing_drop(self._df, columns=columns, how=how)
        self._df = cleaned_df
        self._history.append(record)
        return self

    def duplicate_count(
        self,
        subset: Optional[Union[str, Sequence[str]]] = None,
        keep: Union[str, bool] = "first",
    ) -> int:
        """Return the count of duplicate rows in the dataset.

        Args:
            subset: Column or list of columns to consider. If None, considers all columns.
            keep: 'first', 'last', or False.

        Returns:
            int: Number of duplicate rows.
        """
        return get_duplicate_count(self._df, subset=subset, keep=keep)

    def has_duplicates(
        self,
        subset: Optional[Union[str, Sequence[str]]] = None,
    ) -> bool:
        """Return True if the dataset contains any duplicate rows.

        Args:
            subset: Optional column or sequence of columns to consider.

        Returns:
            bool: True if duplicate rows exist, False otherwise.
        """
        return self.duplicate_count(subset=subset) > 0

    def duplicates(
        self,
        subset: Optional[Union[str, Sequence[str]]] = None,
        keep: Union[str, bool] = "first",
    ) -> pd.DataFrame:
        """Return duplicate rows from the dataset.

        Args:
            subset: Column or list of columns to consider. If None, considers all columns.
            keep: 'first', 'last', or False.

        Returns:
            pd.DataFrame: A DataFrame containing only duplicate rows.
        """
        return get_duplicates(self._df, subset=subset, keep=keep)

    def drop_duplicates(
        self,
        subset: Optional[Union[str, Sequence[str]]] = None,
        keep: Union[str, bool] = "first",
    ) -> Cleaner:
        """Remove duplicate rows from the dataset.

        Args:
            subset: Column or list of columns to consider. If None, considers all columns.
            keep: 'first', 'last', or False. Defaults to 'first'.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = duplicates_drop(self._df, subset=subset, keep=keep)
        self._df = cleaned_df
        self._history.append(record)
        return self

    def remove_duplicates(
        self,
        subset: Optional[Union[str, Sequence[str]]] = None,
        keep: Union[str, bool] = "first",
    ) -> Cleaner:
        """Alias for drop_duplicates."""
        return self.drop_duplicates(subset=subset, keep=keep)

    def clean_column_names(
        self,
        case: Optional[str] = "lower",
        replace_spaces: str = "_",
        remove_special: bool = True,
        strip: bool = True,
        normalize_accents: bool = True,
        rename: Optional[Dict[str, str]] = None,
    ) -> Cleaner:
        """Clean and standardize the DataFrame's column names.

        Args:
            case: Case transformation ('lower', 'upper', 'snake', or None/'preserve'). Defaults to 'lower'.
            replace_spaces: Character or string to replace whitespace and separators with. Defaults to '_'.
            remove_special: Whether to strip unwanted special characters. Defaults to True.
            strip: Whether to strip leading/trailing whitespace and separators. Defaults to True.
            normalize_accents: Whether to normalize accented characters. Defaults to True.
            rename: Optional dictionary mapping old column names directly to new names.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = clean_dataframe_columns(
            self._df,
            case=case,
            replace_spaces=replace_spaces,
            remove_special=remove_special,
            strip=strip,
            normalize_accents=normalize_accents,
            rename=rename,
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def clean_columns(
        self,
        case: Optional[str] = "lower",
        replace_spaces: str = "_",
        remove_special: bool = True,
        strip: bool = True,
        normalize_accents: bool = True,
        rename: Optional[Dict[str, str]] = None,
    ) -> Cleaner:
        """Alias for clean_column_names."""
        return self.clean_column_names(
            case=case,
            replace_spaces=replace_spaces,
            remove_special=remove_special,
            strip=strip,
            normalize_accents=normalize_accents,
            rename=rename,
        )

    def clean_text(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        strip: bool = True,
        collapse_spaces: bool = True,
        remove_invisible: bool = True,
        case: Optional[str] = None,
    ) -> Cleaner:
        """Normalize text in the specified columns or all non-numeric columns.

        Args:
            columns: Target column or list of columns. If None, targets all non-numeric columns.
            strip: Whether to strip leading/trailing whitespace. Defaults to True.
            collapse_spaces: Whether to collapse multiple spaces into one. Defaults to True.
            remove_invisible: Whether to replace tabs, newlines, etc. with space. Defaults to True.
            case: Case transformation ('lower', 'upper', 'title', or None/'preserve'). Defaults to None.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = clean_dataframe_text(
            self._df,
            columns=columns,
            strip=strip,
            collapse_spaces=collapse_spaces,
            remove_invisible=remove_invisible,
            case=case,
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def to_lower(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        strip: bool = True,
    ) -> Cleaner:
        """Convert text columns to lowercase."""
        return self.clean_text(columns=columns, case="lower", strip=strip)

    def to_lowercase(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        strip: bool = True,
    ) -> Cleaner:
        """Alias for to_lower."""
        return self.to_lower(columns=columns, strip=strip)

    def to_upper(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        strip: bool = True,
    ) -> Cleaner:
        """Convert text columns to uppercase."""
        return self.clean_text(columns=columns, case="upper", strip=strip)

    def to_uppercase(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        strip: bool = True,
    ) -> Cleaner:
        """Alias for to_upper."""
        return self.to_upper(columns=columns, strip=strip)

    def to_title(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        strip: bool = True,
    ) -> Cleaner:
        """Convert text columns to title case."""
        return self.clean_text(columns=columns, case="title", strip=strip)

    def to_titlecase(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        strip: bool = True,
    ) -> Cleaner:
        """Alias for to_title."""
        return self.to_title(columns=columns, strip=strip)

    def strip_whitespace(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
    ) -> Cleaner:
        """Strip leading/trailing whitespace and collapse internal spaces in text columns."""
        return self.clean_text(columns=columns, strip=True, collapse_spaces=True, case=None)

    def dtypes(
        self, column: Optional[str] = None
    ) -> Union[Dict[str, str], str]:
        """Return the Pandas data types of all columns or a specific column.

        Args:
            column: Optional column name.

        Returns:
            Dict[str, str] mapping column names to dtype strings, or str for a single column.
        """
        return get_column_dtypes(self._df, column=column)

    def inspect_types(
        self, column: Optional[str] = None
    ) -> Union[Dict[str, str], str]:
        """Inspect and return current column data types.

        Args:
            column: Optional column name.

        Returns:
            Dict[str, str] or str.
        """
        return self.dtypes(column=column)

    def convert_type(
        self,
        column: Union[str, Dict[str, str]],
        target_type: Optional[str] = None,
        errors: str = "raise",
        **kwargs: Any,
    ) -> Cleaner:
        """Convert a column or multiple columns to the specified target type.

        Args:
            column: Column name (str) or a dictionary mapping column names to target types.
            target_type: Target type name ('int', 'float', 'string', 'boolean', 'datetime').
                Required if column is a str.
            errors: How to handle conversion errors: 'raise' (default), 'coerce', or 'ignore'.
            **kwargs: Extra parameters for type converters (e.g. format for datetime).

        Returns:
            Cleaner: self for method chaining.
        """
        if isinstance(column, dict):
            for col, t_type in column.items():
                cleaned_df, record = convert_column_type(
                    self._df, col, t_type, errors=errors, **kwargs
                )
                self._df = cleaned_df
                self._history.append(record)
            return self

        if target_type is None:
            raise ValueError(
                "target_type must be specified when converting a single column."
            )

        cleaned_df, record = convert_column_type(
            self._df, column, target_type, errors=errors, **kwargs
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def convert_types(
        self,
        conversions: Dict[str, str],
        errors: str = "raise",
        **kwargs: Any,
    ) -> Cleaner:
        """Convert multiple columns to their respective target types.

        Args:
            conversions: Dictionary mapping column names to target types.
            errors: 'raise', 'coerce', or 'ignore'.
            **kwargs: Extra parameters for converters.

        Returns:
            Cleaner: self for method chaining.
        """
        return self.convert_type(conversions, errors=errors, **kwargs)

    def check_range(
        self,
        column: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        inclusive: bool = True,
    ) -> ValidationResult:
        """Find values in a column outside the specified numerical range.

        Args:
            column: Name of the column to validate.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            inclusive: Whether bounds are inclusive. Defaults to True.

        Returns:
            ValidationResult: Structured result reporting invalid values, indices, and rows.
        """
        self._range_rules[column] = {
            "min_value": min_value,
            "max_value": max_value,
            "inclusive": inclusive,
        }
        return validate_range(
            self._df,
            column=column,
            min_value=min_value,
            max_value=max_value,
            inclusive=inclusive,
        )

    def check_values(
        self,
        column: str,
        allowed_values: Sequence[Any],
    ) -> ValidationResult:
        """Find values in a column not in the allowed list of values.

        Args:
            column: Name of the column to validate.
            allowed_values: Sequence of allowed values.

        Returns:
            ValidationResult: Structured result reporting invalid values, indices, and rows.
        """
        return validate_values(self._df, column=column, allowed_values=allowed_values)

    def check_rule(
        self,
        column: str,
        rule: Callable[[Any], bool],
        rule_name: str = "custom_rule",
    ) -> ValidationResult:
        """Find values in a column that violate a custom predicate function.

        Args:
            column: Name of the column to validate.
            rule: Callable taking a value and returning True if valid, False if invalid.
            rule_name: Description or name of the rule.

        Returns:
            ValidationResult: Structured result reporting invalid values, indices, and rows.
        """
        return validate_rule(self._df, column=column, rule=rule, rule_name=rule_name)

    def drop_invalid(self, validation_result: ValidationResult) -> Cleaner:
        """Drop rows identified as invalid by a ValidationResult.

        Args:
            validation_result: The ValidationResult containing invalid indices.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = drop_invalid_rows(self._df, validation_result)
        self._df = cleaned_df
        self._history.append(record)
        return self

    def replace_invalid(
        self, validation_result: ValidationResult, value: Any
    ) -> Cleaner:
        """Replace invalid values identified by a ValidationResult with a replacement value.

        Args:
            validation_result: The ValidationResult containing invalid indices.
            value: The replacement value.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = replace_invalid_values(
            self._df, validation_result, value=value
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def clamp_range(
        self,
        column: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
    ) -> Cleaner:
        """Clamp numeric values in a column to the specified range.

        Args:
            column: Column name.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = clamp_column_range(
            self._df, column=column, min_value=min_value, max_value=max_value
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def clamp_ranges(
        self, ranges: Optional[Dict[str, Any]] = None
    ) -> Cleaner:
        """Clamp numeric values for multiple columns to their specified ranges.

        Args:
            ranges: Optional dictionary mapping column names to range specifications
                (e.g. {'marks': (0, 100)} or {'marks': {'min_value': 0, 'max_value': 100}}).
                If None, uses any range rules registered via check_range().

        Returns:
            Cleaner: self for method chaining.
        """
        targets = dict(self._range_rules) if ranges is None else {}
        if ranges is not None:
            for col, spec in ranges.items():
                if isinstance(spec, (tuple, list)) and len(spec) == 2:
                    targets[col] = {"min_value": spec[0], "max_value": spec[1]}
                elif isinstance(spec, dict):
                    targets[col] = spec

        for col, spec in targets.items():
            actual_col = col
            if actual_col not in self._df.columns:
                from tersus.columns import clean_single_column_name
                clean_name = clean_single_column_name(actual_col)
                if clean_name in self._df.columns:
                    actual_col = clean_name
                else:
                    continue
            min_v = spec.get("min_value")
            max_v = spec.get("max_value")
            self.clamp_range(actual_col, min_value=min_v, max_value=max_v)

        return self

    def detect_datetime_columns(self, threshold: float = 0.5) -> List[str]:
        """Detect columns in the dataset that contain datetime or date-like values.

        Args:
            threshold: Minimum fraction of non-empty values that must match date
                patterns (default 0.5).

        Returns:
            List[str]: Column names detected as datetime-like.
        """
        return detect_datetime_columns(self._df, threshold=threshold)

    def is_datetime_column(self, column: str, threshold: float = 0.5) -> bool:
        """Check if a specific column contains datetime or date-like values.

        Args:
            column: Name of the column to evaluate.
            threshold: Minimum fraction of non-empty values matching date patterns.

        Returns:
            bool: True if column is datetime-like.
        """
        if column not in self._df.columns:
            raise KeyError(
                f"Column '{column}' not found in dataset. Available columns: {list(self._df.columns)}"
            )
        return is_datetime_column(self._df[column], threshold=threshold)

    def check_dates(
        self,
        column: str,
        format: Optional[str] = None,
        dayfirst: Optional[bool] = None,
        allow_na: bool = True,
    ) -> DateValidationResult:
        """Identify invalid and ambiguous date values in a column without modifying data.

        Args:
            column: Name of the column to validate.
            format: Optional explicit strftime format or shorthand (e.g. 'YYYY-MM-DD').
            dayfirst: Optional boolean for day-first parsing. When None, ambiguous
                dates like '01/10/2026' are flagged.
            allow_na: Whether missing values are allowed (default True).

        Returns:
            DateValidationResult: Structured validation result containing invalid
                and ambiguous dates, row indices, and reasons.
        """
        return validate_dates(
            self._df,
            column=column,
            format=format,
            dayfirst=dayfirst,
            allow_na=allow_na,
        )

    def validate_dates(
        self,
        column: str,
        format: Optional[str] = None,
        dayfirst: Optional[bool] = None,
        allow_na: bool = True,
    ) -> DateValidationResult:
        """Validate dates in a column (alias for check_dates).

        Args:
            column: Name of the column to validate.
            format: Optional explicit strftime format or shorthand.
            dayfirst: Optional boolean for day-first parsing.
            allow_na: Whether missing values are allowed (default True).

        Returns:
            DateValidationResult: Structured validation result.
        """
        return self.check_dates(
            column=column,
            format=format,
            dayfirst=dayfirst,
            allow_na=allow_na,
        )

    def parse_dates(
        self,
        column: Union[str, Sequence[str]],
        format: Optional[str] = None,
        dayfirst: Optional[bool] = None,
        errors: str = "raise",
    ) -> Cleaner:
        """Parse valid date strings into datetime values in one or more columns.

        Args:
            column: Column name or sequence of column names to parse.
            format: Optional explicit strftime format or shorthand.
            dayfirst: Optional boolean for day-first parsing.
            errors: 'raise' (default), 'coerce', or 'ignore'.

        Returns:
            Cleaner: self for method chaining.
        """
        cols = [column] if isinstance(column, str) else list(column)
        for col in cols:
            cleaned_df, record = parse_column_dates(
                self._df,
                column=col,
                format=format,
                dayfirst=dayfirst,
                errors=errors,
            )
            self._df = cleaned_df
            self._history.append(record)
        return self

    def to_datetime(
        self,
        column: Union[str, Sequence[str]],
        format: Optional[str] = None,
        dayfirst: Optional[bool] = None,
        errors: str = "raise",
    ) -> Cleaner:
        """Convert date strings to datetime (alias for parse_dates).

        Args:
            column: Column name or sequence of column names.
            format: Optional explicit strftime format or shorthand.
            dayfirst: Optional boolean for day-first parsing.
            errors: 'raise', 'coerce', or 'ignore'.

        Returns:
            Cleaner: self for method chaining.
        """
        return self.parse_dates(
            column=column,
            format=format,
            dayfirst=dayfirst,
            errors=errors,
        )

    def standardize_dates(
        self,
        column: Union[str, Sequence[str]],
        output_format: str = "%Y-%m-%d",
        input_format: Optional[str] = None,
        dayfirst: Optional[bool] = None,
        errors: str = "raise",
    ) -> Cleaner:
        """Standardize date strings into a uniform user-selected output format.

        Args:
            column: Column name or sequence of column names to standardize.
            output_format: Desired date format (e.g. '%Y-%m-%d', 'DD/MM/YYYY').
            input_format: Optional explicit input format.
            dayfirst: Optional boolean for day-first parsing.
            errors: 'raise' (default), 'coerce', or 'ignore'.

        Returns:
            Cleaner: self for method chaining.
        """
        cols = [column] if isinstance(column, str) else list(column)
        for col in cols:
            cleaned_df, record = standardize_column_dates(
                self._df,
                column=col,
                output_format=output_format,
                input_format=input_format,
                dayfirst=dayfirst,
                errors=errors,
            )
            self._df = cleaned_df
            self._history.append(record)
        return self

    def format_dates(
        self,
        column: Union[str, Sequence[str]],
        output_format: str = "%Y-%m-%d",
        input_format: Optional[str] = None,
        dayfirst: Optional[bool] = None,
        errors: str = "raise",
    ) -> Cleaner:
        """Standardize date strings (alias for standardize_dates).

        Args:
            column: Column name or sequence of column names.
            output_format: Desired date format.
            input_format: Optional explicit input format.
            dayfirst: Optional boolean for day-first parsing.
            errors: 'raise', 'coerce', or 'ignore'.

        Returns:
            Cleaner: self for method chaining.
        """
        return self.standardize_dates(
            column=column,
            output_format=output_format,
            input_format=input_format,
            dayfirst=dayfirst,
            errors=errors,
        )

    def check_outliers(
        self,
        column: str,
        method: str = "iqr",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> OutlierResult:
        """Inspect a numerical column for potential statistical outliers.

        Supports the IQR (Interquartile Range) and Z-Score (standard score) methods.
        Note: An outlier is not automatically an error or invalid data. This check
        is purely diagnostic and does not mutate the DataFrame.

        Mathematical Formulas:
            IQR Method:
                IQR = Q3 - Q1
                Lower Bound = Q1 - (threshold * IQR)  [default threshold: 1.5]
                Upper Bound = Q3 + (threshold * IQR)
                Condition: x < Lower Bound OR x > Upper Bound

            Z-Score Method:
                mean (μ) = (1 / n) * Σ(x_i)
                std (σ) = sqrt((1 / (n - ddof)) * Σ(x_i - μ)^2)
                z = (x - μ) / σ
                Lower Bound = μ - (threshold * σ)    [default threshold: 3.0]
                Upper Bound = μ + (threshold * σ)
                Condition: |z| > threshold

        Args:
            column: Name of the numerical column to inspect.
            method: 'iqr' (default) or 'zscore'.
            threshold: Multiplier threshold (defaults to 1.5 for IQR, 3.0 for Z-score).
            **kwargs: Extra arguments (e.g. ddof for zscore).

        Returns:
            OutlierResult: Structured result reporting detected outliers, row
                indices, threshold used, and bounds.
        """
        return detect_outliers(
            self._df, column=column, method=method, threshold=threshold, **kwargs
        )

    def detect_outliers(
        self,
        column: str,
        method: str = "iqr",
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> OutlierResult:
        """Alias for check_outliers."""
        return self.check_outliers(
            column=column, method=method, threshold=threshold, **kwargs
        )

    def check_outliers_iqr(
        self, column: str, threshold: float = 1.5
    ) -> OutlierResult:
        """Inspect a numerical column for outliers using the IQR method.

        Formula:
            IQR = Q3 - Q1
            Bounds = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

        Args:
            column: Name of numerical column.
            threshold: IQR multiplier threshold (default 1.5).

        Returns:
            OutlierResult: Structured report of detected outliers.
        """
        return detect_outliers_iqr(self._df, column=column, threshold=threshold)

    def check_outliers_zscore(
        self, column: str, threshold: float = 3.0, ddof: int = 1
    ) -> OutlierResult:
        """Inspect a numerical column for outliers using the Z-Score method.

        Formula:
            z = (x - μ) / σ
            Bounds = [μ - 3*σ, μ + 3*σ]

        Args:
            column: Name of numerical column.
            threshold: Z-score threshold (default 3.0).
            ddof: Delta Degrees of Freedom (default 1).

        Returns:
            OutlierResult: Structured report of detected outliers.
        """
        return detect_outliers_zscore(
            self._df, column=column, threshold=threshold, ddof=ddof
        )

    def drop_outliers(
        self, outlier_result: Union[OutlierResult, str], **kwargs: Any
    ) -> Cleaner:
        """Explicitly drop rows identified as outliers by an OutlierResult.

        Args:
            outlier_result: An OutlierResult or column name string.
            **kwargs: Extra parameters (e.g. method='iqr', factor=1.5).

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = outliers_drop(self._df, outlier_result, **kwargs)
        self._df = cleaned_df
        self._history.append(record)
        return self

    def cap_outliers(
        self, outlier_result: Union[OutlierResult, str], **kwargs: Any
    ) -> Cleaner:
        """Explicitly cap outlier values to the computed statistical boundaries.

        Args:
            outlier_result: An OutlierResult or column name string.
            **kwargs: Extra parameters (e.g. method='iqr', factor=1.5).

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = outliers_cap(self._df, outlier_result, **kwargs)
        self._df = cleaned_df
        self._history.append(record)
        return self

    def clip_outliers(
        self, outlier_result: Union[OutlierResult, str], **kwargs: Any
    ) -> Cleaner:
        """Alias for cap_outliers."""
        return self.cap_outliers(outlier_result, **kwargs)

    def check_email(self, column: str, allow_na: bool = True) -> ValidationResult:
        """Validate email formats in a column, identifying invalid entries.

        Note: Validating format does not verify that the email address actually exists.

        Args:
            column: Name of the column containing email addresses.
            allow_na: Whether missing values are allowed without failing validation (default True).

        Returns:
            ValidationResult: Structured validation report with invalid rows and indices.
        """
        return validate_email(self._df, column=column, allow_na=allow_na)

    def validate_email(self, column: str, allow_na: bool = True) -> ValidationResult:
        """Alias for check_email."""
        return self.check_email(column=column, allow_na=allow_na)

    def clean_email(
        self,
        column: Union[str, Sequence[str]],
        lowercase: bool = True,
        strip_whitespace: bool = True,
        errors: str = "ignore",
    ) -> Cleaner:
        """Normalize email addresses in one or more columns (whitespace, casing).

        Note: Tersus normalizes format syntax only; it does not claim that an email
        address actually exists or is deliverable.

        Args:
            column: Column name or sequence of column names to clean.
            lowercase: Whether to convert email to lowercase (default True).
            strip_whitespace: Whether to strip surrounding whitespace (default True).
            errors: 'ignore' (keep invalid emails normalized), 'coerce' (turn invalid emails into pd.NA),
                or 'raise' (raise ValueError if invalid email found).

        Returns:
            Cleaner: self for method chaining.
        """
        cols = [column] if isinstance(column, str) else list(column)
        for col in cols:
            cleaned_df, record = clean_email_column(
                self._df,
                column=col,
                lowercase=lowercase,
                strip_whitespace=strip_whitespace,
                errors=errors,
            )
            self._df = cleaned_df
            self._history.append(record)
        return self

    def normalize_email(
        self,
        column: Union[str, Sequence[str]],
        lowercase: bool = True,
        strip_whitespace: bool = True,
        errors: str = "ignore",
    ) -> Cleaner:
        """Alias for clean_email."""
        return self.clean_email(
            column=column,
            lowercase=lowercase,
            strip_whitespace=strip_whitespace,
            errors=errors,
        )

    def check_phone(
        self, column: str, country: str = "IN", allow_na: bool = True
    ) -> ValidationResult:
        """Validate phone numbers in a column, identifying invalid entries.

        Note: Validating format does not verify that the phone number is active or connected.

        Args:
            column: Name of the column containing phone numbers.
            country: Country code ('IN' supported for V1).
            allow_na: Whether missing values are allowed without failing validation (default True).

        Returns:
            ValidationResult: Structured validation report with invalid rows and indices.
        """
        return validate_phone(self._df, column=column, country=country, allow_na=allow_na)

    def validate_phone(
        self, column: str, country: str = "IN", allow_na: bool = True
    ) -> ValidationResult:
        """Alias for check_phone."""
        return self.check_phone(column=column, country=country, allow_na=allow_na)

    def clean_phone(
        self,
        column: Union[str, Sequence[str]],
        country: str = "IN",
        output_format: str = "10-digit",
        errors: str = "raise",
    ) -> Cleaner:
        """Normalize phone numbers in one or more columns into standard representation.

        Note: Tersus normalizes format syntax only; it does not claim that a phone
        number is active or reachable.

        Args:
            column: Column name or sequence of column names.
            country: Country code ('IN' supported for V1).
            output_format: '10-digit', '+91', 'spaced', 'hyphenated'.
            errors: 'raise' (default), 'coerce' (turn invalid into pd.NA), or 'ignore'.

        Returns:
            Cleaner: self for method chaining.
        """
        cols = [column] if isinstance(column, str) else list(column)
        for col in cols:
            cleaned_df, record = clean_phone_column(
                self._df,
                column=col,
                country=country,
                output_format=output_format,
                errors=errors,
            )
            self._df = cleaned_df
            self._history.append(record)
        return self

    def normalize_phone(
        self,
        column: Union[str, Sequence[str]],
        country: str = "IN",
        output_format: str = "10-digit",
        errors: str = "raise",
    ) -> Cleaner:
        """Alias for clean_phone."""
        return self.clean_phone(
            column=column,
            country=country,
            output_format=output_format,
            errors=errors,
        )

    def inspect_categories(self, column: str) -> CategoryVariationsResult:
        """Inspect a column to discover possible categorical variations.

        Groups case/whitespace variants and abbreviation candidates for human review.
        Note: This is diagnostic only. Tersus does NOT automatically assume two
        categories mean the same thing without an explicit mapping.

        Args:
            column: Name of the categorical column to inspect.

        Returns:
            CategoryVariationsResult: Report of discovered unique values and candidate variations.
        """
        return find_category_variations(self._df, column=column)

    def find_category_variations(self, column: str) -> CategoryVariationsResult:
        """Alias for inspect_categories."""
        return self.inspect_categories(column=column)

    def map_categories(
        self,
        column: str,
        mapping: Dict[Any, Any],
        case_sensitive: bool = False,
        default: Any = None,
    ) -> Cleaner:
        """Explicitly map categorical values to standard representations.

        Args:
            column: Name of categorical column.
            mapping: Dictionary specifying category transformations.
                Can be {'M': 'Male', 'm': 'Male'} or {'Male': ['M', 'm', 'MALE']}.
            case_sensitive: Whether key lookup is case-sensitive (default False).
            default: Optional value for items not present in mapping.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = map_category_values(
            self._df,
            column=column,
            mapping=mapping,
            case_sensitive=case_sensitive,
            default=default,
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def normalize_categories(
        self,
        column: Union[str, Sequence[str]],
        case: Optional[str] = "title",
        strip_whitespace: bool = True,
    ) -> Cleaner:
        """Safely normalize casing and whitespace for all categorical text in a column.

        Args:
            column: Column name or sequence of column names.
            case: 'title', 'lower', 'upper', or None.
            strip_whitespace: Whether to strip leading/trailing spaces (default True).

        Returns:
            Cleaner: self for method chaining.
        """
        cols = [column] if isinstance(column, str) else list(column)
        for col in cols:
            cleaned_df, record = normalize_category_strings(
                self._df, column=col, case=case, strip_whitespace=strip_whitespace
            )
            self._df = cleaned_df
            self._history.append(record)
        return self

    def clean_invisible_chars(
        self, columns: Optional[Union[str, Sequence[str]]] = None
    ) -> Cleaner:
        """Remove zero-width spaces, control characters, and normalize non-breaking spaces.

        Args:
            columns: Column name or sequence of column names (defaults to all text columns).

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = clean_invisible_chars(self._df, columns=columns)
        self._df = cleaned_df
        self._history.append(record)
        return self

    def normalize_unicode(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        form: str = "NFC",
        strip_accents: bool = False,
    ) -> Cleaner:
        """Apply safe Unicode normalization and optional accent stripping.

        Args:
            columns: Column name or sequence of column names (defaults to all text columns).
            form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD'). Default 'NFC'.
            strip_accents: Whether to remove diacritical accents (e.g. 'Café' -> 'Cafe').

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = normalize_unicode(
            self._df, columns=columns, form=form, strip_accents=strip_accents
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def clean_special_chars(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        allowed_chars: str = "",
        replace_with: str = "",
        keep_alphanumeric: bool = True,
        keep_spaces: bool = True,
    ) -> Cleaner:
        """Remove or replace unwanted special characters and symbols from text columns.

        Args:
            columns: Column name or sequence of column names.
            allowed_chars: Permitted non-alphanumeric symbols (e.g. '-_.@').
            replace_with: Replacement string for disallowed symbols (default '').
            keep_alphanumeric: Whether to retain letters and numbers (default True).
            keep_spaces: Whether to preserve whitespace (default True).

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = clean_special_chars(
            self._df,
            columns=columns,
            allowed_chars=allowed_chars,
            replace_with=replace_with,
            keep_alphanumeric=keep_alphanumeric,
            keep_spaces=keep_spaces,
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def extract_numeric(
        self,
        column: str,
        as_type: str = "float",
        handle_commas: bool = True,
        which: str = "first",
        errors: str = "coerce",
    ) -> Cleaner:
        """Extract numeric values from a messy text column (e.g. '120 km/h' -> 120.0).

        Args:
            column: Name of text column.
            as_type: 'float' (default) or 'int'.
            handle_commas: Whether to remove thousand commas between digits (default True).
            which: 'first' (default) or 'last'.
            errors: 'coerce' (return pd.NA if no number found), 'raise', or 'ignore'.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = extract_numeric_column(
            self._df,
            column=column,
            as_type=as_type,
            handle_commas=handle_commas,
            which=which,
            errors=errors,
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def clean_currency(
        self,
        column: str,
        as_numeric: bool = True,
        errors: str = "coerce",
    ) -> Cleaner:
        """Clean currency strings, stripping symbols, handling accounting parentheses and commas.

        Examples:
            '$1,250.00' -> 1250.0
            '($50.00)' -> -50.0
            '₹ 45,000' -> 45000.0

        Args:
            column: Name of currency column.
            as_numeric: Return float if True, or formatted string if False (default True).
            errors: 'coerce' (default), 'raise', or 'ignore'.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = clean_currency_column(
            self._df, column=column, as_numeric=as_numeric, errors=errors
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def convert_units(
        self,
        column: str,
        from_unit: str,
        to_unit: str,
        round_digits: Optional[int] = None,
        errors: str = "raise",
    ) -> Cleaner:
        """Convert numerical values in a column between units of measurement.

        Supported V1 categories:
        - Length: mm, cm, m, km, in, ft, yd, mi
        - Weight: mg, g, kg, oz, lb, t (tonne)
        - Temperature: c (Celsius), f (Fahrenheit), k (Kelvin)

        Args:
            column: Name of numerical column.
            from_unit: Source unit.
            to_unit: Target unit.
            round_digits: Optional decimal places to round results.
            errors: 'raise' (default), 'coerce', or 'ignore'.

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = convert_column_units(
            self._df,
            column=column,
            from_unit=from_unit,
            to_unit=to_unit,
            round_digits=round_digits,
            errors=errors,
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def check_date_order(
        self,
        start_column: str,
        end_column: str,
        allow_equal: bool = True,
        date_format: Optional[str] = None,
    ) -> ConsistencyResult:
        """Verify that start dates do not occur after end dates.

        Args:
            start_column: Name of start date column.
            end_column: Name of end date column.
            allow_equal: Whether start == end is considered valid (default True).
            date_format: Optional explicit strftime format string.

        Returns:
            ConsistencyResult: Detailed report including inconsistent indices, rows, and suggestions.
        """
        return check_date_order(
            self._df,
            start_column=start_column,
            end_column=end_column,
            allow_equal=allow_equal,
            date_format=date_format,
        )

    def check_value_order(
        self,
        min_column: str,
        max_column: str,
        allow_equal: bool = True,
    ) -> ConsistencyResult:
        """Verify that minimum values do not exceed maximum values.

        Args:
            min_column: Name of lower/minimum value column.
            max_column: Name of upper/maximum value column.
            allow_equal: Whether min == max is considered valid (default True).

        Returns:
            ConsistencyResult: Detailed report including inconsistent indices, rows, and suggestions.
        """
        return check_value_order(
            self._df,
            min_column=min_column,
            max_column=max_column,
            allow_equal=allow_equal,
        )

    def check_min_max(
        self,
        min_column: str,
        max_column: str,
        allow_equal: bool = True,
    ) -> ConsistencyResult:
        """Alias for check_value_order: verify minimum does not exceed maximum."""
        return self.check_value_order(
            min_column=min_column,
            max_column=max_column,
            allow_equal=allow_equal,
        )

    def check_relationship(
        self,
        columns: Sequence[str],
        rule: Union[Callable[[pd.Series], bool], str],
        rule_name: str = "custom_relationship",
        ignore_na: bool = False,
        suggestion: Optional[str] = None,
    ) -> ConsistencyResult:
        """Evaluate a user-defined logical relationship across multiple columns.

        Args:
            columns: Sequence of column names involved in the relationship.
            rule: Either a callable taking a row pd.Series and returning bool (True if consistent),
                  or a string expression evaluatable by pandas eval (e.g. 'total == subtotal + tax').
            rule_name: Human-readable name or expression describing the rule.
            ignore_na: Whether to skip checking rows where any involved column is null (default False).
            suggestion: Optional custom suggestion message to include if inconsistencies are found.

        Security:
            A string `rule` is evaluated with `DataFrame.eval`, which can
            execute arbitrary attribute/method access. Only use trusted,
            developer-authored rule strings; never build `rule` from
            untrusted end-user input. Use the callable form of `rule`
            instead when the source cannot be fully trusted.

        Returns:
            ConsistencyResult: Detailed report including inconsistent indices, rows, and suggestions.
        """
        return check_relationship(
            self._df,
            columns=columns,
            rule=rule,
            rule_name=rule_name,
            ignore_na=ignore_na,
            suggestion=suggestion,
        )

    def resolve_consistency(
        self,
        consistency_result: ConsistencyResult,
        strategy: str = "swap",
        **kwargs: Any,
    ) -> Cleaner:
        """Explicitly resolve logical inconsistencies detected by a ConsistencyResult.

        Strategies:
        - 'swap': Swap the two columns for inconsistent rows.
        - 'drop': Remove inconsistent rows from the dataset.
        - 'set_null': Set a specified target column to NaN on inconsistent rows.
        - 'align': Set target_column equal to source_column on inconsistent rows.
        - 'custom': Apply a user-provided function `func(row) -> dict` returning updated column values.

        Args:
            consistency_result: ConsistencyResult identifying inconsistent rows.
            strategy: 'swap', 'drop', 'set_null', 'align', or 'custom'.
            **kwargs: Extra strategy parameters (col_a, col_b, target_column, source_column, func).

        Returns:
            Cleaner: self for method chaining.
        """
        cleaned_df, record = resolve_consistency(
            self._df, consistency_result=consistency_result, strategy=strategy, **kwargs
        )
        self._df = cleaned_df
        self._history.append(record)
        return self

    def drop_inconsistent(self, consistency_result: ConsistencyResult) -> Cleaner:
        """Drop all rows identified as inconsistent by a ConsistencyResult.

        Args:
            consistency_result: ConsistencyResult identifying inconsistent rows.

        Returns:
            Cleaner: self for method chaining.
        """
        return self.resolve_consistency(consistency_result, strategy="drop")

    def swap_inconsistent(
        self,
        consistency_result: ConsistencyResult,
        col_a: Optional[str] = None,
        col_b: Optional[str] = None,
    ) -> Cleaner:
        """Swap values between two columns for all rows identified as inconsistent.

        Args:
            consistency_result: ConsistencyResult identifying inconsistent rows.
            col_a: First column name (optional, defaults to first column in result).
            col_b: Second column name (optional, defaults to second column in result).

        Returns:
            Cleaner: self for method chaining.
        """
        return self.resolve_consistency(
            consistency_result, strategy="swap", col_a=col_a, col_b=col_b
        )

    def suggest(
        self,
        validation_results: Optional[Sequence[ValidationResult]] = None,
        consistency_results: Optional[Sequence[ConsistencyResult]] = None,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
        outlier_method: str = "iqr",
    ) -> SuggestionReport:
        """Analyze dataset and return structured cleaning suggestions.

        Identifies quality issues across:
        - missing values
        - duplicate rows
        - inconsistent column names
        - whitespace and invisible characters
        - suspicious data types
        - invalid values and range violations
        - invalid date formats
        - potential outliers
        - invalid email addresses
        - phone-format problems
        - category inconsistencies

        This method is purely diagnostic and does NOT mutate the dataset.

        Args:
            validation_results: Optional list of ValidationResult objects.
            consistency_results: Optional list of ConsistencyResult objects.
            rules: Optional dict of range rules (e.g. {'age': {'min_value': 0, 'max_value': 120}}).
            outlier_method: Algorithm for outlier detection ('iqr' or 'zscore'). Default 'iqr'.

        Returns:
            SuggestionReport: Collection of structured Suggestion objects.
        """
        return generate_suggestions(
            self._df,
            validation_results=validation_results,
            consistency_results=consistency_results,
            rules=rules,
            outlier_method=outlier_method,
        )

    def clean(
        self,
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
        return_df: bool = False,
        **kwargs: Any,
    ) -> Union[Cleaner, pd.DataFrame]:
        """Orchestrate multiple cleaning operations in a validated, predictable sequence.

        Uses existing tested cleaning operations and preserves a detailed
        cleaning history. Unrequested operations are NEVER executed automatically.

        Execution Order:
        1. column_names
        2. invisible_chars
        3. whitespace
        4. text
        5. duplicates
        6. types
        7. missing
        8. categories
        9. dates
        10. emails
        11. phones
        12. ranges
        13. outliers

        Example:
            cleaner.clean(
                missing="median",
                duplicates=True,
                column_names=True,
                text=True,
            )

        Args:
            missing: Strategy ('median', 'mean', 'mode', 'drop') or config dict.
            duplicates: bool or config dict (e.g. {'subset': ['id']}).
            column_names: bool or config dict (e.g. {'case': 'snake'}).
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
            return_df: Whether to return self.data (pd.DataFrame) instead of self (default False).
            **kwargs: Extra parameters (validated against supported options).

        Returns:
            Cleaner (self) or pd.DataFrame if return_df=True.
        """
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
            "return_df": return_df,
            **kwargs,
        }
        validate_clean_options(all_options)

        execute_cleaning_pipeline(
            cleaner=self,
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

        return self._df if return_df else self

    def auto_clean(
        self,
        return_df: bool = False,
        **kwargs: Any,
    ) -> Union[Cleaner, pd.DataFrame]:
        """Automatically detect and fix quality issues across the entire dataset.

        Args:
            return_df: Whether to return pandas DataFrame instead of Cleaner instance.
            **kwargs: Extra parameters to override auto defaults.

        Returns:
            Cleaner or pd.DataFrame.
        """
        return self.clean(auto=True, return_df=return_df, **kwargs)

    def apply_suggestions(
        self,
        suggestions: Optional[Sequence[Any]] = None,
        return_df: bool = False,
        **kwargs: Any,
    ) -> Union[Cleaner, pd.DataFrame]:
        """Apply cleaning actions based on suggestions report.

        Args:
            suggestions: Optional sequence of Suggestion objects.
            return_df: Whether to return DataFrame instead of Cleaner.
            **kwargs: Extra parameters passed to clean.

        Returns:
            Cleaner or pd.DataFrame.
        """
        return self.clean(auto=True, return_df=return_df, **kwargs)

    def validate(
        self,
        goals: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
        validation_results: Optional[Sequence[Any]] = None,
        consistency_results: Optional[Sequence[Any]] = None,
        outlier_method: str = "iqr",
    ) -> ValidationReport:
        """Inspect the current dataset after cleaning and evaluate goal satisfaction.

        Clearly distinguishes:
        - issues resolved
        - issues remaining
        - issues requiring manual review

        This method is non-destructive and does NOT modify the DataFrame.

        Args:
            goals: Optional list of cleaning goals to verify (e.g. ['missing', 'duplicates'])
                or dict with target thresholds (e.g. {'missing': 0, 'duplicates': 0}).
                If None, goals are inferred from cleaning operations recorded in history.
            rules: Optional dict of value validation rules (e.g. {'age': {'min_value': 0, 'max_value': 120}}).
            validation_results: Optional sequence of pre-computed ValidationResult objects.
            consistency_results: Optional sequence of pre-computed ConsistencyResult objects.
            outlier_method: Algorithm for outlier detection ('iqr' or 'zscore'). Default 'iqr'.

        Returns:
            ValidationReport: Structured report detailing resolved issues, remaining issues,
                and issues requiring manual review.
        """
        return validate_cleaning(
            cleaner=self,
            goals=goals,
            rules=rules,
            validation_results=validation_results,
            consistency_results=consistency_results,
            outlier_method=outlier_method,
        )

    def report(
        self,
        goals: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
        validation_results: Optional[Sequence[Any]] = None,
        consistency_results: Optional[Sequence[Any]] = None,
        outlier_method: str = "iqr",
        print_report: bool = False,
    ) -> SessionReport:
        """Summarize the complete data-cleaning session.

        Includes:
        1. Dataset information (before and after cleaning)
        2. Issues detected before cleaning
        3. Cleaning operations performed
        4. Number of affected rows/values
        5. Issues remaining after cleaning
        6. Validation status
        7. Warnings requiring manual review

        The returned SessionReport provides structured programmatic data (.to_dict(),
        attributes) and friendly, human-readable console output (str(report), report.print()).

        Args:
            goals: Optional cleaning goals to verify against.
            rules: Optional custom value validation rules.
            validation_results: Optional pre-computed ValidationResults.
            consistency_results: Optional pre-computed ConsistencyResults.
            outlier_method: Algorithm for outlier detection ('iqr' or 'zscore'). Default 'iqr'.
            print_report: If True, prints the human-readable report to console before returning.

        Returns:
            SessionReport: Structured and beginner-friendly session report.
        """
        rep = generate_session_report(
            cleaner=self,
            goals=goals,
            rules=rules,
            validation_results=validation_results,
            consistency_results=consistency_results,
            outlier_method=outlier_method,
        )
        if print_report:
            rep.print()
        return rep

    def save(self, path: Union[str, Path], index: bool = False, **kwargs: Any) -> None:
        """Save the current working DataFrame to a CSV file.

        Args:
            path: Destination file path (str or Path).
            index: Whether to write row names (index). Defaults to False.
            **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_csv.

        Raises:
            TypeError: If path is not a str or Path.
            ValueError: If path is empty or points to an existing directory.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"Invalid path type: '{type(path).__name__}'. Expected a str or Path."
            )

        path_str = str(path).strip()
        if not path_str:
            raise ValueError("Destination path cannot be empty.")

        dest = Path(path)
        if dest.is_dir():
            raise ValueError(
                f"Destination path is an existing directory, not a file: '{path}'"
            )

        if dest.parent and not dest.parent.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)

        self._df.to_csv(dest, index=index, **kwargs)

    def __repr__(self) -> str:
        """Return a concise string representation of the Cleaner."""
        rows, cols = self._df.shape
        return f"Cleaner(rows={rows}, columns={cols})"
