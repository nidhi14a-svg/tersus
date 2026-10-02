"""Cleaner module for Tersus."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from tersus.columns import clean_dataframe_columns
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
from tersus.profiling import ProfileResult, profile_dataframe


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

        self._history: List[CleaningRecord] = []

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
