"""Profiling module for Tersus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfileResult:
    """Structured result containing quality characteristics and statistics of a dataset."""

    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    duplicate_rows: int
    unique_counts: Dict[str, int]
    numeric_stats: Dict[str, Dict[str, Any]]

    @property
    def n_rows(self) -> int:
        """Alias for row_count."""
        return self.row_count

    @property
    def n_columns(self) -> int:
        """Alias for column_count."""
        return self.column_count

    @property
    def duplicate_count(self) -> int:
        """Alias for duplicate_rows."""
        return self.duplicate_rows

    @property
    def numeric_columns(self) -> List[str]:
        """List of columns identified as numeric."""
        return list(self.numeric_stats.keys())

    @property
    def categorical_columns(self) -> List[str]:
        """List of non-numeric / categorical columns."""
        return [col for col in self.columns if col not in self.numeric_stats]

    @property
    def has_missing_values(self) -> bool:
        """Whether any column has missing values."""
        return any(count > 0 for count in self.missing_counts.values())

    @property
    def has_duplicates(self) -> bool:
        """Whether the dataset contains any duplicate rows."""
        return self.duplicate_rows > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile result into a plain dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "missing_counts": dict(self.missing_counts),
            "missing_percentages": dict(self.missing_percentages),
            "duplicate_rows": self.duplicate_rows,
            "unique_counts": dict(self.unique_counts),
            "numeric_stats": {
                col: dict(stats) for col, stats in self.numeric_stats.items()
            },
        }

    def __getitem__(self, item: str) -> Any:
        """Provide dictionary-like access to profile attributes."""
        data = self.to_dict()
        if item in data:
            return data[item]

        aliases: Dict[str, str] = {
            "n_rows": "row_count",
            "rows": "row_count",
            "n_columns": "column_count",
            "cols": "column_count",
            "duplicate_count": "duplicate_rows",
            "n_duplicates": "duplicate_rows",
        }
        if item in aliases:
            return data[aliases[item]]

        raise KeyError(f"Key '{item}' not found in ProfileResult.")

    def __contains__(self, item: str) -> bool:
        """Check if an attribute key exists in the profile result."""
        return item in self.to_dict()

    def __str__(self) -> str:
        """Return a formatted, human-readable summary of the dataset profile."""
        lines = [
            "Dataset Profile:",
            f"  Rows: {self.row_count}",
            f"  Columns: {self.column_count}",
            f"  Duplicate Rows: {self.duplicate_rows}",
            "",
            "Column Details:",
        ]

        if not self.columns:
            lines.append("  (No columns)")
        else:
            for col in self.columns:
                dtype = self.dtypes.get(col, "unknown")
                missing = self.missing_counts.get(col, 0)
                pct = self.missing_percentages.get(col, 0.0)
                unique = self.unique_counts.get(col, 0)
                col_line = (
                    f"  - {col} [{dtype}]: {missing} missing ({pct:.2f}%), "
                    f"{unique} unique"
                )

                if col in self.numeric_stats:
                    stats = self.numeric_stats[col]
                    min_val = stats.get("min")
                    max_val = stats.get("max")
                    mean_val = stats.get("mean")
                    median_val = stats.get("median")
                    if min_val is not None and max_val is not None:
                        col_line += (
                            f" | min: {min_val}, max: {max_val}, "
                            f"mean: {mean_val}, median: {median_val}"
                        )

                lines.append(col_line)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise string representation of ProfileResult."""
        return (
            f"ProfileResult(rows={self.row_count}, columns={self.column_count}, "
            f"duplicates={self.duplicate_rows})"
        )


def _compute_numeric_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute summary statistics for a numeric column."""
    valid = series.dropna()
    if valid.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
        }

    try:
        min_val = float(valid.min())
        max_val = float(valid.max())
        mean_val = round(float(valid.mean()), 2)
        std_val = round(float(valid.std()), 2) if len(valid) > 1 else 0.0
        if pd.isna(std_val):
            std_val = 0.0
        median_val = round(float(valid.median()), 2)
        q25_val = round(float(valid.quantile(0.25)), 2)
        q75_val = round(float(valid.quantile(0.75)), 2)

        return {
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "q25": q25_val,
            "q75": q75_val,
        }
    except Exception:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
        }


def profile_dataframe(df: pd.DataFrame) -> ProfileResult:
    """Inspect a pandas DataFrame and extract structured quality characteristics.

    Args:
        df: The pandas DataFrame to inspect.

    Returns:
        ProfileResult: A structured, deterministic representation of dataset characteristics.
    """
    row_count = int(len(df))
    column_count = int(len(df.columns))
    columns = [str(col) for col in df.columns]

    dtypes: Dict[str, str] = {}
    missing_counts: Dict[str, int] = {}
    missing_percentages: Dict[str, float] = {}
    unique_counts: Dict[str, int] = {}
    numeric_stats: Dict[str, Dict[str, Any]] = {}

    duplicate_rows = int(df.duplicated().sum()) if row_count > 0 else 0

    for col in df.columns:
        col_str = str(col)
        series = df[col]

        dtypes[col_str] = str(series.dtype)

        missing = int(series.isna().sum())
        missing_counts[col_str] = missing

        if row_count > 0:
            missing_percentages[col_str] = round((missing / row_count) * 100.0, 2)
        else:
            missing_percentages[col_str] = 0.0

        unique_counts[col_str] = int(series.nunique(dropna=True))

        # Check if numeric (excluding boolean)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            numeric_stats[col_str] = _compute_numeric_stats(series)

    return ProfileResult(
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        dtypes=dtypes,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        duplicate_rows=duplicate_rows,
        unique_counts=unique_counts,
        numeric_stats=numeric_stats,
    )
