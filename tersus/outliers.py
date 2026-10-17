"""Outlier detection module for Tersus.

Provides statistical methods for identifying potential distributional anomalies:
1. IQR (Interquartile Range) method
2. Z-Score (Standard Score) method

Mathematical Foundations:
-------------------------

1. Interquartile Range (IQR) Method:
   - Q1 = 25th percentile of the numerical distribution
   - Q3 = 75th percentile of the numerical distribution
   - IQR = Q3 - Q1
   - Lower Bound = Q1 - (k * IQR)
   - Upper Bound = Q3 + (k * IQR)
   - Outlier condition: x < Lower Bound OR x > Upper Bound
   - Standard threshold: k = 1.5 (moderate outliers) or k = 3.0 (extreme outliers)

2. Z-Score (Standard Score) Method:
   - mean (μ) = (1 / n) * Σ(x_i)
   - std (σ) = sqrt((1 / (n - ddof)) * Σ(x_i - μ)^2)
   - z = (x - μ) / σ
   - Lower Bound = μ - (c * σ)
   - Upper Bound = μ + (c * σ)
   - Outlier condition: |z| > c (i.e. x < Lower Bound OR x > Upper Bound)
   - Standard threshold: c = 3.0 (or c = 2.0 / 2.5 for tighter sensitivity)

Design Philosophy:
------------------
An outlier is an extreme or unusual value within a statistical distribution.
An outlier is NOT automatically an error or invalid data. The Tersus outlier
detection module identifies potential anomalies for human or analytical review
without silently modifying or deleting the underlying data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from tersus.missing import CleaningRecord


@dataclass
class OutlierResult:
    """Structured report of outlier detection on a dataset column.

    Note: An outlier is a statistical extreme in the distribution, not
    necessarily an error or corrupt data. Review outliers in context
    before deciding whether to filter, transform, or retain them.
    """

    column: str
    method: str
    threshold: float
    outlier_count: int
    outlier_indices: List[Any]
    outlier_values: List[Any]
    outlier_rows: pd.DataFrame
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    statistics: Dict[str, Any]

    @property
    def has_outliers(self) -> bool:
        """Whether any potential outliers were detected."""
        return self.outlier_count > 0

    def __len__(self) -> int:
        """Total number of potential outliers found."""
        return self.outlier_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "column": self.column,
            "method": self.method,
            "threshold": self.threshold,
            "outlier_count": self.outlier_count,
            "outlier_indices": list(self.outlier_indices),
            "outlier_values": list(self.outlier_values),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "statistics": dict(self.statistics),
        }

    def __repr__(self) -> str:
        return (
            f"OutlierResult(column='{self.column}', method='{self.method}', "
            f"threshold={self.threshold}, outlier_count={self.outlier_count}, "
            f"bounds=[{self.lower_bound}, {self.upper_bound}])"
        )

    def __str__(self) -> str:
        status = (
            f"No potential outliers detected (0 items)"
            if not self.has_outliers
            else f"{self.outlier_count} potential outlier(s) detected"
        )
        lines = [
            f"Outlier Detection Report on '{self.column}':",
            f"  Method: {self.method.upper()} (threshold: {self.threshold})",
            f"  Status: {status}",
            f"  Acceptable Range: [{self.lower_bound}, {self.upper_bound}]",
        ]
        if self.statistics:
            stat_strs = [f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in self.statistics.items()]
            lines.append(f"  Distribution Stats: {', '.join(stat_strs)}")

        if self.has_outliers:
            lines.append("  Detected Outlier Samples:")
            for idx, val in zip(self.outlier_indices[:10], self.outlier_values[:10]):
                lines.append(f"    - Row {idx}: {val}")
            if self.outlier_count > 10:
                lines.append(f"    ... and {self.outlier_count - 10} more")

        return "\n".join(lines)


def _check_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Validate that column exists and contains numeric data."""
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"Column '{column}' of dtype '{series.dtype}' is not numeric. "
            f"Outlier detection requires numeric data."
        )

    return series


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    threshold: float = 1.5,
) -> OutlierResult:
    """Detect potential outliers in a column using the Interquartile Range (IQR) method.

    Formula:
        IQR = Q3 - Q1
        Lower Bound = Q1 - (threshold * IQR)
        Upper Bound = Q3 + (threshold * IQR)
        Outlier condition: x < Lower Bound OR x > Upper Bound

    Args:
        df: Pandas DataFrame.
        column: Numerical column name.
        threshold: IQR multiplier threshold (default 1.5).

    Returns:
        OutlierResult: Structured result reporting detected outliers and bounds.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")

    series = _check_numeric_column(df, column)
    valid_series = series.dropna()

    if valid_series.empty:
        return OutlierResult(
            column=column,
            method="iqr",
            threshold=threshold,
            outlier_count=0,
            outlier_indices=[],
            outlier_values=[],
            outlier_rows=df.loc[[]].copy(),
            lower_bound=None,
            upper_bound=None,
            statistics={"q1": None, "q3": None, "iqr": None},
        )

    q1 = float(valid_series.quantile(0.25))
    q3 = float(valid_series.quantile(0.75))
    iqr = float(q3 - q1)

    lower_bound = float(q1 - threshold * iqr)
    upper_bound = float(q3 + threshold * iqr)

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_subset = series[outlier_mask]

    outlier_indices = list(outlier_subset.index)
    outlier_values = list(outlier_subset.values)
    outlier_rows = df.loc[outlier_indices].copy()

    return OutlierResult(
        column=column,
        method="iqr",
        threshold=threshold,
        outlier_count=len(outlier_indices),
        outlier_indices=outlier_indices,
        outlier_values=outlier_values,
        outlier_rows=outlier_rows,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        statistics={
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
        },
    )


def detect_outliers_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0,
    ddof: int = 1,
) -> OutlierResult:
    """Detect potential outliers in a column using the Z-Score (standard score) method.

    Formula:
        mean (μ) = (1 / n) * Σ(x_i)
        std (σ) = sqrt((1 / (n - ddof)) * Σ(x_i - μ)^2)
        z = (x - μ) / σ
        Lower Bound = μ - (threshold * σ)
        Upper Bound = μ + (threshold * σ)
        Outlier condition: |z| > threshold

    Args:
        df: Pandas DataFrame.
        column: Numerical column name.
        threshold: Standard deviation multiplier threshold (default 3.0).
        ddof: Delta Degrees of Freedom for std computation (default 1 for sample std).

    Returns:
        OutlierResult: Structured result reporting detected outliers and bounds.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")

    series = _check_numeric_column(df, column)
    valid_series = series.dropna()

    if valid_series.empty:
        return OutlierResult(
            column=column,
            method="zscore",
            threshold=threshold,
            outlier_count=0,
            outlier_indices=[],
            outlier_values=[],
            outlier_rows=df.loc[[]].copy(),
            lower_bound=None,
            upper_bound=None,
            statistics={"mean": None, "std": None, "ddof": ddof},
        )

    mean = float(valid_series.mean())
    std = float(valid_series.std(ddof=ddof)) if len(valid_series) > ddof else 0.0

    if np.isnan(std) or std == 0.0:
        # Zero variance: all valid values are identical; no outliers possible
        lower_bound = mean
        upper_bound = mean
        outlier_indices = []
        outlier_values = []
        outlier_rows = df.loc[[]].copy()
    else:
        lower_bound = float(mean - threshold * std)
        upper_bound = float(mean + threshold * std)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_subset = series[outlier_mask]

        outlier_indices = list(outlier_subset.index)
        outlier_values = list(outlier_subset.values)
        outlier_rows = df.loc[outlier_indices].copy()

    return OutlierResult(
        column=column,
        method="zscore",
        threshold=threshold,
        outlier_count=len(outlier_indices),
        outlier_indices=outlier_indices,
        outlier_values=outlier_values,
        outlier_rows=outlier_rows,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        statistics={
            "mean": mean,
            "std": std,
            "ddof": ddof,
        },
    )


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: Optional[float] = None,
    **kwargs: Any,
) -> OutlierResult:
    """Unified entry point for outlier detection supporting IQR and Z-Score methods.

    Args:
        df: Pandas DataFrame.
        column: Column name to evaluate.
        method: 'iqr' or 'zscore' (default 'iqr').
        threshold: Threshold multiplier. If None, defaults to 1.5 for IQR and 3.0 for Z-score.
        **kwargs: Additional parameters (e.g. ddof for zscore).

    Returns:
        OutlierResult: Structured result of outlier detection.
    """
    clean_method = method.strip().lower()
    if clean_method == "iqr":
        t = 1.5 if threshold is None else threshold
        return detect_outliers_iqr(df, column=column, threshold=t)
    elif clean_method in ("zscore", "z_score", "z"):
        t = 3.0 if threshold is None else threshold
        ddof = kwargs.get("ddof", 1)
        return detect_outliers_zscore(df, column=column, threshold=t, ddof=ddof)
    else:
        raise ValueError(
            f"Unsupported outlier detection method '{method}'. Supported methods: 'iqr', 'zscore'."
        )


def drop_outliers(
    df: pd.DataFrame,
    outlier_result: Union[OutlierResult, str],
    **kwargs: Any,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Explicitly drop rows identified as outliers by an OutlierResult or column name.

    Args:
        df: Pandas DataFrame.
        outlier_result: An OutlierResult or the column name string.
        **kwargs: Optional parameters when outlier_result is column name (method='iqr', factor=1.5, threshold=None).

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if isinstance(outlier_result, str):
        outlier_result = detect_outliers(
            df,
            column=outlier_result,
            method=kwargs.get("method", "iqr"),
            factor=kwargs.get("factor", 1.5),
            threshold=kwargs.get("threshold", None),
        )

    df_clean = df.drop(index=outlier_result.outlier_indices).reset_index(drop=True)
    count = outlier_result.outlier_count

    msg = (
        f"Dropped {count} outlier row(s) from column '{outlier_result.column}' "
        f"using {outlier_result.method.upper()} method (threshold={outlier_result.threshold})."
    )
    record = CleaningRecord(
        operation="drop_outliers",
        strategy="drop",
        columns=[outlier_result.column],
        affected_count=count,
        details={
            "method": outlier_result.method,
            "threshold": outlier_result.threshold,
            "dropped_indices": outlier_result.outlier_indices,
            "bounds": [outlier_result.lower_bound, outlier_result.upper_bound],
        },
        message=msg,
    )
    return df_clean, record


def cap_outliers(
    df: pd.DataFrame,
    outlier_result: Union[OutlierResult, str],
    **kwargs: Any,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Explicitly cap (clamp) outlier values to the computed statistical boundaries.

    Values below the lower bound are capped to lower_bound;
    values above the upper bound are capped to upper_bound.

    Args:
        df: Pandas DataFrame.
        outlier_result: An OutlierResult or column name string.
        **kwargs: Optional parameters when outlier_result is column name (method='iqr', factor=1.5, threshold=None).

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if isinstance(outlier_result, str):
        outlier_result = detect_outliers(
            df,
            column=outlier_result,
            method=kwargs.get("method", "iqr"),
            factor=kwargs.get("factor", 1.5),
            threshold=kwargs.get("threshold", None),
        )

    col = outlier_result.column
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataset.")

    df_clean = df.copy()
    series = df_clean[col]

    lower = outlier_result.lower_bound
    upper = outlier_result.upper_bound

    clamped = series.clip(lower=lower, upper=upper)
    changed_mask = (series != clamped) & ~series.isna()
    affected_count = int(changed_mask.sum())

    df_clean[col] = clamped

    msg = (
        f"Capped {affected_count} outlier value(s) in column '{col}' to range "
        f"[{lower}, {upper}] ({outlier_result.method.upper()} method)."
    )
    record = CleaningRecord(
        operation="cap_outliers",
        strategy="cap",
        columns=[col],
        affected_count=affected_count,
        details={
            "method": outlier_result.method,
            "threshold": outlier_result.threshold,
            "lower_bound": lower,
            "upper_bound": upper,
        },
        message=msg,
    )
    return df_clean, record
