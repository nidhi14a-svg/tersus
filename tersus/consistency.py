"""Logical consistency checking and resolution module for Tersus.

Provides detection, suggestion, and correction for logical relationships
across columns:
1. Date ordering (e.g. start_date <= end_date)
2. Value ordering / Min-Max (e.g. min_val <= max_val)
3. User-defined column relationships (callable or expression)

Design Principles:
------------------
- Detection: Discovers inconsistencies and returns a structured report (ConsistencyResult).
  Does NOT silently modify or delete any data.
- Suggestion: Recommends actionable, human-readable correction strategies (ConsistencySuggestion)
  without applying them automatically.
- Correction: Explicit user-driven operations (swap, drop, set_null, align, custom)
  that produce a modified DataFrame and a CleaningRecord.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tersus.missing import CleaningRecord


@dataclass
class ConsistencySuggestion:
    """Actionable recommendation for resolving a detected logical inconsistency."""

    strategy: str  # 'swap', 'drop', 'set_null', 'align', 'custom'
    description: str
    target_column: Optional[str] = None
    source_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to dictionary representation."""
        return {
            "strategy": self.strategy,
            "description": self.description,
            "target_column": self.target_column,
            "source_column": self.source_column,
        }

    def __repr__(self) -> str:
        return f"ConsistencySuggestion(strategy='{self.strategy}', description={self.description!r})"

    def __str__(self) -> str:
        return f"[{self.strategy}] {self.description}"


@dataclass
class ConsistencyResult:
    """Structured report of a logical consistency check across dataset columns."""

    rule: str
    columns: List[str]
    is_consistent: bool
    inconsistent_count: int
    inconsistent_indices: List[Any]
    inconsistent_rows: pd.DataFrame
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[ConsistencySuggestion] = field(default_factory=list)

    @property
    def has_inconsistencies(self) -> bool:
        """Whether any inconsistent rows were detected."""
        return not self.is_consistent

    def __bool__(self) -> bool:
        """Evaluates to True if consistent (no violations), False if inconsistent."""
        return self.is_consistent

    def __len__(self) -> int:
        """Total number of inconsistent rows detected."""
        return self.inconsistent_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "rule": self.rule,
            "columns": list(self.columns),
            "is_consistent": self.is_consistent,
            "inconsistent_count": self.inconsistent_count,
            "inconsistent_indices": list(self.inconsistent_indices),
            "details": dict(self.details),
            "suggestions": [s.to_dict() for s in self.suggestions],
        }

    def __repr__(self) -> str:
        return (
            f"ConsistencyResult(rule='{self.rule}', columns={self.columns}, "
            f"is_consistent={self.is_consistent}, inconsistent_count={self.inconsistent_count})"
        )

    def __str__(self) -> str:
        status = (
            "PASSED (0 inconsistent rows)"
            if self.is_consistent
            else f"FAILED ({self.inconsistent_count} inconsistent row(s) found)"
        )
        lines = [
            f"Logical Consistency Check: '{self.rule}':",
            f"  Status: {status}",
            f"  Involved Columns: {self.columns}",
        ]
        if self.inconsistent_count > 0:
            lines.append("  Inconsistent Rows Preview:")
            for idx in self.inconsistent_indices[:5]:
                row_vals = {
                    col: self.inconsistent_rows.loc[idx, col] for col in self.columns
                }
                lines.append(f"    - Row {idx}: {row_vals}")
            if self.inconsistent_count > 5:
                lines.append(f"    ... and {self.inconsistent_count - 5} more")

            if self.suggestions:
                lines.append("  Suggested Resolution Actions:")
                for i, sugg in enumerate(self.suggestions, 1):
                    lines.append(f"    {i}. {sugg}")

        return "\n".join(lines)


def check_date_order(
    df: pd.DataFrame,
    start_column: str,
    end_column: str,
    allow_equal: bool = True,
    date_format: Optional[str] = None,
) -> ConsistencyResult:
    """Verify that start dates do not occur after end dates.

    Args:
        df: Pandas DataFrame.
        start_column: Name of start date column.
        end_column: Name of end date column.
        allow_equal: Whether start == end is considered valid (default True).
        date_format: Optional explicit strftime format string.

    Returns:
        ConsistencyResult: Detailed report including inconsistent indices, rows, and suggestions.
    """
    for col in (start_column, end_column):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in dataset. Available columns: {list(df.columns)}"
            )

    start_series = pd.to_datetime(df[start_column], format=date_format, errors="coerce")
    end_series = pd.to_datetime(df[end_column], format=date_format, errors="coerce")

    # Only compare where both dates are successfully parsed and not null
    both_present = start_series.notna() & end_series.notna()

    if allow_equal:
        inconsistent_mask = both_present & (start_series > end_series)
        rule_desc = f"{start_column} <= {end_column}"
    else:
        inconsistent_mask = both_present & (start_series >= end_series)
        rule_desc = f"{start_column} < {end_column}"

    inconsistent_indices = list(df.index[inconsistent_mask])
    inconsistent_rows = df.loc[inconsistent_indices].copy()
    inconsistent_count = len(inconsistent_indices)

    suggestions: List[ConsistencySuggestion] = []
    if inconsistent_count > 0:
        suggestions.append(
            ConsistencySuggestion(
                strategy="swap",
                description=(
                    f"Swap values between '{start_column}' and '{end_column}' for inverted dates"
                ),
                source_column=start_column,
                target_column=end_column,
            )
        )
        suggestions.append(
            ConsistencySuggestion(
                strategy="align",
                description=f"Align '{end_column}' to match '{start_column}' for inconsistent rows",
                source_column=start_column,
                target_column=end_column,
            )
        )
        suggestions.append(
            ConsistencySuggestion(
                strategy="set_null",
                description=f"Set '{end_column}' to null for inconsistent rows",
                target_column=end_column,
            )
        )
        suggestions.append(
            ConsistencySuggestion(
                strategy="drop",
                description=f"Drop the {inconsistent_count} inconsistent row(s) failing '{rule_desc}'",
            )
        )

    return ConsistencyResult(
        rule=rule_desc,
        columns=[start_column, end_column],
        is_consistent=(inconsistent_count == 0),
        inconsistent_count=inconsistent_count,
        inconsistent_indices=inconsistent_indices,
        inconsistent_rows=inconsistent_rows,
        details={
            "allow_equal": allow_equal,
            "date_format": date_format,
            "compared_rows_count": int(both_present.sum()),
        },
        suggestions=suggestions,
    )


def check_value_order(
    df: pd.DataFrame,
    min_column: str,
    max_column: str,
    allow_equal: bool = True,
) -> ConsistencyResult:
    """Verify that minimum values do not exceed maximum values.

    Args:
        df: Pandas DataFrame.
        min_column: Name of lower/minimum value column.
        max_column: Name of upper/maximum value column.
        allow_equal: Whether min == max is considered valid (default True).

    Returns:
        ConsistencyResult: Detailed report including inconsistent indices, rows, and suggestions.
    """
    for col in (min_column, max_column):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in dataset. Available columns: {list(df.columns)}"
            )

    min_series = pd.to_numeric(df[min_column], errors="coerce")
    max_series = pd.to_numeric(df[max_column], errors="coerce")

    # Only compare where both numeric values are present
    both_present = min_series.notna() & max_series.notna()

    if allow_equal:
        inconsistent_mask = both_present & (min_series > max_series)
        rule_desc = f"{min_column} <= {max_column}"
    else:
        inconsistent_mask = both_present & (min_series >= max_series)
        rule_desc = f"{min_column} < {max_column}"

    inconsistent_indices = list(df.index[inconsistent_mask])
    inconsistent_rows = df.loc[inconsistent_indices].copy()
    inconsistent_count = len(inconsistent_indices)

    suggestions: List[ConsistencySuggestion] = []
    if inconsistent_count > 0:
        suggestions.append(
            ConsistencySuggestion(
                strategy="swap",
                description=(
                    f"Swap values between '{min_column}' and '{max_column}' where minimum exceeds maximum"
                ),
                source_column=min_column,
                target_column=max_column,
            )
        )
        suggestions.append(
            ConsistencySuggestion(
                strategy="align",
                description=f"Align '{max_column}' to match '{min_column}' for inconsistent rows",
                source_column=min_column,
                target_column=max_column,
            )
        )
        suggestions.append(
            ConsistencySuggestion(
                strategy="set_null",
                description=f"Set '{max_column}' to null for inconsistent rows",
                target_column=max_column,
            )
        )
        suggestions.append(
            ConsistencySuggestion(
                strategy="drop",
                description=f"Drop the {inconsistent_count} inconsistent row(s) failing '{rule_desc}'",
            )
        )

    return ConsistencyResult(
        rule=rule_desc,
        columns=[min_column, max_column],
        is_consistent=(inconsistent_count == 0),
        inconsistent_count=inconsistent_count,
        inconsistent_indices=inconsistent_indices,
        inconsistent_rows=inconsistent_rows,
        details={
            "allow_equal": allow_equal,
            "compared_rows_count": int(both_present.sum()),
        },
        suggestions=suggestions,
    )


def check_relationship(
    df: pd.DataFrame,
    columns: Sequence[str],
    rule: Union[Callable[[pd.Series], bool], str],
    rule_name: str = "custom_relationship",
    ignore_na: bool = False,
    suggestion: Optional[str] = None,
) -> ConsistencyResult:
    """Evaluate a user-defined logical relationship across multiple columns.

    Args:
        df: Pandas DataFrame.
        columns: Sequence of column names involved in the relationship.
        rule: Either a callable taking a row pd.Series and returning bool (True if consistent),
              or a string expression evaluatable by pandas eval (e.g. 'total == subtotal + tax').
        rule_name: Human-readable name or expression describing the rule.
        ignore_na: Whether to skip checking rows where any involved column is null (default False).
        suggestion: Optional custom suggestion message to include if inconsistencies are found.

    Security:
        A string `rule` is evaluated with `DataFrame.eval`, which can fall
        back to Python-level evaluation and execute arbitrary attribute or
        method access. This is safe for rule strings written by a trusted
        developer against their own data, but `rule` strings must NEVER be
        constructed from untrusted/end-user input (e.g. values typed into a
        web form) without independent sanitization. Prefer the callable form
        of `rule` when the rule's source cannot be fully trusted.

    Returns:
        ConsistencyResult: Detailed report including inconsistent indices, rows, and suggestions.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in dataset. Available columns: {list(df.columns)}"
            )

    cols_list = list(columns)
    inconsistent_indices: List[Any] = []

    if isinstance(rule, str):
        rule_expr = rule
        if rule_name == "custom_relationship":
            rule_name = rule_expr

        try:
            eval_res = df.eval(rule_expr)
            if isinstance(eval_res, pd.Series):
                if ignore_na:
                    not_na = ~df[cols_list].isna().any(axis=1)
                    inconsistent_mask = not_na & (~eval_res.astype(bool))
                else:
                    inconsistent_mask = ~eval_res.astype(bool)
                inconsistent_indices = list(df.index[inconsistent_mask])
        except Exception as exc:
            raise ValueError(
                f"Failed to evaluate expression rule '{rule_expr}': {exc}"
            ) from exc
    else:
        for idx, row in df.iterrows():
            if ignore_na and row[cols_list].isna().any():
                continue
            try:
                res = rule(row)
                if not bool(res):
                    inconsistent_indices.append(idx)
            except Exception:
                inconsistent_indices.append(idx)

    inconsistent_rows = df.loc[inconsistent_indices].copy()
    inconsistent_count = len(inconsistent_indices)

    suggestions: List[ConsistencySuggestion] = []
    if inconsistent_count > 0:
        if suggestion:
            suggestions.append(
                ConsistencySuggestion(
                    strategy="custom",
                    description=suggestion,
                )
            )
        suggestions.append(
            ConsistencySuggestion(
                strategy="drop",
                description=f"Drop the {inconsistent_count} inconsistent row(s) failing '{rule_name}'",
            )
        )

    return ConsistencyResult(
        rule=rule_name,
        columns=cols_list,
        is_consistent=(inconsistent_count == 0),
        inconsistent_count=inconsistent_count,
        inconsistent_indices=inconsistent_indices,
        inconsistent_rows=inconsistent_rows,
        details={
            "ignore_na": ignore_na,
            "rule_type": "expr" if isinstance(rule, str) else "callable",
        },
        suggestions=suggestions,
    )


def resolve_consistency(
    df: pd.DataFrame,
    consistency_result: ConsistencyResult,
    strategy: str = "swap",
    **kwargs: Any,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Explicitly correct logical inconsistencies detected by a ConsistencyResult.

    Supported strategies:
    - 'swap': Swap the two columns for inconsistent rows (e.g. min <-> max, start <-> end).
    - 'drop': Remove inconsistent rows from the DataFrame.
    - 'set_null': Set a designated column to NaN on inconsistent rows.
    - 'align': Set the target column equal to the source column on inconsistent rows.
    - 'custom': Apply a user-provided function `func(row) -> dict` returning updated column values.

    Args:
        df: Pandas DataFrame.
        consistency_result: The ConsistencyResult to resolve.
        strategy: 'swap', 'drop', 'set_null', 'align', or 'custom'.
        **kwargs: Strategy-specific parameters:
            - col_a, col_b for 'swap' (optional, defaults to first two columns of result)
            - target_column for 'set_null' or 'align'
            - source_column for 'align'
            - func for 'custom'

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    valid_strategies = ("swap", "drop", "set_null", "align", "custom")
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. Supported strategies: "
            f"'swap', 'drop', 'set_null', 'align', 'custom'."
        )

    if not consistency_result.inconsistent_indices:
        # Nothing to resolve
        return df.copy(), CleaningRecord(
            operation="resolve_consistency",
            strategy=strategy,
            columns=consistency_result.columns,
            affected_count=0,
            details={"rule": consistency_result.rule, "strategy": strategy},
            message=f"No inconsistencies to resolve for rule '{consistency_result.rule}'.",
        )

    idx = consistency_result.inconsistent_indices
    df_clean = df.copy()

    if strategy == "swap":
        col_a = kwargs.get("col_a") or (
            consistency_result.columns[0]
            if len(consistency_result.columns) >= 2
            else None
        )
        col_b = kwargs.get("col_b") or (
            consistency_result.columns[1]
            if len(consistency_result.columns) >= 2
            else None
        )
        if not col_a or not col_b:
            raise ValueError(
                "Strategy 'swap' requires two columns (col_a and col_b). "
                f"Columns available: {consistency_result.columns}"
            )
        temp = df_clean.loc[idx, col_a].copy()
        df_clean.loc[idx, col_a] = df_clean.loc[idx, col_b]
        df_clean.loc[idx, col_b] = temp

    elif strategy == "drop":
        df_clean = df_clean.drop(index=idx).reset_index(drop=True)

    elif strategy == "set_null":
        target_col = kwargs.get("target_column") or consistency_result.columns[-1]
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset.")
        df_clean.loc[idx, target_col] = np.nan

    elif strategy == "align":
        source_col = kwargs.get("source_column") or consistency_result.columns[0]
        target_col = kwargs.get("target_column") or consistency_result.columns[1]
        if source_col not in df.columns or target_col not in df.columns:
            raise KeyError("Both source_column and target_column must exist in dataset.")
        df_clean.loc[idx, target_col] = df_clean.loc[idx, source_col]

    elif strategy == "custom":
        func = kwargs.get("func")
        if not callable(func):
            raise ValueError(
                "Strategy 'custom' requires a callable 'func(row) -> dict'."
            )
        for row_idx in idx:
            updates = func(df_clean.loc[row_idx])
            if isinstance(updates, dict):
                for k, v in updates.items():
                    df_clean.loc[row_idx, k] = v
    else:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. Supported strategies: "
            f"'swap', 'drop', 'set_null', 'align', 'custom'."
        )

    record = CleaningRecord(
        operation="resolve_consistency",
        strategy=strategy,
        columns=consistency_result.columns,
        affected_count=len(idx),
        details={
            "rule": consistency_result.rule,
            "strategy": strategy,
            "affected_indices": idx,
            **{k: str(v) if callable(v) else v for k, v in kwargs.items()},
        },
        message=(
            f"Resolved logical consistency rule '{consistency_result.rule}' on "
            f"{len(idx)} row(s) using strategy '{strategy}'."
        ),
    )
    return df_clean, record
