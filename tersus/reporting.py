"""Reporting system module for Tersus.

Provides a comprehensive, beginner-friendly summary of the complete cleaning session,
including dataset information, issues detected before cleaning, operations performed,
affected rows/values, remaining issues, validation status, and manual review warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import pandas as pd

from tersus.verification import RemainingIssue, ResolvedIssue, ValidationReport, validate_cleaning

if TYPE_CHECKING:
    from tersus.cleaner import Cleaner
    from tersus.missing import CleaningRecord


@dataclass(frozen=True)
class DatasetInfo:
    """Overview of dataset characteristics before and after cleaning."""

    initial_rows: int
    current_rows: int
    initial_columns: int
    current_columns: int
    rows_change: int
    columns: List[str]
    initial_columns_list: List[str]
    dtypes: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset info to a dictionary."""
        return {
            "initial_rows": self.initial_rows,
            "current_rows": self.current_rows,
            "initial_columns": self.initial_columns,
            "current_columns": self.current_columns,
            "rows_change": self.rows_change,
            "columns": list(self.columns),
            "initial_columns_list": list(self.initial_columns_list),
            "dtypes": dict(self.dtypes),
        }


@dataclass(frozen=True)
class OperationSummary:
    """Summary of an individual cleaning operation performed during the session."""

    step: int
    operation: str
    strategy: str
    columns: List[str]
    affected_count: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert operation summary to a dictionary."""
        return {
            "step": self.step,
            "operation": self.operation,
            "strategy": self.strategy,
            "columns": list(self.columns),
            "affected_count": self.affected_count,
            "message": self.message,
            "details": dict(self.details),
        }

    def __str__(self) -> str:
        col_str = ", ".join(self.columns) if self.columns else "all columns"
        return (
            f"[Step {self.step}] {self.operation} (strategy: {self.strategy})\n"
            f"       Target: {col_str}\n"
            f"       Affected: {self.affected_count} row(s)/value(s)\n"
            f"       Details: {self.message}"
        )


@dataclass
class SessionReport:
    """Comprehensive summary of a Tersus data-cleaning session.

    Contains:
    1. Dataset information (before and after)
    2. Issues detected before cleaning
    3. Cleaning operations performed
    4. Number of affected rows/values
    5. Issues remaining after cleaning
    6. Validation status
    7. Warnings requiring manual review
    """

    dataset_info: DatasetInfo
    before_issues: Dict[str, int]
    operations_performed: List[OperationSummary]
    total_affected_count: int
    remaining_issues: List[RemainingIssue]
    validation: ValidationReport
    manual_review_warnings: List[RemainingIssue]
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether validation passed and goals were satisfied."""
        return self.validation.is_valid

    @property
    def goals_satisfied(self) -> bool:
        """Whether selected cleaning goals were met."""
        return self.validation.goals_satisfied

    @property
    def resolved_issues(self) -> List[ResolvedIssue]:
        """List of issues resolved during the session."""
        return self.validation.resolved_issues

    @property
    def has_remaining_issues(self) -> bool:
        """Whether any issues remain in the dataset."""
        return len(self.remaining_issues) > 0

    @property
    def has_manual_review_warnings(self) -> bool:
        """Whether any issues require manual human review."""
        return len(self.manual_review_warnings) > 0

    @property
    def operation_count(self) -> int:
        """Number of cleaning operations performed."""
        return len(self.operations_performed)

    def print(self) -> None:
        """Print the human-readable report to console."""
        print(str(self))

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete session report to a structured dictionary."""
        return {
            "dataset_info": self.dataset_info.to_dict(),
            "before_issues": dict(self.before_issues),
            "operations_performed": [op.to_dict() for op in self.operations_performed],
            "total_affected_count": self.total_affected_count,
            "remaining_issues": [rem.to_dict() for rem in self.remaining_issues],
            "validation": self.validation.to_dict(),
            "manual_review_warnings": [warn.to_dict() for warn in self.manual_review_warnings],
            "is_valid": self.is_valid,
            "goals_satisfied": self.goals_satisfied,
            "details": dict(self.details),
        }

    def __repr__(self) -> str:
        return (
            f"<SessionReport: {self.dataset_info.current_rows} rows, "
            f"{len(self.operations_performed)} operations, "
            f"status={'PASSED' if self.is_valid else 'ACTION REQUIRED'}>"
        )

    def __str__(self) -> str:
        """Render a beginner-friendly, structured console report."""
        divider = "=" * 80
        subdivider = "-" * 80
        lines: List[str] = [
            divider,
            "                    TERSUS DATA CLEANING SESSION REPORT                         ",
            divider,
            "",
            "1. DATASET OVERVIEW",
            subdivider,
        ]

        # 1. Dataset information
        info = self.dataset_info
        lines.append(f"  - Initial Shape: {info.initial_rows:,} rows x {info.initial_columns} columns")
        change_str = ""
        if info.rows_change != 0:
            sign = "+" if info.rows_change > 0 else ""
            change_str = f" ({sign}{info.rows_change:,} rows)"
        lines.append(f"  - Current Shape: {info.current_rows:,} rows x {info.current_columns} columns{change_str}")

        # Format column list cleanly
        cols = info.columns
        if len(cols) <= 8:
            cols_str = ", ".join(cols) if cols else "(none)"
        else:
            cols_str = ", ".join(cols[:8]) + f" ... (+{len(cols) - 8} more)"
        lines.append(f"  - Columns:       {cols_str}")

        lines.extend(["", "2. ISSUES DETECTED BEFORE CLEANING", subdivider])
        # 2. Issues detected before cleaning
        if self.before_issues:
            total_before = sum(self.before_issues.values())
            lines.append(f"  Found {total_before} potential data quality issue(s) before cleaning:")
            for issue, count in self.before_issues.items():
                label = issue.replace("_", " ").capitalize()
                lines.append(f"  - {label:<30} {count}")
        else:
            lines.append("  [OK] No data quality issues were detected before cleaning.")

        lines.extend(["", "3. CLEANING OPERATIONS PERFORMED", subdivider])
        # 3. Cleaning operations performed & 4. Affected counts
        if self.operations_performed:
            lines.append(
                f"  Executed {len(self.operations_performed)} operation(s) "
                f"affecting {self.total_affected_count:,} row(s)/value(s) in total:"
            )
            for op in self.operations_performed:
                lines.append(f"  {op}")
        else:
            lines.append("  (No cleaning operations were performed during this session.)")

        lines.extend(["", "4. ISSUES REMAINING AFTER CLEANING", subdivider])
        # 5. Issues remaining after cleaning
        if self.remaining_issues:
            lines.append(f"  {len(self.remaining_issues)} issue(s) remain in the dataset:")
            for rem in self.remaining_issues:
                label = rem.issue.replace("_", " ").capitalize()
                col_part = f" (in '{rem.column}')" if rem.column else ""
                manual_marker = " [MANUAL REVIEW RECOMMENDED]" if rem.requires_manual_review else ""
                lines.append(f"  - {label}{col_part}: {rem.count} item(s) [{rem.severity} severity]{manual_marker}")
                lines.append(f"    Suggested Action: {rem.suggested_action}")
        else:
            lines.append("  [OK] All detected issues have been completely resolved! The dataset is clean.")

        lines.extend(["", "5. VALIDATION STATUS", subdivider])
        # 6. Validation status
        status_word = "PASSED" if self.is_valid else "ACTION REQUIRED"
        goals_word = "YES" if self.goals_satisfied else "NO"
        status_tag = "[OK]" if self.is_valid else "[!]"
        lines.append(f"  {status_tag} Status: {status_word} (Goals Satisfied: {goals_word})")

        goal_evals = self.validation.details.get("goal_evaluations", {})
        if goal_evals:
            lines.append("    Goal Evaluations:")
            for g_name, g_info in goal_evals.items():
                res_word = "PASSED" if g_info.get("passed") else "FAILED"
                t_val = g_info.get("target")
                a_val = g_info.get("actual")
                lines.append(f"    - {g_name}: {res_word} (target: {t_val}, actual: {a_val})")
        else:
            if self.is_valid:
                lines.append(
                    "    Validation confirms that all cleaning targets selected for this session "
                    "have been satisfied."
                )
            else:
                lines.append(
                    "    Validation indicates that one or more cleaning targets were not satisfied "
                    "or issues remain."
                )

        lines.extend(["", "6. WARNINGS REQUIRING MANUAL REVIEW", subdivider])
        # 7. Warnings requiring manual review
        if self.manual_review_warnings:
            lines.append(
                f"  [!] The following {len(self.manual_review_warnings)} issue(s) require manual human review:"
            )
            for idx, warn in enumerate(self.manual_review_warnings, start=1):
                col_part = f"[{warn.column}] " if warn.column else ""
                lines.append(f"  {idx}. {col_part}{warn.description}")
                lines.append(f"     Recommendation: {warn.suggested_action}")
        else:
            lines.append("  [OK] No manual review warnings. All automated transformations are verified.")

        lines.append(divider)
        return "\n".join(lines)


# Type alias for convenience
CleaningReport = SessionReport


def generate_session_report(
    cleaner: Union[Cleaner, pd.DataFrame],
    initial_df: Optional[pd.DataFrame] = None,
    history: Optional[Sequence[CleaningRecord]] = None,
    goals: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
    rules: Optional[Dict[str, Dict[str, Any]]] = None,
    validation_results: Optional[Sequence[Any]] = None,
    consistency_results: Optional[Sequence[Any]] = None,
    outlier_method: str = "iqr",
) -> SessionReport:
    """Generate a comprehensive session report for a cleaning session.

    This function does NOT duplicate detection or cleaning logic. It inspects
    the cleaner's baseline data, current data, cleaning history, and validation
    results to construct a structured, beginner-friendly report.

    Args:
        cleaner: Cleaner instance or pandas DataFrame.
        initial_df: Baseline DataFrame before cleaning (used if cleaner is a DataFrame).
        history: Sequence of CleaningRecords (used if cleaner is a DataFrame).
        goals: Optional cleaning goals to verify.
        rules: Optional custom value validation rules.
        validation_results: Optional pre-computed ValidationResults.
        consistency_results: Optional pre-computed ConsistencyResults.
        outlier_method: Algorithm for outlier detection ('iqr' or 'zscore'). Default 'iqr'.

    Returns:
        SessionReport: Structured and human-readable session summary.
    """
    # 1. Safely extract dataframes and history
    if hasattr(cleaner, "initial_data") and hasattr(cleaner, "data"):
        init_data = cleaner.initial_data.copy()
        current_data = cleaner.data.copy()
        cleaning_history = list(getattr(cleaner, "history", []))
        # Reuse existing cleaner.validate()
        val_report = cleaner.validate(
            goals=goals,
            rules=rules,
            validation_results=validation_results,
            consistency_results=consistency_results,
            outlier_method=outlier_method,
        )
    elif isinstance(cleaner, pd.DataFrame):
        current_data = cleaner.copy()
        init_data = initial_df.copy() if initial_df is not None else cleaner.copy()
        cleaning_history = list(history) if history is not None else []
        val_report = validate_cleaning(
            cleaner=current_data,
            initial_df=init_data,
            goals=goals,
            rules=rules,
            validation_results=validation_results,
            consistency_results=consistency_results,
            outlier_method=outlier_method,
        )
    else:
        raise TypeError(
            f"Expected cleaner to be a Cleaner instance or pd.DataFrame, got '{type(cleaner).__name__}'."
        )

    # 2. Construct dataset information
    init_rows = len(init_data)
    curr_rows = len(current_data)
    init_cols = len(init_data.columns)
    curr_cols = len(current_data.columns)
    rows_change = curr_rows - init_rows
    cols_list = [str(c) for c in current_data.columns]
    init_cols_list = [str(c) for c in init_data.columns]
    dtypes_dict = {str(col): str(dtype) for col, dtype in current_data.dtypes.items()}

    dataset_info = DatasetInfo(
        initial_rows=init_rows,
        current_rows=curr_rows,
        initial_columns=init_cols,
        current_columns=curr_cols,
        rows_change=rows_change,
        columns=cols_list,
        initial_columns_list=init_cols_list,
        dtypes=dtypes_dict,
    )

    # 3. Issues detected before cleaning
    before_issues = dict(val_report.before_summary)

    # 4. Cleaning operations performed & affected rows/values
    operations_performed: List[OperationSummary] = []
    total_affected = 0
    for step, rec in enumerate(cleaning_history, start=1):
        total_affected += rec.affected_count
        operations_performed.append(
            OperationSummary(
                step=step,
                operation=rec.operation,
                strategy=rec.strategy,
                columns=list(rec.columns),
                affected_count=rec.affected_count,
                message=rec.message,
                details=dict(rec.details),
            )
        )

    # 5. Issues remaining after cleaning
    remaining_issues = list(val_report.remaining_issues)

    # 6. Warnings requiring manual review
    manual_review_warnings = list(val_report.manual_review_issues)

    details = {
        "resolved_count": len(val_report.resolved_issues),
        "remaining_count": len(val_report.remaining_issues),
        "manual_review_count": len(val_report.manual_review_issues),
        "history_length": len(cleaning_history),
    }

    return SessionReport(
        dataset_info=dataset_info,
        before_issues=before_issues,
        operations_performed=operations_performed,
        total_affected_count=total_affected,
        remaining_issues=remaining_issues,
        validation=val_report,
        manual_review_warnings=manual_review_warnings,
        details=details,
    )
