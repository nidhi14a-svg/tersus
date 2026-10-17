"""Post-cleaning validation and goal verification module for Tersus.

Inspects the dataset after cleaning to determine whether selected cleaning goals
have been satisfied, distinguishing resolved issues, remaining issues, and issues
requiring manual review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import pandas as pd

from tersus.suggestions import Suggestion, generate_suggestions

if TYPE_CHECKING:
    from tersus.cleaner import Cleaner
    from tersus.consistency import ConsistencyResult
    from tersus.validation import ValidationResult


GOAL_ISSUE_MAPPING = {
    "missing": "missing_values",
    "missing_values": "missing_values",
    "no_missing": "missing_values",
    "duplicates": "duplicate_rows",
    "duplicate_rows": "duplicate_rows",
    "no_duplicates": "duplicate_rows",
    "column_names": "inconsistent_column_name",
    "inconsistent_column_name": "inconsistent_column_name",
    "clean_column_names": "inconsistent_column_name",
    "text": "whitespace_problems",
    "whitespace": "whitespace_problems",
    "whitespace_problems": "whitespace_problems",
    "invisible_chars": "whitespace_problems",
    "outliers": "potential_outlier",
    "potential_outlier": "potential_outlier",
    "dates": "invalid_dates",
    "invalid_dates": "invalid_dates",
    "emails": "invalid_email",
    "invalid_email": "invalid_email",
    "phones": "invalid_phone",
    "invalid_phone": "invalid_phone",
    "categories": "category_inconsistencies",
    "category_inconsistencies": "category_inconsistencies",
    "range": "range_violations",
    "range_violations": "range_violations",
    "types": "suspicious_dtype",
    "suspicious_dtype": "suspicious_dtype",
    "consistency": "consistency_violation",
    "consistency_violation": "consistency_violation",
}

OPERATION_GOAL_MAPPING = {
    "fill_missing": "missing_values",
    "drop_missing": "missing_values",
    "drop_duplicates": "duplicate_rows",
    "clean_column_names": "inconsistent_column_name",
    "clean_text": "whitespace_problems",
    "strip_whitespace": "whitespace_problems",
    "clean_invisible_chars": "whitespace_problems",
    "convert_type": "suspicious_dtype",
    "standardize_dates": "invalid_dates",
    "clean_email": "invalid_email",
    "clean_phone": "invalid_phone",
    "map_categories": "category_inconsistencies",
    "normalize_categories": "category_inconsistencies",
    "clamp_range": "range_violations",
    "clamp_ranges": "range_violations",
    "cap_outliers": "potential_outlier",
}


@dataclass
class ResolvedIssue:
    """Record of an issue that was present before cleaning and successfully resolved."""

    issue: str
    column: Optional[str]
    before_count: int
    after_count: int
    resolved_count: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue": self.issue,
            "column": self.column,
            "before_count": self.before_count,
            "after_count": self.after_count,
            "resolved_count": self.resolved_count,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return (
            f"ResolvedIssue(issue='{self.issue}', column={self.column!r}, "
            f"resolved_count={self.resolved_count})"
        )

    def __str__(self) -> str:
        col_str = f"[{self.column}] " if self.column else ""
        return (
            f"[RESOLVED] {col_str}{self.issue}: resolved {self.resolved_count} item(s) "
            f"({self.before_count} -> {self.after_count})"
        )


@dataclass
class RemainingIssue:
    """Record of an issue that remains in the dataset after cleaning."""

    issue: str
    column: Optional[str]
    count: int
    severity: str
    description: str
    requires_manual_review: bool
    suggested_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue": self.issue,
            "column": self.column,
            "count": self.count,
            "severity": self.severity,
            "description": self.description,
            "requires_manual_review": self.requires_manual_review,
            "suggested_action": self.suggested_action,
        }

    def __repr__(self) -> str:
        return (
            f"RemainingIssue(issue='{self.issue}', column={self.column!r}, "
            f"count={self.count}, severity='{self.severity}', "
            f"requires_manual_review={self.requires_manual_review})"
        )

    def __str__(self) -> str:
        col_str = f"[{self.column}] " if self.column else ""
        flag = "MANUAL REVIEW" if self.requires_manual_review else "REMAINING"
        return (
            f"[{flag}] {col_str}{self.issue} ({self.count} items, {self.severity}): "
            f"{self.description}"
        )


@dataclass
class ValidationReport:
    """Structured report returned by cleaner.validate() evaluating cleaning goals.

    Clearly distinguishes:
    - issues resolved
    - issues remaining
    - issues requiring manual review
    """

    is_valid: bool
    goals_satisfied: bool
    resolved_issues: List[ResolvedIssue]
    remaining_issues: List[RemainingIssue]
    manual_review_issues: List[RemainingIssue]
    before_summary: Dict[str, int]
    after_summary: Dict[str, int]
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Evaluates truthiness based on whether goals are satisfied."""
        return self.goals_satisfied

    def __len__(self) -> int:
        """Total number of remaining issues."""
        return len(self.remaining_issues)

    @property
    def has_remaining_issues(self) -> bool:
        """Whether any issues remain in the dataset."""
        return len(self.remaining_issues) > 0

    @property
    def has_manual_review_issues(self) -> bool:
        """Whether any issues require manual human review."""
        return len(self.manual_review_issues) > 0

    @property
    def has_resolved_issues(self) -> bool:
        """Whether any issues were resolved during cleaning."""
        return len(self.resolved_issues) > 0

    def resolved_by_issue(self, issue: str) -> List[ResolvedIssue]:
        """Filter resolved issues by issue name."""
        return [r for r in self.resolved_issues if r.issue == issue]

    def remaining_by_issue(self, issue: str) -> List[RemainingIssue]:
        """Filter remaining issues by issue name."""
        return [r for r in self.remaining_issues if r.issue == issue]

    def manual_by_issue(self, issue: str) -> List[RemainingIssue]:
        """Filter manual review issues by issue name."""
        return [r for r in self.manual_review_issues if r.issue == issue]

    def summary(self) -> str:
        """Return a formatted string summary highlighting Before, After, and Remaining issues."""
        lines = ["Before:"]
        if self.before_summary:
            for issue, count in self.before_summary.items():
                label = issue.replace("_", " ").capitalize()
                lines.append(f"{label}: {count}")
        else:
            lines.append("  (no issues detected)")

        lines.append("")
        lines.append("After:")
        if self.after_summary:
            for issue, count in self.after_summary.items():
                label = issue.replace("_", " ").capitalize()
                lines.append(f"{label}: {count}")
        else:
            lines.append("  (clean)")

        if self.remaining_issues:
            lines.append("")
            lines.append("Remaining:")
            for item in self.remaining_issues:
                label = item.issue.replace("_", " ").capitalize()
                col_part = f" ({item.column})" if item.column else ""
                manual_marker = " [manual review]" if item.requires_manual_review else ""
                lines.append(f"{label}{col_part}: {item.count}{manual_marker}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to a dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "goals_satisfied": self.goals_satisfied,
            "before_summary": dict(self.before_summary),
            "after_summary": dict(self.after_summary),
            "resolved_issues": [r.to_dict() for r in self.resolved_issues],
            "remaining_issues": [r.to_dict() for r in self.remaining_issues],
            "manual_review_issues": [r.to_dict() for r in self.manual_review_issues],
            "details": dict(self.details),
        }

    def __repr__(self) -> str:
        return (
            f"ValidationReport(goals_satisfied={self.goals_satisfied}, "
            f"resolved_count={len(self.resolved_issues)}, "
            f"remaining_count={len(self.remaining_issues)}, "
            f"manual_review_count={len(self.manual_review_issues)})"
        )

    def __str__(self) -> str:
        status = "SATISFIED" if self.goals_satisfied else "NOT SATISFIED"
        lines = [
            "Tersus Cleaning Validation Report:",
            "=" * 60,
            f"Goals Satisfied: {status}",
            f"Validation Status: {'PASSED' if self.is_valid else 'ACTION REQUIRED'}",
            "",
            "Before Cleaning:",
        ]
        if self.before_summary:
            for issue, count in self.before_summary.items():
                lines.append(f"  - {issue.replace('_', ' ').capitalize()}: {count}")
        else:
            lines.append("  (no initial quality issues detected)")

        lines.extend(["", "After Cleaning:"])
        if self.after_summary:
            for issue, count in self.after_summary.items():
                lines.append(f"  - {issue.replace('_', ' ').capitalize()}: {count}")
        else:
            lines.append("  (clean)")

        if self.resolved_issues:
            lines.extend(["", f"Issues Resolved ({len(self.resolved_issues)}):"])
            for r in self.resolved_issues:
                lines.append(f"  {r}")

        if self.manual_review_issues:
            lines.extend(["", f"Issues Requiring Manual Review ({len(self.manual_review_issues)}):"])
            for m in self.manual_review_issues:
                lines.append(f"  {m}")

        remaining_non_manual = [r for r in self.remaining_issues if not r.requires_manual_review]
        if remaining_non_manual:
            lines.extend(["", f"Other Remaining Issues ({len(remaining_non_manual)}):"])
            for r in remaining_non_manual:
                lines.append(f"  {r}")

        lines.append("=" * 60)
        return "\n".join(lines)


def validate_cleaning(
    cleaner: Union[Cleaner, pd.DataFrame],
    initial_df: Optional[pd.DataFrame] = None,
    goals: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
    rules: Optional[Dict[str, Dict[str, Any]]] = None,
    validation_results: Optional[Sequence[Any]] = None,
    consistency_results: Optional[Sequence[Any]] = None,
    outlier_method: str = "iqr",
) -> ValidationReport:
    """Validate dataset state after cleaning against baseline quality characteristics and goals.

    This function is strictly non-destructive and does NOT modify the DataFrame.

    Args:
        cleaner: Cleaner instance or current working DataFrame.
        initial_df: Baseline DataFrame before cleaning. Required if cleaner is a DataFrame.
            If cleaner is a Cleaner instance, its initial_data is used automatically.
        goals: Optional sequence of goal names (e.g. ['missing', 'duplicates'])
            or dict of target thresholds (e.g. {'missing': 0, 'duplicates': 0}).
            If None, goals are inferred from cleaning operations in cleaner.history.
        rules: Optional dict of value validation rules (e.g. {'age': {'min_value': 0, 'max_value': 120}}).
        validation_results: Optional sequence of pre-computed ValidationResult objects.
        consistency_results: Optional sequence of pre-computed ConsistencyResult objects.
        outlier_method: Method for outlier detection ('iqr' or 'zscore'). Default 'iqr'.

    Returns:
        ValidationReport: Structured report detailing resolved issues, remaining issues,
            and issues requiring manual review.
    """
    # Extract DataFrames and history safely
    if hasattr(cleaner, "initial_data") and hasattr(cleaner, "data"):
        init_data = cleaner.initial_data.copy()
        current_data = cleaner.data.copy()
        history = list(getattr(cleaner, "history", []))
    elif isinstance(cleaner, pd.DataFrame):
        current_data = cleaner.copy()
        init_data = initial_df.copy() if initial_df is not None else cleaner.copy()
        history = []
    else:
        raise TypeError(
            f"Expected cleaner to be a Cleaner instance or pd.DataFrame, got '{type(cleaner).__name__}'."
        )

    # Generate suggestions for initial and current datasets
    initial_report = generate_suggestions(
        init_data,
        outlier_method=outlier_method,
        rules=rules,
        validation_results=validation_results,
        consistency_results=consistency_results,
    )
    current_report = generate_suggestions(
        current_data,
        outlier_method=outlier_method,
        rules=rules,
        validation_results=validation_results,
        consistency_results=consistency_results,
    )

    # 1. Compute before and after summaries
    before_summary: Dict[str, int] = {}
    for s in initial_report:
        before_summary[s.issue] = before_summary.get(s.issue, 0) + s.count

    after_summary: Dict[str, int] = {}
    # Seed with all issues present before cleaning so resolved issues explicitly display count 0
    for issue in before_summary:
        after_summary[issue] = 0

    for s in current_report:
        after_summary[s.issue] = after_summary.get(s.issue, 0) + s.count

    # 2. Build column rename mapping from cleaner history if any
    column_mapping: Dict[str, str] = {}
    for rec in history:
        if rec.operation == "clean_column_names" and isinstance(rec.details, dict) and "renamed" in rec.details:
            for old_c, new_c in rec.details["renamed"].items():
                column_mapping[old_c] = new_c

    # 3. Identify resolved issues by comparing before and after counts per (issue, column)
    resolved_issues: List[ResolvedIssue] = []

    # Map (issue, col) -> before count
    initial_groups: Dict[tuple, int] = {}
    for s in initial_report:
        key = (s.issue, s.column)
        initial_groups[key] = initial_groups.get(key, 0) + s.count

    # Map (issue, col) -> current count
    current_groups: Dict[tuple, int] = {}
    for s in current_report:
        key = (s.issue, s.column)
        current_groups[key] = current_groups.get(key, 0) + s.count

    for (issue, col), b_count in initial_groups.items():
        # Handle renamed columns: find corresponding column name in current dataset
        target_col = col
        if col is not None and col in column_mapping and issue != "inconsistent_column_name":
            target_col = column_mapping[col]

        a_count = current_groups.get((issue, target_col), 0)
        if a_count < b_count:
            resolved_count = b_count - a_count
            col_desc = f"in column '{col}'" if col else "in dataset"
            desc = (
                f"Resolved {resolved_count} {issue.replace('_', ' ')} item(s) {col_desc} "
                f"({b_count} -> {a_count})."
            )
            resolved_issues.append(
                ResolvedIssue(
                    issue=issue,
                    column=col,
                    before_count=b_count,
                    after_count=a_count,
                    resolved_count=resolved_count,
                    description=desc,
                )
            )

    # 4. Identify remaining issues and manual review issues
    remaining_issues: List[RemainingIssue] = []
    manual_review_issues: List[RemainingIssue] = []

    for s in current_report:
        requires_manual = not s.safe_to_automate
        rem = RemainingIssue(
            issue=s.issue,
            column=s.column,
            count=s.count,
            severity=s.severity,
            description=s.description,
            requires_manual_review=requires_manual,
            suggested_action=s.suggestion,
        )
        remaining_issues.append(rem)
        if requires_manual:
            manual_review_issues.append(rem)

    # 5. Determine whether cleaning goals were satisfied
    goals_satisfied = True
    goal_evaluations: Dict[str, Any] = {}

    if goals is not None:
        # User explicitly supplied goals
        if isinstance(goals, dict):
            for goal_key, target_val in goals.items():
                target_issue = GOAL_ISSUE_MAPPING.get(goal_key.lower(), goal_key)
                actual_val = after_summary.get(target_issue, 0)
                passed = actual_val <= target_val
                goal_evaluations[goal_key] = {
                    "issue": target_issue,
                    "target": target_val,
                    "actual": actual_val,
                    "passed": passed,
                }
                if not passed:
                    goals_satisfied = False
        else:
            for goal_item in goals:
                target_issue = GOAL_ISSUE_MAPPING.get(str(goal_item).lower(), str(goal_item))
                actual_val = after_summary.get(target_issue, 0)
                passed = actual_val == 0
                goal_evaluations[str(goal_item)] = {
                    "issue": target_issue,
                    "target": 0,
                    "actual": actual_val,
                    "passed": passed,
                }
                if not passed:
                    goals_satisfied = False
        is_valid = goals_satisfied
    else:
        # Infer goals from cleaning operations in cleaner history
        inferred_issues = set()
        for rec in history:
            if rec.operation in OPERATION_GOAL_MAPPING:
                inferred_issues.add(OPERATION_GOAL_MAPPING[rec.operation])

        if inferred_issues:
            for target_issue in inferred_issues:
                actual_val = after_summary.get(target_issue, 0)
                passed = actual_val == 0
                goal_evaluations[target_issue] = {
                    "issue": target_issue,
                    "target": 0,
                    "actual": actual_val,
                    "passed": passed,
                }
                if not passed:
                    goals_satisfied = False
            is_valid = goals_satisfied
        else:
            # No operations performed; goals satisfied if dataset has 0 automated errors
            automated_remaining = [r for r in remaining_issues if not r.requires_manual_review]
            goals_satisfied = len(automated_remaining) == 0
            is_valid = goals_satisfied

    details = {
        "initial_row_count": len(init_data),
        "current_row_count": len(current_data),
        "goal_evaluations": goal_evaluations,
        "cleaner_history_length": len(history),
    }

    return ValidationReport(
        is_valid=is_valid,
        goals_satisfied=goals_satisfied,
        resolved_issues=resolved_issues,
        remaining_issues=remaining_issues,
        manual_review_issues=manual_review_issues,
        before_summary=before_summary,
        after_summary=after_summary,
        details=details,
    )
