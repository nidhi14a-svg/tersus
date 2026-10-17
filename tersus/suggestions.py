"""Suggestion engine module for Tersus.

Analyzes dataset profiling, quality inspection, and validation results to generate
understandable, structured cleaning suggestions without automatically mutating the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tersus.categories import find_category_variations
from tersus.columns import clean_single_column_name
from tersus.consistency import ConsistencyResult
from tersus.dates import detect_datetime_columns, is_datetime_column, validate_dates
from tersus.duplicates import get_duplicate_count
from tersus.emails import validate_email
from tersus.missing import get_missing_counts, get_missing_percentages
from tersus.outliers import detect_outliers
from tersus.phones import validate_phone
from tersus.profiling import ProfileResult, profile_dataframe
from tersus.text import INVISIBLE_REGEX
from tersus.validation import ValidationResult, validate_range


@dataclass
class Suggestion:
    """Individual data cleaning suggestion produced by the Tersus suggestion engine."""

    issue: str
    column: Optional[str]
    count: int
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    suggestion: str
    safe_to_automate: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def issue_type(self) -> str:
        """Alias for issue."""
        return self.issue

    @property
    def affected_column(self) -> Optional[str]:
        """Alias for column."""
        return self.column

    @property
    def affected_count(self) -> int:
        """Alias for count."""
        return self.count

    @property
    def suggested_action(self) -> str:
        """Alias for suggestion."""
        return self.suggestion

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion into a dictionary."""
        return {
            "issue": self.issue,
            "column": self.column,
            "count": self.count,
            "severity": self.severity,
            "description": self.description,
            "suggestion": self.suggestion,
            "safe_to_automate": self.safe_to_automate,
            "details": dict(self.details),
        }

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-like access to suggestion attributes."""
        d = self.to_dict()
        aliases = {
            "issue_type": "issue",
            "affected_column": "column",
            "affected_count": "count",
            "suggested_action": "suggestion",
        }
        lookup = aliases.get(key, key)
        if lookup in d:
            return d[lookup]
        raise KeyError(f"Key '{key}' not found in Suggestion.")

    def __repr__(self) -> str:
        return (
            f"Suggestion(issue='{self.issue}', column={self.column!r}, "
            f"count={self.count}, severity='{self.severity}', "
            f"safe_to_automate={self.safe_to_automate})"
        )

    def __str__(self) -> str:
        col_str = f"[{self.column}]" if self.column else "[Dataset]"
        safe_str = "SAFE" if self.safe_to_automate else "MANUAL"
        return (
            f"{col_str} {self.issue.upper()} ({self.severity.upper()}, {safe_str}): "
            f"{self.description} -> Suggested action: {self.suggestion}"
        )


@dataclass
class SuggestionReport:
    """Structured collection of suggestions generated from dataset quality analysis."""

    suggestions: List[Suggestion]
    summary: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.suggestions)

    def __iter__(self):
        return iter(self.suggestions)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Suggestion, List[Suggestion]]:
        return self.suggestions[idx]

    @property
    def has_suggestions(self) -> bool:
        """Whether any suggestions were generated."""
        return len(self.suggestions) > 0

    @property
    def safe_suggestions(self) -> List[Suggestion]:
        """Filter suggestions that are safe to automate."""
        return [s for s in self.suggestions if s.safe_to_automate]

    @property
    def manual_review_suggestions(self) -> List[Suggestion]:
        """Filter suggestions requiring human or domain review."""
        return [s for s in self.suggestions if not s.safe_to_automate]

    def by_column(self, column: str) -> List[Suggestion]:
        """Filter suggestions for a specific column."""
        return [s for s in self.suggestions if s.column == column]

    def by_issue(self, issue: str) -> List[Suggestion]:
        """Filter suggestions by issue type."""
        return [s for s in self.suggestions if s.issue == issue]

    def by_severity(self, severity: str) -> List[Suggestion]:
        """Filter suggestions by severity level."""
        return [s for s in self.suggestions if s.severity.lower() == severity.lower()]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert suggestions list into a Pandas DataFrame."""
        if not self.suggestions:
            return pd.DataFrame(
                columns=[
                    "issue",
                    "column",
                    "count",
                    "severity",
                    "description",
                    "suggestion",
                    "safe_to_automate",
                ]
            )
        data = [
            {
                "issue": s.issue,
                "column": s.column,
                "count": s.count,
                "severity": s.severity,
                "description": s.description,
                "suggestion": s.suggestion,
                "safe_to_automate": s.safe_to_automate,
            }
            for s in self.suggestions
        ]
        return pd.DataFrame(data)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert suggestions to a list of dictionaries."""
        return [s.to_dict() for s in self.suggestions]

    def __repr__(self) -> str:
        return f"SuggestionReport(total_suggestions={len(self.suggestions)})"

    def __str__(self) -> str:
        if not self.suggestions:
            return "No cleaning suggestions. The dataset appears clean."

        lines = [f"Tersus Cleaning Suggestions ({len(self.suggestions)} total):"]
        lines.append("=" * 60)
        for i, s in enumerate(self.suggestions, 1):
            lines.append(f"{i}. {s}")
        lines.append("=" * 60)
        return "\n".join(lines)


def generate_suggestions(
    df: pd.DataFrame,
    profile: Optional[ProfileResult] = None,
    validation_results: Optional[Sequence[ValidationResult]] = None,
    consistency_results: Optional[Sequence[ConsistencyResult]] = None,
    rules: Optional[Dict[str, Dict[str, Any]]] = None,
    outlier_method: str = "iqr",
) -> SuggestionReport:
    """Analyze a DataFrame and produce structured, understandable cleaning suggestions.

    This function does NOT modify the DataFrame or apply automated changes.

    Args:
        df: Pandas DataFrame to analyze.
        profile: Optional pre-computed ProfileResult.
        validation_results: Optional sequence of ValidationResult objects.
        consistency_results: Optional sequence of ConsistencyResult objects.
        rules: Optional dictionary mapping column names to validation kwargs (e.g. min_value, max_value).
        outlier_method: Outlier detection algorithm ('iqr' or 'zscore'). Default 'iqr'.

    Returns:
        SuggestionReport: Structured report of detected issues and recommendations.
    """
    suggestions: List[Suggestion] = []

    if df.empty:
        return SuggestionReport(suggestions=[], summary={"total": 0})

    # Ensure profile is available
    if profile is None:
        profile = profile_dataframe(df)

    n_rows = len(df)

    # 1. Duplicate rows
    if profile.duplicate_rows > 0:
        suggestions.append(
            Suggestion(
                issue="duplicate_rows",
                column=None,
                count=profile.duplicate_rows,
                severity="medium",
                description=f"Found {profile.duplicate_rows} exact duplicate row(s) in the dataset.",
                suggestion="drop duplicate rows",
                safe_to_automate=True,
                details={"duplicate_rows": profile.duplicate_rows},
            )
        )

    # 2. Inconsistent column names
    for col in df.columns:
        cleaned_col = clean_single_column_name(col)
        if cleaned_col != col:
            suggestions.append(
                Suggestion(
                    issue="inconsistent_column_name",
                    column=col,
                    count=1,
                    severity="low",
                    description=(
                        f"Column name '{col}' contains spaces, uppercase characters, or special symbols."
                    ),
                    suggestion=f"rename column to '{cleaned_col}'",
                    safe_to_automate=True,
                    details={"cleaned_name": cleaned_col},
                )
            )

    # 3. Missing values
    for col in df.columns:
        m_count = profile.missing_counts.get(col, 0)
        if m_count > 0:
            m_ratio = m_count / n_rows if n_rows > 0 else 0.0
            m_pct_display = f"{m_ratio * 100.0:.1f}%"
            is_numeric = col in profile.numeric_stats

            if m_count == n_rows:
                # 100% missing
                suggestions.append(
                    Suggestion(
                        issue="missing_values",
                        column=col,
                        count=m_count,
                        severity="high",
                        description=f"Column '{col}' is entirely empty (100% missing values).",
                        suggestion="drop all-null column",
                        safe_to_automate=True,
                        details={"percentage": 1.0, "is_all_null": True},
                    )
                )
            elif is_numeric:
                # Low missing rate: median imputation is typically safe
                if m_ratio <= 0.15:
                    suggestions.append(
                        Suggestion(
                            issue="missing_values",
                            column=col,
                            count=m_count,
                            severity="low",
                            description=(
                                f"Column '{col}' has {m_count} missing numerical value(s) ({m_pct_display})."
                            ),
                            suggestion="fill with median",
                            safe_to_automate=True,
                            details={"percentage": m_ratio, "recommended_strategy": "median"},
                        )
                    )
                elif m_ratio >= 0.60:
                    suggestions.append(
                        Suggestion(
                            issue="missing_values",
                            column=col,
                            count=m_count,
                            severity="high",
                            description=(
                                f"Column '{col}' has a high missing rate ({m_count} values, {m_pct_display})."
                            ),
                            suggestion="review missing values before imputing, or consider dropping column",
                            safe_to_automate=False,
                            details={"percentage": m_ratio, "recommended_strategy": "review"},
                        )
                    )
                else:
                    suggestions.append(
                        Suggestion(
                            issue="missing_values",
                            column=col,
                            count=m_count,
                            severity="medium",
                            description=(
                                f"Column '{col}' has {m_count} missing value(s) ({m_pct_display})."
                            ),
                            suggestion="fill with median or mean",
                            safe_to_automate=True,
                            details={"percentage": m_ratio, "recommended_strategy": "median"},
                        )
                    )
            else:
                # Categorical / text missing values
                suggestions.append(
                    Suggestion(
                        issue="missing_values",
                        column=col,
                        count=m_count,
                        severity="medium",
                        description=(
                            f"Column '{col}' has {m_count} missing categorical/text value(s) ({m_pct_display})."
                        ),
                        suggestion="fill with mode or a user-provided constant",
                        safe_to_automate=False,
                        details={"percentage": m_ratio, "recommended_strategy": "mode"},
                    )
                )

    # 4. Whitespace and invisible-character problems
    for col in df.columns:
        if col not in profile.numeric_stats:
            series = df[col].dropna()
            if series.empty:
                continue

            str_series = series.astype(str)
            whitespace_mask = str_series.apply(
                lambda s: (
                    s != s.strip()
                    or bool(re.search(r" {2,}", s))
                    or bool(INVISIBLE_REGEX.search(s))
                    or bool(re.search(r"[\t\n\r]", s))
                )
            )
            ws_count = int(whitespace_mask.sum())
            if ws_count > 0:
                suggestions.append(
                    Suggestion(
                        issue="whitespace_problems",
                        column=col,
                        count=ws_count,
                        severity="low",
                        description=(
                            f"Column '{col}' has {ws_count} value(s) with leading/trailing spaces, "
                            f"multiple spaces, or invisible characters."
                        ),
                        suggestion="strip whitespace and clean invisible characters",
                        safe_to_automate=True,
                        details={"affected_rows": ws_count},
                    )
                )

    # 5. Suspicious data types
    for col in df.columns:
        if col not in profile.numeric_stats:
            series = df[col].dropna()
            if series.empty:
                continue

            str_series = series.astype(str)
            # Check if strings are numeric numbers
            parsed_num = pd.to_numeric(str_series, errors="coerce")
            valid_num_count = int(parsed_num.notna().sum())

            if len(str_series) > 0 and (valid_num_count / len(str_series)) >= 0.85:
                # Mostly or completely numeric stored as string/object
                non_null_nums = parsed_num.dropna()
                is_int = (
                    bool((non_null_nums % 1 == 0).all())
                    if not non_null_nums.empty
                    else False
                )
                target = "int" if is_int else "float"
                suggestions.append(
                    Suggestion(
                        issue="suspicious_dtype",
                        column=col,
                        count=valid_num_count,
                        severity="medium",
                        description=(
                            f"Column '{col}' is stored as object/string, but "
                            f"{valid_num_count}/{len(str_series)} values appear to be numeric."
                        ),
                        suggestion=f"convert data type to {target}",
                        safe_to_automate=False,
                        details={"detected_type": target, "numeric_ratio": valid_num_count / len(str_series)},
                    )
                )
            elif is_datetime_column(df[col]):
                suggestions.append(
                    Suggestion(
                        issue="suspicious_dtype",
                        column=col,
                        count=len(str_series),
                        severity="medium",
                        description=(
                            f"Column '{col}' is stored as object/string, but contains date/datetime values."
                        ),
                        suggestion="convert data type to datetime",
                        safe_to_automate=False,
                        details={"detected_type": "datetime"},
                    )
                )

    # 6. Potential outliers (numeric columns)
    for col in profile.numeric_columns:
        try:
            outlier_res = detect_outliers(df, col, method=outlier_method)
            if outlier_res.outlier_count > 0:
                suggestions.append(
                    Suggestion(
                        issue="potential_outlier",
                        column=col,
                        count=outlier_res.outlier_count,
                        severity="medium",
                        description=(
                            f"Column '{col}' has {outlier_res.outlier_count} potential outlier(s) "
                            f"detected by {outlier_res.method.upper()} "
                            f"(bounds: [{outlier_res.lower_bound}, {outlier_res.upper_bound}])."
                        ),
                        suggestion="review manually",
                        safe_to_automate=False,
                        details={
                            "method": outlier_res.method,
                            "lower_bound": outlier_res.lower_bound,
                            "upper_bound": outlier_res.upper_bound,
                            "indices": outlier_res.outlier_indices,
                        },
                    )
                )
        except Exception:
            pass

    # 7. Invalid dates
    date_candidates = detect_datetime_columns(df)
    for col in date_candidates:
        try:
            d_res = validate_dates(df, col)
            if d_res.invalid_count > 0:
                suggestions.append(
                    Suggestion(
                        issue="invalid_dates",
                        column=col,
                        count=d_res.invalid_count,
                        severity="high",
                        description=(
                            f"Column '{col}' contains {d_res.invalid_count} unparseable or invalid date value(s)."
                        ),
                        suggestion="review and correct invalid date formats",
                        safe_to_automate=False,
                        details={"invalid_indices": d_res.invalid_indices},
                    )
                )
        except Exception:
            pass

    # 8. Invalid email formats
    for col in df.columns:
        if col not in profile.numeric_stats:
            col_lower = col.lower()
            if "email" in col_lower or df[col].astype(str).str.contains("@").any():
                try:
                    e_res = validate_email(df, col)
                    if e_res.invalid_count > 0:
                        suggestions.append(
                            Suggestion(
                                issue="invalid_email",
                                column=col,
                                count=e_res.invalid_count,
                                severity="high",
                                description=(
                                    f"Column '{col}' contains {e_res.invalid_count} invalid email address format(s)."
                                ),
                                suggestion="review or correct invalid email formats",
                                safe_to_automate=False,
                                details={"invalid_indices": e_res.invalid_indices},
                            )
                        )
                except Exception:
                    pass

    # 9. Phone-format problems
    for col in df.columns:
        if col not in profile.numeric_stats:
            col_lower = col.lower()
            if any(term in col_lower for term in ("phone", "mobile", "tel", "contact")):
                try:
                    p_res = validate_phone(df, col)
                    if p_res.invalid_count > 0:
                        suggestions.append(
                            Suggestion(
                                issue="invalid_phone",
                                column=col,
                                count=p_res.invalid_count,
                                severity="medium",
                                description=(
                                    f"Column '{col}' contains {p_res.invalid_count} invalid or non-standard phone number(s)."
                                ),
                                suggestion="review or standardize phone numbers",
                                safe_to_automate=False,
                                details={"invalid_indices": p_res.invalid_indices},
                            )
                        )
                except Exception:
                    pass

    # 10. Category inconsistencies
    for col in profile.categorical_columns:
        if col in df.columns:
            u_count = profile.unique_counts.get(col, df[col].nunique())
            if 1 < u_count <= 40 and u_count < n_rows:
                try:
                    c_res = find_category_variations(df, col)
                    if c_res.has_variations:
                        variant_count = sum(
                            len(v) for v in c_res.variations.values() if len(v) > 1
                        )
                        suggestions.append(
                            Suggestion(
                                issue="category_inconsistencies",
                                column=col,
                                count=variant_count,
                                severity="medium",
                                description=(
                                    f"Column '{col}' contains potential category variations: "
                                    f"{list(c_res.variations.keys())}."
                                ),
                                suggestion=f"review and apply category mapping",
                                safe_to_automate=False,
                                details={"suggested_mapping": c_res.suggested_mapping},
                            )
                        )
                except Exception:
                    pass

    # 11. Range violations from user rules
    if rules:
        for col, rule_kwargs in rules.items():
            if col in df.columns:
                try:
                    v_res = validate_range(df, col, **rule_kwargs)
                    if not v_res.is_valid:
                        suggestions.append(
                            Suggestion(
                                issue="range_violations",
                                column=col,
                                count=v_res.invalid_count,
                                severity="high",
                                description=(
                                    f"Column '{col}' has {v_res.invalid_count} value(s) "
                                    f"violating range rule '{v_res.rule}'."
                                ),
                                suggestion="clamp values to range or review invalid rows",
                                safe_to_automate=False,
                                details={"rule": v_res.rule, "invalid_indices": v_res.invalid_indices},
                            )
                        )
                except Exception:
                    pass

    # 12. External validation results (range, values, custom rules)
    if validation_results:
        for vr in validation_results:
            if not vr.is_valid:
                issue_name = (
                    "range_violations"
                    if any(op in vr.rule for op in ("<", ">", "<=", ">="))
                    else "invalid_values"
                )
                suggestions.append(
                    Suggestion(
                        issue=issue_name,
                        column=vr.column,
                        count=vr.invalid_count,
                        severity="high",
                        description=(
                            f"Column '{vr.column}' has {vr.invalid_count} value(s) failing rule '{vr.rule}'."
                        ),
                        suggestion="review and resolve invalid values",
                        safe_to_automate=False,
                        details={"rule": vr.rule, "invalid_indices": vr.invalid_indices},
                    )
                )

    # 13. Logical consistency results
    if consistency_results:
        for cr in consistency_results:
            if not cr.is_consistent:
                rec_action = (
                    cr.suggestions[0].description
                    if cr.suggestions
                    else "review and resolve logical contradiction"
                )
                suggestions.append(
                    Suggestion(
                        issue="logical_inconsistency",
                        column=", ".join(cr.columns),
                        count=cr.inconsistent_count,
                        severity="high",
                        description=(
                            f"Logical consistency rule '{cr.rule}' failed for "
                            f"{cr.inconsistent_count} row(s)."
                        ),
                        suggestion=rec_action,
                        safe_to_automate=False,
                        details={"rule": cr.rule, "inconsistent_indices": cr.inconsistent_indices},
                    )
                )

    severity_counts = {
        "low": sum(1 for s in suggestions if s.severity == "low"),
        "medium": sum(1 for s in suggestions if s.severity == "medium"),
        "high": sum(1 for s in suggestions if s.severity == "high"),
        "critical": sum(1 for s in suggestions if s.severity == "critical"),
    }

    summary = {
        "total": len(suggestions),
        "safe_to_automate_count": sum(1 for s in suggestions if s.safe_to_automate),
        "manual_review_count": sum(1 for s in suggestions if not s.safe_to_automate),
        "by_severity": severity_counts,
    }

    return SuggestionReport(suggestions=suggestions, summary=summary)
