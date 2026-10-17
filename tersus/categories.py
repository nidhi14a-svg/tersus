"""Categorical-value normalization and variation inspection module for Tersus.

Features:
- Identification and clustering of possible category variations (e.g. Male, male, M, MALE).
- Explicit user-guided categorical mapping (does NOT automatically assume variations mean the same thing).
- Safe preservation of missing values.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from tersus.missing import CleaningRecord

KNOWN_CITY_ALIASES: Dict[str, str] = {
    "bangalore": "Bangalore",
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "madras": "Chennai",
    "chennai": "Chennai",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "delhi": "Delhi",
    "new delhi": "Delhi",
}


@dataclass
class CategoryVariationsResult:
    """Structured report of category variations discovered in a column.

    Note: Variations are grouped as candidates for human inspection.
    Tersus does NOT automatically assume two categories have the same meaning
    without explicit user confirmation or mapping.
    """

    column: str
    unique_count: int
    unique_values: List[Any]
    variations: Dict[str, List[Any]]
    suggested_mapping: Dict[Any, str]

    @property
    def has_variations(self) -> bool:
        """Whether any category has multiple detected variations."""
        return any(len(variants) > 1 for variants in self.variations.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "column": self.column,
            "unique_count": self.unique_count,
            "unique_values": list(self.unique_values),
            "variations": {k: list(v) for k, v in self.variations.items()},
            "suggested_mapping": dict(self.suggested_mapping),
            "has_variations": self.has_variations,
        }

    def __repr__(self) -> str:
        return (
            f"CategoryVariationsResult(column='{self.column}', "
            f"unique_count={self.unique_count}, has_variations={self.has_variations})"
        )

    def __str__(self) -> str:
        lines = [
            f"Category Variations Report on '{self.column}':",
            f"  Total Unique Values: {self.unique_count}",
        ]
        multi_variants = {k: v for k, v in self.variations.items() if len(v) > 1}
        if multi_variants:
            lines.append("  Detected Potential Variations (for human review):")
            for key, variants in multi_variants.items():
                lines.append(f"    - '{key}': {variants}")
        else:
            lines.append("  No obvious variations detected among categories.")

        return "\n".join(lines)


def find_category_variations(
    df: pd.DataFrame, column: str
) -> CategoryVariationsResult:
    """Inspect a column to discover possible categorical variations.

    Groups:
    1. Case and whitespace variations (e.g. 'Male', 'male', 'MALE', ' male ')
    2. Single-letter abbreviation/prefix candidates (e.g. 'M' and 'Male')

    Note: This is a diagnostic function only. It does NOT automatically merge
    or modify any data without an explicit user mapping.

    Args:
        df: Pandas DataFrame.
        column: Column name to inspect.

    Returns:
        CategoryVariationsResult: Report of discovered unique values and candidate variations.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    series = df[column]
    non_null_values = [v for v in series if v is not None and not pd.isna(v)]
    unique_values = list(dict.fromkeys(non_null_values))

    # Step 1: Group by stripped lowercase string representation
    base_groups = defaultdict(list)
    for u in unique_values:
        key = str(u).strip().lower()
        base_groups[key].append(u)

    # Step 2: Cluster abbreviations / single-character prefixes
    clusters = defaultdict(list)
    used = set()

    # Sort keys by length descending so longer words serve as cluster anchors
    sorted_keys = sorted(base_groups.keys(), key=lambda k: len(k), reverse=True)

    for key in sorted_keys:
        if key in used:
            continue
        vals = list(base_groups[key])

        # Look for 1-character prefixes (e.g. 'm' matching 'male', 'f' matching 'female')
        for other_key in sorted_keys:
            if other_key != key and other_key not in used:
                if len(other_key) == 1 and key.startswith(other_key):
                    vals.extend(base_groups[other_key])
                    used.add(other_key)

        clusters[key] = vals
        used.add(key)

    # Step 3: Prepare suggested mapping (for user inspection, NOT automatically applied)
    suggested: Dict[Any, str] = {}
    is_city_col = any(term in column.lower() for term in ("city", "location", "town"))

    for key, variants in clusters.items():
        if len(variants) > 1 or (is_city_col and key in KNOWN_CITY_ALIASES):
            # Pick canonical representation: longest title-cased or uppercase variant
            candidates = [v.strip() for v in variants if isinstance(v, str) and v.strip()]
            if is_city_col and key in KNOWN_CITY_ALIASES:
                canonical = KNOWN_CITY_ALIASES[key]
            else:
                canonical = (
                    key.title()
                    if not candidates
                    else max(candidates, key=lambda s: (len(s), s.istitle()))
                )
            for v in variants:
                suggested[v] = str(canonical)

    if is_city_col:
        # Cross-unify city aliases in suggested mapping
        for v in unique_values:
            k = str(v).strip().lower()
            if k in KNOWN_CITY_ALIASES:
                suggested[v] = KNOWN_CITY_ALIASES[k]

    return CategoryVariationsResult(
        column=column,
        unique_count=len(unique_values),
        unique_values=unique_values,
        variations=dict(clusters),
        suggested_mapping=suggested,
    )


def map_category_values(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[Any, Any],
    case_sensitive: bool = False,
    default: Any = None,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Explicitly map categorical values to standard representations.

    Supports:
    1. Direct value-to-value: {'M': 'Male', 'm': 'Male', 'F': 'Female'}
    2. Canonical-to-variants list: {'Male': ['M', 'male', 'MALE'], 'Female': ['F', 'female']}

    Important: Tersus does not assume categories mean the same thing;
    the user must explicitly provide the mapping.

    Args:
        df: Pandas DataFrame.
        column: Column to map.
        mapping: Dictionary specifying category transformations.
        case_sensitive: Whether key lookup is case-sensitive (default False).
        default: If provided, values not found in mapping are replaced with default.
            If None (default), unmapped values remain unchanged.

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")

    # Invert canonical-to-list mappings if provided
    resolved_map: Dict[Any, Any] = {}
    for k, v in mapping.items():
        if isinstance(v, (list, tuple, set)):
            # e.g. 'Male': ['M', 'male', 'MALE'] -> 'M': 'Male', etc.
            for item in v:
                resolved_map[item] = k
        else:
            resolved_map[k] = v

    lookup_map: Dict[Any, Any] = {}
    for k, v in resolved_map.items():
        if not case_sensitive and isinstance(k, str):
            lookup_map[k.strip().lower()] = v
        else:
            lookup_map[k] = v

    series = df[column]
    mapped_values: List[Any] = []
    affected_count = 0

    for idx, val in series.items():
        if val is None or pd.isna(val):
            mapped_values.append(val)
            continue

        target_val = None
        found = False

        if not case_sensitive and isinstance(val, str):
            lookup_key = val.strip().lower()
            if lookup_key in lookup_map:
                target_val = lookup_map[lookup_key]
                found = True
        elif val in lookup_map:
            target_val = lookup_map[val]
            found = True

        if found:
            mapped_values.append(target_val)
            if target_val != val:
                affected_count += 1
        else:
            if default is not None:
                mapped_values.append(default)
                if default != val:
                    affected_count += 1
            else:
                mapped_values.append(val)

    df_clean = df.copy()
    df_clean[column] = pd.Series(mapped_values, index=series.index, dtype="object")

    record = CleaningRecord(
        operation="map_categories",
        strategy="explicit_mapping",
        columns=[column],
        affected_count=affected_count,
        details={
            "mapping": mapping,
            "case_sensitive": case_sensitive,
            "default": default,
        },
        message=f"Mapped {affected_count} categorical value(s) in column '{column}'.",
    )
    return df_clean, record


def normalize_category_strings(
    df: pd.DataFrame,
    column: str,
    case: Optional[str] = "title",
    strip_whitespace: bool = True,
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Safely normalize casing and whitespace for all categorical text in a column.

    Args:
        df: Pandas DataFrame.
        column: Column to normalize.
        case: 'title', 'lower', 'upper', or None.
        strip_whitespace: Whether to strip leading/trailing spaces (default True).

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")

    series = df[column]
    cleaned_values: List[Any] = []
    affected_count = 0

    for idx, val in series.items():
        if val is None or pd.isna(val):
            cleaned_values.append(val)
            continue

        if not isinstance(val, str):
            cleaned_values.append(val)
            continue

        s = val
        if strip_whitespace:
            s = s.strip()

        if case == "lower":
            s = s.lower()
        elif case == "upper":
            s = s.upper()
        elif case == "title":
            s = s.title()

        cleaned_values.append(s)
        if s != val:
            affected_count += 1

    df_clean = df.copy()
    df_clean[column] = pd.Series(cleaned_values, index=series.index, dtype="object")

    record = CleaningRecord(
        operation="normalize_categories",
        strategy="case_and_whitespace",
        columns=[column],
        affected_count=affected_count,
        details={"case": case, "strip_whitespace": strip_whitespace},
        message=f"Normalized {affected_count} category string(s) in column '{column}'.",
    )
    return df_clean, record
