"""Tests for categorical-value normalization and variation inspection."""

import numpy as np
import pandas as pd
import pytest

from tersus import CategoryVariationsResult, Cleaner
from tersus.categories import (
    find_category_variations,
    map_category_values,
    normalize_category_strings,
)


def test_specification_variation_identification():
    """Verify specification example:

    Male
    male
    M
    MALE

    Ensure candidate variations are detected without automatically mutating data.
    """
    df = pd.DataFrame(
        {
            "gender": ["Male", "male", "M", "MALE", "Female", "female", "F", "Other", None],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.inspect_categories("gender")

    assert isinstance(result, CategoryVariationsResult)
    assert result.has_variations is True
    assert result.unique_count == 8

    # Verify grouping of variations
    variations = result.variations
    assert "male" in variations
    assert "female" in variations

    male_variants = variations["male"]
    for expected in ["Male", "male", "M", "MALE"]:
        assert expected in male_variants

    female_variants = variations["female"]
    for expected in ["Female", "female", "F"]:
        assert expected in female_variants

    assert "Other" in variations["other"]

    # CRITICAL: DataFrame must NOT be automatically modified!
    # Tersus does NOT automatically assume two categories mean the same thing.
    assert cleaner.data["gender"].iloc[0] == "Male"
    assert cleaner.data["gender"].iloc[1] == "male"
    assert cleaner.data["gender"].iloc[2] == "M"
    assert cleaner.data["gender"].iloc[3] == "MALE"
    assert len(cleaner.history) == 0


def test_explicit_user_mapping():
    """User explicitly provides a mapping to unify variations."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "gender": ["Male", "male", "M", "MALE", "Female", "female", "F", "Other", None],
        }
    )
    cleaner = Cleaner(df)

    # Explicit user mapping dictionary
    mapping = {
        "M": "Male",
        "male": "Male",
        "MALE": "Male",
        "F": "Female",
        "female": "Female",
    }
    cleaner.map_categories("gender", mapping=mapping)

    expected = ["Male", "Male", "Male", "Male", "Female", "Female", "Female", "Other", None]
    for actual, exp in zip(cleaner.data["gender"], expected):
        if exp is None:
            assert actual is None or pd.isna(actual)
        else:
            assert actual == exp

    assert len(cleaner.history) == 1
    assert cleaner.last_operation.operation == "map_categories"
    assert cleaner.last_operation.affected_count == 5


def test_canonical_to_list_mapping():
    """Support convenient target-to-variants list mapping."""
    df = pd.DataFrame(
        {
            "status": ["Active", "active", "ACT", "Pending", "pend", "Closed", None],
        }
    )
    cleaner = Cleaner(df)

    cleaner.map_categories(
        "status",
        mapping={
            "Active": ["active", "ACT"],
            "Pending": ["pend"],
        },
    )

    expected = ["Active", "Active", "Active", "Pending", "Pending", "Closed", None]
    for actual, exp in zip(cleaner.data["status"], expected):
        if exp is None:
            assert actual is None or pd.isna(actual)
        else:
            assert actual == exp


def test_mapping_with_default_fallback():
    df = pd.DataFrame({"color": ["Red", "Blue", "Green", "Unknown_Color", None]})
    cleaner = Cleaner(df)

    cleaner.map_categories(
        "color",
        mapping={"Red": "RED", "Blue": "BLUE", "Green": "GREEN"},
        default="OTHER",
    )
    assert cleaner.data["color"].iloc[0] == "RED"
    assert cleaner.data["color"].iloc[1] == "BLUE"
    assert cleaner.data["color"].iloc[2] == "GREEN"
    assert cleaner.data["color"].iloc[3] == "OTHER"
    assert pd.isna(cleaner.data["color"].iloc[4])


def test_case_sensitive_option():
    df = pd.DataFrame({"code": ["m", "M", "Male"]})

    # Case-sensitive: only exact match replaces
    c_sens = Cleaner(df)
    c_sens.map_categories("code", mapping={"M": "Male_Code"}, case_sensitive=True)
    assert c_sens.data["code"].tolist() == ["m", "Male_Code", "Male"]

    # Case-insensitive (default): both m and M match
    c_insens = Cleaner(df)
    c_insens.map_categories("code", mapping={"m": "Male_Code"}, case_sensitive=False)
    assert c_insens.data["code"].iloc[0] == "Male_Code"
    assert c_insens.data["code"].iloc[1] == "Male_Code"


def test_normalize_categories_title_and_trim():
    df = pd.DataFrame(
        {
            "department": ["  engineering  ", "SALES", "marketing", None, "Human Resources"]
        }
    )
    cleaner = Cleaner(df)
    cleaner.normalize_categories("department", case="title", strip_whitespace=True)

    assert cleaner.data["department"].iloc[0] == "Engineering"
    assert cleaner.data["department"].iloc[1] == "Sales"
    assert cleaner.data["department"].iloc[2] == "Marketing"
    assert pd.isna(cleaner.data["department"].iloc[3])
    assert cleaner.data["department"].iloc[4] == "Human Resources"


def test_single_letter_clustering_known_ambiguity_is_not_auto_applied():
    """Negative/limitation case for the single-letter-prefix clustering heuristic.

    'N' is ambiguous between 'No' and 'None'. The heuristic will cluster it
    with one of them (documented in find_category_variations' docstring as a
    known limitation) rather than raising an error, and this is only ever a
    *suggestion* -- it must never be applied automatically.
    """
    df = pd.DataFrame({"resp": ["N", "No", "None", "Yes"]})
    cleaner = Cleaner(df)
    result = cleaner.inspect_categories("resp")

    # 'N' is clustered with exactly one of its ambiguous candidates, not both,
    # and 'Yes' (unrelated) is never pulled into any cluster.
    clustered_with_n = [key for key, vals in result.variations.items() if "N" in vals]
    assert len(clustered_with_n) == 1
    assert clustered_with_n[0] in ("no", "none")
    # 'Yes' has no single-letter sibling in this column, so it must stay in
    # its own untouched cluster rather than being pulled into another one.
    assert result.variations["yes"] == ["Yes"]

    # Critically: this is diagnostic only. The underlying data is untouched
    # until the user explicitly calls map_category_values with their own
    # reviewed mapping.
    assert cleaner.data["resp"].tolist() == ["N", "No", "None", "Yes"]


def test_category_variations_result_formatting():
    df = pd.DataFrame({"cat": ["A", "a", "B"]})
    cleaner = Cleaner(df)
    res = cleaner.inspect_categories("cat")

    assert res.has_variations is True
    d = res.to_dict()
    assert d["column"] == "cat"
    assert d["unique_count"] == 3
    assert "a" in d["variations"]

    s = str(res)
    assert "Category Variations Report on 'cat'" in s
    assert "Detected Potential Variations" in s
