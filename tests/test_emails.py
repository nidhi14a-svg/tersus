"""Tests for email validation and normalization module."""

import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, ValidationResult
from tersus.emails import (
    clean_email_column,
    is_valid_email,
    normalize_email,
    validate_email,
)


def test_valid_email_formats():
    valid_emails = [
        "alice@example.com",
        "bob.smith@company.org",
        "carol+dev@sub.domain.co.in",
        "user_123@domain.io",
        "first-last@dept.school.edu",
        "admin@123domain.com",
    ]
    for email in valid_emails:
        assert is_valid_email(email) is True, f"Expected {email} to be valid"


def test_invalid_email_formats():
    invalid_emails = [
        "plainaddress",
        "@missinguser.com",
        "missingdomain@",
        "user@.com",
        "user@domain",
        "user@domain.c",
        "user..name@domain.com",
        ".user@domain.com",
        "user.@domain.com",
        "user@domain..com",
        "user name@domain.com",
        "user@@domain.com",
        "12345",
        "hello@world",
    ]
    for email in invalid_emails:
        assert is_valid_email(email) is False, f"Expected {email} to be invalid"


def test_normalize_email():
    # Whitespace stripping and lowercase normalization
    assert normalize_email("  Alice@Example.COM  ") == "alice@example.com"
    assert normalize_email("Bob.Smith@Domain.Org") == "bob.smith@domain.org"

    # Missing values safely preserved
    assert pd.isna(normalize_email(None))
    assert pd.isna(normalize_email(np.nan))
    assert pd.isna(normalize_email(""))
    assert pd.isna(normalize_email("   "))
    assert pd.isna(normalize_email("none"))


def test_email_validation_in_cleaner():
    df = pd.DataFrame(
        {
            "user": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "email": [
                "alice@example.com",
                "bad_email_at_nowhere",
                None,
                "dave@@domain.com",
                "eve@sub.domain.org",
            ],
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_email("email")

    assert isinstance(res, ValidationResult)
    assert res.is_valid is False
    assert res.has_violations is True
    assert res.invalid_count == 2
    assert res.invalid_indices == [1, 3]
    assert res.invalid_values == ["bad_email_at_nowhere", "dave@@domain.com"]

    # Verify invalid_rows
    assert len(res.invalid_rows) == 2
    assert res.invalid_rows["user"].tolist() == ["Bob", "David"]

    # Original DataFrame is NOT modified
    assert len(cleaner.data) == 5
    assert cleaner.data["email"].iloc[1] == "bad_email_at_nowhere"


def test_email_cleaning_and_normalization():
    df = pd.DataFrame(
        {
            "email": [
                "  Alice@Example.COM  ",
                "Bob.Smith@Sub.Domain.ORG",
                None,
                "  Carol+work@Gmail.com  ",
            ]
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_email("email")

    expected = [
        "alice@example.com",
        "bob.smith@sub.domain.org",
        pd.NA,
        "carol+work@gmail.com",
    ]
    for actual, exp in zip(cleaner.data["email"], expected):
        if pd.isna(exp):
            assert pd.isna(actual)
        else:
            assert actual == exp

    assert len(cleaner.history) == 1
    assert cleaner.last_operation.operation == "clean_email"


def test_clean_email_error_modes():
    df = pd.DataFrame(
        {
            "email": ["alice@example.com", "not_an_email", "bob@example.com"]
        }
    )

    # 1. errors="raise"
    c_raise = Cleaner(df)
    with pytest.raises(ValueError, match="Invalid email format"):
        c_raise.clean_email("email", errors="raise")

    # 2. errors="coerce"
    c_coerce = Cleaner(df)
    c_coerce.clean_email("email", errors="coerce")
    assert c_coerce.data["email"].iloc[0] == "alice@example.com"
    assert pd.isna(c_coerce.data["email"].iloc[1])
    assert c_coerce.data["email"].iloc[2] == "bob@example.com"

    # 3. errors="ignore"
    c_ignore = Cleaner(df)
    c_ignore.clean_email("email", errors="ignore")
    assert c_ignore.data["email"].iloc[1] == "not_an_email"


def test_email_drop_invalid():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "email": ["alice@example.com", "bademail", "bob@example.com"],
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_email("email")

    cleaner.drop_invalid(res)
    assert len(cleaner.data) == 2
    assert cleaner.data["id"].tolist() == [1, 3]


def test_email_strict_allow_na_false():
    df = pd.DataFrame({"email": ["alice@example.com", None, "bob@example.com"]})
    cleaner = Cleaner(df)

    res_allow = cleaner.check_email("email", allow_na=True)
    assert res_allow.is_valid is True

    res_strict = cleaner.check_email("email", allow_na=False)
    assert res_strict.is_valid is False
    assert res_strict.invalid_count == 1
    assert res_strict.invalid_indices == [1]
