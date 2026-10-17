"""Tests for phone-number validation and normalization (Indian V1 support)."""

import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, ValidationResult
from tersus.phones import (
    clean_phone_column,
    format_indian_phone,
    is_valid_phone,
    parse_indian_phone,
    validate_phone,
)


def test_valid_indian_phones_various_formats():
    valid_numbers = [
        "9876543210",               # 10 digits
        "09876543210",              # Leading STD trunk 0
        "+919876543210",            # Country code +91
        "919876543210",             # Country code 91
        "+91 98765 43210",          # Spaces
        "+91-98765-43210",          # Hyphens
        "(098765) 43210",           # Parentheses
        "98765 43210",              # National with space
        "98765-43210",              # National with hyphen
        9876543210,                 # Integer format
        "8765432109",               # Starting with 8
        "7654321098",               # Starting with 7
        "6543210987",               # Starting with 6
    ]
    for num in valid_numbers:
        assert is_valid_phone(num, country="IN") is True, f"Expected {num} to be valid"
        core, err = parse_indian_phone(num)
        assert err is None
        assert core is not None
        assert len(core) == 10
        assert core[0] in "6789"


def test_invalid_indian_phones():
    invalid_numbers = [
        "12345",                    # Too short
        "987654321",                # 9 digits
        "1234567890",               # Invalid starting digit (1)
        "2345678901",               # Invalid starting digit (2)
        "5555555555",               # Invalid starting digit (5)
        "98765432101234",           # Too long
        "abcdefghij",               # Letters
        "+91 98765 callme",         # Mixed text
        "0000000000",               # Repeating dummy
        "9999999999",               # Repeating dummy
    ]
    for num in invalid_numbers:
        assert is_valid_phone(num, country="IN") is False, f"Expected {num} to be invalid"


def test_phone_formatting_styles():
    core = "9876543210"
    assert format_indian_phone(core, "10-digit") == "9876543210"
    assert format_indian_phone(core, "+91") == "+919876543210"
    assert format_indian_phone(core, "spaced") == "+91 98765 43210"
    assert format_indian_phone(core, "hyphenated") == "+91-98765-43210"
    assert format_indian_phone(core, "national-spaced") == "98765 43210"


def test_phone_validation_in_cleaner():
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "mobile": ["+91 98765 43210", "12345", None, "08765432100"],
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_phone("mobile", country="IN")

    assert isinstance(res, ValidationResult)
    assert res.is_valid is False
    assert res.has_violations is True
    assert res.invalid_count == 1
    assert res.invalid_indices == [1]
    assert res.invalid_values == ["12345"]

    # Non-destructive
    assert len(cleaner.data) == 4


def test_clean_phone_and_normalization():
    df = pd.DataFrame(
        {
            "mobile": [
                "+91 98765 43210",
                "08765432100",
                None,
                "76543-21098",
            ]
        }
    )
    cleaner = Cleaner(df)

    # Standardize to 10-digit national
    cleaner.clean_phone("mobile", country="IN", output_format="10-digit")
    assert cleaner.data["mobile"].iloc[0] == "9876543210"
    assert cleaner.data["mobile"].iloc[1] == "8765432100"
    assert pd.isna(cleaner.data["mobile"].iloc[2])
    assert cleaner.data["mobile"].iloc[3] == "7654321098"

    # Standardize to +91 international
    cleaner.clean_phone("mobile", country="IN", output_format="+91")
    assert cleaner.data["mobile"].iloc[0] == "+919876543210"
    assert cleaner.data["mobile"].iloc[1] == "+918765432100"
    assert pd.isna(cleaner.data["mobile"].iloc[2])
    assert cleaner.data["mobile"].iloc[3] == "+917654321098"


def test_clean_phone_error_modes():
    df = pd.DataFrame({"mobile": ["9876543210", "bad_phone", "8765432109"]})

    # 1. errors="raise" (default)
    c_raise = Cleaner(df)
    with pytest.raises(ValueError, match="Invalid phone number"):
        c_raise.clean_phone("mobile", errors="raise")

    # 2. errors="coerce"
    c_coerce = Cleaner(df)
    c_coerce.clean_phone("mobile", errors="coerce")
    assert c_coerce.data["mobile"].iloc[0] == "9876543210"
    assert pd.isna(c_coerce.data["mobile"].iloc[1])
    assert c_coerce.data["mobile"].iloc[2] == "8765432109"

    # 3. errors="ignore"
    c_ignore = Cleaner(df)
    c_ignore.clean_phone("mobile", errors="ignore")
    assert c_ignore.data["mobile"].iloc[1] == "bad_phone"


def test_phone_drop_invalid():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "mobile": ["9876543210", "12345", "7654321098"],
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_phone("mobile")

    cleaner.drop_invalid(res)
    assert len(cleaner.data) == 2
    assert cleaner.data["id"].tolist() == [1, 3]


def test_missing_phone_handling():
    df = pd.DataFrame(
        {
            "mobile": ["9876543210", None, np.nan, "", "   "]
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_phone("mobile", allow_na=True)
    assert res.is_valid is True
    assert res.invalid_count == 0

    res_strict = cleaner.check_phone("mobile", allow_na=False)
    assert res_strict.is_valid is False
    assert res_strict.invalid_count == 4
