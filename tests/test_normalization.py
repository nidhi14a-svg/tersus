"""Tests for invisible characters, Unicode, special characters, numeric extraction, and currency cleaning."""

import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner
from tersus.text import (
    clean_currency_value,
    clean_invisible_characters,
    clean_special_characters,
    extract_numeric_value,
    normalize_unicode_text,
)


def test_invisible_character_cleaning():
    # String containing zero-width spaces (\u200B), byte order mark (\uFEFF), non-breaking space (\u00A0), and null bytes
    raw = "Hello\u200B \uFEFFWorld\u00A0Test\x00!\t\nNext"
    cleaned = clean_invisible_characters(raw)
    assert cleaned == "Hello World Test! Next"

    # Missing and non-string preservation
    assert clean_invisible_characters(None) is None
    assert pd.isna(clean_invisible_characters(np.nan))
    assert clean_invisible_characters(123) == 123


def test_clean_invisible_chars_in_cleaner():
    df = pd.DataFrame(
        {
            "text": ["Item\u200B1", "Value\u00A0Here", None, "Pure Text"],
            "num": [10, 20, 30, 40],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_invisible_chars("text")

    assert cleaner.data["text"].iloc[0] == "Item1"
    assert cleaner.data["text"].iloc[1] == "Value Here"
    assert pd.isna(cleaner.data["text"].iloc[2])
    assert cleaner.data["text"].iloc[3] == "Pure Text"
    assert len(cleaner.history) == 1


def test_unicode_normalization_and_accent_stripping():
    # 1. Standard Unicode normalization (NFC)
    assert normalize_unicode_text("Café", form="NFC") == "Café"

    # 2. Stripping accents safely
    assert normalize_unicode_text("Café", strip_accents=True) == "Cafe"
    assert normalize_unicode_text("naïve", strip_accents=True) == "naive"
    assert normalize_unicode_text("résumé", strip_accents=True) == "resume"
    assert normalize_unicode_text("Schrödinger", strip_accents=True) == "Schrodinger"

    # 3. Compatibility decomposition NFKC (e.g. full-width digits '１２３' -> '123')
    assert normalize_unicode_text("１２３", form="NFKC") == "123"

    # 4. Missing value preservation
    assert normalize_unicode_text(None) is None
    assert pd.isna(normalize_unicode_text(np.nan))


def test_normalize_unicode_in_cleaner():
    df = pd.DataFrame(
        {
            "names": ["Renée", "François", "Björn", None],
        }
    )
    cleaner = Cleaner(df)
    cleaner.normalize_unicode("names", strip_accents=True)

    assert cleaner.data["names"].iloc[0] == "Renee"
    assert cleaner.data["names"].iloc[1] == "Francois"
    assert cleaner.data["names"].iloc[2] == "Bjorn"
    assert pd.isna(cleaner.data["names"].iloc[3])


def test_special_character_cleaning():
    raw = "Product #123 (Special Edition!)"
    # Default: keep only alphanumeric and spaces
    assert clean_special_characters(raw) == "Product 123 Special Edition"

    # With allowed characters '-_.'
    raw2 = "user_name-123@domain.com"
    assert clean_special_characters(raw2, allowed_chars="_-.") == "user_name-123domain.com"
    assert clean_special_characters(raw2, allowed_chars="_-.@") == "user_name-123@domain.com"

    # With replacement string
    assert clean_special_characters("item:value", replace_with=" ") == "item value"


def test_clean_special_chars_in_cleaner():
    df = pd.DataFrame(
        {
            "title": ["Special @ Offer!", "50% OFF (Today Only)", None],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_special_chars("title", allowed_chars="%")

    assert cleaner.data["title"].iloc[0] == "Special Offer"
    assert cleaner.data["title"].iloc[1] == "50% OFF Today Only"
    assert pd.isna(cleaner.data["title"].iloc[2])


def test_numeric_extraction():
    # Various noisy strings
    assert extract_numeric_value("120 km/h") == 120.0
    assert extract_numeric_value("Score: 95.5%") == 95.5
    assert extract_numeric_value("-15.2 degrees") == -15.2
    assert extract_numeric_value("$4,500.50") == 4500.5
    assert extract_numeric_value("Age 25 years", as_type="int") == 25
    assert extract_numeric_value(42) == 42.0

    # Missing and non-numeric
    assert pd.isna(extract_numeric_value(None))
    assert pd.isna(extract_numeric_value("No numbers here", errors="coerce"))

    # Errors = raise
    with pytest.raises(ValueError, match="No numeric value found"):
        extract_numeric_value("No numbers here", errors="raise")


def test_extract_numeric_in_cleaner():
    df = pd.DataFrame(
        {
            "speed": ["120 km/h", "85.5 km/h", "N/A", None, "60 km/h"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.extract_numeric("speed", as_type="float", errors="coerce")

    assert cleaner.data["speed"].iloc[0] == 120.0
    assert cleaner.data["speed"].iloc[1] == 85.5
    assert pd.isna(cleaner.data["speed"].iloc[2])
    assert pd.isna(cleaner.data["speed"].iloc[3])
    assert cleaner.data["speed"].iloc[4] == 60.0


def test_currency_cleaning():
    assert clean_currency_value("$1,250.00") == 1250.0
    assert clean_currency_value("€ 120.50") == 120.5
    assert clean_currency_value("£99.99") == 99.99
    assert clean_currency_value("₹ 45,000") == 45000.0
    assert clean_currency_value("USD 1,500") == 1500.0

    # Accounting negative with parentheses
    assert clean_currency_value("($50.00)") == -50.0
    assert clean_currency_value("-$25.50") == -25.5
    assert clean_currency_value("(100.00)") == -100.0

    # Missing and unparseable
    assert pd.isna(clean_currency_value(None))
    assert pd.isna(clean_currency_value("not_a_currency", errors="coerce"))


def test_clean_currency_in_cleaner():
    df = pd.DataFrame(
        {
            "price": ["$1,250.00", "($50.00)", "€ 120.50", None],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_currency("price", as_numeric=True)

    assert cleaner.data["price"].iloc[0] == 1250.0
    assert cleaner.data["price"].iloc[1] == -50.0
    assert cleaner.data["price"].iloc[2] == 120.5
    assert pd.isna(cleaner.data["price"].iloc[3])
    assert cleaner.data["price"].dtype == "float64"


def test_extract_numeric_int_and_which():
    df = pd.DataFrame(
        {
            "items": ["Order #1024 with 5 units", "Order #2048 with 12 units", None],
        }
    )
    cleaner = Cleaner(df)
    # Extract first integer (order id)
    cleaner.extract_numeric("items", as_type="int", which="first")
    assert cleaner.data["items"].iloc[0] == 1024
    assert cleaner.data["items"].iloc[1] == 2048
    assert pd.isna(cleaner.data["items"].iloc[2])
    assert str(cleaner.data["items"].dtype) == "Int64"

    # Extract last integer (unit count)
    cleaner2 = Cleaner(df)
    cleaner2.extract_numeric("items", as_type="int", which="last")
    assert cleaner2.data["items"].iloc[0] == 5
    assert cleaner2.data["items"].iloc[1] == 12
    assert pd.isna(cleaner2.data["items"].iloc[2])


def test_clean_currency_non_numeric_format():
    df = pd.DataFrame(
        {
            "raw_cost": ["$1,000", "($250)", "€50.5"],
        }
    )
    cleaner = Cleaner(df)
    cleaner.clean_currency("raw_cost", as_numeric=False)
    assert cleaner.data["raw_cost"].iloc[0] == "1000.0"
    assert cleaner.data["raw_cost"].iloc[1] == "-250.0"
    assert cleaner.data["raw_cost"].iloc[2] == "50.5"


def test_clean_invisible_all_zero_width_varieties():
    # ZWSP (\u200B), ZWNJ (\u200C), ZWJ (\u200D), Word Joiner (\u2060), LTR mark (\u200E), RTL mark (\u200F)
    tricky = "A\u200Bb\u200Cc\u200Dd\u2060e\u200Ef\u200Fg"
    assert clean_invisible_characters(tricky) == "Abcdefg"

