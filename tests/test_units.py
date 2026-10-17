"""Mathematical unit tests for basic unit conversion in Tersus (V1)."""

import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner
from tersus.units import (
    convert_column_units,
    convert_single_unit,
)


def test_length_conversions():
    # Base: meters
    # 1 km = 1000 m
    assert convert_single_unit(1, from_unit="km", to_unit="m") == 1000.0
    # 100 cm = 1 m
    assert convert_single_unit(100, from_unit="cm", to_unit="m") == 1.0
    # 1000 mm = 1 m
    assert convert_single_unit(1000, from_unit="mm", to_unit="m") == 1.0
    # 1 ft = 12 in = 0.3048 m
    assert convert_single_unit(1, from_unit="ft", to_unit="in") == 12.0
    # 1 yd = 3 ft = 36 in
    assert convert_single_unit(1, from_unit="yd", to_unit="ft") == 3.0
    # 1 mi = 1609.344 m = 1.609344 km
    assert convert_single_unit(1, from_unit="mi", to_unit="km", round_digits=6) == 1.609344


def test_weight_conversions():
    # Base: kilograms
    # 1 kg = 1000 g
    assert convert_single_unit(1, from_unit="kg", to_unit="g") == 1000.0
    # 1 g = 1000 mg
    assert convert_single_unit(1, from_unit="g", to_unit="mg") == 1000.0
    # 1 t = 1000 kg
    assert convert_single_unit(1, from_unit="t", to_unit="kg") == 1000.0
    # 1 lb = 16 oz (exact: 0.45359237 / 0.028349523125)
    assert round(convert_single_unit(1, from_unit="lb", to_unit="oz"), 6) == 16.0
    # 1000 kg to t
    assert convert_single_unit(1000, from_unit="kg", to_unit="t") == 1.0


def test_temperature_conversions():
    # 0 °C = 32 °F = 273.15 K
    assert convert_single_unit(0, from_unit="c", to_unit="f") == 32.0
    assert convert_single_unit(0, from_unit="c", to_unit="k") == 273.15

    # 100 °C = 212 °F = 373.15 K
    assert convert_single_unit(100, from_unit="c", to_unit="f") == 212.0
    assert convert_single_unit(100, from_unit="c", to_unit="k") == 373.15

    # -40 °C = -40 °F
    assert convert_single_unit(-40, from_unit="c", to_unit="f") == -40.0
    assert convert_single_unit(-40, from_unit="f", to_unit="c") == -40.0

    # 77 °F = 25 °C
    assert convert_single_unit(77, from_unit="f", to_unit="c") == 25.0

    # 300 K = 26.85 °C
    assert round(convert_single_unit(300, from_unit="k", to_unit="c"), 2) == 26.85


def test_incompatible_unit_categories_raise_error():
    with pytest.raises(ValueError, match="Cannot convert between incompatible unit categories"):
        convert_single_unit(10, from_unit="km", to_unit="kg")

    with pytest.raises(ValueError, match="Cannot convert between incompatible unit categories"):
        convert_single_unit(10, from_unit="c", to_unit="m")


def test_unsupported_unit_raises_error():
    with pytest.raises(ValueError, match="Unsupported unit"):
        convert_single_unit(10, from_unit="lightyears", to_unit="m")


def test_cleaner_convert_units_in_dataframe():
    df = pd.DataFrame(
        {
            "distance_km": [5.0, 10.0, 42.195, None],
            "temp_f": [32.0, 212.0, 77.0, np.nan],
            "weight_lb": [150.0, 200.0, None, 100.0],
        }
    )
    cleaner = Cleaner(df)

    # 1. Convert km to meters
    cleaner.convert_units("distance_km", from_unit="km", to_unit="m")
    assert cleaner.data["distance_km"].iloc[0] == 5000.0
    assert cleaner.data["distance_km"].iloc[1] == 10000.0
    assert cleaner.data["distance_km"].iloc[2] == 42195.0
    assert pd.isna(cleaner.data["distance_km"].iloc[3])

    # 2. Convert Fahrenheit to Celsius
    cleaner.convert_units("temp_f", from_unit="f", to_unit="c", round_digits=1)
    assert cleaner.data["temp_f"].iloc[0] == 0.0
    assert cleaner.data["temp_f"].iloc[1] == 100.0
    assert cleaner.data["temp_f"].iloc[2] == 25.0
    assert pd.isna(cleaner.data["temp_f"].iloc[3])

    # 3. Convert lbs to kg
    cleaner.convert_units("weight_lb", from_unit="lb", to_unit="kg", round_digits=2)
    assert cleaner.data["weight_lb"].iloc[0] == round(150.0 * 0.45359237, 2)
    assert cleaner.data["weight_lb"].iloc[1] == round(200.0 * 0.45359237, 2)
    assert pd.isna(cleaner.data["weight_lb"].iloc[2])

    assert len(cleaner.history) == 3


def test_convert_units_with_embedded_strings():
    df = pd.DataFrame({"distance": ["5 km", "10 km", "42.195 km", None]})
    cleaner = Cleaner(df)
    cleaner.convert_units("distance", from_unit="km", to_unit="m")

    assert cleaner.data["distance"].iloc[0] == 5000.0
    assert cleaner.data["distance"].iloc[1] == 10000.0
    assert cleaner.data["distance"].iloc[2] == 42195.0
    assert pd.isna(cleaner.data["distance"].iloc[3])


def test_convert_units_error_handling():
    df = pd.DataFrame({"distance": ["5 km", "invalid_val", None]})

    # errors='raise' (default)
    cleaner = Cleaner(df)
    with pytest.raises(ValueError, match="Unit conversion failed"):
        cleaner.convert_units("distance", from_unit="km", to_unit="m", errors="raise")

    # errors='coerce'
    cleaner = Cleaner(df)
    cleaner.convert_units("distance", from_unit="km", to_unit="m", errors="coerce")
    assert cleaner.data["distance"].iloc[0] == 5000.0
    assert pd.isna(cleaner.data["distance"].iloc[1])
    assert pd.isna(cleaner.data["distance"].iloc[2])

    # errors='ignore'
    cleaner = Cleaner(df)
    cleaner.convert_units("distance", from_unit="km", to_unit="m", errors="ignore")
    assert cleaner.data["distance"].iloc[0] == 5000.0
    assert cleaner.data["distance"].iloc[1] == "invalid_val"

