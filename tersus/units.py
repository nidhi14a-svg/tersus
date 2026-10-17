"""Basic unit conversion module for Tersus.

V1 Scope:
---------
Deliberately focused on three fundamental unit categories:
1. Length: mm, cm, m, km, in, ft, yd, mi
2. Weight/Mass: mg, g, kg, oz, lb, t (metric ton)
3. Temperature: C (Celsius), F (Fahrenheit), K (Kelvin)

Mathematical Formulas:
----------------------
1. Length (relative to standard base unit: meter):
   - mm = 0.001 m
   - cm = 0.01 m
   - m = 1.0 m
   - km = 1000.0 m
   - in = 0.0254 m
   - ft = 0.3048 m
   - yd = 0.9144 m
   - mi = 1609.344 m

2. Weight (relative to standard base unit: kilogram):
   - mg = 1e-6 kg
   - g = 0.001 kg
   - kg = 1.0 kg
   - oz = 0.028349523125 kg
   - lb = 0.45359237 kg
   - t (tonne) = 1000.0 kg

3. Temperature:
   - To Celsius:
       From C: C
       From F: (F - 32) * 5/9
       From K: K - 273.15
   - From Celsius:
       To C: C
       To F: (C * 9/5) + 32
       To K: C + 273.15

Design Philosophy:
------------------
All unit conversions must be explicit. No automatic or destructive unit
assumptions are made without user specification of source and target units.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from tersus.missing import CleaningRecord

# Length conversion factors to meters
LENGTH_TO_METERS: Dict[str, float] = {
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "yd": 0.9144,
    "yard": 0.9144,
    "yards": 0.9144,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
}

# Weight conversion factors to kilograms
WEIGHT_TO_KG: Dict[str, float] = {
    "mg": 0.000001,
    "milligram": 0.000001,
    "milligrams": 0.000001,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "kgs": 1.0,
    "oz": 0.028349523125,
    "ounce": 0.028349523125,
    "ounces": 0.028349523125,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "t": 1000.0,
    "tonne": 1000.0,
    "tonnes": 1000.0,
    "metric_ton": 1000.0,
}

# Temperature unit aliases
TEMP_UNITS = {
    "c": "c",
    "celsius": "c",
    "°c": "c",
    "f": "f",
    "fahrenheit": "f",
    "°f": "f",
    "k": "k",
    "kelvin": "k",
    "°k": "k",
}


def _get_unit_category(unit: str) -> Tuple[str, str]:
    """Identify category ('length', 'weight', or 'temperature') and normalized unit name."""
    norm = unit.strip().lower()
    if norm in LENGTH_TO_METERS:
        return "length", norm
    elif norm in WEIGHT_TO_KG:
        return "weight", norm
    elif norm in TEMP_UNITS:
        return "temperature", TEMP_UNITS[norm]
    else:
        raise ValueError(
            f"Unsupported unit '{unit}'. Supported units for V1: "
            f"length (mm, cm, m, km, in, ft, yd, mi), "
            f"weight (mg, g, kg, oz, lb, t), "
            f"temperature (c, f, k)."
        )


def convert_single_unit(
    val: Any,
    from_unit: str,
    to_unit: str,
    round_digits: Optional[int] = None,
) -> Any:
    """Convert a single numerical value from one unit to another.

    Args:
        val: Numerical or string value to convert.
        from_unit: Source unit (e.g. 'km', 'lb', 'c').
        to_unit: Target unit (e.g. 'm', 'kg', 'f').
        round_digits: Optional decimal places to round the output.

    Returns:
        Converted float value, or pd.NA if input is missing.
    """
    if val is None or pd.isna(val):
        return np.nan

    cat_from, norm_from = _get_unit_category(from_unit)
    cat_to, norm_to = _get_unit_category(to_unit)

    if cat_from != cat_to:
        raise ValueError(
            f"Cannot convert between incompatible unit categories '{cat_from}' "
            f"('{from_unit}') and '{cat_to}' ('{to_unit}')."
        )

    # Extract numerical magnitude if string
    if isinstance(val, (int, float)):
        num = float(val)
    else:
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "null", "<na>"):
            return np.nan
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            raise ValueError(f"No numeric value found in '{val}' for unit conversion.")
        num = float(m.group(0))

    if cat_from == "length":
        meters = num * LENGTH_TO_METERS[norm_from]
        result = meters / LENGTH_TO_METERS[norm_to]

    elif cat_from == "weight":
        kgs = num * WEIGHT_TO_KG[norm_from]
        result = kgs / WEIGHT_TO_KG[norm_to]

    elif cat_from == "temperature":
        # First convert to Celsius
        if norm_from == "c":
            celsius = num
        elif norm_from == "f":
            celsius = (num - 32.0) * (5.0 / 9.0)
        else:  # 'k'
            celsius = num - 273.15

        # Convert from Celsius to target
        if norm_to == "c":
            result = celsius
        elif norm_to == "f":
            result = (celsius * (9.0 / 5.0)) + 32.0
        else:  # 'k'
            result = celsius + 273.15
    else:
        raise ValueError(f"Unknown category '{cat_from}'.")

    if round_digits is not None:
        result = round(result, round_digits)
    else:
        result = round(result, 10)

    return result


def convert_column_units(
    df: pd.DataFrame,
    column: str,
    from_unit: str,
    to_unit: str,
    round_digits: Optional[int] = None,
    errors: str = "raise",
) -> tuple[pd.DataFrame, CleaningRecord]:
    """Convert all values in a column from one unit to another.

    Args:
        df: Pandas DataFrame.
        column: Column name to convert.
        from_unit: Source unit.
        to_unit: Target unit.
        round_digits: Optional decimal places to round results.
        errors: 'raise' (default), 'coerce' (set conversion failures to np.nan), or 'ignore'.

    Returns:
        Tuple of (cleaned_df, cleaning_record).
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataset. Available columns: {list(df.columns)}"
        )
    if errors not in ("raise", "coerce", "ignore"):
        raise ValueError(f"errors must be 'raise', 'coerce', or 'ignore', got '{errors}'.")

    cat_from, norm_from = _get_unit_category(from_unit)
    cat_to, norm_to = _get_unit_category(to_unit)
    if cat_from != cat_to:
        raise ValueError(
            f"Cannot convert between incompatible unit categories '{cat_from}' "
            f"('{from_unit}') and '{cat_to}' ('{to_unit}')."
        )

    series = df[column]
    converted_values: List[Any] = []
    affected_count = 0

    for idx, val in series.items():
        if val is None or pd.isna(val):
            converted_values.append(np.nan)
            continue

        try:
            res = convert_single_unit(
                val, from_unit=from_unit, to_unit=to_unit, round_digits=round_digits
            )
            converted_values.append(res)
            affected_count += 1
        except Exception as exc:
            if errors == "raise":
                raise ValueError(
                    f"Unit conversion failed at row {idx} in column '{column}': {exc}"
                ) from exc
            elif errors == "coerce":
                converted_values.append(np.nan)
            else:  # errors == "ignore"
                converted_values.append(val)

    df_clean = df.copy()
    s = pd.Series(converted_values, index=series.index)
    if errors != "ignore":
        s = s.astype("float64")
    else:
        try:
            s = s.astype("float64")
        except (ValueError, TypeError):
            s = s.astype("object")
    df_clean[column] = s

    record = CleaningRecord(
        operation="convert_units",
        strategy="unit_transformation",
        columns=[column],
        affected_count=affected_count,
        details={
            "from_unit": from_unit,
            "to_unit": to_unit,
            "category": cat_from,
            "round_digits": round_digits,
        },
        message=(
            f"Converted {affected_count} value(s) in column '{column}' "
            f"from '{from_unit}' to '{to_unit}'."
        ),
    )
    return df_clean, record
