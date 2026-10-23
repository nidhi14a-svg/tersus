"""Mathematical and integration unit tests for Tersus outlier detection.

Formulas Documented and Verified:
---------------------------------
1. Interquartile Range (IQR) Method:
   - Q1 = 25th percentile
   - Q3 = 75th percentile
   - IQR = Q3 - Q1
   - Lower Bound = Q1 - (threshold * IQR)
   - Upper Bound = Q3 + (threshold * IQR)
   - Outlier condition: x < Lower Bound OR x > Upper Bound

2. Z-Score (Standard Score) Method:
   - mean (μ) = (1 / n) * Σ(x_i)
   - std (σ) = sqrt((1 / (n - ddof)) * Σ(x_i - μ)^2)
   - z = (x - μ) / σ
   - Lower Bound = μ - (threshold * σ)
   - Upper Bound = μ + (threshold * σ)
   - Outlier condition: |z| > threshold (x < Lower Bound OR x > Upper Bound)
"""

import numpy as np
import pandas as pd
import pytest

from tersus import Cleaner, CleaningRecord, OutlierResult
from tersus.outliers import (
    cap_outliers,
    detect_outliers,
    detect_outliers_iqr,
    detect_outliers_zscore,
    drop_outliers,
)


def test_manual_calculation_iqr_symmetric():
    """Manual Calculation Dataset 1 (IQR Method):

    Dataset: [-50, 10, 20, 30, 40, 50, 150] (7 sorted values)

    Step-by-step Manual Derivation:
    - n = 7
    - Q1 (25th percentile):
        index = 0.25 * (7 - 1) = 1.5
        Q1 = val[1] + 0.5 * (val[2] - val[1]) = 10 + 0.5 * (20 - 10) = 15.0
    - Q3 (75th percentile):
        index = 0.75 * (7 - 1) = 4.5
        Q3 = val[4] + 0.5 * (val[5] - val[4]) = 40 + 0.5 * (50 - 40) = 45.0
    - IQR = Q3 - Q1 = 45.0 - 15.0 = 30.0
    - Multiplier threshold: k = 1.5
    - Lower Bound = 15.0 - (1.5 * 30.0) = 15.0 - 45.0 = -30.0
    - Upper Bound = 45.0 + (1.5 * 30.0) = 45.0 + 45.0 = 90.0

    Outlier Evaluation:
    - Row 0: -50 < -30.0  -> OUTLIER
    - Row 1:  10 >= -30.0 and <= 90.0 -> In bounds
    - Row 2:  20 >= -30.0 and <= 90.0 -> In bounds
    - Row 3:  30 >= -30.0 and <= 90.0 -> In bounds
    - Row 4:  40 >= -30.0 and <= 90.0 -> In bounds
    - Row 5:  50 >= -30.0 and <= 90.0 -> In bounds
    - Row 6: 150 > 90.0   -> OUTLIER

    Expected Result:
    - outlier_count = 2
    - outlier_indices = [0, 6]
    - outlier_values = [-50, 150]
    """
    df = pd.DataFrame(
        {
            "id": ["p0", "p1", "p2", "p3", "p4", "p5", "p6"],
            "measurement": [-50, 10, 20, 30, 40, 50, 150],
        }
    )
    cleaner = Cleaner(df)
    result = cleaner.check_outliers_iqr("measurement", threshold=1.5)

    assert isinstance(result, OutlierResult)
    assert result.method == "iqr"
    assert result.threshold == 1.5
    assert result.statistics["q1"] == 15.0
    assert result.statistics["q3"] == 45.0
    assert result.statistics["iqr"] == 30.0
    assert result.lower_bound == -30.0
    assert result.upper_bound == 90.0

    assert result.has_outliers is True
    assert result.outlier_count == 2
    assert result.outlier_indices == [0, 6]
    assert result.outlier_values == [-50, 150]

    # Verify outlier_rows slice
    assert isinstance(result.outlier_rows, pd.DataFrame)
    assert len(result.outlier_rows) == 2
    assert result.outlier_rows["id"].tolist() == ["p0", "p6"]

    # CRITICAL: Outlier detection must NOT modify the working DataFrame!
    assert len(cleaner.data) == 7
    assert cleaner.data["measurement"].iloc[0] == -50
    assert cleaner.data["measurement"].iloc[6] == 150


def test_manual_calculation_iqr_single_outlier_and_custom_threshold():
    """Manual Calculation Dataset 2 (IQR Method with 8 items and custom threshold):

    Dataset: [10, 12, 14, 16, 18, 20, 22, 100] (8 items)

    Step-by-step Manual Derivation:
    - n = 8
    - Q1: index = 0.25 * 7 = 1.75 -> 12 + 0.75 * (14 - 12) = 13.5
    - Q3: index = 0.75 * 7 = 5.25 -> 20 + 0.25 * (22 - 20) = 20.5
    - IQR = 20.5 - 13.5 = 7.0

    With threshold k = 1.5:
    - Lower Bound = 13.5 - 1.5 * 7.0 = 13.5 - 10.5 = 3.0
    - Upper Bound = 20.5 + 1.5 * 7.0 = 20.5 + 10.5 = 31.0
    - Outlier: row 7 (value 100 > 31.0) -> 1 outlier

    With custom extreme threshold k = 12.0:
    - Upper Bound = 20.5 + 12.0 * 7.0 = 20.5 + 84.0 = 104.5
    - 100 <= 104.5 -> 0 outliers!
    """
    df = pd.DataFrame({"score": [10, 12, 14, 16, 18, 20, 22, 100]})
    cleaner = Cleaner(df)

    res_k15 = cleaner.check_outliers("score", method="iqr", threshold=1.5)
    assert res_k15.statistics["q1"] == 13.5
    assert res_k15.statistics["q3"] == 20.5
    assert res_k15.statistics["iqr"] == 7.0
    assert res_k15.lower_bound == 3.0
    assert res_k15.upper_bound == 31.0
    assert res_k15.outlier_count == 1
    assert res_k15.outlier_indices == [7]
    assert res_k15.outlier_values == [100]

    # Extreme threshold k = 12.0: no outliers
    res_k12 = cleaner.check_outliers("score", method="iqr", threshold=12.0)
    assert res_k12.outlier_count == 0
    assert res_k12.has_outliers is False


def test_manual_calculation_zscore_exact_integers():
    """Manual Calculation Dataset 3 (Z-Score Method):

    Dataset: 9 values: [10, 10, 10, 10, 10, 10, 10, 10, 100]

    Step-by-step Manual Derivation:
    - n = 9
    - Sum = (8 * 10) + 100 = 80 + 100 = 180
    - Mean (μ) = 180 / 9 = 20.0 (Exact integer)
    - Sample Variance (ddof=1):
        Sum of squared differences:
        8 * (10 - 20)^2 + (100 - 20)^2 = 8 * (-10)^2 + (80)^2
                                       = 8 * 100 + 6400 = 800 + 6400 = 7200
        Variance = 7200 / (9 - 1) = 7200 / 8 = 900.0 (Exact integer)
    - Sample Standard Deviation (σ) = sqrt(900.0) = 30.0 (Exact integer)

    Z-Scores:
    - For 10:  z = (10 - 20) / 30 = -10 / 30 = -1/3 ≈ -0.3333
    - For 100: z = (100 - 20) / 30 = 80 / 30 = 8/3 ≈ 2.6667

    Case A: Threshold c = 2.0
    - Lower Bound = 20 - (2.0 * 30) = 20 - 60 = -40.0
    - Upper Bound = 20 + (2.0 * 30) = 20 + 60 = 80.0
    - Evaluation: 100 > 80.0 (z = 2.6667 > 2.0) -> OUTLIER at row 8
    - Count = 1

    Case B: Standard Default Threshold c = 3.0
    - Lower Bound = 20 - (3.0 * 30) = 20 - 90 = -70.0
    - Upper Bound = 20 + (3.0 * 30) = 20 + 90 = 110.0
    - Evaluation: 100 <= 110.0 (z = 2.6667 <= 3.0) -> NOT an outlier!
    - Count = 0
    """
    df = pd.DataFrame(
        {
            "sensor": [10, 10, 10, 10, 10, 10, 10, 10, 100],
        }
    )
    cleaner = Cleaner(df)

    # Case A: threshold = 2.0
    res_c2 = cleaner.check_outliers_zscore("sensor", threshold=2.0, ddof=1)
    assert res_c2.method == "zscore"
    assert res_c2.threshold == 2.0
    assert res_c2.statistics["mean"] == 20.0
    assert res_c2.statistics["std"] == 30.0
    assert res_c2.lower_bound == -40.0
    assert res_c2.upper_bound == 80.0
    assert res_c2.has_outliers is True
    assert res_c2.outlier_count == 1
    assert res_c2.outlier_indices == [8]
    assert res_c2.outlier_values == [100]

    # Case B: default threshold = 3.0
    res_c3 = cleaner.check_outliers_zscore("sensor", threshold=3.0, ddof=1)
    assert res_c3.threshold == 3.0
    assert res_c3.lower_bound == -70.0
    assert res_c3.upper_bound == 110.0
    assert res_c3.has_outliers is False
    assert res_c3.outlier_count == 0
    assert res_c3.outlier_indices == []


def test_outlier_is_not_an_error_concept():
    """An outlier is a statistical anomaly, not an error.

    Ensure detection does NOT modify the dataset, and distinguish
    between validation errors and outlier reports.
    """
    df = pd.DataFrame(
        {
            "employee": ["E1", "E2", "E3", "E4", "CEO"],
            "salary": [50000, 52000, 48000, 51000, 1000000],
        }
    )
    cleaner = Cleaner(df)

    # 1. Detection is read-only
    res = cleaner.check_outliers("salary", method="iqr")
    assert res.has_outliers is True
    assert res.outlier_indices == [4]
    assert res.outlier_values == [1000000]

    # 2. DataFrame remains completely unaltered
    assert len(cleaner.data) == 5
    assert cleaner.data["salary"].iloc[4] == 1000000
    assert len(cleaner.history) == 0  # No cleaning mutation recorded


def test_user_action_drop_outliers():
    """User explicitly chooses to drop outliers after reviewing."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "val": [-50, 10, 20, 30, 40, 50, 150],
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_outliers_iqr("val")

    assert res.outlier_count == 2
    assert res.outlier_indices == [0, 6]

    # Explicit user action: drop outliers
    cleaner.drop_outliers(res)
    assert len(cleaner.data) == 5
    assert cleaner.data["val"].tolist() == [10, 20, 30, 40, 50]
    assert len(cleaner.history) == 1
    assert cleaner.last_operation.operation == "drop_outliers"
    assert cleaner.last_operation.affected_count == 2


def test_user_action_cap_outliers():
    """User explicitly chooses to cap (winsorize/clamp) outliers to statistical bounds."""
    df = pd.DataFrame(
        {
            "val": [-50, 10, 20, 30, 40, 50, 150],
        }
    )
    cleaner = Cleaner(df)
    res = cleaner.check_outliers_iqr("val")  # bounds: [-30.0, 90.0]

    cleaner.cap_outliers(res)
    assert len(cleaner.data) == 7
    # -50 should be clamped to -30.0; 150 should be clamped to 90.0
    expected = [-30.0, 10.0, 20.0, 30.0, 40.0, 50.0, 90.0]
    assert cleaner.data["val"].tolist() == expected
    assert cleaner.last_operation.operation == "cap_outliers"
    assert cleaner.last_operation.affected_count == 2


def test_outlier_detection_with_missing_values():
    """Missing values (NaN/None) are ignored in stats and NOT flagged as outliers."""
    df = pd.DataFrame(
        {
            "val": [10, np.nan, 20, None, 30, 40, 50, 150],
        }
    )
    cleaner = Cleaner(df)

    res_iqr = cleaner.check_outliers_iqr("val")
    # NaNs should not be in outlier_indices
    assert 1 not in res_iqr.outlier_indices
    assert 3 not in res_iqr.outlier_indices
    assert res_iqr.outlier_indices == [7]
    assert res_iqr.outlier_values == [150]


def test_zero_variance_dataset():
    """Datasets with 0 variance (all values identical) have no outliers."""
    df = pd.DataFrame({"constant": [5.0, 5.0, 5.0, 5.0, 5.0]})
    cleaner = Cleaner(df)

    res_iqr = cleaner.check_outliers_iqr("constant")
    assert res_iqr.outlier_count == 0
    assert res_iqr.has_outliers is False
    assert res_iqr.statistics["iqr"] == 0.0

    res_z = cleaner.check_outliers_zscore("constant")
    assert res_z.outlier_count == 0
    assert res_z.has_outliers is False
    assert res_z.statistics["std"] == 0.0


def test_error_on_non_numeric_column():
    """Attempting outlier detection on non-numeric columns raises TypeError."""
    df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
    cleaner = Cleaner(df)

    with pytest.raises(TypeError, match="is not numeric"):
        cleaner.check_outliers("name", method="iqr")

    with pytest.raises(TypeError, match="is not numeric"):
        cleaner.check_outliers("name", method="zscore")


def test_error_on_mixed_type_object_column():
    """A column holding genuinely numeric values but stored as object dtype
    (e.g. mixed int/float/str-number entries) is still rejected, since
    pandas reports it as non-numeric dtype. Outlier detection requires an
    explicit numeric dtype -- users must convert_type() first rather than
    relying on implicit coercion that could silently misinterpret values.
    """
    df = pd.DataFrame({"val": [1, 2.5, "3", 4]}, dtype=object)
    assert df["val"].dtype == object
    cleaner = Cleaner(df)

    with pytest.raises(TypeError, match="is not numeric"):
        cleaner.check_outliers("val", method="iqr")


def test_error_on_missing_column():
    """Missing column raises KeyError."""
    df = pd.DataFrame({"val": [1, 2, 3]})
    cleaner = Cleaner(df)

    with pytest.raises(KeyError, match="not found in dataset"):
        cleaner.check_outliers("non_existent")


def test_error_on_invalid_threshold_or_method():
    """Invalid thresholds or unknown methods raise ValueError."""
    df = pd.DataFrame({"val": [1, 2, 3]})
    cleaner = Cleaner(df)

    with pytest.raises(ValueError, match="threshold must be positive"):
        cleaner.check_outliers_iqr("val", threshold=0)

    with pytest.raises(ValueError, match="threshold must be positive"):
        cleaner.check_outliers_zscore("val", threshold=-1.5)

    with pytest.raises(ValueError, match="Unsupported outlier detection method"):
        cleaner.check_outliers("val", method="unknown_method")


def test_empty_dataframe_safety():
    """Empty DataFrames are handled safely."""
    empty_df = pd.DataFrame({"val": pd.Series([], dtype="float64")})
    cleaner = Cleaner(empty_df)

    res_iqr = cleaner.check_outliers_iqr("val")
    assert res_iqr.outlier_count == 0
    assert res_iqr.has_outliers is False
    assert res_iqr.lower_bound is None

    res_z = cleaner.check_outliers_zscore("val")
    assert res_z.outlier_count == 0
    assert res_z.has_outliers is False
    assert res_z.lower_bound is None


def test_outlier_result_formatting_and_dict():
    """Test OutlierResult representations."""
    df = pd.DataFrame({"val": [-50, 10, 20, 30, 40, 50, 150]})
    cleaner = Cleaner(df)
    res = cleaner.check_outliers_iqr("val")

    d = res.to_dict()
    assert d["column"] == "val"
    assert d["method"] == "iqr"
    assert d["threshold"] == 1.5
    assert d["outlier_count"] == 2
    assert d["outlier_indices"] == [0, 6]

    s = str(res)
    assert "Outlier Detection Report on 'val'" in s
    assert "2 potential outlier(s) detected" in s
    assert "Row 0: -50" in s
    assert "Row 6: 150" in s
    assert len(res) == 2
