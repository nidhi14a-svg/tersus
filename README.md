# Tersus

> **A higher-level data-quality and data-cleaning workflow library built on top of Pandas.**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Built On](https://img.shields.io/badge/built%20on-Pandas%20%26%20NumPy-150458.svg)](https://pandas.pydata.org/)
[![Tests](https://img.shields.io/badge/tests-239%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 📌 What is Tersus?

**Tersus is a higher-level data-quality and cleaning workflow built on top of Pandas.**

> [!IMPORTANT]
> **Tersus does NOT replace Pandas.**  
> Pandas is a powerful general-purpose data manipulation library. However, real-world data cleaning often requires repetitive boilerplate code: diagnosing null values, identifying duplicates, detecting statistical outliers, normalizing irregular strings, remembering which columns were cleaned, and validating whether quality targets were met.
>
> Tersus builds on top of Pandas to give students, data analysts, and engineers a **structured, auditable, and non-destructive quality lifecycle**:
>
> **`load → profile → suggest → clean → validate → report → save`**

---

## 🚀 The 7-Step Cleaning Lifecycle

Instead of running ad-hoc transformations that risk silently altering or deleting valid data, Tersus separates **detection**, **recommendation**, **controlled cleaning**, and **verification**:

```text
  ┌──────────┐     ┌───────────┐     ┌───────────┐     ┌─────────┐
  │ 1. Load  │ ──> │ 2. Profile│ ──> │ 3. Suggest│ ──> │ 4. Clean│
  └──────────┘     └───────────┘     └───────────┘     └─────────┘
                                                            │
  ┌──────────┐     ┌───────────┐     ┌───────────┐          │
  │ 7. Save  │ <── │ 6. Report │ <── │ 5.Validate│ <────────┘
  └──────────┘     └───────────┘     └───────────┘
```

1. **Load**: Initialize the dataset from a CSV path or an existing Pandas `DataFrame`.
2. **Profile**: Inspect dataset dimensions, data types, missing rates, duplicate counts, and summary statistics without mutating the data.
3. **Suggest**: Analyze quality metrics to generate prioritised, beginner-friendly cleaning suggestions.
4. **Clean**: Execute explicitly chosen cleaning operations in a predictable, reproducible sequence with full history tracking.
5. **Validate**: Automatically check whether cleaning goals have been achieved, distinguishing resolved issues from those needing manual review.
6. **Report**: Summarize the complete cleaning session into human-readable console output and structured JSON-serializable records.
7. **Save**: Export the cleaned data to CSV with full confidence.

---

## 📦 Installation

Tersus requires **Python 3.9 or higher** and is built on standard `pandas` and `numpy`.

### From Source (Editable Mode)

```bash
# Clone the repository
git clone https://github.com/your-username/tersus.git
cd tersus

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# or: .\.venv\Scripts\activate  # On Windows

# Install Tersus in editable mode
pip install -e .

# Install development dependencies (pytest)
pip install -e ".[dev]"
```

---

## 💡 Complete End-to-End Example

Here is a complete, working example demonstrating all 7 steps of the lifecycle:

```python
import pandas as pd
from tersus import Cleaner

# 1. LOAD: Read CSV or wrap an existing Pandas DataFrame
cleaner = Cleaner("example/students.csv")
print(f"Loaded dataset: {cleaner}")

# 2. PROFILE: Non-destructive quality inspection
profile = cleaner.profile()
print(f"Missing values: {sum(profile.missing_counts.values())}")
print(f"Duplicate rows: {profile.duplicate_rows}")

# 3. SUGGEST: Get actionable, prioritized cleaning advice
suggestions = cleaner.suggest()
for s in suggestions[:3]:
    print(f"[{s.severity.upper()}] {s.issue} in '{s.column}': {s.suggestion}")

# 4. CLEAN: Execute controlled transformations (orchestrated or individual)
cleaner.clean(
    column_names=True,      # Standardize column headers to snake_case
    missing="median",        # Impute numeric missing values with median
    duplicates=True,        # Remove duplicate records
    text=True,              # Strip whitespace and invisible characters
)

# 5. VALIDATE: Determine whether quality goals were satisfied
validation = cleaner.validate(goals=["missing", "duplicates"])
print(f"Goals satisfied: {validation.goals_satisfied}")
print(f"Issues resolved: {len(validation.resolved_issues)}")
print(f"Issues remaining: {len(validation.remaining_issues)}")

# 6. REPORT: Display a comprehensive, beginner-friendly summary
cleaner.report(print_report=True)

# 7. SAVE: Export the cleaned DataFrame to CSV
cleaner.save("example/output.csv")
print("Cleaned data successfully saved to example/output.csv!")
```

---

## ✨ Features Overview

| Feature Category | Description | Primary Methods |
| :--- | :--- | :--- |
| **Data Profiling** | Inspect row/column shapes, data types, missingness, and distributions | `cleaner.profile()` |
| **Suggestion Engine** | Generate rule-based cleaning suggestions with severity levels | `cleaner.suggest()` |
| **Missing-Value Handling** | Impute (mean, median, mode, constant) or drop missing values | `cleaner.fill_missing()`, `cleaner.drop_missing()` |
| **Duplicate Detection** | Identify and drop duplicate rows with flexible subset matching | `cleaner.duplicates()`, `cleaner.drop_duplicates()` |
| **Column Normalization** | Clean headers (snake_case, strip whitespace, remove special symbols) | `cleaner.clean_column_names()` |
| **Text Normalization** | Strip whitespace, remove invisible Unicode characters, normalize text | `cleaner.clean_text()`, `cleaner.clean_invisible_chars()` |
| **Controlled Types** | Safe data-type conversion (int, float, string, boolean, datetime) | `cleaner.convert_type()`, `cleaner.convert_types()` |
| **Value Validation** | Detect out-of-range values or violations of custom rules | `cleaner.check_range()`, `cleaner.check_values()`, `cleaner.check_rule()` |
| **Date Processing** | Detect datetime columns, validate dates, standardize output formats | `cleaner.detect_datetime_columns()`, `cleaner.standardize_dates()` |
| **Outlier Detection** | Mathematical outlier detection via IQR or Z-score without silent drops | `cleaner.detect_outliers()`, `cleaner.cap_outliers()` |
| **Domain Validation** | Validate email format syntax and normalize Indian phone numbers (+91) | `cleaner.check_emails()`, `cleaner.check_phones()` |
| **Category Normalization** | Identify spelling/case variations and apply explicit category mappings | `cleaner.find_category_variations()`, `cleaner.map_categories()` |
| **Logical Consistency** | Check cross-column rules (start_date <= end_date, min <= max) | `cleaner.check_date_order()`, `cleaner.check_value_order()` |
| **Physical Units** | Explicit conversions for length, weight, and temperature | `cleaner.convert_units()` |
| **Orchestration** | Declarative single-call cleaning pipeline with audit trail | `cleaner.clean()` |
| **Goal Verification** | Evaluate before vs after state against targeted goals | `cleaner.validate()` |
| **Session Reporting** | Human-readable console audit and structured dictionary reports | `cleaner.report()` |

---

## 🛠️ API Examples

### 1. Data Profiling & Suggestions

```python
cleaner = Cleaner(df)

# Profile returns summary statistics and quality metrics
profile = cleaner.profile()
print(profile.missing_counts)
print(profile.numeric_stats)

# Suggest evaluates the profile and returns actionable recommendations
suggestions = cleaner.suggest()
for item in suggestions:
    print(item.issue, item.column, item.suggestion, item.safe_to_automate)
```

### 2. Missing Values & Duplicate Handling

```python
# Impute numeric missing values using median
cleaner.fill_missing("age", strategy="median")

# Fill categorical missing values with a constant
cleaner.fill_missing("city", strategy="constant", fill_value="Unknown")

# Remove duplicate rows based on specific columns
cleaner.drop_duplicates(subset=["student_id"])
```

### 3. Column & Text Cleaning

```python
# Standardize column headers: " Student Name (%) " -> "student_name"
cleaner.clean_column_names(case="lower", replace_spaces="_", remove_special=True)

# Clean whitespace, tabs, and invisible zero-width characters in text
cleaner.clean_text(columns=["name", "city"])
cleaner.clean_invisible_chars("name")
```

### 4. Value Validation & Clamping

```python
# Check for exam marks outside [0, 100]
result = cleaner.check_range("marks", min_value=0, max_value=100)
if not result.is_valid:
    print(f"Found {result.invalid_count} invalid mark(s): {result.invalid_values}")
    # Replace invalid values or clamp them
    cleaner.clamp_range("marks", min_value=0, max_value=100)
```

### 5. Date Validation & Standardization

```python
# Check for invalid dates without guessing ambiguous formats
date_check = cleaner.check_dates("joining_date", dayfirst=True)
print(f"Invalid dates found: {date_check.invalid_count}")

# Standardize date strings to YYYY-MM-DD
cleaner.standardize_dates("joining_date", output_format="%Y-%m-%d", dayfirst=True)
```

### 6. Mathematical Outlier Detection

An outlier is not automatically an error. Tersus separates detection from modification:

```python
# IQR method (1.5 * IQR)
outliers_iqr = cleaner.detect_outliers("salary", method="iqr", factor=1.5)
print(f"Detected {outliers_iqr.outlier_count} outliers between [{outliers_iqr.lower_bound}, {outliers_iqr.upper_bound}]")

# Z-score method (threshold = 3.0 standard deviations)
outliers_z = cleaner.detect_outliers("salary", method="zscore", threshold=3.0)

# Cap outliers to bounds rather than silently deleting them
cleaner.cap_outliers("salary", method="iqr")
```

### 7. Domain Validation: Emails & Phone Numbers

```python
# Validate email formats
email_check = cleaner.check_emails("email")
print(f"Invalid email addresses: {email_check.invalid_count}")

# Validate and format Indian phone numbers to E.164 (+919876543210)
cleaner.clean_phone("phone_number", style="e164", country="IN")
```

### 8. Physical Units Conversion

```python
# Convert height from inches to centimeters
cleaner.convert_units("height", from_unit="in", to_unit="cm")

# Convert temperature from Fahrenheit to Celsius
cleaner.convert_units("temp", from_unit="degF", to_unit="degC")
```

### 9. Logical Consistency Checks

```python
# Verify that project start date is on or before completion date
result = cleaner.check_date_order("start_date", "end_date", allow_equal=True)
if not result.is_consistent:
    print(f"Inconsistent date order in {result.inconsistent_count} row(s)")
```

### 10. Declarative Pipeline & Session Reporting

```python
# Execute multiple operations in a single, predictable call
cleaner.clean(
    column_names=True,
    missing="median",
    duplicates=True,
    text=True,
)

# Inspect the structured session report
report = cleaner.report()
print(report)  # Formatted human-readable output
data_dict = report.to_dict()  # Structured dictionary for logging or APIs
```

---

## 🏗️ Architecture Explanation

Tersus is engineered with a strict **functional core and stateful orchestrator pattern**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                       Tersus Architecture Overview                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Stateful Orchestrator: Cleaner]                                       │
│    • Holds self._initial_df (baseline snapshot)                         │
│    • Holds self._df (current working Pandas DataFrame)                  │
│    • Holds self._history (immutable audit log of CleaningRecords)       │
│                                                                         │
│  [Stateless Functional Modules]                                         │
│    • profiling.py     ─ Non-destructive dataset metrics                 │
│    • suggestions.py   ─ Heuristic cleaning recommendations              │
│    • missing.py       ─ Imputation & null removal algorithms            │
│    • duplicates.py    ─ Deduplication logic                             │
│    • columns.py       ─ Name cleaning & sanitization                    │
│    • text.py          ─ Unicode, regex & whitespace cleaning            │
│    • types.py         ─ Controlled data-type conversions                │
│    • validation.py    ─ Rule & range checking                           │
│    • dates.py         ─ Date parsing & standardization                  │
│    • outliers.py      ─ Mathematical outlier detection                  │
│    • emails.py        ─ RFC-compliant format validation                 │
│    • phones.py        ─ Indian phone normalization                      │
│    • categories.py    ─ Variation detection & mapping                   │
│    • consistency.py   ─ Cross-column logical consistency                │
│    • units.py         ─ Exact physical unit conversions                 │
│    • orchestration.py ─ Validated multi-step pipeline runner            │
│    • verification.py  ─ Post-cleaning goal verification                 │
│    • reporting.py     ─ Human-readable & structured session summaries   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Non-Destructive Baseline**: Initial data is captured on load (`cleaner.initial_data`) so that post-cleaning validation (`cleaner.validate()`) can accurately measure how many issues were resolved.
2. **Immutable Audit History**: Every cleaning operation records a `CleaningRecord` containing the operation name, strategy, affected columns, count of modified rows/values, and an audit message.
3. **No Silent Destruction**: Outliers, ambiguous dates, or invalid values are never deleted automatically. The user is provided detection indices and explicit resolution methods (`replace_invalid`, `clamp_range`, `cap_outliers`, `drop_invalid`).
4. **Clean Decoupling**: Every functional module can be imported and used independently with raw Pandas DataFrames without needing the `Cleaner` wrapper.

---

## 📋 The Cleaning Report Output

When you call `cleaner.report()` or `print(report)`, Tersus outputs a beginner-friendly summary structured into 6 clear sections:

```text
================================================================================
                    TERSUS DATA CLEANING SESSION REPORT                         
================================================================================

1. DATASET OVERVIEW
--------------------------------------------------------------------------------
  - Initial Shape: 31 rows x 10 columns
  - Current Shape: 30 rows x 10 columns (-1 rows)
  - Columns:       student_id, name, city, department, subject, score, attendance, fee ...

2. ISSUES DETECTED BEFORE CLEANING
--------------------------------------------------------------------------------
  Found 39 potential data quality issue(s) before cleaning:
  - Duplicate rows                 1
  - Missing values                 6
  - Suspicious dtype               31
  - Potential outlier              1

3. CLEANING OPERATIONS PERFORMED
--------------------------------------------------------------------------------
  Executed 4 operation(s) affecting 4 row(s)/value(s) in total:
  [Step 1] clean_column_names (strategy: lower)
       Target: student_id, name, city, department, subject, score, attendance, fee, joined, status
       Affected: 0 row(s)/value(s)
       Details: No column names were changed.
  [Step 2] drop_duplicates (strategy: keep_first)
       Target: student_id, name, city, department, subject, score, attendance, fee, joined, status
       Affected: 1 row(s)/value(s)
       Details: Removed 1 duplicate row(s) (subset: all columns, keep='first').
  [Step 3] clean_text (strategy: normalize)
       Target: student_id, name, city, department, subject, joined, status
       Affected: 0 row(s)/value(s)
       Details: Normalized text in 7 column(s): 0 value(s) modified.
  [Step 4] fill_missing (strategy: median)
       Target: score, fee
       Affected: 3 row(s)/value(s)
       Details: Filled 3 missing value(s) in 'score', 'fee' using median strategy.

4. ISSUES REMAINING AFTER CLEANING
--------------------------------------------------------------------------------
  3 issue(s) remain in the dataset:
  - Missing values (in 'city'): 3 item(s) [medium severity] [MANUAL REVIEW RECOMMENDED]
    Suggested Action: fill with mode or a user-provided constant
  - Suspicious dtype (in 'joined'): 30 item(s) [medium severity] [MANUAL REVIEW RECOMMENDED]
    Suggested Action: convert data type to datetime
  - Potential outlier (in 'fee'): 1 item(s) [medium severity] [MANUAL REVIEW RECOMMENDED]
    Suggested Action: review manually

5. VALIDATION STATUS
--------------------------------------------------------------------------------
  [!] Status: ACTION REQUIRED (Goals Satisfied: NO)
    Goal Evaluations:
    - duplicate_rows: PASSED (target: 0, actual: 0)
    - missing_values: FAILED (target: 0, actual: 3)
    - inconsistent_column_name: PASSED (target: 0, actual: 0)
    - whitespace_problems: PASSED (target: 0, actual: 0)

6. WARNINGS REQUIRING MANUAL REVIEW
--------------------------------------------------------------------------------
  [!] The following 3 issue(s) require manual human review:
  1. [city] Column 'city' has 3 missing categorical/text value(s) (10.0%).
     Recommendation: fill with mode or a user-provided constant
  2. [joined] Column 'joined' is stored as object/string, but contains date/datetime values.
     Recommendation: convert data type to datetime
  3. [fee] Column 'fee' has 1 potential outlier(s) detected by IQR (bounds: [37500.0, 49500.0]).
     Recommendation: review manually
================================================================================
```

---

## ⚠️ Known Limitations

Being honest about what a tool can and cannot do is essential for reliable data science:

1. **In-Memory Scale**: Tersus processes data in-memory using Pandas. It is designed for datasets that fit comfortably in system RAM (typically up to several gigabytes). It is not a distributed big-data framework like Spark, Dask, or Polars.
2. **File Format Scope**: In V1, the primary file format supported for automated loading and saving is CSV. For Parquet, Excel, SQL, or JSON, load your data with Pandas first (`pd.read_parquet(...)`) and pass the resulting DataFrame directly into `Cleaner(df)`.
3. **Domain Validations**: Phone number parsing in V1 is optimized specifically for Indian phone formats (`+91`). Additional regional formats can be validated using `cleaner.check_rule()`.
4. **Heuristics vs. Domain Truth**:
   - Outliers are identified statistically via IQR and Z-scores; Tersus flags potential anomalies, but domain knowledge is required to decide whether an extreme value is an error or a rare valid record.
   - Ambiguous date strings (e.g., `01/02/2024` which could mean January 2 or February 1) are flagged rather than guessed unless `dayfirst` is explicitly set.

---

## 🧪 Testing Instructions

Tersus is thoroughly tested with **239 unit and integration tests** covering typical data, edge cases, missing data, and invalid inputs.

To run the complete test suite:

```bash
# Run all tests with standard output
pytest

# Run tests with verbose output
pytest -v

# Run a specific test module
pytest tests/test_reporting.py -v
pytest tests/test_verification.py -v
pytest tests/test_orchestration.py -v
pytest tests/test_outliers.py -v
```

### Test Suite Breakdown

- `test_cleaner.py`: Core `Cleaner` initialization, accessors, and state management.
- `test_profiling.py`: Profile metrics, missing value counting, and distributions.
- `test_suggestions.py`: Recommendation rules, severity levels, and suggestion reports.
- `test_missing.py`: Missing value detection, strategies (`mean`, `median`, `mode`, `constant`, `drop`).
- `test_duplicates.py`: Exact and subset row deduplication.
- `test_columns.py`: Case styles, character sanitization, and duplicate column handling.
- `test_text.py`: Unicode normalization, whitespace trimming, and invisible character cleaning.
- `test_types.py`: Safe conversions and non-destructive error handling.
- `test_validation.py`: Numerical bounds, allowed sets, and custom predicate rules.
- `test_dates.py`: DateTime candidate detection, ambiguity flagging, and date formatting.
- `test_outliers.py`: IQR and Z-score calculations with manual mathematical verification.
- `test_emails.py`: Email validation and formatting.
- `test_phones.py`: Indian phone number normalization and styles.
- `test_categories.py`: Category clustering, variation detection, and explicit mapping.
- `test_consistency.py`: Multi-column logical constraints and ordering rules.
- `test_units.py`: Length, weight, and temperature conversions.
- `test_orchestration.py`: Multi-step `clean()` pipeline execution and option validation.
- `test_verification.py`: Post-cleaning goal verification and issue classification.
- `test_reporting.py`: 7-part session reporting, formatting, and serializability.
- `test_package.py`: Module exports and package versioning.

---

## 📄 License

Tersus is an open-source educational project distributed under the MIT License. Built with ❤️ for clean, reliable data.