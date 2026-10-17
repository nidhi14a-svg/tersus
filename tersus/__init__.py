"""Tersus - A Python data-cleaning and data-quality library built on top of Pandas."""

from tersus.categories import CategoryVariationsResult
from tersus.cleaner import Cleaner
from tersus.consistency import ConsistencyResult, ConsistencySuggestion
from tersus.dates import DateValidationResult
from tersus.missing import CleaningRecord
from tersus.orchestration import clean
from tersus.outliers import OutlierResult
from tersus.profiling import ProfileResult
from tersus.suggestions import Suggestion, SuggestionReport
from tersus.validation import ValidationResult
from tersus.reporting import (
    CleaningReport,
    DatasetInfo,
    OperationSummary,
    SessionReport,
    generate_session_report,
    generate_session_report as report,
)
from tersus.verification import (
    RemainingIssue,
    ResolvedIssue,
    ValidationReport,
    validate_cleaning,
    validate_cleaning as validate,
)

__version__ = "0.1.0"

__all__ = [
    "CategoryVariationsResult",
    "Cleaner",
    "CleaningRecord",
    "CleaningReport",
    "ConsistencyResult",
    "ConsistencySuggestion",
    "DatasetInfo",
    "DateValidationResult",
    "OperationSummary",
    "OutlierResult",
    "ProfileResult",
    "RemainingIssue",
    "ResolvedIssue",
    "SessionReport",
    "Suggestion",
    "SuggestionReport",
    "ValidationReport",
    "ValidationResult",
    "__version__",
    "clean",
    "generate_session_report",
    "report",
    "validate",
    "validate_cleaning",
]

