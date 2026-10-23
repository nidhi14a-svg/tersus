import re
from pathlib import Path

import pandas as pd
import pytest
import tersus
from tersus import Cleaner


def test_package_version():
    assert hasattr(tersus, "__version__")
    assert tersus.__version__ == "0.1.0"


def test_cleaner_import_and_instantiation():
    cleaner = Cleaner(pd.DataFrame({"a": [1]}))
    assert isinstance(cleaner, Cleaner)


def test_readme_documented_cleaner_methods_actually_exist():
    """Regression guard: every `cleaner.<method>(` snippet documented in
    README.md must correspond to a real Cleaner method. Catches doc/API
    drift like the previous check_emails()/check_phones() vs. the actual
    check_email()/check_phone() mismatch, which would break any user
    copy-pasting the README's own examples.
    """
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    documented_methods = sorted(set(re.findall(r"cleaner\.([a-zA-Z_]+)\(", readme_text)))
    assert documented_methods, "Expected to find cleaner.<method>( examples in README.md"

    missing = [m for m in documented_methods if not hasattr(Cleaner, m)]
    assert missing == [], f"README documents Cleaner methods that don't exist: {missing}"
