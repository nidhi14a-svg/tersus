import pandas as pd
import tersus
from tersus import Cleaner


def test_package_version():
    assert hasattr(tersus, "__version__")
    assert tersus.__version__ == "0.1.0"


def test_cleaner_import_and_instantiation():
    cleaner = Cleaner(pd.DataFrame({"a": [1]}))
    assert isinstance(cleaner, Cleaner)
