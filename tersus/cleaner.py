"""Cleaner module for Tersus."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pandas as pd

from tersus.profiling import ProfileResult, profile_dataframe


class Cleaner:
    """Core data cleaning and quality orchestrator for Tersus.

    Accepts either a CSV file path or a pandas DataFrame, stores an internal
    working copy of the data, and provides accessors and export functionality.
    """

    def __init__(self, data: Union[str, Path, pd.DataFrame, None] = None) -> None:
        """Initialize the Cleaner with a CSV file path or pandas DataFrame.

        Args:
            data: A file path to a CSV file (str or Path) or an existing pandas DataFrame.

        Raises:
            ValueError: If data is None, empty, points to a directory, or has an unsupported format.
            FileNotFoundError: If the specified CSV file does not exist.
            TypeError: If data is neither a valid file path nor a pandas DataFrame.
        """
        if data is None:
            raise ValueError(
                "A data source is required. Expected a CSV file path (str or Path) or a pandas DataFrame."
            )

        if isinstance(data, pd.DataFrame):
            self._df: pd.DataFrame = data.copy()
        elif isinstance(data, (str, Path)):
            path_str = str(data).strip()
            if not path_str:
                raise ValueError("CSV file path cannot be empty.")

            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"File not found: '{data}'")
            if path.is_dir():
                raise ValueError(f"Expected a CSV file, but got a directory: '{data}'")
            if path.suffix.lower() != ".csv":
                raise ValueError(
                    f"Unsupported file format '{path.suffix}'. Tersus Cleaner currently only supports CSV files."
                )

            try:
                self._df = pd.read_csv(path)
            except Exception as exc:
                raise ValueError(f"Failed to read CSV file '{data}': {exc}") from exc
        else:
            raise TypeError(
                f"Invalid input type: '{type(data).__name__}'. Expected a pandas DataFrame or a CSV file path (str or Path)."
            )

    @property
    def data(self) -> pd.DataFrame:
        """Access the current working DataFrame.

        Returns:
            pd.DataFrame: The current working DataFrame.
        """
        return self._df

    def profile(self) -> ProfileResult:
        """Inspect the dataset and identify its basic quality characteristics.

        Returns:
            ProfileResult: A structured object containing dataset quality statistics.
        """
        return profile_dataframe(self._df)

    def save(self, path: Union[str, Path], index: bool = False, **kwargs: Any) -> None:
        """Save the current working DataFrame to a CSV file.

        Args:
            path: Destination file path (str or Path).
            index: Whether to write row names (index). Defaults to False.
            **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_csv.

        Raises:
            TypeError: If path is not a str or Path.
            ValueError: If path is empty or points to an existing directory.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"Invalid path type: '{type(path).__name__}'. Expected a str or Path."
            )

        path_str = str(path).strip()
        if not path_str:
            raise ValueError("Destination path cannot be empty.")

        dest = Path(path)
        if dest.is_dir():
            raise ValueError(
                f"Destination path is an existing directory, not a file: '{path}'"
            )

        if dest.parent and not dest.parent.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)

        self._df.to_csv(dest, index=index, **kwargs)

    def __repr__(self) -> str:
        """Return a concise string representation of the Cleaner."""
        rows, cols = self._df.shape
        return f"Cleaner(rows={rows}, columns={cols})"
