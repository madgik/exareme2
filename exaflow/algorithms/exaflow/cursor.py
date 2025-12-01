from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Iterable

import pandas as pd
import pyarrow as pa


class Cursor(ABC):
    """
    Lightweight interface exposed to algorithm UDFs so they can decide whether
    to materialize the full dataset or iterate over Arrow batches.
    """

    @abstractmethod
    def load_all_dataframe(self) -> pd.DataFrame:
        """Materialize the full dataset as a pandas DataFrame."""

    @abstractmethod
    def arrow_streaming_factory(self) -> Callable[[], Iterable[pa.Table]]:
        """Return a factory that yields Arrow Tables when iterated."""
