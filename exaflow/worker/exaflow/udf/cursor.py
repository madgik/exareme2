from __future__ import annotations

from typing import Callable
from typing import Iterable
from typing import Optional

import pandas as pd
import pyarrow as pa

from exaflow.algorithms.exaflow.cursor import Cursor
from exaflow.algorithms.exaflow.longitudinal_transformer import (
    apply_longitudinal_transformation,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.pandas_utils import ensure_pandas_dataframe
from exaflow.worker.exaflow.udf.udf_db import load_algorithm_arrow_streaming_factory
from exaflow.worker.exaflow.udf.udf_db import load_algorithm_arrow_table


class DuckDBCursor(Cursor):
    """
    Concrete Cursor backed by DuckDB queries. DataFrame materialization and the
    streaming factory are both cached so algorithms can call either method
    multiple times without triggering duplicate work.
    """

    def __init__(
        self,
        inputdata: Inputdata,
        *,
        dropna: bool,
        include_dataset: bool,
        extra_columns: Optional[Iterable[str]],
        preprocessing: Optional[dict],
    ) -> None:
        self._inputdata = inputdata
        self._dropna = dropna
        self._include_dataset = include_dataset
        self._extra_columns = tuple(extra_columns) if extra_columns else None
        self._preprocessing = preprocessing or {}

        self._dataframe_cache: Optional[pd.DataFrame] = None
        self._factory_cache: Optional[Callable[[], Iterable[pa.Table]]] = None

    def load_all_dataframe(self) -> pd.DataFrame:
        if self._dataframe_cache is None:
            table = load_algorithm_arrow_table(
                self._inputdata,
                dropna=self._dropna,
                include_dataset=self._include_dataset,
                extra_columns=self._extra_columns,
            )
            df = ensure_pandas_dataframe(table)
            transformer_payload = self._preprocessing.get("longitudinal_transformer")
            if transformer_payload:
                df = apply_longitudinal_transformation(df, transformer_payload)
            self._dataframe_cache = df
        return self._dataframe_cache

    def arrow_streaming_factory(self) -> Callable[[], Iterable[pa.Table]]:
        if self._factory_cache is None:
            self._factory_cache = load_algorithm_arrow_streaming_factory(
                self._inputdata,
                dropna=self._dropna,
                include_dataset=self._include_dataset,
                extra_columns=self._extra_columns,
            )
        return self._factory_cache
