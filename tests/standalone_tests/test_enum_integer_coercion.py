import pandas as pd

from exaflow.algorithms.exareme3.anova_oneway import anova_oneway_local_step
from exaflow.algorithms.exareme3.utils.registry import get_udf_registry_key
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker import config as worker_config
from exaflow.worker.exareme3.udf import udf_db
from exaflow.worker.exareme3.udf import udf_service


class DummyAggClient:
    def __init__(self, *args, **kwargs):
        pass

    def sum(self, values):
        return values

    def min(self, values):
        return values

    def max(self, values):
        return values


def test_integer_enum_codes_flow_through_anova_oneway(monkeypatch):
    df = pd.DataFrame(
        {
            "ppmicategory": ["0", "1", "0", "1", None],
            "age": [60, 61, 62, 63, 64],
        }
    )
    metadata = {
        "ppmicategory": {
            "is_categorical": True,
            "enumerations": {0: "HC", 1: "PD"},
        }
    }

    class FakeArrowTable:
        def __init__(self, data):
            self._data = data
            self.num_rows = data.shape[0]

        def to_pandas(self):
            return self._data

    def fake_load_algorithm_arrow_table(inputdata, dropna=True, **_kwargs):
        data = df.copy()
        if dropna:
            columns = list(dict.fromkeys((inputdata.x or []) + (inputdata.y or [])))
            data = data.dropna(subset=columns)
        return FakeArrowTable(data)

    monkeypatch.setattr(udf_service, "AggregationClient", DummyAggClient)
    monkeypatch.setattr(
        udf_db, "load_algorithm_arrow_table", fake_load_algorithm_arrow_table
    )
    monkeypatch.setattr(
        udf_service, "load_algorithm_arrow_table", fake_load_algorithm_arrow_table
    )
    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 0)

    inputdata = Inputdata(
        data_model="test_model",
        datasets=["test_dataset"],
        x=["ppmicategory"],
        y=["age"],
    )
    udf_registry_key = get_udf_registry_key(anova_oneway_local_step)
    result = udf_service.run_udf(
        request_id="test-request",
        udf_registry_key=udf_registry_key,
        kw_args={
            "x_var": "ppmicategory",
            "y_var": "age",
            "covar_enums": [0, 1],
        },
        system_args={
            "inputdata": inputdata.json(),
            "metadata": metadata,
            "drop_na": True,
            "check_min_rows": True,
            "add_dataset_variable": False,
        },
    )

    assert result["categories"] == [0, 1]
    assert result["group_stats_index"] == [0, 1]
