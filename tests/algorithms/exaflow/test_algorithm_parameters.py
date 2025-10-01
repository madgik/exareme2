from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.utils.inputdata_utils import Inputdata


class _DummyAlgorithm(Algorithm, algname="dummy"):
    def run(self, metadata):  # pragma: no cover - not used in the assertion
        return metadata


def test_algorithm_parameters_are_available():
    inputdata = Inputdata(
        data_model="dm:0.1",
        datasets=["ds1"],
        y=["col"],
    )
    params = {"threshold": 0.5}

    algorithm = _DummyAlgorithm(
        inputdata=inputdata,
        engine=object(),
        parameters=params,
    )

    assert algorithm.parameters == params

    algorithm_without_params = _DummyAlgorithm(
        inputdata=inputdata,
        engine=object(),
    )

    assert algorithm_without_params.parameters == {}
