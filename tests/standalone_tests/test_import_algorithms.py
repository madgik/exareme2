import pytest

from exareme2 import Algorithm
from exareme2.algorithms.specifications import AlgorithmSpecification
from exareme2.algorithms.specifications import InputDataSpecification
from exareme2.algorithms.specifications import InputDataSpecifications
from exareme2.algorithms.specifications import InputDataStatType
from exareme2.algorithms.specifications import InputDataType
from exareme2.controller import _get_algorithms_specifications


def test_algorithm_specifications_with_same_name_raises_error():
    spec = AlgorithmSpecification(
        name="algorithm_name",
        desc="",
        label="",
        enabled=True,
        inputdata=InputDataSpecifications(
            y=InputDataSpecification(
                label="",
                desc="",
                types=[InputDataType.TEXT],
                stattypes=[InputDataStatType.NOMINAL],
                notblank=True,
                multiple=True,
            )
        ),
        parameters=None,
    )

    class Algorithm1(Algorithm, algname="algorithm1"):
        @classmethod
        def get_specification(cls):
            return spec

        def run(self, data, metadata):
            pass

    class Algorithm2(Algorithm, algname="algorithm1"):
        @classmethod
        def get_specification(cls):
            return spec

        def run(self, data, metadata):
            pass

    algorithms = [Algorithm1, Algorithm2]

    with pytest.raises(
        ValueError,
        match="The algorithm name .* exists more than once in the algorithm specifications.",
    ):
        _get_algorithms_specifications(algorithms)
