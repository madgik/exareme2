from mipengine.algorithms.specifications import AlgorithmSpecification
from mipengine.algorithms.specifications import InputDataSpecification
from mipengine.algorithms.specifications import InputDataSpecifications
from mipengine.algorithms.specifications import InputDataStatType
from mipengine.algorithms.specifications import InputDataType
from mipengine.algorithms.specifications import ParameterEnumSpecification
from mipengine.algorithms.specifications import ParameterEnumType
from mipengine.algorithms.specifications import ParameterSpecification
from mipengine.algorithms.specifications import ParameterType
from mipengine.algorithms.specifications import TransformerSpecification
from mipengine.controller.api.specifications_dtos import AlgorithmSpecificationDTO
from mipengine.controller.api.specifications_dtos import InputDataSpecificationDTO
from mipengine.controller.api.specifications_dtos import InputDataSpecificationsDTO
from mipengine.controller.api.specifications_dtos import ParameterEnumSpecificationDTO
from mipengine.controller.api.specifications_dtos import ParameterSpecificationDTO
from mipengine.controller.api.specifications_dtos import TransformerSpecificationDTO
from mipengine.controller.api.specifications_dtos import (
    _convert_algorithm_specification_to_dto,
)
from mipengine.controller.api.specifications_dtos import (
    _convert_transformer_specification_to_dto,
)
from mipengine.controller.api.specifications_dtos import (
    _get_algorithm_specifications_dtos,
)


def test_convert_algorithm_specification_to_dto():
    algo_spec = AlgorithmSpecification(
        name="sample_algorithm",
        desc="sample_algorithm",
        label="sample_algorithm",
        enabled=True,
        inputdata=InputDataSpecifications(
            y=InputDataSpecification(
                label="y",
                desc="y",
                types=[InputDataType.REAL],
                stattypes=[InputDataStatType.NUMERICAL],
                notblank=True,
                multiple=False,
            ),
            x=InputDataSpecification(
                label="x",
                desc="x",
                types=[InputDataType.TEXT],
                stattypes=[InputDataStatType.NOMINAL],
                notblank=True,
                multiple=False,
            ),
        ),
        parameters={
            "sample_param": ParameterSpecification(
                label="sample_param",
                desc="sample_param",
                types=[ParameterType.TEXT],
                notblank=False,
                multiple=False,
                enums=ParameterEnumSpecification(
                    type=ParameterEnumType.LIST,
                    source=["a", "b", "c"],
                ),
            ),
        },
    )

    expected_dto = AlgorithmSpecificationDTO(
        name="sample_algorithm",
        desc="sample_algorithm",
        label="sample_algorithm",
        inputdata=InputDataSpecificationsDTO(
            data_model=InputDataSpecificationDTO(
                label="Data model of the data.",
                desc="The data model that the algorithm will run on.",
                types=[InputDataType.TEXT],
                notblank=True,
                multiple=False,
            ),
            datasets=InputDataSpecificationDTO(
                label="Set of data to use.",
                desc="The set of data to run the algorithm on.",
                types=[InputDataType.TEXT],
                notblank=True,
                multiple=True,
            ),
            filter=InputDataSpecificationDTO(
                label="filter on the data.",
                desc="Features used in my algorithm.",
                types=[InputDataType.JSONOBJECT],
                notblank=False,
                multiple=False,
            ),
            y=InputDataSpecificationDTO(
                label="y",
                desc="y",
                types=[InputDataType.REAL],
                notblank=True,
                multiple=False,
                stattypes=[InputDataStatType.NUMERICAL],
            ),
            x=InputDataSpecificationDTO(
                label="x",
                desc="x",
                types=[InputDataType.TEXT],
                notblank=True,
                multiple=False,
                stattypes=[InputDataStatType.NOMINAL],
            ),
        ),
        parameters={
            "sample_param": ParameterSpecificationDTO(
                label="sample_param",
                desc="sample_param",
                types=[ParameterType.TEXT],
                notblank=False,
                multiple=False,
                enums=ParameterEnumSpecificationDTO(
                    type=ParameterEnumType.LIST, source=["a", "b", "c"]
                ),
            )
        },
        preprocessing=[],
    )

    dto = _convert_algorithm_specification_to_dto(
        spec=algo_spec,
        transformers=[],
    )

    assert dto == expected_dto


def test_convert_transformer_specification_to_dto():
    transformer_spec = TransformerSpecification(
        name="sample_transformer",
        desc="sample_transformer",
        label="sample_transformer",
        enabled=True,
        parameters={
            "sample_param": ParameterSpecification(
                label="sample_param",
                desc="sample_param",
                types=[ParameterType.TEXT],
                notblank=False,
                multiple=False,
                enums=ParameterEnumSpecification(
                    type=ParameterEnumType.LIST,
                    source=["a", "b", "c"],
                ),
            ),
        },
    )

    expected_dto = TransformerSpecificationDTO(
        name="sample_transformer",
        desc="sample_transformer",
        label="sample_transformer",
        parameters={
            "sample_param": ParameterSpecificationDTO(
                label="sample_param",
                desc="sample_param",
                types=[ParameterType.TEXT],
                notblank=False,
                multiple=False,
                enums=ParameterEnumSpecificationDTO(
                    type=ParameterEnumType.LIST, source=["a", "b", "c"]
                ),
            )
        },
    )

    dto = _convert_transformer_specification_to_dto(
        spec=transformer_spec,
    )

    assert dto == expected_dto


def test_get_algorithm_specifications_dtos_compatible_algorithms():
    algo_specs = [
        AlgorithmSpecification(
            name="sample_algorithm",
            desc="sample_algorithm",
            label="sample_algorithm",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=False,
                ),
            ),
        )
    ]

    transformer_specs = [
        TransformerSpecification(
            name="sample_transformer",
            desc="sample_transformer",
            label="sample_transformer",
            enabled=True,
            parameters={
                "sample_param": ParameterSpecification(
                    label="sample_param",
                    desc="sample_param",
                    types=[ParameterType.TEXT],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST,
                        source=["a", "b", "c"],
                    ),
                ),
            },
            compatible_algorithms=["sample_algorithm"],
        )
    ]

    dtos = _get_algorithm_specifications_dtos(
        algorithms_specs=algo_specs,
        transformers_specs=transformer_specs,
    )

    dto = dtos.__root__[0]

    assert dto.preprocessing
    assert dto.preprocessing[0].name == "sample_transformer"


def test_get_algorithm_specifications_dtos_empty_compatible_algorithms():
    algo_specs = [
        AlgorithmSpecification(
            name="sample_algorithm",
            desc="sample_algorithm",
            label="sample_algorithm",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=False,
                ),
            ),
        )
    ]

    transformer_specs = [
        TransformerSpecification(
            name="sample_transformer",
            desc="sample_transformer",
            label="sample_transformer",
            enabled=True,
            parameters={
                "sample_param": ParameterSpecification(
                    label="sample_param",
                    desc="sample_param",
                    types=[ParameterType.TEXT],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST,
                        source=["a", "b", "c"],
                    ),
                ),
            },
        )
    ]

    dtos = _get_algorithm_specifications_dtos(
        algorithms_specs=algo_specs,
        transformers_specs=transformer_specs,
    )

    dto = dtos.__root__[0]

    assert dto.preprocessing
    assert dto.preprocessing[0].name == "sample_transformer"
