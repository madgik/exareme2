from exareme2.algorithms.specifications import AlgorithmSpecification
from exareme2.algorithms.specifications import InputDataSpecification
from exareme2.algorithms.specifications import InputDataSpecifications
from exareme2.algorithms.specifications import InputDataStatType
from exareme2.algorithms.specifications import InputDataType
from exareme2.algorithms.specifications import ParameterEnumSpecification
from exareme2.algorithms.specifications import ParameterEnumType
from exareme2.algorithms.specifications import ParameterSpecification
from exareme2.algorithms.specifications import ParameterType
from exareme2.algorithms.specifications import TransformerSpecification
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmType
from exareme2.controller.services.api.algorithm_request_dtos import TransformerType
from exareme2.controller.services.api.algorithm_spec_dtos import (
    AlgorithmSpecificationDTO,
)
from exareme2.controller.services.api.algorithm_spec_dtos import (
    InputDataSpecificationDTO,
)
from exareme2.controller.services.api.algorithm_spec_dtos import (
    InputDataSpecificationsDTO,
)
from exareme2.controller.services.api.algorithm_spec_dtos import (
    ParameterEnumSpecificationDTO,
)
from exareme2.controller.services.api.algorithm_spec_dtos import (
    ParameterSpecificationDTO,
)
from exareme2.controller.services.api.algorithm_spec_dtos import (
    TransformerSpecificationDTO,
)
from exareme2.controller.services.api.algorithm_spec_dtos import (
    _convert_algorithm_specification_to_dto,
)
from exareme2.controller.services.api.algorithm_spec_dtos import (
    _convert_transformer_specification_to_dto,
)
from exareme2.controller.services.api.algorithm_spec_dtos import (
    _get_algorithm_specifications_dtos,
)


def test_convert_algorithm_specification_to_dto():
    algo_spec = AlgorithmSpecification(
        name="sample_algorithm",
        desc="sample_algorithm",
        label="sample_algorithm",
        enabled=True,
        type=AlgorithmType.EXAREME2,
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
        type=AlgorithmType.EXAREME2,
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
        flags=["smpc"],
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
        type=TransformerType.EXAREME2_TRANSFORMER,
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
        type=TransformerType.EXAREME2_TRANSFORMER,
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
            type=AlgorithmType.EXAREME2,
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
            type=TransformerType.EXAREME2_TRANSFORMER,
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
            type=AlgorithmType.EXAREME2,
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
            type=TransformerType.EXAREME2_TRANSFORMER,
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
