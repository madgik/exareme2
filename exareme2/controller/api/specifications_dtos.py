from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

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
from exareme2.controller import algorithms_specifications
from exareme2.controller import transformers_specifications


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


class InputDataSpecificationDTO(ImmutableBaseModel):
    label: str
    desc: str
    types: List[InputDataType]
    notblank: bool
    multiple: bool
    stattypes: Optional[List[InputDataStatType]]
    enumslen: Optional[int]


class InputDataSpecificationsDTO(ImmutableBaseModel):
    data_model: InputDataSpecificationDTO
    datasets: InputDataSpecificationDTO
    filter: InputDataSpecificationDTO
    y: InputDataSpecificationDTO
    x: Optional[InputDataSpecificationDTO]


class ParameterEnumSpecificationDTO(ImmutableBaseModel):
    type: ParameterEnumType
    source: List[Any]


class ParameterSpecificationDTO(ImmutableBaseModel):
    label: str
    desc: str
    types: List[ParameterType]
    notblank: bool
    multiple: bool
    default: Any
    enums: Optional[ParameterEnumSpecificationDTO]
    dict_keys_enums: Optional[ParameterEnumSpecificationDTO]
    dict_values_enums: Optional[ParameterEnumSpecificationDTO]
    min: Optional[float]
    max: Optional[float]


class TransformerSpecificationDTO(ImmutableBaseModel):
    name: str
    desc: str
    label: str
    parameters: Optional[Dict[str, ParameterSpecificationDTO]]


class AlgorithmSpecificationDTO(ImmutableBaseModel):
    name: str
    desc: str
    label: str
    inputdata: InputDataSpecificationsDTO
    parameters: Optional[Dict[str, ParameterSpecificationDTO]]
    preprocessing: Optional[List[TransformerSpecificationDTO]]


class AlgorithmSpecificationsDTO(ImmutableBaseModel):
    __root__: List[AlgorithmSpecificationDTO]


class TransformerSpecificationsDTO(ImmutableBaseModel):
    __root__: List[TransformerSpecificationDTO]


def _convert_inputdata_specification_to_dto(self: InputDataSpecification):
    # The only difference of the DTO is that it's stattypes is Optional,
    # due to the fact that the datasets/data_model variables are added.
    return InputDataSpecificationDTO(
        label=self.label,
        desc=self.desc,
        types=self.types,
        stattypes=self.stattypes,
        notblank=self.notblank,
        multiple=self.multiple,
        enumslen=self.enumslen,
    )


def _get_data_model_input_data_specification_DTO():
    return InputDataSpecificationDTO(
        label="Data model of the data.",
        desc="The data model that the algorithm will run on.",
        types=[InputDataType.TEXT],
        notblank=True,
        multiple=False,
        stattypes=None,
        enumslen=None,
    )


def _get_datasets_input_data_specification_DTO():
    return InputDataSpecificationDTO(
        label="Set of data to use.",
        desc="The set of data to run the algorithm on.",
        types=[InputDataType.TEXT],
        notblank=True,
        multiple=True,
        stattypes=None,
        enumslen=None,
    )


def _get_filters_input_data_specification_DTO():
    return InputDataSpecificationDTO(
        label="filter on the data.",
        desc="Features used in my algorithm.",
        types=[InputDataType.JSONOBJECT],
        notblank=False,
        multiple=False,
        stattypes=None,
        enumslen=None,
    )


def _convert_inputdata_specifications_to_dto(spec: InputDataSpecifications):
    # In the DTO the datasets, data_model and filter parameters are added from the engine.
    # These parameters are not added by the algorithm developer.
    y = _convert_inputdata_specification_to_dto(spec.y)
    x = _convert_inputdata_specification_to_dto(spec.x) if spec.x else None
    return InputDataSpecificationsDTO(
        y=y,
        x=x,
        data_model=_get_data_model_input_data_specification_DTO(),
        datasets=_get_datasets_input_data_specification_DTO(),
        filter=_get_filters_input_data_specification_DTO(),
    )


def _convert_parameter_enum_specification_to_dto(spec: ParameterEnumSpecification):
    return ParameterEnumSpecificationDTO(
        type=spec.type,
        source=spec.source,
    )


def _convert_parameter_specification_to_dto(spec: ParameterSpecification):
    return ParameterSpecificationDTO(
        label=spec.label,
        desc=spec.desc,
        types=spec.types,
        notblank=spec.notblank,
        multiple=spec.multiple,
        default=spec.default,
        enums=(
            _convert_parameter_enum_specification_to_dto(spec.enums)
            if spec.enums
            else None
        ),
        dict_keys_enums=(
            _convert_parameter_enum_specification_to_dto(spec.dict_keys_enums)
            if spec.dict_keys_enums
            else None
        ),
        dict_values_enums=(
            _convert_parameter_enum_specification_to_dto(spec.dict_values_enums)
            if spec.dict_values_enums
            else None
        ),
        min=spec.min,
        max=spec.max,
    )


def _convert_transformer_specification_to_dto(spec: TransformerSpecification):
    return TransformerSpecificationDTO(
        name=spec.name,
        desc=spec.desc,
        label=spec.label,
        parameters=(
            {
                name: _convert_parameter_specification_to_dto(value)
                for name, value in spec.parameters.items()
            }
            if spec.parameters
            else None
        ),
    )


def _get_algorithm_compatible_transformers(
    algo_name: str, transformers: List[TransformerSpecification]
) -> List[TransformerSpecification]:
    compatible_transformers = []
    for transformer in transformers:
        if (
            not transformer.compatible_algorithms
            or algo_name in transformer.compatible_algorithms
        ):
            compatible_transformers.append(transformer)
    return compatible_transformers


def _convert_algorithm_specification_to_dto(
    spec: AlgorithmSpecification, transformers: List[TransformerSpecification]
):
    """
    Converting to a DTO does not include the flags.
    The preprocessing specifications is added from the transformers that are compatible with the specific algorithm.
    """
    return AlgorithmSpecificationDTO(
        name=spec.name,
        desc=spec.desc,
        label=spec.label,
        inputdata=_convert_inputdata_specifications_to_dto(spec.inputdata),
        parameters=(
            {
                name: _convert_parameter_specification_to_dto(value)
                for name, value in spec.parameters.items()
            }
            if spec.parameters
            else None
        ),
        preprocessing=[
            _convert_transformer_specification_to_dto(spec)
            for spec in _get_algorithm_compatible_transformers(spec.name, transformers)
        ],
    )


def _get_algorithm_specifications_dtos(
    algorithms_specs: List[AlgorithmSpecification],
    transformers_specs: List[TransformerSpecification],
) -> AlgorithmSpecificationsDTO:
    return AlgorithmSpecificationsDTO(
        __root__=[
            _convert_algorithm_specification_to_dto(spec, transformers_specs)
            for spec in algorithms_specs
        ]
    )


algorithm_specifications_dtos = _get_algorithm_specifications_dtos(
    algorithms_specifications.values(), transformers_specifications.values()
)