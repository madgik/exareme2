import json
import logging
from abc import ABC
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from exareme2 import ALGORITHM_FOLDERS
from exareme2.algorithms.specifications import AlgorithmSpecification
from exareme2.algorithms.specifications import AlgorithmType
from exareme2.algorithms.specifications import InputDataSpecification
from exareme2.algorithms.specifications import InputDataSpecifications
from exareme2.algorithms.specifications import InputDataStatType
from exareme2.algorithms.specifications import InputDataType
from exareme2.algorithms.specifications import ParameterEnumSpecification
from exareme2.algorithms.specifications import ParameterEnumType
from exareme2.algorithms.specifications import ParameterSpecification
from exareme2.algorithms.specifications import ParameterType
from exareme2.algorithms.specifications import TransformerSpecification
from exareme2.algorithms.specifications import TransformerType
from exareme2.controller.services.api.algorithm_request_dtos import (
    AlgorithmRequestSystemFlags,
)


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
    validation_datasets: Optional[InputDataSpecificationDTO]


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
    flags: Optional[List[str]]
    type: AlgorithmType


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


def _get_data_model_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Data model of the data.",
        desc="The data model that the algorithm will run on.",
        types=[InputDataType.TEXT],
        notblank=True,
        multiple=False,
        stattypes=None,
        enumslen=None,
    )


def _get_valiadtion_datasets_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Set of data to validate.",
        desc="The set of data to validate the algorithm model on.",
        types=[InputDataType.TEXT],
        notblank=True,
        multiple=True,
        stattypes=None,
        enumslen=None,
    )


def _get_datasets_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Set of data to use.",
        desc="The set of data to run the algorithm on.",
        types=[InputDataType.TEXT],
        notblank=True,
        multiple=True,
        stattypes=None,
        enumslen=None,
    )


def _get_filters_input_data_specification_dto():
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
    validation_datasets = (
        _get_valiadtion_datasets_input_data_specification_dto()
        if spec.validation
        else None
    )
    return InputDataSpecificationsDTO(
        y=y,
        x=x,
        validation_datasets=validation_datasets,
        data_model=_get_data_model_input_data_specification_dto(),
        datasets=_get_datasets_input_data_specification_dto(),
        filter=_get_filters_input_data_specification_dto(),
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
    Converting to a DTO has the following additions:
    1) The preprocessing specifications are added from the transformers that are compatible with the specific algorithm.
    2) The system specific flags are added.
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
        flags=[AlgorithmRequestSystemFlags.SMPC],
        type=spec.type,
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


class Specifications:
    def __init__(self):
        (
            self.enabled_algorithms,
            self.enabled_transformers,
        ) = self.load_and_parse_specifications()

    def load_and_parse_specifications(self):
        all_algorithms, all_transformers = {}, {}
        for specs_path in self.get_specs_paths():
            self.parse_specifications(specs_path, all_algorithms, all_transformers)
        return self.filter_enabled_specifications(all_algorithms, all_transformers)

    @staticmethod
    def get_specs_paths():
        return [Path(specs_path.strip()) for specs_path in ALGORITHM_FOLDERS.split(",")]

    def parse_specifications(self, specs_path, all_algorithms, all_transformers):
        for spec_property_path in specs_path.glob("*.json"):
            self.process_spec_file(spec_property_path, all_algorithms, all_transformers)

    def process_spec_file(self, spec_property_path, all_algorithms, all_transformers):
        try:
            spec_content = self.read_spec_file(spec_property_path)
            self.process_spec_content(
                spec_content, spec_property_path.name, all_algorithms, all_transformers
            )
        except Exception as error:
            logging.error(f"Error processing {spec_property_path.name}: {error}")
            raise

    @staticmethod
    def read_spec_file(spec_property_path):
        with open(spec_property_path, "r") as specifications_file:
            return specifications_file.read()

    @staticmethod
    def process_spec_content(spec_content, spec_name, all_algorithms, all_transformers):
        try:
            spec_json = json.loads(spec_content)
            spec_type = spec_json["type"]
            if TransformerType.EXAREME2_TRANSFORMER.value in spec_type:
                transformer_spec = TransformerSpecification.parse_raw(spec_content)
                all_transformers[transformer_spec.name] = transformer_spec
            else:
                algorithm_specification = AlgorithmSpecification.parse_raw(spec_content)
                all_algorithms[
                    (algorithm_specification.name, algorithm_specification.type)
                ] = algorithm_specification
        except KeyError as e:
            logging.error(f"Missing key {e} in {spec_name}")
            raise

    @staticmethod
    def filter_enabled_specifications(all_algorithms, all_transformers):
        enabled_algorithms = {k: v for k, v in all_algorithms.items() if v.enabled}
        enabled_transformers = {k: v for k, v in all_transformers.items() if v.enabled}
        return enabled_algorithms, enabled_transformers


specifications = Specifications()


algorithm_specifications_dtos = _get_algorithm_specifications_dtos(
    list(specifications.enabled_algorithms.values()),
    list(specifications.enabled_transformers.values()),
)
