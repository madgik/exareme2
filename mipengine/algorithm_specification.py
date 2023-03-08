from enum import Enum
from enum import unique
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import root_validator


@unique
class InputDataType(Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"
    JSONOBJECT = "jsonObject"


@unique
class InputDataStatType(Enum):
    NUMERICAL = "numerical"
    NOMINAL = "nominal"


@unique
class ParameterType(str, Enum):
    REAL = "real"
    INT = "int"
    TEXT = "text"
    BOOLEAN = "boolean"
    DICT = "dict"


@unique
class ParameterEnumType(str, Enum):
    LIST = "list"
    INPUT_VAR_CDE_ENUMS = "input_var_CDE_enums"
    FIXED_VAR_CDE_ENUMS = "fixed_var_CDE_enums"
    INPUT_VAR_NAMES = "input_var_names"


class ImmutableBaseModel(BaseModel):
    class Config:
        allow_mutation = False


class InputDataSpecification(ImmutableBaseModel):
    label: str
    desc: str
    types: List[InputDataType]
    stattypes: List[InputDataStatType]
    notblank: bool
    multiple: bool
    enumslen: Optional[int]


class InputDataSpecifications(ImmutableBaseModel):
    y: InputDataSpecification
    x: Optional[InputDataSpecification]


class ParameterEnumSpecification(ImmutableBaseModel):
    type: ParameterEnumType
    source: Any


class ParameterSpecification(ImmutableBaseModel):
    label: str
    desc: str
    types: List[ParameterType]
    notblank: bool
    multiple: bool
    default: Any
    enums: Optional[ParameterEnumSpecification]
    dict_keys_enums: Optional[ParameterEnumSpecification]
    dict_values_enums: Optional[ParameterEnumSpecification]
    min: Optional[float]
    max: Optional[float]


def _validate_parameter_with_enums_type_input_var_CDE_enums(param_value, cls_values):
    if param_value.enums.source not in ["x", "y"]:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type 'input_var_CDE_enums' "
            f"that supports only 'x' or 'y' as source. Value given: '{param_value.enums.source}'."
        )
    if param_value.multiple:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type 'input_var_CDE_enums' "
            f"that doesn't support 'multiple=True', in the parameter."
        )
    inputdata_var = (
        cls_values["inputdata"].x
        if param_value.enums.source == "x"
        else cls_values["inputdata"].y
    )
    if inputdata_var.multiple:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type "
            f"'{ParameterEnumType.INPUT_VAR_CDE_ENUMS.value}' "
            f"that doesn't support 'multiple=True' in it's linked inputdata var '{inputdata_var.label}'."
        )


def _validate_parameter_with_enums_type_input_var_names(param_value, cls_values):
    if param_value.types != [ParameterType.TEXT]:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has enums type "
            f"'{ParameterEnumType.INPUT_VAR_NAMES.value}' that supports ONLY 'types=[\"text\"]' but the 'types' "
            f"provided were {[t.value for t in param_value.types]}."
        )


def _validate_parameter_enums(param_value, cls_values):
    if not param_value.enums:
        return
    if param_value.enums.type == ParameterEnumType.INPUT_VAR_CDE_ENUMS:
        _validate_parameter_with_enums_type_input_var_CDE_enums(param_value, cls_values)
    if param_value.enums.type == ParameterEnumType.INPUT_VAR_NAMES:
        _validate_parameter_with_enums_type_input_var_names(param_value, cls_values)


def _validate_parameter_type_dict(param_value, cls_values):
    if ParameterType.DICT in param_value.types:
        if len(param_value.types) > 1:
            raise ValueError(
                f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' cannot use 'dict' type "
                f"combined with other types. Types provided: {param_value.types}. "
            )


def _validate_parameter_type_dict_keys_enums(param_value, cls_values):
    if not param_value.dict_keys_enums:
        return

    if ParameterType.DICT not in param_value.types:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has the property 'dict_keys_enums' "
            f"but the allowed 'types' is not '{ParameterType.DICT}'."
        )


def _validate_parameter_type_dict_values_enums(param_value, cls_values):
    if not param_value.dict_values_enums:
        return

    if ParameterType.DICT not in param_value.types:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has the property 'dict_values_enums' "
            f"but the allowed 'types' is not '{ParameterType.DICT}'."
        )


def _validate_parameter_type_dict_enums_not_allowed(param_value, cls_values):
    if not param_value.enums:
        return

    if ParameterType.DICT in param_value.types:
        raise ValueError(
            f"In algorithm '{cls_values['label']}', parameter '{param_value.label}' has the property 'enums' "
            f"but since the 'types' is '{ParameterType.DICT}', you should use 'dict_keys_enums' and 'dict_values_enums'."
        )


def _validate_parameter_type_dict_enums(param_value, cls_values):
    _validate_parameter_type_dict_keys_enums(param_value, cls_values)
    _validate_parameter_type_dict_values_enums(param_value, cls_values)
    _validate_parameter_type_dict_enums_not_allowed(param_value, cls_values)


class AlgorithmSpecification(ImmutableBaseModel):
    name: str
    desc: str
    label: str
    enabled: bool
    inputdata: InputDataSpecifications
    parameters: Optional[Dict[str, ParameterSpecification]]
    flags: Optional[Dict[str, bool]]

    @root_validator
    def validate_parameter_enums_logic(cls, cls_values):
        if not cls_values["parameters"]:
            return cls_values

        for param_value in cls_values["parameters"].values():
            _validate_parameter_enums(param_value, cls_values)
            _validate_parameter_type_dict(param_value, cls_values)
            _validate_parameter_type_dict_enums(param_value, cls_values)
        return cls_values