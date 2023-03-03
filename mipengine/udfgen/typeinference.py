from typing import Dict
from typing import TypeVar
from typing import Union

from mipengine.udfgen.decorator import UDFBadCall
from mipengine.udfgen.helpers import compose_mappings
from mipengine.udfgen.helpers import mapping_inverse
from mipengine.udfgen.helpers import merge_mappings_consistently
from mipengine.udfgen.iotypes import ParametrizedType
from mipengine.udfgen.iotypes import TableType

KnownTypeParams = Union[type, int]
UnknownTypeParams = TypeVar
TypeParamsInference = Dict[UnknownTypeParams, KnownTypeParams]


def infer_output_type(
    passed_input_types: Dict[str, ParametrizedType],
    declared_input_types: Dict[str, ParametrizedType],
    declared_output_type: ParametrizedType,
) -> ParametrizedType:
    verify_declared_and_passed_param_types_match(
        declared_input_types, passed_input_types
    )

    inferred_input_typeparams = infer_unknown_input_typeparams(
        declared_input_types,
        passed_input_types,
    )
    known_output_typeparams = dict(**declared_output_type.known_typeparams)
    inferred_output_typeparams = compose_mappings(
        declared_output_type.unknown_typeparams,
        inferred_input_typeparams,
    )
    known_output_typeparams.update(inferred_output_typeparams)
    inferred_output_type = type(declared_output_type)(**known_output_typeparams)
    return inferred_output_type


def infer_unknown_input_typeparams(
    declared_input_types: "Dict[str, ParametrizedType]",
    passed_input_types: Dict[str, ParametrizedType],
) -> TypeParamsInference:
    typeparams_inference_mappings = [
        map_unknown_to_known_typeparams(
            input_type.unknown_typeparams,
            passed_input_types[name].known_typeparams,
        )
        for name, input_type in declared_input_types.items()
        if input_type.is_generic
    ]
    distinct_inferred_typeparams = merge_mappings_consistently(
        typeparams_inference_mappings
    )
    return distinct_inferred_typeparams


def map_unknown_to_known_typeparams(
    unknown_params: Dict[str, UnknownTypeParams],
    known_params: Dict[str, KnownTypeParams],
) -> TypeParamsInference:
    return compose_mappings(mapping_inverse(unknown_params), known_params)


def verify_declared_and_passed_param_types_match(
    declared_types: Dict[str, TableType],
    passed_types: Dict[str, TableType],
) -> None:
    passed_param_args = {
        name: type
        for name, type in passed_types.items()
        if isinstance(type, ParametrizedType)
    }
    for argname, type in passed_param_args.items():
        known_params = declared_types[argname].known_typeparams
        verify_declared_typeparams_match_passed_type(known_params, type)


def verify_declared_typeparams_match_passed_type(
    known_typeparams: Dict[str, KnownTypeParams],
    passed_type: ParametrizedType,
) -> None:
    for name, param in known_typeparams.items():
        if not hasattr(passed_type, name):
            raise UDFBadCall(f"{passed_type} has no typeparam {name}.")
        if getattr(passed_type, name) != param:
            raise UDFBadCall(
                "InputType's known typeparams do not match typeparams passed "
                f"in {passed_type}: {param}, {getattr(passed_type, name)}."
            )
