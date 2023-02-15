import pytest
from pydantic import ValidationError

from mipengine.controller.algorithm_specifications import AlgorithmSpecification
from mipengine.controller.algorithm_specifications import InputDataSpecification
from mipengine.controller.algorithm_specifications import InputDataSpecifications
from mipengine.controller.algorithm_specifications import ParameterEnumSpecification
from mipengine.controller.algorithm_specifications import ParameterSpecification
from mipengine.controller.api.algorithm_specifications_dtos import InputDataStatType
from mipengine.controller.api.algorithm_specifications_dtos import InputDataType
from mipengine.controller.api.algorithm_specifications_dtos import ParameterEnumType
from mipengine.controller.api.algorithm_specifications_dtos import ParameterType


def test_validate_parameter_spec_inputdata_CDE_enums_source_is_x_or_y():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has enums type 'inputdata_CDE_enums' "
        "that supports only 'x' or 'y' as source. Value given: 'not_x_or_y'.*"
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=False,
                )
            ),
            parameters={
                "inputdata_cde_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUTDATA_CDE_ENUMS, source="not_x_or_y"
                    ),
                ),
            },
        )


def test_validate_parameter_spec_inputdata_CDE_enums_multiple_false():
    exception_type = ValidationError
    exception_message = (
        ".*In algorithm 'sample_algo', parameter 'sample_label' has enums type 'inputdata_CDE_enums' "
        "that doesn't support 'multiple=True', in the parameter.*"
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=False,
                )
            ),
            parameters={
                "inputdata_cde_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    notblank=False,
                    multiple=True,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUTDATA_CDE_ENUMS, source="y"
                    ),
                ),
            },
        )


def test_validate_parameter_spec_inputdata_CDE_enums_inputdata_has_multiple_false():
    exception_type = ValidationError
    exception_message = (
        ".* In algorithm 'sample_algo', parameter 'sample_label' has enums type 'inputdata_CDE_enums' "
        "that doesn't support 'multiple=True' in it's linked inputdata var 'y'.*"
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=True,
                )
            ),
            parameters={
                "inputdata_cde_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.TEXT],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUTDATA_CDE_ENUMS, source="y"
                    ),
                ),
            },
        )


def test_validate_parameter_spec_inputdata_CDEs_type_must_be_text():
    exception_type = ValidationError
    exception_message = (
        """.* In algorithm 'sample_algo', parameter 'sample_label' has enums type 'inputdata_CDEs' """
        """that supports ONLY '.*' but the 'types' provided were .*"""
    )
    with pytest.raises(exception_type, match=exception_message):
        AlgorithmSpecification(
            name="sample_algo",
            desc="sample",
            label="sample_algo",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    notblank=True,
                    multiple=True,
                )
            ),
            parameters={
                "inputdata_cdes_enum_param": ParameterSpecification(
                    label="sample_label",
                    desc="sample",
                    types=[ParameterType.INT],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUTDATA_CDES, source=["y"]
                    ),
                ),
            },
        )
