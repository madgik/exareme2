from mipengine.algorithms.base_classes.transformer import Transformer
from mipengine.algorithms.specifications.parameter_specification import (
    ParameterEnumSpecification,
)
from mipengine.algorithms.specifications.parameter_specification import (
    ParameterEnumType,
)
from mipengine.algorithms.specifications.parameter_specification import (
    ParameterSpecification,
)
from mipengine.algorithms.specifications.parameter_specification import ParameterType
from mipengine.algorithms.specifications.transformer_specification import (
    TransformerSpecification,
)


class LongitudinalTransformation(Transformer, stepname="longitudinal_transform"):
    @classmethod
    def get_specification(cls):
        return TransformerSpecification(
            name=cls.stepname,
            desc="longitudinal_transform",
            label="longitudinal_transform",
            enabled=True,
            parameters={
                "visit1": ParameterSpecification(
                    label="visit1",
                    desc="visit1",
                    types=[ParameterType.TEXT],
                    notblank=True,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.FIXED_VAR_CDE_ENUMS, source="visitID"
                    ),
                ),
                "visit2": ParameterSpecification(
                    label="visit2",
                    desc="visit2",
                    types=[ParameterType.TEXT],
                    notblank=True,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.FIXED_VAR_CDE_ENUMS, source="visitID"
                    ),
                ),
                "strategy": ParameterSpecification(
                    label="strategy",
                    desc="strategy",
                    types=[ParameterType.DICT],
                    notblank=True,
                    multiple=False,
                    dict_keys_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_NAMES, source=["x", "y"]
                    ),
                    dict_values_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST, source=["diff", "first", "second"]
                    ),
                ),
            },
        )
