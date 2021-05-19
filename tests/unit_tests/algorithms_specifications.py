from mipengine.controller.algorithms_specifications import AlgorithmSpecifications
from mipengine.controller.algorithms_specifications import AlgorithmsSpecifications
from mipengine.controller.algorithms_specifications import GenericParameterSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecifications

algorithms_specifications = AlgorithmsSpecifications()
algorithms_specifications.crossvalidation = None
algorithms_specifications.enabled_algorithms = {
    "test_algorithm1": AlgorithmSpecifications(
        name="test algorithm1",
        desc="test algorithm1",
        label="test algorithm1",
        enabled=True,
        inputdata=InputDataSpecifications(
            x=InputDataSpecification(
                label="features",
                desc="Features",
                types=["real"],
                stattypes=["numerical"],
                notblank=True,
                multiple=True,
                enumslen=None,
            ),
            y=InputDataSpecification(
                label="target",
                desc="Target variable",
                types=["text"],
                stattypes=["nominal"],
                notblank=True,
                multiple=False,
                enumslen=2,
            ),
        ),
        parameters={
            "parameter1": GenericParameterSpecification(
                label="paremeter1",
                desc="parameter 1",
                type="real",
                notblank=True,
                multiple=True,
                default=1,
                enums=[1, 2, 3],
            ),
            "parameter2": GenericParameterSpecification(
                label="paremeter2",
                desc="parameter 2",
                type="real",
                notblank=False,
                multiple=False,
                default=None,
                min=2,
                max=5,
            ),
        },
        flags={"formula": False, "crossvalidation": False},
    )
}
