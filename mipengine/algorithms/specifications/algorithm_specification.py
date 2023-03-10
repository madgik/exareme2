from enum import Enum
from enum import unique
from typing import Dict
from typing import Optional

from mipengine.algorithms.specifications.inputdata_specification import (
    InputDataSpecifications,
)
from mipengine.algorithms.specifications.parameter_specification import (
    ParameterSpecification,
)
from mipengine.algorithms.specifications.pipeline_step_specification import (
    PipelineStepSpecification,
)


@unique
class AlgorithmName(Enum):
    ANOVA = "anova"
    ANOVA_ONEWAY = "anova_oneway"
    DESCRIPTIVE_STATS = "descriptive_stats"
    LINEAR_REGRESSION = "linear_regression"
    LINEAR_REGRESSION_CV = "linear_regression_cv"
    LOGISTIC_REGRESSION = "logistic_regression"
    LOGISTIC_REGRESSION_CV = "logistic_regression_cv"
    MULTIPLE_HISTOGRAMS = "multiple_histograms"
    PCA = "pca"
    PEARSON_CORRELATION = "pearson_correlation"
    TTEST_INDEPENDENT = "ttest_independent"
    TTEST_ONESAMPLE = "ttest_onesample"
    TTEST_PAIRED = "ttest_paired"


class AlgorithmSpecification(PipelineStepSpecification):
    name: str
    desc: str
    label: str
    enabled: bool
    inputdata: InputDataSpecifications
    parameters: Optional[Dict[str, ParameterSpecification]]
    flags: Optional[Dict[str, bool]]
