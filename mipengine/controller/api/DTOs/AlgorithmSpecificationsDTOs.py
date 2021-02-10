from dataclasses import dataclass
from typing import List, Optional, Dict

from dataclasses_json import dataclass_json

from mipengine.controller.algorithms_specifications import CROSSVALIDATION_ALGORITHM_NAME, \
    GenericParameterSpecification, AlgorithmSpecifications, AlgorithmsSpecifications
from mipengine.controller.utils import Singleton

INPUTDATA_PATHOLOGY_PARAMETER_NAME = "pathology"
INPUTDATA_DATASET_PARAMETER_NAME = "dataset"
INPUTDATA_FILTERS_PARAMETER_NAME = "filter"
INPUTDATA_X_PARAMETER_NAME = "x"
INPUTDATA_Y_PARAMETER_NAME = "y"


@dataclass_json
@dataclass
class InputDataSpecificationDTO:
    """
    InputDataParameterDTO is different from the InputDataParameter
    on the stattypes field.
    It is optional on the DTOs, due to the dataset and pathology parameters.
    """
    label: str
    desc: str
    types: List[str]
    notblank: bool
    multiple: bool
    stattypes: Optional[List[str]] = None
    enumslen: Optional[int] = None


def get_pathology_parameter():
    return InputDataSpecificationDTO(
        "Pathology of the data.",
        "The pathology that the algorithm will run on.",
        ["text"],
        True,
        False,
        None,
        None,
    )


def get_dataset_parameter():
    return InputDataSpecificationDTO(
        "Set of data to use.",
        "The set of data to run the algorithm on.",
        ["text"],
        True,
        True,
        None,
        None,
    )


def get_filter_parameter():
    return InputDataSpecificationDTO(
        "Filter on the data.",
        "Features used in my algorithm.",
        ["jsonObject"],
        False,
        False,
        None,
        None,
    )


@dataclass_json
@dataclass
class GenericParameterSpecificationDTO(GenericParameterSpecification):
    """
    GenericParameterDTO is identical to the GenericParameterSpecification
    but exists for consistency and future use if needed.
    """
    pass


@dataclass_json
@dataclass
class CrossValidationSpecificationsDTO:
    """
    CrossValidationDTO is a nested object, that contains
    all the information need to run crossvalidation on an algorithm.
    """
    desc: str
    label: str
    parameters: Dict[str, GenericParameterSpecification]


@dataclass_json
@dataclass
class AlgorithmSpecificationDTO:
    """
    AlgorithmDTO is used to provide the UI the requirements
    of each algorithm.
    System variables are added and unnecessary fields are removed.
    """
    name: str
    desc: str
    label: str
    inputdata: Dict[str, InputDataSpecificationDTO]
    parameters: Optional[Dict[str, GenericParameterSpecification]] = None
    crossvalidation: Optional[CrossValidationSpecificationsDTO] = None

    def __init__(self, algorithm: AlgorithmSpecifications, crossvalidation: AlgorithmSpecifications):
        self.name = algorithm.name
        self.desc = algorithm.desc
        self.label = algorithm.label
        self.parameters = algorithm.parameters
        self.inputdata = {name: InputDataSpecificationDTO(
            parameter.label,
            parameter.desc,
            parameter.types,
            parameter.notblank,
            parameter.multiple,
            parameter.stattypes,
            parameter.enumslen,
        )
            for name, parameter in algorithm.inputdata.items()
        }

        # Adding the 3 "system" input data parameters
        self.inputdata[INPUTDATA_PATHOLOGY_PARAMETER_NAME] = get_pathology_parameter()
        self.inputdata[INPUTDATA_DATASET_PARAMETER_NAME] = get_dataset_parameter()
        self.inputdata[INPUTDATA_FILTERS_PARAMETER_NAME] = get_filter_parameter()

        # Adding the crossvalidation algorithm as a nested algorithm
        if (CROSSVALIDATION_ALGORITHM_NAME in algorithm.flags.keys()
                and algorithm.flags[CROSSVALIDATION_ALGORITHM_NAME]):
            self.crossvalidation = CrossValidationSpecificationsDTO(
                crossvalidation.desc,
                crossvalidation.label,
                crossvalidation.parameters,
            )


class AlgorithmSpecificationsDTOs(metaclass=Singleton):
    algorithms_list = List[AlgorithmSpecificationDTO]
    algorithms_dict = Dict[str, AlgorithmSpecificationDTO]

    def __init__(self):
        algorithms_specifications = AlgorithmsSpecifications()
        self.algorithms_list = [AlgorithmSpecificationDTO(algorithm, algorithms_specifications.crossvalidation)
                                for algorithm in algorithms_specifications.enabled_algorithms.values()]

        self.algorithms_dict = {
            algorithm.name: AlgorithmSpecificationDTO(algorithm, algorithms_specifications.crossvalidation)
            for algorithm in algorithms_specifications.enabled_algorithms.values()}
