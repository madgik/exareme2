from dataclasses import dataclass
from typing import List, Optional, Dict

from dataclasses_json import dataclass_json

from controller.algorithms import GenericParameter, Algorithm, CROSSVALIDATION_ALGORITHM_NAME, Algorithms
from controller.utils import Singleton

INPUTDATA_PATHOLOGY_PARAMETER_NAME = "pathology"
INPUTDATA_DATASET_PARAMETER_NAME = "dataset"
INPUTDATA_FILTERS_PARAMETER_NAME = "filter"
INPUTDATA_X_PARAMETER_NAME = "x"
INPUTDATA_Y_PARAMETER_NAME = "y"


@dataclass_json
@dataclass
class InputDataParameterDTO:
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
    return InputDataParameterDTO(
        "Pathology of the data.",
        "The pathology that the algorithm will run on.",
        ["text"],
        True,
        False,
        None,
        None,
    )


def get_dataset_parameter():
    return InputDataParameterDTO(
        "Set of data to use.",
        "The set of data to run the algorithm on.",
        ["text"],
        True,
        True,
        None,
        None,
    )


def get_filter_parameter():
    return InputDataParameterDTO(
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
class GenericParameterDTO(GenericParameter):
    """
    GenericParameterDTO is identical to the GenericParameter
    but exists for consistency and future use if needed.
    """
    pass


@dataclass_json
@dataclass
class CrossValidationParametersDTO:
    """
    CrossValidationDTO is a nested object, that contains
    all the information need to run crossvalidation on an algorithm.
    """
    desc: str
    label: str
    parameters: Dict[str, GenericParameter]


@dataclass_json
@dataclass
class AlgorithmDTO:
    """
    AlgorithmDTO is used to provide the UI the requirements
    of each algorithm.
    System variables are added and unnecessary fields are removed.
    """
    name: str
    desc: str
    label: str
    inputdata: Dict[str, InputDataParameterDTO]
    parameters: Optional[Dict[str, GenericParameter]] = None
    crossvalidation: Optional[CrossValidationParametersDTO] = None

    def __init__(self, algorithm: Algorithm, crossvalidation: Algorithm):
        self.name = algorithm.name
        self.desc = algorithm.desc
        self.label = algorithm.label
        self.parameters = algorithm.parameters
        self.inputdata = {name: InputDataParameterDTO(
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
            self.crossvalidation = CrossValidationParametersDTO(
                crossvalidation.desc,
                crossvalidation.label,
                crossvalidation.parameters,
            )


class AlgorithmSpecifications(metaclass=Singleton):
    algorithms_list = List[AlgorithmDTO]
    algorithms_dict = Dict[str, AlgorithmDTO]

    def __init__(self):
        self.algorithms_list = [AlgorithmDTO(algorithm, Algorithms().crossvalidation)
                                for algorithm in Algorithms().available.values()]

        self.algorithms_dict = {algorithm.name: AlgorithmDTO(algorithm, Algorithms().crossvalidation)
                                for algorithm in Algorithms().available.values()}
