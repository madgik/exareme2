from dataclasses import dataclass
from typing import List, Optional, Dict

from dataclasses_json import dataclass_json

from controller.algorithms import GenericParameter, Algorithm, CROSSVALIDATION_ALGORITHM_NAME


@dataclass_json
@dataclass
class InputDataParameterDTO:
    """
    InputDataParameterDTO is different from the InputDataParameter
    on the stattypes field.
    It is optional on the DTO, due to the dataset and pathology parameters.
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
class CrossValidationAlgorithmDTO:
    """
    CrossValidationAlgorithmDTO is different from the Algorithm class
    because it doesn't have the enabled flag and the name.
    """
    desc: str
    label: str
    parameters: Optional[Dict[str, GenericParameter]] = None


@dataclass_json
@dataclass
class AlgorithmDTO:
    """
    AlgorithmDTO is different from the Algorithm class
    because it doesn't have the enabled flag and
    the crossvalidation algorithm is nested, not a flag.
    """
    name: str
    desc: str
    label: str
    inputdata: Optional[Dict[str, InputDataParameterDTO]] = None
    parameters: Optional[Dict[str, GenericParameter]] = None
    crossvalidation: Optional[CrossValidationAlgorithmDTO] = None

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
        self.inputdata["pathology"] = get_pathology_parameter()
        self.inputdata["dataset"] = get_dataset_parameter()
        self.inputdata["filter"] = get_filter_parameter()

        # Adding the crossvalidation algorithm as a nested algorithm
        if (CROSSVALIDATION_ALGORITHM_NAME in algorithm.flags.keys()
                and algorithm.flags[CROSSVALIDATION_ALGORITHM_NAME]):
            self.crossvalidation = CrossValidationAlgorithmDTO(
                crossvalidation.desc,
                crossvalidation.label,
                crossvalidation.parameters,
            )
