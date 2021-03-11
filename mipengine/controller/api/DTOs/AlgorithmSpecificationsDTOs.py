from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

from dataclasses_json import dataclass_json

from mipengine.controller.algorithms_specifications import AlgorithmSpecifications
from mipengine.controller.algorithms_specifications import CROSSVALIDATION_ALGORITHM_NAME
from mipengine.controller.algorithms_specifications import GenericParameterSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecifications
from mipengine.controller.algorithms_specifications import algorithms_specifications


@dataclass_json
@dataclass
class InputDataSpecificationDTO:
    """
    InputDataSpecificationDTO is different from the InputDataSpecification
    on the stattypes field.
    It is optional on the DTOs, due to the datasets and pathology parameters.
    """
    label: str
    desc: str
    types: List[str]
    notblank: bool
    multiple: bool
    stattypes: Optional[List[str]] = None
    enumslen: Optional[int] = None


@dataclass_json
@dataclass
class InputDataSpecificationsDTO:
    """
    InputDataSpecificationsDTO is a superset of InputDataSpecifications
    containing pathology, dataset and filter.
    """
    pathology: InputDataSpecificationDTO
    datasets: InputDataSpecificationDTO
    filter: InputDataSpecificationDTO
    x: Optional[InputDataSpecificationDTO] = None
    y: Optional[InputDataSpecificationDTO] = None

    def __init__(self, input_data_spec: InputDataSpecifications):
        self.x = InputDataSpecificationDTO(
            label=input_data_spec.x.label,
            desc=input_data_spec.x.desc,
            types=input_data_spec.x.types,
            notblank=input_data_spec.x.notblank,
            multiple=input_data_spec.x.multiple,
            stattypes=input_data_spec.x.stattypes,
            enumslen=input_data_spec.x.enumslen,
        )
        self.y = InputDataSpecificationDTO(
            label=input_data_spec.y.label,
            desc=input_data_spec.y.desc,
            types=input_data_spec.y.types,
            notblank=input_data_spec.y.notblank,
            multiple=input_data_spec.y.multiple,
            stattypes=input_data_spec.y.stattypes,
            enumslen=input_data_spec.y.enumslen,
        )
        self.pathology = InputDataSpecificationDTO(
            label="Pathology of the data.",
            desc="The pathology that the algorithm will run on.",
            types=["text"],
            notblank=True,
            multiple=False,
            stattypes=None,
            enumslen=None,
        )
        self.datasets = InputDataSpecificationDTO(
            label="Set of data to use.",
            desc="The set of data to run the algorithm on.",
            types=["text"],
            notblank=True,
            multiple=True,
            stattypes=None,
            enumslen=None,
        )
        self.filter = InputDataSpecificationDTO(
            label="filter on the data.",
            desc="Features used in my algorithm.",
            types=["jsonObject"],
            notblank=False,
            multiple=False,
            stattypes=None,
            enumslen=None,
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
    inputdata: InputDataSpecificationsDTO
    parameters: Optional[Dict[str, GenericParameterSpecification]] = None
    crossvalidation: Optional[CrossValidationSpecificationsDTO] = None

    def __init__(self, algorithm: AlgorithmSpecifications, crossvalidation: AlgorithmSpecifications):
        self.name = algorithm.name
        self.desc = algorithm.desc
        self.label = algorithm.label
        self.parameters = algorithm.parameters
        self.inputdata = InputDataSpecificationsDTO(algorithm.inputdata)

        # Adding the crossvalidation algorithm as a nested algorithm
        if (CROSSVALIDATION_ALGORITHM_NAME in algorithm.flags.keys()
                and algorithm.flags[CROSSVALIDATION_ALGORITHM_NAME]):
            self.crossvalidation = CrossValidationSpecificationsDTO(
                crossvalidation.desc,
                crossvalidation.label,
                crossvalidation.parameters,
            )


class AlgorithmSpecificationsDTOs:
    algorithms_list = List[AlgorithmSpecificationDTO]
    algorithms_dict = Dict[str, AlgorithmSpecificationDTO]

    def __init__(self):
        self.algorithms_list = [AlgorithmSpecificationDTO(algorithm, algorithms_specifications.crossvalidation)
                                for algorithm in algorithms_specifications.enabled_algorithms.values()]

        self.algorithms_dict = {
            algorithm.name: AlgorithmSpecificationDTO(algorithm, algorithms_specifications.crossvalidation)
            for algorithm in algorithms_specifications.enabled_algorithms.values()}


algorithm_specificationsDTOs = AlgorithmSpecificationsDTOs()
