from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

from dataclasses_json import dataclass_json

from mipengine.controller.algorithms_specifications import AlgorithmSpecifications
from mipengine.controller.algorithms_specifications import GenericParameterSpecification
from mipengine.controller.algorithms_specifications import algorithms_specifications


@dataclass_json
@dataclass
class InputDataSpecificationDTO:
    """
    InputDataSpecificationDTO is different from the InputDataSpecification
    on the stattypes field.
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

    def __init__(self, algorithm: AlgorithmSpecifications):
        self.name = algorithm.name
        self.desc = algorithm.desc
        self.label = algorithm.label
        self.parameters = algorithm.parameters
        self.inputdata = {}
        for inputdata_name, inputdata_spec in algorithm.inputdata.items():
            self.inputdata[inputdata_name] = InputDataSpecificationDTO(
                label=inputdata_spec.label,
                desc=inputdata_spec.desc,
                types=inputdata_spec.types,
                notblank=inputdata_spec.notblank,
                multiple=inputdata_spec.multiple,
                stattypes=inputdata_spec.stattypes,
                enumslen=inputdata_spec.enumslen,
            )
        self.inputdata["pathology"] = InputDataSpecificationDTO(
            label="Pathology of the data.",
            desc="The pathology that the algorithm will run on.",
            types=["text"],
            notblank=True,
            multiple=False,
            stattypes=None,
            enumslen=None,
        )
        self.inputdata["datasets"] = InputDataSpecificationDTO(
            label="Set of data to use.",
            desc="The set of data to run the algorithm on.",
            types=["text"],
            notblank=True,
            multiple=True,
            stattypes=None,
            enumslen=None,
        )
        self.inputdata["filter"] = InputDataSpecificationDTO(
            label="filter on the data.",
            desc="Features used in my algorithm.",
            types=["jsonObject"],
            notblank=False,
            multiple=False,
            stattypes=None,
            enumslen=None,
        )


class AlgorithmSpecificationsDTOs:
    algorithms_list = List[AlgorithmSpecificationDTO]
    algorithms_dict = Dict[str, AlgorithmSpecificationDTO]

    def __init__(self):
        self.algorithms_list = [
            AlgorithmSpecificationDTO(algorithm)
            for algorithm in algorithms_specifications.enabled_algorithms.values()
        ]

        self.algorithms_dict = {
            algorithm.name: AlgorithmSpecificationDTO(algorithm)
            for algorithm in algorithms_specifications.enabled_algorithms.values()
        }


algorithm_specificationsDTOs = AlgorithmSpecificationsDTOs()
