from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

from dataclasses_json import dataclass_json

from mipengine.controller.algorithms_specifications import AlgorithmSpecifications
from mipengine.controller.algorithms_specifications import ParameterSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecifications
from mipengine.controller.algorithms_specifications import algorithms_specifications


@dataclass_json
@dataclass
class InputDataSpecificationDTO:
    """
    InputDataSpecificationDTO is different from the InputDataSpecification
    on the stattypes field.
    It is optional on the DTOs, due to the datasets and data_model parameters.
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
    containing data_model, dataset and filter.
    """

    data_model: InputDataSpecificationDTO
    datasets: InputDataSpecificationDTO
    filter: InputDataSpecificationDTO
    x: Optional[InputDataSpecificationDTO] = None
    y: Optional[InputDataSpecificationDTO] = None

    def __init__(self, input_data_spec: InputDataSpecifications):
        if input_data_spec.x:
            self.x = InputDataSpecificationDTO(
                label=input_data_spec.x.label,
                desc=input_data_spec.x.desc,
                types=input_data_spec.x.types,
                notblank=input_data_spec.x.notblank,
                multiple=input_data_spec.x.multiple,
                stattypes=input_data_spec.x.stattypes,
                enumslen=input_data_spec.x.enumslen,
            )
        if input_data_spec.y:
            self.y = InputDataSpecificationDTO(
                label=input_data_spec.y.label,
                desc=input_data_spec.y.desc,
                types=input_data_spec.y.types,
                notblank=input_data_spec.y.notblank,
                multiple=input_data_spec.y.multiple,
                stattypes=input_data_spec.y.stattypes,
                enumslen=input_data_spec.y.enumslen,
            )
        self.data_model = InputDataSpecificationDTO(
            label="data_model of the data.",
            desc="The data_model that the algorithm will run on.",
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
class ParameterSpecificationDTO(ParameterSpecification):
    """
    ParameterDTO is identical to the ParameterSpecification
    but exists for consistency and future use if needed.
    """

    pass


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
    parameters: Optional[Dict[str, ParameterSpecification]] = None

    def __init__(
        self,
        algorithm: AlgorithmSpecifications,
    ):
        self.name = algorithm.name
        self.desc = algorithm.desc
        self.label = algorithm.label
        self.parameters = algorithm.parameters
        self.inputdata = InputDataSpecificationsDTO(algorithm.inputdata)


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
