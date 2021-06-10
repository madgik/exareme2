import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from dataclasses_json import dataclass_json

from mipengine.common.resources import pathologies_metadata


@dataclass_json
@dataclass
class MetadataEnumeration:
    code: str
    label: str


@dataclass_json
@dataclass
class MetadataVariable:
    code: str
    label: str
    sql_type: str
    isCategorical: bool
    enumerations: Optional[List[MetadataEnumeration]] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self):
        allowed_types = {"int", "real", "text"}
        if self.sql_type not in allowed_types:
            raise ValueError(
                f"Metadata sql_type can be one of the following: {allowed_types}"
            )


@dataclass_json
@dataclass
class MetadataGroup:
    """
    MetadataGroup is used to map the pathology metadata .json to an object.
    """

    code: str
    label: str
    variables: Optional[List[MetadataVariable]] = field(default_factory=list)
    groups: Optional[List["MetadataGroup"]] = field(default_factory=list)

    def __iter__(self):
        yield self
        for subgroup in self.groups:
            yield from subgroup


@dataclass
class CommonDataElement:
    label: str
    sql_type: str
    categorical: bool
    enumerations: Optional[Set] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __init__(self, variable: MetadataVariable):
        self.label = variable.label
        self.sql_type = variable.sql_type
        self.categorical = variable.isCategorical
        if variable.enumerations:
            self.enumerations = {
                enumeration.code for enumeration in variable.enumerations
            }
        self.min = variable.min
        self.max = variable.max


class CommonDataElements:
    pathologies: Dict[str, Dict[str, CommonDataElement]]

    def __init__(self):
        metadata_path = Path(pathologies_metadata.__file__).parent

        self.pathologies = {}
        for pathology_metadata_filepath in metadata_path.glob("*.json"):
            try:
                pathology_metadata: MetadataGroup = MetadataGroup.from_json(
                    open(pathology_metadata_filepath).read()
                )
                self.pathologies[pathology_metadata.code] = {
                    variable.code: CommonDataElement(variable)
                    for group in pathology_metadata
                    for variable in group.variables
                }
            except Exception as e:
                logging.error(f"Parsing metadata file: {pathology_metadata_filepath}")
                raise e
                # Adding the subject code cde that doesn't exist in the metadata
            self.pathologies[pathology_metadata.code][
                "subjectcode"
            ] = CommonDataElement(
                MetadataVariable(
                    code="subjectcode",
                    label="The unique identifier of the record",
                    sql_type="text",
                    isCategorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                )
            )


common_data_elements = CommonDataElements()
