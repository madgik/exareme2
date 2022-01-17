import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from dataclasses_json import dataclass_json

PATHOLOGY_METADATA_FILENAME = "CDEsMetadata.json"


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
    is_categorical: bool
    enumerations: Optional[Set] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __init__(self, variable: MetadataVariable):
        self.label = variable.label
        self.sql_type = variable.sql_type
        self.is_categorical = variable.isCategorical
        if variable.enumerations:
            self.enumerations = {
                enumeration.code for enumeration in variable.enumerations
            }
        self.min = variable.min
        self.max = variable.max


class CommonDataElements:
    pathologies: Dict[str, Dict[str, CommonDataElement]]

    def __init__(self, cdes_metadata_path: str = None):
        self.pathologies = {}

        if not cdes_metadata_path:
            return

        cdes_metadata_path = Path(cdes_metadata_path)

        cdes_pathology_metadata_folders = [
            pathology_folder
            for pathology_folder in cdes_metadata_path.iterdir()
            if pathology_folder.is_dir()
        ]

        for pathology_metadata_folder in cdes_pathology_metadata_folders:
            pathology_metadata_filepath = (
                pathology_metadata_folder / PATHOLOGY_METADATA_FILENAME
            )
            try:
                with open(pathology_metadata_filepath) as file:
                    contents = file.read()
                    pathology_metadata = MetadataGroup.from_json(contents)
                self.pathologies[pathology_metadata.code] = {
                    variable.code: CommonDataElement(variable)
                    for group in pathology_metadata
                    for variable in group.variables
                }
            except Exception as e:
                logging.error(
                    f"Error parsing metadata file: {pathology_metadata_filepath}"
                )
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
