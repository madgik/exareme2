import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict

from dataclasses_json import dataclass_json

# TODO How can we read the pathologies' metadata  files without relative paths?
RELATIVE_METADATA_PATH = "./mipengine/controller/resources/pathologies_metadata"


@dataclass_json
@dataclass
class MetadataEnumeration:
    code: str
    label: str


@dataclass_json
@dataclass
class MetadataVariable:
    code: str
    sql_type: str
    isCategorical: bool
    enumerations: Optional[List[MetadataEnumeration]] = None

    def __post_init__(self):
        allowed_types = {"int", "real", "text"}
        if self.sql_type not in allowed_types:
            raise ValueError(f"Metadata sql_type can be one of the following: {allowed_types}")


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


@dataclass
class CommonDataElement:
    sql_type: str
    categorical: bool
    enumerations: Optional[Set] = None

    def __init__(self, variable: MetadataVariable):
        self.sql_type = variable.sql_type
        self.categorical = variable.isCategorical
        if variable.enumerations:
            self.enumerations = {enumeration.code for enumeration in variable.enumerations}


@dataclass
class CommonDataElements:
    pathologies: Dict[str, Dict[str, CommonDataElement]]

    def __init__(self):

        def iterate_metadata_groups(metadata_group: MetadataGroup) -> Dict[str, CommonDataElement]:
            group_elements: Dict[str, CommonDataElement] = {}
            for variable in metadata_group.variables:
                group_elements[variable.code] = CommonDataElement(variable)
            for sub_group in metadata_group.groups:
                group_elements.update(iterate_metadata_groups(sub_group))
            return group_elements

        pathology_metadata_files = [os.path.join(RELATIVE_METADATA_PATH, json_file)
                                    for json_file in os.listdir(RELATIVE_METADATA_PATH)
                                    if json_file.endswith('.json')]

        self.pathologies = {}
        for pathology_metadata_file in pathology_metadata_files:
            try:
                pathology_metadata = MetadataGroup.from_json(open(pathology_metadata_file).read())
                self.pathologies[pathology_metadata.code] = iterate_metadata_groups(pathology_metadata)

            except Exception as e:
                logging.error(f"Parsing metadata file: {pathology_metadata_file}")
                raise e


CommonDataElements()
