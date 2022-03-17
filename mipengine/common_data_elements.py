import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from dataclasses_json import dataclass_json

DATA_MODEL_METADATA_FILENAME = "CDEsMetadata.json"


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
    MetadataGroup is used to map the data_model metadata .json to an object.
    """

    code: str
    label: str
    version: Optional[str] = None
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
    data_models: Dict[str, Dict[str, CommonDataElement]]

    def __init__(self, cdes_metadata_path: str = None):
        self.data_models = {}

        if not cdes_metadata_path:
            return

        cdes_metadata_path = Path(cdes_metadata_path)

        cdes_data_model_metadata_folders = [
            data_model_folder
            for data_model_folder in cdes_metadata_path.iterdir()
            if data_model_folder.is_dir()
        ]
        for data_model_metadata_folder in cdes_data_model_metadata_folders:
            data_model_metadata_filepath = (
                data_model_metadata_folder / DATA_MODEL_METADATA_FILENAME
            )
            try:
                with open(data_model_metadata_filepath) as file:
                    contents = file.read()
                    data_model_metadata = MetadataGroup.from_json(contents)
                self.data_models[
                    f"{data_model_metadata.code}:{data_model_metadata.version}"
                ] = {
                    variable.code: CommonDataElement(variable)
                    for group in data_model_metadata
                    for variable in group.variables
                }
            except Exception as e:
                logging.error(
                    f"Error parsing metadata file: {data_model_metadata_filepath}"
                )
                raise e