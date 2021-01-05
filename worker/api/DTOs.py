import json
from abc import ABC
from enum import Enum

from typing import List, Union


class DTO(ABC):
    @classmethod
    def from_json(cls, json_string):
        json_dict = json.loads(json_string)
        return cls(**json_dict)


class TableSchemaTypes(Enum):
    INT = 'INT'
    FLOAT = 'FLOAT'
    TEXT = 'TEXT'


class TableIdentifier(DTO):

    def __init__(self, name: str, schema: List[TableSchemaTypes], location: str):
        self.name = name
        self.schema = schema
        self.location = location

    def serialize(self):
        return {
            'name': self.name,
            'schema': [schema_type.name for schema_type in self.schema],
            'location': self.location
        }


class DistributedTableIdentifier:
    def __init__(self, table_identifiers: List[TableIdentifier]):
        self.table_identifiers = table_identifiers

    def serialize(self):
        return [
            table_identifier.serialize()
            for table_identifier in self.table_identifiers
        ]


class TableData(DTO):
    def __init__(self,
                 schema: List[TableSchemaTypes],
                 data: List[Union[int, str, float, bool]]):
        self.schema = schema
        self.data = data


class TableView(DTO):
    def __init__(self, columns: List[str], datasets: List[str], filters: str):
        self.columns = columns
        self.datasets = datasets
        self.filters = filters  # TODO proper filters object, not string


class UDF(DTO):
    def __init__(self,
                 name: str,
                 input: List[
                     Union[
                         str,
                         int,
                         float,
                         bool,
                         TableIdentifier,
                         DistributedTableIdentifier
                     ]]):
        self.name = name
        self.input = input
