from typing import List

from mipengine.common.common_data_elements import common_data_elements
from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface import common_action
from mipengine.node.monetdb_interface.common_action import connection
from mipengine.node.monetdb_interface.common_action import cursor


def get_views_names(context_id: str) -> List[str]:
    return common_action.get_tables_names("view", context_id)


@validate_identifier_names
def create_view(view_name: str, pathology: str, datasets: List[str], columns: List[str], filters: str):
    filter_clause = ""
    if filters is not None:
        validate_proper_filter(pathology, filters)
        filter_clause = f"AND {build_filter_clause(filters)}"
    dataset_names = ','.join(f"'{dataset}'" for dataset in datasets)
    columns = ', '.join(columns)
    print(
        f"""CREATE VIEW {view_name}
         AS SELECT {columns} 
         FROM {pathology}_data 
         WHERE 
         dataset IN ({dataset_names}) 
         {filter_clause}""")
    cursor.execute(
        f"""CREATE VIEW {view_name}
         AS SELECT {columns} 
         FROM {pathology}_data 
         WHERE 
         dataset IN ({dataset_names}) 
         {filter_clause}""")
    connection.commit()


FILTER_OPERATORS = {
    "equal": lambda a, b: f"{a} = {b}",
    "not_equal": lambda a, b: f"{a} <> {b}",
    "less": lambda a, b: f"{a} < {b}",
    "greater": lambda a, b: f"{a} > {b}",
    "less_or_equal": lambda a, b: f"{a} <= {b}",
    "greater_or_equal": lambda a, b: f"{a} >= {b}",
    "between": lambda a, b: f"{a} BETWEEN {b[0]} AND {b[1]}",
    "not_between": lambda a, b: f"NOT {a} BETWEEN {b[0]} AND {b[1]}",
    "is_null": lambda a, b: f"{a} IS NULL",
    "is_not_null": lambda a, b: f"{a} IS NOT NULL",
}


def build_filter_clause(rules):
    if "condition" in rules:
        cond = rules["condition"]
        check_proper_condition(cond)
        rules = rules["rules"]
        return f" {cond} ".join([build_filter_clause(rule) for rule in rules])
    elif "id" in rules:
        var_name = rules["id"]
        op = FILTER_OPERATORS[rules["operator"]]
        val = rules["value"]
        if rules["input"] == "text":
            val = f"'{val}'"
        return op(var_name, val)


def validate_proper_filter(pathology_name: str, rules):
    if "condition" in rules:
        check_proper_condition(rules["condition"])
        rules = rules["rules"]
        [validate_proper_filter(pathology_name, rule) for rule in rules]
    elif "id" in rules:
        column_name = rules["id"]
        val = rules["value"]
        check_proper_operator(rules["operator"])
        check_if_column_exists(pathology_name, column_name)
        print(val)
        print(rules["operator"])
        if val is not None:
            if type(val) is list:
                for item in val:
                    check_if_value_has_proper_type(pathology_name, column_name, item)
            else:
                check_if_value_has_proper_type(pathology_name, column_name, val)


def check_proper_condition(condition: str):
    if condition not in ["OR", "AND"]:
        raise ValueError(f"Condition: {condition} is not acceptable.")


def check_proper_operator(operator: str):
    if operator not in FILTER_OPERATORS:
        raise ValueError(f"Operator: {operator} is not acceptable.")


def check_if_column_exists(pathology_name: str, column: str):
    pathology_common_data_elements = common_data_elements.pathologies[pathology_name]
    if column not in pathology_common_data_elements.keys():
        raise KeyError(f'Column {column} does not exist in the metadata!')


def _convert_type_name_to_class(type: str):
    """ Converts MonetDB's types to MIP Engine's types
        int ->  class int
        float  -> class float
        text  -> class str
        bool -> class bool
        clob -> class str
        """
    type_mapping = {
        "int": int,
        "float": float,
        "text": str,
        "bool": bool,
        "clob": str,
        "real": float,
    }

    if type not in type_mapping.keys():
        raise ValueError(f"Type {type} cannot be converted to class type.")

    return type_mapping.get(str(type).lower())


def check_if_value_has_proper_type(pathology_name: str, column: str, value):
    pathology_common_data_elements = common_data_elements.pathologies[pathology_name]
    column_sql_type = pathology_common_data_elements[column].sql_type

    if type(value) is not _convert_type_name_to_class(column_sql_type):
        raise KeyError(f'Value {value} should be {column_sql_type} but was {type(value)}')
