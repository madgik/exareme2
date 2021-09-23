from mipengine.common_data_elements import CommonDataElements
from mipengine.datatypes import convert_mip_type_to_python_type

FILTER_OPERATORS = {
    "equal": lambda column, value: f"{column} = {value}",
    "not_equal": lambda column, value: f"{column} <> {value}",
    "less": lambda column, value: f"{column} < {value}",
    "greater": lambda column, value: f"{column} > {value}",
    "less_or_equal": lambda column, value: f"{column} <= {value}",
    "greater_or_equal": lambda column, value: f"{column} >= {value}",
    "between": lambda column, values: f"{column} BETWEEN {values[0]} AND {values[1]}",
    "not_between": lambda column, value: f"NOT {column} BETWEEN {value[0]} AND {value[1]}",
    "is_null": lambda column, value: f"{column} IS NULL",
    "is_not_null": lambda column, value: f"{column} IS NOT NULL",
    "in": lambda column, values: f"{column} IN ({','.join(str(value) for value in values)})",
    "not_in": lambda column, values: f"{column} NOT IN ({','.join(str(value) for value in values)})",
}

__all__ = ["build_filter_clause", "validate_filter"]


def build_filter_clause(rules):
    """
    Converts and returns a given filter in jQuery format to an sql clause.
    This function will not check the validity of the
    filters (the only exception is the SQL Injection which will be handled by padantic)
    """
    if rules is None:
        return

    if "condition" in rules:
        _check_condition(rules["condition"])
        cond = rules["condition"]
        rules = rules["rules"]
        return f" {cond} ".join([build_filter_clause(rule) for rule in rules])

    if "id" in rules:
        column_name = rules["id"]
        op = FILTER_OPERATORS[rules["operator"]]
        value = _format_value_if_string(rules["type"], rules["value"])
        return op(column_name, value)

    raise ValueError(f"Filters did not contain the keys: 'condition' or 'id'.")


def validate_filter(
    common_data_elements: CommonDataElements, pathology_name: str, rules: dict
):
    """
    Validates a given filter in jQuery format.
    This function will check the validity of:
        1. The type of the filter
        2. The column name (if it exists in the metadata of the pathology)
        3. The pathology name (if the pathology exists)
        4. All the conditions that the filter contains
        5. All the operators that the filter contains
        6. The type of the given value (if the column of the pathology and the value are the same type)
    """
    if rules is None:
        return

    _check_filter_type(rules)
    _check_pathology_exists(common_data_elements, pathology_name)

    if "condition" in rules:
        _check_condition(rules["condition"])
        rules = rules["rules"]
        for rule in rules:
            validate_filter(common_data_elements, pathology_name, rule)
    elif "id" in rules:
        column_name = rules["id"]
        val = rules["value"]
        _check_operator(rules["operator"])
        _check_column_exists(common_data_elements, pathology_name, column_name)
        _check_value_type(common_data_elements, pathology_name, column_name, val)
    else:
        raise ValueError(
            f"Invalid filters format. Filters did not contain the keys: 'condition' or 'id'."
        )


def _format_value_if_string(column_type, val):
    if column_type == "string":
        return [f"'{item}'" for item in val] if isinstance(val, list) else f"'{val}'"
    return val


def _check_filter_type(rules):
    if not isinstance(rules, dict):
        raise TypeError(f"Filter type can only be dict but was:{type(rules)}")


def _check_condition(condition: str):
    if condition not in ["OR", "AND"]:
        raise ValueError(f"Condition: {condition} is not acceptable.")


def _check_operator(operator: str):
    if operator not in FILTER_OPERATORS:
        raise ValueError(f"Operator: {operator} is not acceptable.")


def _check_column_exists(common_data_elements, pathology_name: str, column: str):
    pathology_common_data_elements = common_data_elements.pathologies[pathology_name]
    if column not in pathology_common_data_elements.keys():
        raise KeyError(
            f"Column {column} does not exist in the metadata of the {pathology_name}!"
        )


def _check_pathology_exists(common_data_elements, pathology_name: str):
    if pathology_name not in common_data_elements.pathologies.keys():
        raise KeyError(f"Pathology:{pathology_name} does not exist in the metadata!")


def _check_value_type(common_data_elements, pathology_name: str, column: str, value):
    if value is None:
        return

    if isinstance(value, list):
        [
            _check_value_type(common_data_elements, pathology_name, column, item)
            for item in value
        ]
    elif isinstance(value, (int, str, float)):
        _check_value_column_same_type(
            common_data_elements, pathology_name, column, value
        )
    else:
        raise TypeError(
            f"Value {value} should be of type int, str, float but was {type(value)}"
        )


def _check_value_column_same_type(common_data_elements, pathology_name, column, value):
    pathology_common_data_elements = common_data_elements.pathologies[pathology_name]
    column_sql_type = pathology_common_data_elements[column].sql_type
    if type(value) is not convert_mip_type_to_python_type(column_sql_type):
        raise TypeError(
            f"{column}'s type: {column_sql_type} was different from the type of the given value:{type(value)}"
        )
