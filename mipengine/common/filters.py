from mipengine.common.common_data_elements import common_data_elements

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


def build_filter_clause(rules):
    """
    Converts and returns a given filter in jQuery format to an sql clause. This function will not check the validity of the
    filters (the only exception is the SQL Injection which will be handled by padantic)
    """
    if rules is None:
        return

    if "condition" in rules:
        check_proper_condition(rules["condition"])
        cond = rules["condition"]
        rules = rules["rules"]
        return f" {cond} ".join([build_filter_clause(rule) for rule in rules])

    if "id" in rules:
        column_name = rules["id"]
        op = FILTER_OPERATORS[rules["operator"]]
        val = rules["value"]
        if rules["type"] == "string":
            val = [f"'{item}'" for item in val] if isinstance(val, list) else f"'{val}'"
        return op(column_name, val)

    raise ValueError(f"Invalid filters format. Filters did not contain the keys: 'condition' or 'id'.")


def validate_proper_filter(pathology_name: str, rules):
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

    check_filter_type(rules)
    check_pathology_exists(pathology_name)

    if "condition" in rules:
        check_proper_condition(rules["condition"])
        rules = rules["rules"]
        [validate_proper_filter(pathology_name, rule) for rule in rules]
    elif "id" in rules:
        column_name = rules["id"]
        val = rules["value"]
        check_proper_operator(rules["operator"])
        check_column_exists(pathology_name, column_name)
        check_value_type(pathology_name, column_name, val)
    else:
        raise ValueError(f"Invalid filters format. Filters did not contain the keys: 'condition' or 'id'.")


def check_filter_type(rules):
    if not isinstance(rules, dict):
        raise TypeError(f"Filter type can only be dict but was:{type(rules)}")


def check_proper_condition(condition: str):
    if condition not in ["OR", "AND"]:
        raise ValueError(f"Condition: {condition} is not acceptable.")


def check_proper_operator(operator: str):
    if operator not in FILTER_OPERATORS:
        raise ValueError(f"Operator: {operator} is not acceptable.")


def check_column_exists(pathology_name: str, column: str):
    pathology_common_data_elements = common_data_elements.pathologies[pathology_name]
    if column not in pathology_common_data_elements.keys():
        raise KeyError(f"Column {column} does not exist in the metadata of the {pathology_name}!")


def check_pathology_exists(pathology_name: str):
    if pathology_name not in common_data_elements.pathologies.keys():
        raise KeyError(f"Pathology:{pathology_name} does not exist in the metadata!")


def _convert_mip_type_to_class_type(mip_type: str):
    """
    Converts MIP's types to the according class.
    """
    type_mapping = {
        "int": int,
        "real": float,
        "text": str,
    }

    if mip_type not in type_mapping.keys():
        raise KeyError(f"MIP type '{mip_type}' cannot be converted to a python class type.")

    return type_mapping.get(mip_type)


def check_value_type(pathology_name: str, column: str, value):
    if value is None:
        return
    elif isinstance(value, list):
        for item in value:
            check_value_type(pathology_name, column, item)
    elif isinstance(value, (int, str, float)):
        pathology_common_data_elements = common_data_elements.pathologies[pathology_name]
        column_sql_type = pathology_common_data_elements[column].sql_type

        if type(value) is not _convert_mip_type_to_class_type(column_sql_type):
            raise TypeError(
                f"{column}'s type: {column_sql_type} was different from the type of the given value:{type(value)}"
            )
    else:
        raise TypeError(
            f"Value {value} should be of type int, str, float but was {type(value)}"
        )