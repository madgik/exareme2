from mipengine.common.common_data_elements import common_data_elements

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
        raise KeyError(f"Column {column} does not exist in the metadata!")


def _convert_mip_type_to_class_type(type: str):
    """
    Converts MIP's types to the according class.
    """
    type_mapping = {
        "int": int,
        "real": float,
        "text": str,
    }

    if type not in type_mapping.keys():
        raise ValueError(
            f"MIP type '{type}' cannot be converted to a python class type."
        )

    return type_mapping.get(type)


def check_if_value_has_proper_type(pathology_name: str, column: str, value):
    pathology_common_data_elements = common_data_elements.pathologies[pathology_name]
    column_sql_type = pathology_common_data_elements[column].sql_type

    if type(value) is not _convert_mip_type_to_class_type(column_sql_type):
        raise KeyError(
            f"Value {value} should be {column_sql_type} but was {type(value)}"
        )
