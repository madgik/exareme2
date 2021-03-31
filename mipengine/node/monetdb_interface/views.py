from typing import List

from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface import common_action
from mipengine.node.monetdb_interface.common_action import connection
from mipengine.node.monetdb_interface.common_action import cursor


def get_views_names(context_id: str) -> List[str]:
    return common_action.get_tables_names("view", context_id)


@validate_identifier_names
def create_view(view_name: str, pathology: str, datasets: List[str], columns: List[str], filters: str):
    filter_clause = ""
    if filters:
        filter_clause = f"AND {build_filter_clause(filters)}"
    dataset_names = ','.join(f"'{dataset}'" for dataset in datasets)
    columns = ', '.join(columns)
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
        rules = rules["rules"]
        return f" {cond} ".join([build_filter_clause(rule) for rule in rules])
    elif "id" in rules:
        var_name = rules["id"]
        op = FILTER_OPERATORS[rules["operator"]]
        val = rules["value"]
        if rules["input"] == "text":
            val = f"'{val}'"
        return op(var_name, val)
