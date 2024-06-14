import pandas as pd


def apply_filter(df, filter_rule):
    condition = filter_rule.get("condition", "AND")
    rules = filter_rule.get("rules", [])

    if condition == "AND":
        filtered_df = df
        for rule in rules:
            filtered_df = apply_single_rule(filtered_df, rule)
        return filtered_df
    elif condition == "OR":
        filtered_dfs = []
        for rule in rules:
            filtered_dfs.append(apply_single_rule(df, rule))
        return pd.concat(filtered_dfs).drop_duplicates().reset_index(drop=True)


def apply_single_rule(df, rule):
    if "condition" in rule:
        return apply_filter(df, rule)
    else:
        id = rule["id"]
        operator = rule["operator"]
        value = rule.get("value", None)

        operator_functions = {
            "equal": lambda df, id, value: df[df[id] == value],
            "not_equal": lambda df, id, value: df[df[id] != value],
            "greater": lambda df, id, value: df[df[id] > value],
            "less": lambda df, id, value: df[df[id] < value],
            "greater_or_equal": lambda df, id, value: df[df[id] >= value],
            "less_or_equal": lambda df, id, value: df[df[id] <= value],
            "between": lambda df, id, value: df[df[id].between(value[0], value[1])],
            "not_between": lambda df, id, value: df[
                ~df[id].between(value[0], value[1])
            ],
            "is_not_null": lambda df, id, value: df[df[id].notnull()],
            "is_null": lambda df, id, value: df[df[id].isnull()],
            "in": lambda df, id, value: df[df[id].isin(value)],
        }

        if operator in operator_functions:
            return operator_functions[operator](df, id, value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")
