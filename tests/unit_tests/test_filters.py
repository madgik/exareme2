import pytest

from mipengine.common.filters import build_filter_clause, validate_proper_filter

PATHOLOGY = "tbi"

equal_filter = {
    "condition": "AND",
    "rules": [
        {
            "id": "age_value",
            "field": "age_value",
            "type": "int",
            "input": "number",
            "operator": "equal",
            "value": 17,
        }
    ],
    "valid": True,
}
equal_result = "age_value = 17"

not_equal_filter = {
    "condition": "AND",
    "rules": [
        {
            "id": "pupil_reactivity_right_eye_result",
            "field": "pupil_reactivity_right_eye_result",
            "type": "string",
            "input": "text",
            "operator": "not_equal",
            "value": "Nonreactive",
        }
    ],
    "valid": True,
}
not_equal_result = "pupil_reactivity_right_eye_result <> 'Nonreactive'"

equal_or_not_equal_filter = {
    "condition": "OR",
    "rules": [
        {
            "id": "age_value",
            "field": "age_value",
            "type": "int",
            "input": "number",
            "operator": "equal",
            "value": 17,
        },
        {
            "id": "pupil_reactivity_right_eye_result",
            "field": "pupil_reactivity_right_eye_result",
            "type": "string",
            "input": "text",
            "operator": "not_equal",
            "value": "Nonreactive",
        },
    ],
    "valid": True,
}
equal_or_not_equal_result = (
    "age_value = 17 OR pupil_reactivity_right_eye_result <> 'Nonreactive'"
)

equal_not_equal_cases = [
    (equal_filter, equal_result),
    (not_equal_filter, not_equal_result),
    (equal_or_not_equal_filter, equal_or_not_equal_result),
]


@pytest.mark.parametrize("test_input,expected", equal_not_equal_cases)
def test_equal_or_not_equal(test_input, expected):
    assert build_filter_clause(test_input) == expected
    validate_proper_filter(PATHOLOGY, test_input)


less_and_greater_filter = {
    "condition": "AND",
    "rules": [
        {
            "id": "age_value",
            "field": "age_value",
            "type": "int",
            "input": "number",
            "operator": "less",
            "value": 50,
        },
        {
            "id": "age_value",
            "field": "age_value",
            "type": "int",
            "input": "number",
            "operator": "greater",
            "value": 20,
        },
    ],
    "valid": True,
}

less_and_greater_result = "age_value < 50 AND age_value > 20"
less_and_greater_cases = [(less_and_greater_filter, less_and_greater_result)]


@pytest.mark.parametrize("test_input,expected", less_and_greater_cases)
def test_less_and_greater(test_input, expected):
    assert build_filter_clause(test_input) == expected
    validate_proper_filter(PATHOLOGY, test_input)


between_and_not_between_filter = {
    "condition": "AND",
    "rules": [
        {
            "id": "age_value",
            "field": "age_value",
            "type": "int",
            "input": "number",
            "operator": "not_between",
            "value": [60, 90],
        },
        {
            "id": "mortality_core",
            "field": "mortality_core",
            "type": "double",
            "input": "number",
            "operator": "between",
            "value": [0.3, 0.8],
        },
    ],
    "valid": True,
}

between_and_not_between_result = (
    "NOT age_value BETWEEN 60 AND 90 AND mortality_core BETWEEN 0.3 AND 0.8"
)
between_and_not_between_cases = [
    (between_and_not_between_filter, between_and_not_between_result)
]


@pytest.mark.parametrize("test_input,expected", between_and_not_between_cases)
def test_between_and_not_between(test_input, expected):
    assert build_filter_clause(test_input) == expected
    validate_proper_filter(PATHOLOGY, test_input)


is_null_or_is_not_null_filter = {
    "condition": "OR",
    "rules": [
        {
            "id": "gose_score",
            "field": "gose_score",
            "type": "text",
            "input": "text",
            "operator": "is_null",
            "value": None,
        },
        {
            "id": "gcs_total_score",
            "field": "gcs_total_score",
            "type": "int",
            "input": "number",
            "operator": "is_not_null",
            "value": None,
        },
    ],
    "valid": True,
}

is_null_or_is_not_null_result = "gose_score IS NULL OR gcs_total_score IS NOT NULL"
is_null_or_is_not_null_cases = [
    (is_null_or_is_not_null_filter, is_null_or_is_not_null_result)
]


@pytest.mark.parametrize("test_input,expected", is_null_or_is_not_null_cases)
def test_is_null_or_is_not_null(test_input, expected):
    assert build_filter_clause(test_input) == expected
    validate_proper_filter(PATHOLOGY, test_input)


all_filter = {
    "condition": "OR",
    "rules": [
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "gender_type",
                    "field": "gender_type",
                    "type": "string",
                    "input": "text",
                    "operator": "equal",
                    "value": "F",
                },
                {
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "age_value",
                            "field": "age_value",
                            "type": "int",
                            "input": "number",
                            "operator": "between",
                            "value": [20, 30],
                        },
                        {
                            "id": "gose_score",
                            "field": "gose_score",
                            "type": "text",
                            "input": "text",
                            "operator": "is_not_null",
                            "value": None,
                        },
                    ],
                },
            ],
        },
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "gender_type",
                    "field": "gender_type",
                    "type": "string",
                    "input": "text",
                    "operator": "not_equal",
                    "value": "F",
                },
                {
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "mortality_core",
                            "field": "mortality_core",
                            "type": "double",
                            "input": "number",
                            "operator": "greater",
                            "value": 0.5,
                        },
                        {
                            "id": "mortality_core",
                            "field": "mortality_core",
                            "type": "double",
                            "input": "number",
                            "operator": "less_or_equal",
                            "value": 0.8,
                        },
                    ],
                },
            ],
        },
    ],
    "valid": True,
}

all_result = (
    "gender_type = 'F' AND age_value BETWEEN 20 AND 30 AND gose_score IS NOT NULL OR gender_type <> 'F' AND "
    "mortality_core > 0.5 AND mortality_core <= 0.8"
)
all_cases = [(all_filter, all_result)]


@pytest.mark.parametrize("test_input,expected", all_cases)
def test_all(test_input, expected):
    assert build_filter_clause(test_input) == expected
    validate_proper_filter(PATHOLOGY, test_input)
