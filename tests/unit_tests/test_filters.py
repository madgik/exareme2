import pytest

from mipengine.filters import build_filter_clause, validate_proper_filter

PATHOLOGY = "tbi"

all_success_cases = [
    (
        {
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
        },
        "age_value = 17",
    ),
    (
        {
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
        },
        "pupil_reactivity_right_eye_result <> 'Nonreactive'",
    ),
    (
        {
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
        },
        "age_value = 17 OR pupil_reactivity_right_eye_result <> 'Nonreactive'",
    ),
    (
        {
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
        },
        "age_value < 50 AND age_value > 20",
    ),
    (
        {
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
        },
        "NOT age_value BETWEEN 60 AND 90 AND mortality_core BETWEEN 0.3 AND 0.8",
    ),
    (
        {
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
        },
        "gose_score IS NULL OR gcs_total_score IS NOT NULL",
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "age_value",
                    "field": "age_value",
                    "type": "integer",
                    "input": "number",
                    "operator": "in",
                    "value": [17, 19],
                }
            ],
            "valid": True,
        },
        "age_value IN (17,19)",
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "gender_type",
                    "field": "gender_type",
                    "type": "string",
                    "input": "text",
                    "operator": "in",
                    "value": ["F", "M"],
                }
            ],
            "valid": True,
        },
        "gender_type IN ('F','M')",
    ),
    (
        {
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
        },
        (
            "gender_type = 'F' AND age_value BETWEEN 20 AND 30 AND gose_score IS NOT NULL OR gender_type <> 'F' AND "
            "mortality_core > 0.5 AND mortality_core <= 0.8"
        ),
    ),
]


@pytest.mark.parametrize("test_input,expected", all_success_cases)
def test_build_filter_clause(test_input, expected):
    assert build_filter_clause(test_input) == expected


@pytest.mark.parametrize("test_input,expected", all_success_cases)
def test_validate_proper_filter(test_input, expected):
    validate_proper_filter(PATHOLOGY, test_input)


all_build_filter_clause_fail_cases = [
    (
        {
            "condition": "ANDOR",
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
        },
        ValueError,
    ),
    (0, TypeError),
    ("not_a_filter", ValueError),
    ({"data": 0}, ValueError),
    ({0}, ValueError),
]

all_validate_proper_filter_fail_cases = [
    (
        {
            "condition": "ANDOR",
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
        },
        ValueError,
    ),
    (0, TypeError),
    ("not_a_filter", TypeError),
    ({"data": 0}, ValueError),
    ({0}, TypeError),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "age_value",
                    "field": "age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "invalid operator",
                    "value": 17,
                }
            ],
            "valid": True,
        },
        ValueError,
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "invalid column",
                    "field": "age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "equal",
                    "value": 17,
                }
            ],
            "valid": True,
        },
        KeyError,
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "age_value",
                    "field": "age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "equal",
                    "value": "17",
                }
            ],
            "valid": True,
        },
        TypeError,
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "age_value",
                    "field": "age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "equal",
                    "value": True,
                }
            ],
            "valid": True,
        },
        TypeError,
    ),
]


@pytest.mark.parametrize(
    "test_input,expected_error", all_validate_proper_filter_fail_cases
)
def test_validate_proper_filter_fail_cases(test_input, expected_error):
    with pytest.raises(expected_error):
        validate_proper_filter(PATHOLOGY, test_input)


@pytest.mark.parametrize(
    "test_input,expected_error", all_build_filter_clause_fail_cases
)
def test_build_filter_clause_fail_cases(test_input, expected_error):
    with pytest.raises(expected_error):
        build_filter_clause(test_input)


invalid_pathology_case = ["non_existing_pathology", 0, True]


@pytest.mark.parametrize("pathology", invalid_pathology_case)
def test_validate_proper_filter_fail_cases(pathology):
    with pytest.raises(KeyError):
        validate_proper_filter(
            pathology,
            {
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
            },
        )
