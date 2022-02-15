from unittest.mock import patch

import pytest

from mipengine.common_data_elements import CommonDataElement
from mipengine.common_data_elements import CommonDataElements
from mipengine.common_data_elements import MetadataEnumeration
from mipengine.common_data_elements import MetadataVariable
from mipengine.filters import FilterError
from mipengine.filters import build_filter_clause
from mipengine.filters import validate_filter

data_model = "test_data_model1:0.1"


@pytest.fixture()
def common_data_elements():
    common_data_elements = CommonDataElements()
    common_data_elements.data_models = {
        "test_data_model1:0.1": {
            "test_age_value": CommonDataElement(
                MetadataVariable(
                    code="test_age_value",
                    label="test test_age_value",
                    sql_type="int",
                    isCategorical=False,
                    enumerations=None,
                    min=0,
                    max=130,
                )
            ),
            "test_pupil_reactivity_right_eye_result": CommonDataElement(
                MetadataVariable(
                    code="test_pupil_reactivity_right_eye_result",
                    label="test test_pupil_reactivity_right_eye_result",
                    sql_type="text",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(code="Sluggish", label="Sluggish"),
                        MetadataEnumeration(code="Nonreactive", label="Nonreactive"),
                        MetadataEnumeration(code="Brisk", label="Brisk"),
                        MetadataEnumeration(code="Untestable", label="Untestable"),
                        MetadataEnumeration(code="Unknown", label="Unknown"),
                    ],
                    min=None,
                    max=None,
                )
            ),
            "test_mortality_core": CommonDataElement(
                MetadataVariable(
                    code="test_mortality_core",
                    label="test_mortality_core",
                    sql_type="real",
                    isCategorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                )
            ),
            "test_gose_score": CommonDataElement(
                MetadataVariable(
                    code="test_gose_score",
                    label="test_gose_score",
                    sql_type="text",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(code="1", label="Dead"),
                        MetadataEnumeration(code="2", label="Vegetative State"),
                        MetadataEnumeration(code="3", label="Lower Severe Disability"),
                        MetadataEnumeration(code="4", label="Upper Severe Disability"),
                        MetadataEnumeration(
                            code="5", label="Lower Moderate Disability"
                        ),
                        MetadataEnumeration(
                            code="6", label="Upper Moderate Disability"
                        ),
                        MetadataEnumeration(code="7", label="Lower Good Recovery"),
                        MetadataEnumeration(code="8", label="Upper Good Recovery"),
                    ],
                    min=None,
                    max=None,
                )
            ),
            "test_gcs_total_score": CommonDataElement(
                MetadataVariable(
                    code="test_gcs_total_score",
                    label="test_gcs_total_score",
                    sql_type="text",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(code="3", label="3"),
                        MetadataEnumeration(code="4", label="4"),
                        MetadataEnumeration(code="5", label="5"),
                        MetadataEnumeration(code="6", label="6"),
                        MetadataEnumeration(code="7", label="7"),
                        MetadataEnumeration(code="8", label="8"),
                        MetadataEnumeration(code="9", label="9"),
                        MetadataEnumeration(code="10", label="10"),
                        MetadataEnumeration(code="11", label="11"),
                        MetadataEnumeration(code="12", label="12"),
                        MetadataEnumeration(code="13", label="13"),
                        MetadataEnumeration(code="14", label="14"),
                        MetadataEnumeration(code="15", label="15"),
                        MetadataEnumeration(code="untestable", label="untestable"),
                        MetadataEnumeration(code="unknown", label="unknown"),
                    ],
                    min=None,
                    max=None,
                )
            ),
            "test_gender_type": CommonDataElement(
                MetadataVariable(
                    code="test_gender_type",
                    label="test_gender_type",
                    sql_type="text",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(code="M", label="Male"),
                        MetadataEnumeration(code="F", label="Female"),
                    ],
                    min=None,
                    max=None,
                )
            ),
        },
    }

    return common_data_elements


all_success_cases = [
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "equal",
                    "value": 17,
                }
            ],
            "valid": True,
        },
        "test_age_value = 17",
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_pupil_reactivity_right_eye_result",
                    "field": "test_pupil_reactivity_right_eye_result",
                    "type": "string",
                    "input": "text",
                    "operator": "not_equal",
                    "value": "Nonreactive",
                }
            ],
            "valid": True,
        },
        "test_pupil_reactivity_right_eye_result <> 'Nonreactive'",
    ),
    (
        {
            "condition": "OR",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "equal",
                    "value": 17,
                },
                {
                    "id": "test_pupil_reactivity_right_eye_result",
                    "field": "test_pupil_reactivity_right_eye_result",
                    "type": "string",
                    "input": "text",
                    "operator": "not_equal",
                    "value": "Nonreactive",
                },
            ],
            "valid": True,
        },
        "test_age_value = 17 OR test_pupil_reactivity_right_eye_result <> 'Nonreactive'",
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "less",
                    "value": 50,
                },
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "greater",
                    "value": 20,
                },
            ],
            "valid": True,
        },
        "test_age_value < 50 AND test_age_value > 20",
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "not_between",
                    "value": [60, 90],
                },
                {
                    "id": "test_mortality_core",
                    "field": "test_mortality_core",
                    "type": "double",
                    "input": "number",
                    "operator": "between",
                    "value": [0.3, 0.8],
                },
            ],
            "valid": True,
        },
        "NOT test_age_value BETWEEN 60 AND 90 AND test_mortality_core BETWEEN 0.3 AND 0.8",
    ),
    (
        {
            "condition": "OR",
            "rules": [
                {
                    "id": "test_gose_score",
                    "field": "test_gose_score",
                    "type": "text",
                    "input": "text",
                    "operator": "is_null",
                    "value": None,
                },
                {
                    "id": "test_gcs_total_score",
                    "field": "test_gcs_total_score",
                    "type": "int",
                    "input": "number",
                    "operator": "is_not_null",
                    "value": None,
                },
            ],
            "valid": True,
        },
        "test_gose_score IS NULL OR test_gcs_total_score IS NOT NULL",
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "integer",
                    "input": "number",
                    "operator": "in",
                    "value": [17, 19],
                }
            ],
            "valid": True,
        },
        "test_age_value IN (17,19)",
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_gender_type",
                    "field": "test_gender_type",
                    "type": "string",
                    "input": "text",
                    "operator": "in",
                    "value": ["F", "M"],
                }
            ],
            "valid": True,
        },
        "test_gender_type IN ('F','M')",
    ),
    (
        {
            "condition": "OR",
            "rules": [
                {
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "test_gender_type",
                            "field": "test_gender_type",
                            "type": "string",
                            "input": "text",
                            "operator": "equal",
                            "value": "F",
                        },
                        {
                            "condition": "AND",
                            "rules": [
                                {
                                    "id": "test_age_value",
                                    "field": "test_age_value",
                                    "type": "int",
                                    "input": "number",
                                    "operator": "between",
                                    "value": [20, 30],
                                },
                                {
                                    "id": "test_gose_score",
                                    "field": "test_gose_score",
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
                            "id": "test_gender_type",
                            "field": "test_gender_type",
                            "type": "string",
                            "input": "text",
                            "operator": "not_equal",
                            "value": "F",
                        },
                        {
                            "condition": "AND",
                            "rules": [
                                {
                                    "id": "test_mortality_core",
                                    "field": "test_mortality_core",
                                    "type": "double",
                                    "input": "number",
                                    "operator": "greater",
                                    "value": 0.5,
                                },
                                {
                                    "id": "test_mortality_core",
                                    "field": "test_mortality_core",
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
            "test_gender_type = 'F' AND test_age_value BETWEEN 20 AND 30 AND test_gose_score IS NOT NULL OR test_gender_type <> 'F' AND "
            "test_mortality_core > 0.5 AND test_mortality_core <= 0.8"
        ),
    ),
]


@pytest.mark.parametrize("test_input,expected", all_success_cases)
def test_build_filter_clause(test_input, expected):
    assert build_filter_clause(test_input) == expected


@pytest.mark.parametrize("test_input,expected", all_success_cases)
def test_validate_filter(test_input, expected, common_data_elements):
    validate_filter(common_data_elements, data_model, test_input)


all_build_filter_clause_fail_cases = [
    {
        "condition": "ANDOR",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": 17,
            }
        ],
        "valid": True,
    },
    0,
    "not_a_filter",
    {"data": 0},
    {0},
]


@pytest.mark.parametrize("test_input", all_build_filter_clause_fail_cases)
def test_build_filter_clause_fail_cases(test_input):
    with pytest.raises(FilterError):
        build_filter_clause(test_input)


all_validate_filter_fail_cases = [
    {
        "condition": "ANDOR",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": 17,
            }
        ],
        "valid": True,
    },
    0,
    "not_a_filter",
    {"data": 0},
    {0},
    {
        "condition": "AND",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "invalid operator",
                "value": 17,
            }
        ],
        "valid": True,
    },
    FilterError,
    {
        "condition": "AND",
        "rules": [
            {
                "id": "invalid column",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": 17,
            }
        ],
        "valid": True,
    },
    {
        "condition": "AND",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": "17",
            }
        ],
        "valid": True,
    },
    {
        "condition": "AND",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": True,
            }
        ],
        "valid": True,
    },
]


@pytest.mark.parametrize("test_input", all_validate_filter_fail_cases)
def test_validate_filter_fail_cases_bad_filter(test_input, common_data_elements):
    with pytest.raises(FilterError):
        validate_filter(common_data_elements, data_model, test_input)
