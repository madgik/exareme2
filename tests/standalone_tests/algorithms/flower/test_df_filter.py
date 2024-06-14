import pandas as pd
import pytest

from exareme2.algorithms.flower.df_filter import apply_filter

# Sample DataFrame
data = {
    "test_age_value": [17, 25, 35, 45, 60, 75],
    "test_pupil_reactivity_right_eye_result": [
        "Reactive",
        "Nonreactive",
        "Reactive",
        "Nonreactive",
        "Reactive",
        "Nonreactive",
    ],
    "test_mortality_core": [0.2, 0.4, 0.6, 0.8, 0.9, 0.5],
    "test_gose_score": [
        None,
        "Moderate Disability",
        "Severe Disability",
        None,
        "Lower Severe Disability",
        None,
    ],
    "test_gcs_total_score": [None, 15, 13, None, 14, 15],
    "test_gender_type": ["F", "M", "F", "M", "F", "M"],
}
sample_dataframe = pd.DataFrame(data)

# Expected Outputs
expected_outputs = [
    # Case 1: test_age_value == 17
    sample_dataframe[sample_dataframe["test_age_value"] == 17].reset_index(drop=True),
    # Case 2: test_pupil_reactivity_right_eye_result != "Nonreactive"
    sample_dataframe[
        sample_dataframe["test_pupil_reactivity_right_eye_result"] != "Nonreactive"
    ].reset_index(drop=True),
    # Case 3: test_age_value == 17 OR test_pupil_reactivity_right_eye_result != "Nonreactive"
    pd.concat(
        [
            sample_dataframe[sample_dataframe["test_age_value"] == 17],
            sample_dataframe[
                sample_dataframe["test_pupil_reactivity_right_eye_result"]
                != "Nonreactive"
            ],
        ]
    )
    .drop_duplicates()
    .reset_index(drop=True),
    # Case 4: test_age_value < 50 AND test_age_value > 20
    sample_dataframe[
        (sample_dataframe["test_age_value"] < 50)
        & (sample_dataframe["test_age_value"] > 20)
    ].reset_index(drop=True),
    # Case 5: test_age_value NOT between [60, 90] AND test_mortality_core between [0.3, 0.8]
    sample_dataframe[
        (~sample_dataframe["test_age_value"].between(60, 90))
        & (sample_dataframe["test_mortality_core"].between(0.3, 0.8))
    ].reset_index(drop=True),
    # Case 6: test_gose_score is null OR test_gcs_total_score is not null
    pd.concat(
        [
            sample_dataframe[sample_dataframe["test_gose_score"].isnull()],
            sample_dataframe[sample_dataframe["test_gcs_total_score"].notnull()],
        ]
    )
    .drop_duplicates()
    .reset_index(drop=True),
    # Case 7: test_age_value in [17, 19]
    sample_dataframe[sample_dataframe["test_age_value"].isin([17, 19])].reset_index(
        drop=True
    ),
    # Case 8: test_gender_type in ["F", "M"]
    sample_dataframe[sample_dataframe["test_gender_type"].isin(["F", "M"])].reset_index(
        drop=True
    ),
    # Case 9: Complex OR condition
    pd.concat(
        [
            sample_dataframe[
                (sample_dataframe["test_gender_type"] == "F")
                & (sample_dataframe["test_age_value"].between(20, 30))
                & (sample_dataframe["test_gose_score"].notnull())
            ],
            sample_dataframe[
                (sample_dataframe["test_gender_type"] != "F")
                & (sample_dataframe["test_mortality_core"] > 0.5)
                & (sample_dataframe["test_mortality_core"] <= 0.8)
            ],
        ]
    )
    .drop_duplicates()
    .reset_index(drop=True),
]

# Test cases with the expected outputs
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
        expected_outputs[0],
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
        expected_outputs[1],
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
        expected_outputs[2],
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
        expected_outputs[3],
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
        expected_outputs[4],
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
        expected_outputs[5],
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
        expected_outputs[6],
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
        expected_outputs[7],
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
        expected_outputs[8],
    ),
]


# Testing function
@pytest.mark.parametrize("test_input,expected", all_success_cases)
def test_apply_filter(test_input, expected):
    result = apply_filter(sample_dataframe, test_input)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )
