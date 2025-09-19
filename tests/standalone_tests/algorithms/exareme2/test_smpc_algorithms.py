import json
import re

import pytest
import requests

from tests.standalone_tests.conftest import SMPC_ALGORITHMS_URL


def get_parametrization_list_success_cases():
    parametrization_list = []

    # ~~~~~~~~~~success case 1~~~~~~~~~~
    algorithm_name = "smpc_standard_deviation_int_only"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "edsd0",
                "edsd1",
                "edsd2",
                "edsd3",
                "edsd4",
                "edsd5",
                "edsd6",
                "edsd7",
                "edsd8",
                "edsd9",
            ],
            "y": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": [
                            "edsd0",
                            "edsd1",
                            "edsd2",
                            "edsd3",
                            "edsd4",
                            "edsd5",
                            "edsd6",
                            "edsd7",
                            "edsd8",
                            "edsd9",
                        ],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
        "type": "exareme2",
    }
    expected_response = {
        "title": "Standard Deviation",
        "columns": [
            {"name": "variable", "data": ["lefthippocampus"], "type": "STR"},
            {"name": "std_deviation", "data": [0.35797549230253856], "type": "FLOAT"},
            {"name": "min_value", "data": [1.0], "type": "FLOAT"},
            {"name": "max_value", "data": [4.0], "type": "FLOAT"},
        ],
    }
    parametrization_list.append(
        pytest.param(
            algorithm_name,
            request_dict,
            expected_response,
            id="smpc std dev ints only without smpc flag",
        )
    )
    # END ~~~~~~~~~~success case 1~~~~~~~~~~

    # ~~~~~~~~~~success case 2~~~~~~~~~~
    algorithm_name = "smpc_standard_deviation_int_only"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "edsd0",
                "edsd1",
                "edsd2",
                "edsd3",
                "edsd4",
                "edsd5",
                "edsd6",
                "edsd7",
                "edsd8",
                "edsd9",
            ],
            "y": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": [
                            "edsd0",
                            "edsd1",
                            "edsd2",
                            "edsd3",
                            "edsd4",
                            "edsd5",
                            "edsd6",
                            "edsd7",
                            "edsd8",
                            "edsd9",
                        ],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
        "flags": {
            "smpc": True,
        },
        "type": "exareme2",
    }
    expected_response = {
        "title": "Standard Deviation",
        "columns": [
            {"name": "variable", "data": ["lefthippocampus"], "type": "STR"},
            {"name": "std_deviation", "data": [0.35797549230253856], "type": "FLOAT"},
            {"name": "min_value", "data": [1.0], "type": "FLOAT"},
            {"name": "max_value", "data": [4.0], "type": "FLOAT"},
        ],
    }
    parametrization_list.append(
        pytest.param(
            algorithm_name,
            request_dict,
            expected_response,
            id="smpc std dev ints only with smpc flag",
        )
    )
    # END ~~~~~~~~~~success case 2~~~~~~~~~~

    # ~~~~~~~~~~success case 3~~~~~~~~~~
    algorithm_name = "smpc_standard_deviation"
    expected_response = {
        "title": "Standard Deviation",
        "columns": [
            {"name": "variable", "data": ["lefthippocampus"], "type": "STR"},
            {"name": "std_deviation", "data": [0.36345069556304765], "type": "FLOAT"},
            {"name": "min_value", "data": [1.3047], "type": "FLOAT"},
            {"name": "max_value", "data": [4.4519], "type": "FLOAT"},
        ],
    }
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "edsd0",
                "edsd1",
                "edsd2",
                "edsd3",
                "edsd4",
                "edsd5",
                "edsd6",
                "edsd7",
                "edsd8",
                "edsd9",
            ],
            "y": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": [
                            "edsd0",
                            "edsd1",
                            "edsd2",
                            "edsd3",
                            "edsd4",
                            "edsd5",
                            "edsd6",
                            "edsd7",
                            "edsd8",
                            "edsd9",
                        ],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
        "type": "exareme2",
    }
    parametrization_list.append(
        pytest.param(
            algorithm_name,
            request_dict,
            expected_response,
            id="smpc std dev floats/ints without smpc flag",
        )
    )
    # END ~~~~~~~~~~success case 3~~~~~~~~~~

    # ~~~~~~~~~~success case 4~~~~~~~~~~
    algorithm_name = "smpc_standard_deviation"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "edsd0",
                "edsd1",
                "edsd2",
                "edsd3",
                "edsd4",
                "edsd5",
                "edsd6",
                "edsd7",
                "edsd8",
                "edsd9",
            ],
            "y": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": [
                            "edsd0",
                            "edsd1",
                            "edsd2",
                            "edsd3",
                            "edsd4",
                            "edsd5",
                            "edsd6",
                            "edsd7",
                            "edsd8",
                            "edsd9",
                        ],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
        "flags": {
            "smpc": True,
        },
        "type": "exareme2",
    }
    expected_response = {
        "title": "Standard Deviation",
        "columns": [
            {"name": "variable", "data": ["lefthippocampus"], "type": "STR"},
            {"name": "std_deviation", "data": [0.36345069556304765], "type": "FLOAT"},
            {"name": "min_value", "data": [1.3047], "type": "FLOAT"},
            {"name": "max_value", "data": [4.4519], "type": "FLOAT"},
        ],
    }
    parametrization_list.append(
        pytest.param(
            algorithm_name,
            request_dict,
            expected_response,
            id="smpc std dev floats/ints with smpc flag",
        )
    )
    # END ~~~~~~~~~~success case 4~~~~~~~~~~
    return parametrization_list


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.smpc_cluster
@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_success_cases(),
)
def test_post_smpc_algorithm(
    smpc_cluster,
    monetdb_smpc_globalworker,
    smpc_globalworker_worker_service,
    monetdb_smpc_localworker1,
    smpc_localworker1_worker_service,
    load_data_smpc_localworker1_with_monetdb,
    smpc_localworker2_worker_service,
    load_data_smpc_localworker2_with_monetdb,
    smpc_controller_service,
    algorithm_name,
    request_dict,
    expected_response,
):
    algorithm_url = SMPC_ALGORITHMS_URL + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200, f"Response message: {response.text}"
    assert json.loads(response.text) == expected_response


def get_parametrization_list_exception_cases():
    parametrization_list = []
    algorithm_name = "smpc_standard_deviation"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "edsd0",
                "edsd1",
                "edsd2",
                "edsd3",
                "edsd4",
                "edsd5",
                "edsd6",
                "edsd7",
                "edsd8",
                "edsd9",
            ],
            "y": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": [
                            "edsd0",
                            "edsd1",
                            "edsd2",
                            "edsd3",
                            "edsd4",
                            "edsd5",
                            "edsd6",
                            "edsd7",
                            "edsd8",
                            "edsd9",
                        ],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
        "flags": {
            "smpc": False,
        },
        "type": "exareme2",
    }

    expected_response = (
        462,
        "The computation cannot be made without SMPC.",
    )

    parametrization_list.append((algorithm_name, request_dict, expected_response))

    return parametrization_list


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_exception_cases(),
)
def test_post_smpc_algorithm_exception(
    monetdb_smpc_globalworker,
    smpc_globalworker_worker_service,
    monetdb_smpc_localworker1,
    smpc_localworker1_worker_service,
    load_data_smpc_localworker1_with_monetdb,
    monetdb_smpc_localworker2,
    smpc_localworker2_worker_service,
    load_data_smpc_localworker2_with_monetdb,
    smpc_controller_service,
    algorithm_name,
    request_dict,
    expected_response,
):
    algorithm_url = SMPC_ALGORITHMS_URL + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    exp_response_status, exp_response_message = expected_response
    assert (
        response.status_code == exp_response_status
    ), f"Response message: {response.text}"
    assert re.search(exp_response_message, response.text)
