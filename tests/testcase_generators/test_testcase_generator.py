from unittest import mock
import io
import json

import pytest

from tests.testcase_generators.testcase_generator import (
    EnumFromCDE,
    EnumFromList,
    FloatParameter,
    NumericalInputDataVariables,
    InputGenerator,
    TestCaseGenerator,
    DB,
    IntegerParameter,
    make_parameters,
)


TESTCASEGEN_PATH = "tests.testcase_generators.testcase_generator"


@pytest.fixture
def specs_file_xy_numerical():
    return io.StringIO(
        """
        {
            "inputdata": {
                "y": {
                    "types": ["real"],
                    "stattypes": ["numerical"],
                    "notblank": true,
                    "multiple": true
                },
                "x": {
                    "types": ["real"],
                    "stattypes": ["numerical"],
                    "notblank": false,
                    "multiple": true
                }
            },
            "parameters": {}
        }
        """
    )


@pytest.fixture
def specs_file_y_numerical():
    return io.StringIO(
        """
        {
            "inputdata": {
                "y": {
                    "types": ["real"],
                    "stattypes": ["numerical"],
                    "notblank": true,
                    "multiple": true
                }
            },
            "parameters": {}
        }
        """
    )


def test_numerical_vars_draw_single_required():
    var = NumericalInputDataVariables(notblank=True, multiple=False)
    assert var.draw()


def test_numerical_vars_draw_single_not_required():
    var = NumericalInputDataVariables(notblank=False, multiple=False)

    with mock.patch(f"{TESTCASEGEN_PATH}.coin", lambda: True):
        assert var.draw()

    with mock.patch(f"{TESTCASEGEN_PATH}.coin", lambda: False):
        assert not var.draw()


def test_numerical_vars_draw_multiple_required():
    var = NumericalInputDataVariables(notblank=True, multiple=True)

    with mock.patch(f"{TESTCASEGEN_PATH}.triangular", lambda: 2):
        assert len(var.draw()) == 2


def test_algorithm_specs_numerical_vars(specs_file_xy_numerical):
    input_gen = InputGenerator(specs_file_xy_numerical)
    input_ = input_gen.draw()
    inputdata = input_["inputdata"]
    parameters = input_["parameters"]
    assert "x" in inputdata.keys() and "y" in inputdata
    assert inputdata["y"] is not None
    assert not parameters


def test_get_input_data(specs_file_y_numerical):
    input_ = {
        "inputdata": {
            "y": ("lefthippocampus", "righthippocampus"),
            "data_model": "dementia",
            "data_model_version": "0.1",
            "datasets": ("desd-synthdata",),
            "filters": "",
        },
        "para meters": {},
    }

    class ConcreteTestCaseGenerator(TestCaseGenerator):
        def compute_expected_output(self, input_data, parameters):
            return "Result"

    testcase_gen = ConcreteTestCaseGenerator(specs_file_y_numerical)
    y_data, _ = testcase_gen.get_input_data(input_)
    assert set(y_data.columns) == set(input_["inputdata"]["y"])


def test_testcase_generator_single_testcase(specs_file_xy_numerical):
    class ConcreteTestCaseGenerator(TestCaseGenerator):
        def compute_expected_output(self, input_data, parameters):
            return "Result"

    testcase_gen = ConcreteTestCaseGenerator(specs_file_xy_numerical)
    testcase = testcase_gen.generate_test_case()
    assert testcase["input"]
    assert testcase["output"]
    assert testcase["input"]["parameters"] == {}
    assert testcase["input"]["inputdata"]["y"] is not None


def test_testcase_generator_multiple_testcases(specs_file_xy_numerical):
    class ConcreteTestCaseGenerator(TestCaseGenerator):
        def compute_expected_output(self, input_data, parameters):
            return "Result"

    testcase_gen = ConcreteTestCaseGenerator(specs_file_xy_numerical)
    testcases = testcase_gen.generate_test_cases(num_test_cases=2)["test_cases"]
    assert len(testcases) == 2


def test_testcase_generator_write_to_file(specs_file_xy_numerical):
    class ConcreteTestCaseGenerator(TestCaseGenerator):
        def compute_expected_output(self, input_data, parameters):
            return "Result"

    testcase_gen = ConcreteTestCaseGenerator(specs_file_xy_numerical)
    expected_file = io.StringIO()
    testcase_gen.write_test_cases(expected_file, num_test_cases=2)

    contents = json.loads(expected_file.getvalue())
    assert len(contents["test_cases"]) == 2


def test_db_get_numerical_variables():
    numerical_vars = DB().get_numerical_variables()
    assert numerical_vars != []


def test_db_get_nominal_variables():
    nominal_vars = DB().get_nominal_variables()
    assert nominal_vars != []


def test_db_get_datasets():
    datasets = DB().get_datasets()
    assert datasets != []


def test_db_get_data_table():
    data_table = DB().get_data_table()
    assert data_table is not None


def test_db_get_enumerations():
    enums = DB().get_enumerations("gender")
    assert set(enums) == {"F", "M"}


def test_db_get_data_table_replicas():
    data_table_once = DB().get_data_table(replicas=1)
    data_table_twice = DB().get_data_table(replicas=2)
    assert len(data_table_once) != 0
    assert len(data_table_once) * 2 == len(data_table_twice)


def test_int_parameter():
    assert 1 <= IntegerParameter(min=1, max=5).draw() <= 5


def test_float_parameter():
    assert 2.71 <= FloatParameter(min=2.71, max=3.14).draw() <= 3.14


def test_enum_from_list_parameter():
    enums = ["a", "b", "c"]
    assert EnumFromList(enums).draw() in enums


def test_enum_from_cde_parameter():
    varname = "gender"
    assert EnumFromCDE(varname).draw() in {"F", "M"}


def test_make_enum_from_cde_parameter():
    properties = {"type": "enum_from_cde", "variable_name": "y"}
    y = ["a_variable"]
    enum = make_parameters(properties, y)
    assert enum.varname == "a_variable"


def test_make_enum_from_cde_parameter__error_multiple_vars():
    properties = {"type": "enum_from_cde", "variable_name": "y"}
    y = ["a_variable", "another_one"]
    with pytest.raises(AssertionError):
        make_parameters(properties, y)


def test_make_enum_from_cde_parameter__error_unknown_type():
    properties = {"type": "unknown"}
    with pytest.raises(TypeError):
        make_parameters(properties)
