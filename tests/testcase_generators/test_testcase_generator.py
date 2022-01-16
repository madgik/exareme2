from unittest import mock
import io
import json

import pytest

from tests.testcase_generators.testcase_generator import (
    NumericalInputDataVariables,
    InputGenerator,
    TestCaseGenerator,
    DB,
)


# XXX Uncomment next statement to test TestCaseGenerator. However, comment
# again before commiting. We don't need to test the TestCaseGenerator on a
# regular basis. We only test while developing new features.
pytestmark = pytest.mark.skip(
    "These tests rely on a MonetDB container already loaded with data. They are "
    "useful for developing TestCaseGenerator features only and are not related to "
    "the MIP engine."
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
            "pathology": "dementia",
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


def test_db_get_data_table_replicas():
    data_table_once = DB().get_data_table(replicas=1)
    data_table_twice = DB().get_data_table(replicas=2)
    assert len(data_table_once) != 0
    assert len(data_table_once) * 2 == len(data_table_twice)
