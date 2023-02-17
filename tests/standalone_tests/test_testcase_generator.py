import io
import json

import pytest

from tests.testcase_generators.testcase_generator import DB
from tests.testcase_generators.testcase_generator import EnumFromCDE
from tests.testcase_generators.testcase_generator import EnumFromList
from tests.testcase_generators.testcase_generator import FloatParameter
from tests.testcase_generators.testcase_generator import InputGenerator
from tests.testcase_generators.testcase_generator import IntegerParameter
from tests.testcase_generators.testcase_generator import MixedTypeMultipleVar
from tests.testcase_generators.testcase_generator import MixedTypeSingleVar
from tests.testcase_generators.testcase_generator import SingleTypeMultipleVar
from tests.testcase_generators.testcase_generator import SingleTypeSingleVar
from tests.testcase_generators.testcase_generator import TestCaseGenerator
from tests.testcase_generators.testcase_generator import make_parameters

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


@pytest.fixture
def specs_file_xy_mixed():
    return io.StringIO(
        """
        {
            "inputdata": {
                "y": {
                    "types": ["text", "int"],
                    "stattypes": ["nominal"],
                    "notblank": true,
                    "multiple": false
                },
                "x": {
                    "types": ["real", "text", "int"],
                    "stattypes": ["numerical", "nominal"],
                    "notblank": true,
                    "multiple": true
                }
            },
            "parameters": {}
        }
        """
    )


def test_single_type_singe_var():
    pool = ["x1", "x2", "x3"]
    var = SingleTypeSingleVar(notblank=True, pool=pool)
    result = var.draw()
    assert result == ("x3",)


# Since the classes SingleTypeMultipleVar, MixedTypeSingleVar and
# MixedTypeMultipleVar implement a non-deterministic draw method, I test them
# using property-based testing. This means that I run 100 draws and verify that
# some property holds, rather than comparing with some known expected result.


def test_single_type_singe_var_notblank_is_false():
    results = []
    for _ in range(100):
        pool = ["x1", "x2", "x3"]
        var = SingleTypeSingleVar(notblank=False, pool=pool)
        result = var.draw()
        results.append(result == ("x3",))

    assert len(set(results)) == 2  # this has probability 2^(-99) to fail


def test_single_type_multi_var():
    for _ in range(100):
        pool = ["x1", "x2", "x3"]
        var = SingleTypeMultipleVar(notblank=True, pool=pool)
        result = var.draw()
        assert list(result) + pool == ["x1", "x2", "x3"]


def test_mixed_type_single_var():
    for _ in range(100):
        pool1 = ["x1", "x2", "x3"]
        pool2 = ["y1", "y2", "y3"]
        var = MixedTypeSingleVar(notblank=True, pool1=pool1, pool2=pool2)
        result = var.draw()
        assert result == ("x3",) or result == ("y3",)


def test_mixed_type_multi_var():
    for _ in range(100):
        pool1 = ["x1", "x2", "x3"]
        pool2 = ["y1", "y2", "y3"]
        all_vars = set(pool1) | set(pool2)

        var = MixedTypeMultipleVar(notblank=True, pool1=pool1, pool2=pool2)
        result = var.draw()
        assert len(result) + len(pool1) + len(pool2) == len(all_vars)
        assert set(result) | set(pool1) | set(pool2) == all_vars
        assert set(result) & {"x1", "x2", "x3"}  # assert pool1 is always used


@pytest.mark.slow
def test_algorithm_specs_numerical_vars(specs_file_xy_numerical):
    numericals = set(DB().get_numerical_variables())
    input_gen = InputGenerator(specs_file_xy_numerical)

    for _ in range(100):
        input_ = input_gen.draw()
        y, x = input_["inputdata"]["y"], input_["inputdata"]["x"]

        assert len(y) >= 1
        assert set(x) <= numericals
        assert set(y) <= numericals
        assert not set(x) & set(y)


@pytest.mark.slow
def test_algorithm_specs_mixed_vars(specs_file_xy_mixed):
    nominals = set(DB().get_nominal_variables())
    numericals = set(DB().get_numerical_variables())
    input_gen = InputGenerator(specs_file_xy_mixed)

    for _ in range(100):
        input_ = input_gen.draw()
        y, x = input_["inputdata"]["y"], input_["inputdata"]["x"]

        assert len(y) == 1
        assert len(x) >= 1
        assert set(y) <= nominals
        assert set(x) <= numericals | nominals
        assert not set(x) & set(y)


@pytest.mark.slow
def test_get_input_data(specs_file_y_numerical):
    class ConcreteTestCaseGenerator(TestCaseGenerator):
        def compute_expected_output(self, input_data, parameters, metadata):
            return "Result"

    testcase_gen = ConcreteTestCaseGenerator(specs_file_y_numerical)

    for _ in range(100):
        testcase = testcase_gen.generate_test_case()
        input_ = testcase["input"]
        y_data, _ = testcase_gen.get_input_data(input_)
        assert set(y_data.columns) == set(input_["inputdata"]["y"])


@pytest.mark.slow
def test_testcase_generator_single_testcase(specs_file_xy_numerical):
    class ConcreteTestCaseGenerator(TestCaseGenerator):
        def compute_expected_output(self, input_data, parameters, metadata):
            return "Result"

    testcase_gen = ConcreteTestCaseGenerator(specs_file_xy_numerical)

    for _ in range(100):
        testcase = testcase_gen.generate_test_case()
        assert testcase["input"]
        assert testcase["output"]
        assert testcase["input"]["parameters"] == {}
        assert testcase["input"]["inputdata"]["y"] is not None


@pytest.mark.slow
def test_testcase_generator_multiple_testcases(specs_file_xy_numerical):
    class ConcreteTestCaseGenerator(TestCaseGenerator):
        def compute_expected_output(self, input_data, parameters, metadata):
            return "Result"

    testcase_gen = ConcreteTestCaseGenerator(specs_file_xy_numerical)
    testcases = testcase_gen.generate_test_cases(num_test_cases=2)["test_cases"]
    assert len(testcases) == 2


@pytest.mark.slow
def test_testcase_generator_write_to_file(specs_file_xy_numerical):
    class ConcreteTestCaseGenerator(TestCaseGenerator):
        def compute_expected_output(self, input_data, parameters, metadata):
            return "Result"

    testcase_gen = ConcreteTestCaseGenerator(specs_file_xy_numerical)
    expected_file = io.StringIO()
    testcase_gen.write_test_cases(expected_file, num_test_cases=2)

    contents = json.loads(expected_file.getvalue())
    assert len(contents["test_cases"]) == 2


@pytest.mark.slow
def test_db_get_numerical_variables():
    numerical_vars = DB().get_numerical_variables()
    assert numerical_vars != []


@pytest.mark.slow
def test_db_get_nominal_variables():
    nominal_vars = DB().get_nominal_variables()
    assert nominal_vars != []


@pytest.mark.slow
def test_db_get_datasets():
    datasets = DB().get_datasets()
    assert datasets != []


@pytest.mark.slow
def test_db_get_data_table():
    data_table = DB().get_data_table()
    assert data_table is not None
    assert len(data_table) > 0


@pytest.mark.slow
def test_db_get_enumerations():
    enums = DB().get_enumerations("gender")
    assert set([enum["code"] for enum in enums]) == {"F", "M"}


def test_int_parameter():
    assert 1 <= IntegerParameter(min=1, max=5).draw() <= 5


def test_float_parameter():
    assert 2.71 <= FloatParameter(min=2.71, max=3.14).draw() <= 3.14


def test_enum_from_list_parameter():
    enums = ["a", "b", "c"]
    assert EnumFromList(enums).draw() in enums


@pytest.mark.slow
def test_enum_from_cde_parameter():
    varname = "gender"
    enum = EnumFromCDE(varname).draw()
    assert enum["code"] in {"F", "M"}


def test_make_enum_from_cde_parameter():
    properties = {
        "types": ["int", "text"],
        "enums": {"type": "input_var_CDE_enums", "source": "y"},
    }
    variable_groups = {"y": ["a_variable"], "x": []}
    enum = make_parameters(properties, variable_groups)
    assert enum.varname == "a_variable"


def test_make_enum_from_cde_parameter__error_multiple_vars():
    properties = {
        "types": ["int", "text"],
        "enums": {"type": "input_var_CDE_enums", "source": "y"},
    }
    variable_groups = {"y": ["a_variable", "another_one"], "x": []}
    with pytest.raises(AssertionError):
        make_parameters(properties, variable_groups)


def test_make_enum_from_cde_parameter__error_unknown_type():
    properties = {"types": ["unknown"]}
    with pytest.raises(TypeError):
        make_parameters(properties, variable_groups={})
