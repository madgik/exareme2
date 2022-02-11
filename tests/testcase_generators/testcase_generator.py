from abc import ABC, abstractmethod
import random
import json
from functools import partial, cached_property
import re

from tqdm import tqdm
import pandas as pd
import pymonetdb

from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


TESTING_DATAMODEL = "dementia"
TESTING_DATAMODEL_VERSION = "0.1"
DATA_TABLENAME = f""""{TESTING_DATAMODEL}:{TESTING_DATAMODEL_VERSION}".primary_data"""
METADATA_TABLENAME = (
    f""""{TESTING_DATAMODEL}:{TESTING_DATAMODEL_VERSION}".variables_metadata"""
)

MAX_TABLE_SIZE = 60
MIN_TABLE_SIZE = 1
TABLE_SIZE_MODE = 10

# XXX Change according to your local setup
DB_IP = "172.17.0.1"
DB_PORT = 50001
DB_USER = "monetdb"
DB_PASS = "monetdb"
DB_FARM = "db"


class DB(MonetDB):
    def refresh_connection(self):
        self._connection = pymonetdb.connect(
            hostname=DB_IP,
            port=DB_PORT,
            username=DB_USER,
            password=DB_PASS,
            database=DB_FARM,
        )

    def get_numerical_variables(self):
        query = f"""select code from {METADATA_TABLENAME} """
        query += """ where metadata LIKE '%"isCategorical": false%' AND (metadata LIKE '%"sql_type": "real"%' OR metadata LIKE '%"sql_type": "int"%'); """
        variables = pd.read_sql(query, self._connection)
        return variables["code"].tolist()

    def get_nominal_variables(self):
        query = f"""select code from {METADATA_TABLENAME} where metadata LIKE '%"isCategorical": true%'; """
        variables = pd.read_sql(query, self._connection)
        return variables["code"].tolist()

    def get_data_table(self, replicas=2):
        """Loads the whole data table from the DB.

        Parameters
        ----------
        replicas: int
            Number of times the data will be replicated. Useful for testing in
            federated environment with an equal number of local nodes.

        """
        query = f"SELECT * FROM {DATA_TABLENAME}"
        data_table = pd.read_sql(query, self._connection)
        data_table = pd.concat([data_table for _ in range(replicas)])
        return data_table

    def get_datasets(self):
        query = f"SELECT DISTINCT dataset FROM {DATA_TABLENAME}"
        datasets = pd.read_sql(query, self._connection)
        return datasets["dataset"].tolist()

    def get_enumerations(self, varname):
        query = f"SELECT enumerations FROM {METADATA_TABLENAME} WHERE code='{varname}'"
        result = self.execute_and_fetchall(query)
        enums = re.split(r"\s*,\s*", result[0][0])
        return enums


def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def coin():
    outcome = random.choice([True, False])
    return outcome


def triangular():
    return int(random.triangular(MIN_TABLE_SIZE, MAX_TABLE_SIZE, TABLE_SIZE_MODE))


class InputDataVariable(ABC):
    db = DB()

    def __init__(self, notblank, multiple):
        self._notblank = notblank
        self._multiple = multiple

    @property
    @abstractmethod
    def all_variables(self):
        pass

    def draw(self):
        if not self._notblank and not coin():
            return
        num = triangular() if self._multiple else 1
        choice = random_permutation(self.all_variables, r=num)
        return choice


class NumericalInputDataVariables(InputDataVariable):
    @cached_property
    def all_variables(self):
        """Gets the names of all numerical variables available."""
        numerical_vars = self.db.get_numerical_variables()
        return numerical_vars


class NominalInputDataVariables:
    pass


class MixedInputDataVariables:
    pass


def make_input_data_variables(properties):
    if properties["stattypes"] == ["numerical"]:
        return NumericalInputDataVariables(
            properties["notblank"],
            properties["multiple"],
        )
    elif properties["stattypes"] == ["nominal"]:
        raise NotImplementedError
    elif set(properties["stattypes"]) == {"numerical", "nominal"}:
        raise NotImplementedError


class AlgorithmParameter(ABC):
    @abstractmethod
    def draw(self):
        pass


class IntegerParameter(AlgorithmParameter):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def draw(self):
        return random.randint(self.min, self.max)


class FloatParameter(AlgorithmParameter):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def draw(self):
        return random.uniform(self.min, self.max)


class EnumFromList(AlgorithmParameter):
    def __init__(self, enums):
        self.enums = enums

    def draw(self):
        return random.choice(self.enums)


class EnumFromCDE(AlgorithmParameter):
    db = DB()

    def __init__(self, varname):
        self.varname = varname

    @cached_property
    def enums(self):
        return self.db.get_enumerations(self.varname)

    def draw(self):
        return random.choice(self.enums)


def make_parameters(properties, y=None, x=None):
    if properties["type"] == "int":
        return IntegerParameter(min=properties["min"], max=properties["max"])
    if properties["type"] == "float":
        return FloatParameter(min=properties["min"], max=properties["max"])
    if properties["type"] == "enum_from_list":
        return EnumFromList(enums=properties["enumerations"])
    if properties["type"] == "enum_from_cde":
        return make_enum_from_cde_parameter(properties, y, x)
    else:
        raise TypeError(f"Unknown paremeter type: {properties['type']}.")


def make_enum_from_cde_parameter(properties, y=None, x=None):
    if properties["variable_name"] == "y":
        assert len(y) == 1, "Can't instantiate EnumFromList for multiple variables."
        varname = y[0]
        return EnumFromCDE(varname=varname)
    elif properties["variable_name"] == "x":
        assert len(x) == 1, "Can't instantiate EnumFromList for multiple variables."
        varname = x[0]
        return EnumFromCDE(varname=varname)
    else:
        raise ValueError(
            "Parameters enum_from_cde can only accept x or y as variable_name."
        )


class InputGenerator:
    def __init__(self, specs_file):
        specs = json.load(specs_file)
        self.inputdata_gens = {
            name: make_input_data_variables(properties)
            for name, properties in specs["inputdata"].items()
        }
        y, x = specs["inputdata"].get("y"), specs["inputdata"].get("y")

        self.parameter_gens = {
            name: make_parameters(properties, y, x)
            for name, properties in specs["parameters"].items()
        }
        self.datasets_gen = DatasetsGenerator()
        self.filters_gen = FiltersGenerator()
        self._seen = set()

    def draw(self):
        """Draws a random instance of an algorithm input, based on the
        algorithm specs."""
        for _ in range(100):
            inputdata = {
                name: inputdata_vars.draw()
                for name, inputdata_vars in self.inputdata_gens.items()
            }
            # removes vars found in both y and x, from x
            if inputdata["x"] != None:
                diff = set(inputdata["y"]) & set(inputdata["x"])
                inputdata["x"] = tuple(set(inputdata["x"]) - diff)

            inputdata["data_model_code"] = TESTING_DATAMODEL
            inputdata["data_model_version"] = TESTING_DATAMODEL_VERSION
            inputdata["datasets"] = self.datasets_gen.draw()
            inputdata["filters"] = self.filters_gen.draw()
            parameters = {
                name: param.draw() for name, param in self.parameter_gens.items()
            }
            input_ = {"inputdata": inputdata, "parameters": parameters}
            input_key = (*inputdata.values(), *parameters.values())
            if input_key not in self._seen:
                self._seen.add(input_key)
                return input_
        else:
            raise ValueError("Cannot find inputdata that has not already be generated.")


class DatasetsGenerator:
    db = DB()

    @cached_property
    def all_datasets(self):
        return self.db.get_datasets()

    def draw(self):
        num = len(self.all_datasets)
        num_datasets = random.randint(1, num)
        return random_permutation(self.all_datasets, r=num_datasets)


class FiltersGenerator:
    def draw(self):
        # TODO Implement random filters generator
        return ""


tqdm = partial(tqdm, desc="Generating test cases", unit=" test cases")


class TestCaseGenerator(ABC):
    """Test case generator ABC.

    For each new algorithm we wish to test a subclass must be created.
    Subclasses must implement `compute_expected_output` where the algorithm
    expected ouput is computed according to the SOA implementation.

    Parameters
    ----------
    specs_file
        Pointer to algorithm specs file in json format
    replicas: int
        Number of times the data used in the expected output will be
        replicated. Useful for testing in federated environment with an equal
        number of local nodes.

    Examples
    --------
    >>> class MyAlgorithmTCG(TestCaseGenerator):
            def compute_expected_output(self, inputdata, input_parameters):
                ...  # SOA computation
                return {"some_result": 42, "other_result": 24}
    >>> with open('my_algoritm.json') as file:
            tcg = MyAlgorithmTCG(file, replicas=2)
    >>> with open('my_algoritm_expected.json') as file:
            tcg.write_test_cases(file)

    """

    __test__ = False

    def __init__(self, specs_file, replicas=2):
        self.input_gen = InputGenerator(specs_file)
        self.all_data = DB().get_data_table(replicas)

    def generate_input(self):
        return self.input_gen.draw()

    @abstractmethod
    def compute_expected_output(self, input_data, input_parameters=None):
        pass

    def get_input_data(self, input_):
        inputdata = input_["inputdata"]
        datasets = list(inputdata["datasets"])

        y = list(inputdata["y"])
        x = inputdata.get("x", None)
        x = list(x) if x else []

        variables = list(set(y + x))
        full_data = self.all_data[variables + ["dataset"]]
        full_data = full_data[full_data.dataset.isin(datasets)]
        del full_data["dataset"]
        full_data = full_data.dropna()
        if len(full_data) == 0:
            return None

        y_data = full_data[y]
        x_data = full_data[x] if x else None
        return y_data, x_data

    def generate_test_case(self):
        for _ in range(1000):
            input_ = self.generate_input()
            input_data = self.get_input_data(input_)
            if input_data is not None:
                break
        else:
            raise ValueError(
                "Cannot find inputdata values resulting in non-empty data."
            )
        parameters = input_["parameters"]
        output = self.compute_expected_output(input_data, parameters)
        return {"input": input_, "output": output}

    def generate_test_cases(self, num_test_cases=100):
        test_cases = [self.generate_test_case() for _ in tqdm(range(num_test_cases))]
        return {"test_cases": test_cases}

    def write_test_cases(self, file, num_test_cases=100):
        test_cases = self.generate_test_cases(num_test_cases)
        json.dump(test_cases, file, indent=4)
