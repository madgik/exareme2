from abc import ABC, abstractmethod
import random
import json

import pandas as pd


def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


# TODO This is a list of all numerical variables in our CDEs. It is used to
# generate testcases with the correct variable types. It should be removed and
# replaced with an actual call to the DB, or by reading the CDEs file.
NUMERICAL_VARS = [
    "av45",
    "fdg",
    "pib",
    "brainstem",
    "tiv",
    "_3rdventricle",
    "_4thventricle",
    "csfglobal",
    "leftinflatvent",
    "leftlateralventricle",
    "rightinflatvent",
    "rightlateralventricle",
    "cerebellarvermallobulesiv",
    "cerebellarvermallobulesviiix",
    "cerebellarvermallobulesvivii",
    "leftcerebellumexterior",
    "rightcerebellumexterior",
    "leftamygdala",
    "rightamygdala",
    "leftaccumbensarea",
    "leftbasalforebrain",
    "leftcaudate",
    "leftpallidum",
    "leftputamen",
    "rightaccumbensarea",
    "rightbasalforebrain",
    "rightcaudate",
    "rightpallidum",
    "rightputamen",
    "leftventraldc",
    "rightventraldc",
    "leftaorganteriororbitalgyrus",
    "leftcocentraloperculum",
    "leftfofrontaloperculum",
    "leftfrpfrontalpole",
    "leftgregyrusrectus",
    "leftlorglateralorbitalgyrus",
    "leftmfcmedialfrontalcortex",
    "leftmfgmiddlefrontalgyrus",
    "leftmorgmedialorbitalgyrus",
    "leftmprgprecentralgyrusmedialsegment",
    "leftmsfgsuperiorfrontalgyrusmedialsegment",
    "leftopifgopercularpartoftheinferiorfrontalgyrus",
    "leftorifgorbitalpartoftheinferiorfrontalgyrus",
    "leftpoparietaloperculum",
    "leftporgposteriororbitalgyrus",
    "leftprgprecentralgyrus",
    "leftscasubcallosalarea",
    "leftsfgsuperiorfrontalgyrus",
    "leftsmcsupplementarymotorcortex",
    "lefttrifgtriangularpartoftheinferiorfrontalgyrus",
    "rightaorganteriororbitalgyrus",
    "rightcocentraloperculum",
    "rightfofrontaloperculum",
    "rightfrpfrontalpole",
    "rightgregyrusrectus",
    "rightlorglateralorbitalgyrus",
    "rightmfcmedialfrontalcortex",
    "rightmfgmiddlefrontalgyrus",
    "rightmorgmedialorbitalgyrus",
    "rightmprgprecentralgyrusmedialsegment",
    "rightmsfgsuperiorfrontalgyrusmedialsegment",
    "rightopifgopercularpartoftheinferiorfrontalgyrus",
    "rightorifgorbitalpartoftheinferiorfrontalgyrus",
    "rightpoparietaloperculum",
    "rightporgposteriororbitalgyrus",
    "rightprgprecentralgyrus",
    "rightscasubcallosalarea",
    "rightsfgsuperiorfrontalgyrus",
    "rightsmcsupplementarymotorcortex",
    "righttrifgtriangularpartoftheinferiorfrontalgyrus",
    "leftainsanteriorinsula",
    "leftpinsposteriorinsula",
    "rightainsanteriorinsula",
    "rightpinsposteriorinsula",
    "leftacgganteriorcingulategyrus",
    "leftententorhinalarea",
    "lefthippocampus",
    "leftmcggmiddlecingulategyrus",
    "leftpcggposteriorcingulategyrus",
    "leftphgparahippocampalgyrus",
    "leftthalamusproper",
    "rightacgganteriorcingulategyrus",
    "rightententorhinalarea",
    "righthippocampus",
    "rightmcggmiddlecingulategyrus",
    "rightpcggposteriorcingulategyrus",
    "rightphgparahippocampalgyrus",
    "rightthalamusproper",
    "leftcalccalcarinecortex",
    "leftcuncuneus",
    "leftioginferioroccipitalgyrus",
    "leftliglingualgyrus",
    "leftmogmiddleoccipitalgyrus",
    "leftocpoccipitalpole",
    "leftofugoccipitalfusiformgyrus",
    "leftsogsuperioroccipitalgyrus",
    "rightcalccalcarinecortex",
    "rightcuncuneus",
    "rightioginferioroccipitalgyrus",
    "rightliglingualgyrus",
    "rightmogmiddleoccipitalgyrus",
    "rightocpoccipitalpole",
    "rightofugoccipitalfusiformgyrus",
    "rightsogsuperioroccipitalgyrus",
    "leftangangulargyrus",
    "leftmpogpostcentralgyrusmedialsegment",
    "leftpcuprecuneus",
    "leftpogpostcentralgyrus",
    "leftsmgsupramarginalgyrus",
    "leftsplsuperiorparietallobule",
    "rightangangulargyrus",
    "rightmpogpostcentralgyrusmedialsegment",
    "rightpcuprecuneus",
    "rightpogpostcentralgyrus",
    "rightsmgsupramarginalgyrus",
    "rightsplsuperiorparietallobule",
    "leftfugfusiformgyrus",
    "leftitginferiortemporalgyrus",
    "leftmtgmiddletemporalgyrus",
    "leftppplanumpolare",
    "leftptplanumtemporale",
    "leftstgsuperiortemporalgyrus",
    "lefttmptemporalpole",
    "leftttgtransversetemporalgyrus",
    "rightfugfusiformgyrus",
    "rightitginferiortemporalgyrus",
    "rightmtgmiddletemporalgyrus",
    "rightppplanumpolare",
    "rightptplanumtemporale",
    "rightstgsuperiortemporalgyrus",
    "righttmptemporalpole",
    "rightttgtransversetemporalgyrus",
    "leftcerebellumwhitematter",
    "leftcerebralwhitematter",
    "opticchiasm",
    "rightcerebellumwhitematter",
    "rightcerebralwhitematter",
    "subjectage",
    "ab1_42",
    "ab1_40",
    "t_tau",
    "p_tau",
]


class TestCaseGenerator(ABC):
    """Parent class for per algorithm test case generators.

    Subclasses should implement compute_expected_output where the expected
    output for any given algorithm is computed, starting from the input_data
    and input_parameters."""

    def __init__(self, expected_path, dataset_path, variable_types, num_test_cases=100):
        self.expected_path = expected_path
        self.all_data = pd.read_csv(dataset_path)
        self.variable_types = variable_types
        self.num_test_cases = num_test_cases
        self._seen_inputs = []

    def draw_variables(self):
        if self.variable_types == "numerical":
            all_variables = [
                varname
                for varname in NUMERICAL_VARS
                if varname in self.all_data.columns
            ]
        elif self.variable_types == "nominal":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown variable type: {self.variable_types}")
        var_num = random.randint(1, len(all_variables))
        input_vars = list(random_permutation(all_variables, r=var_num))
        return input_vars

    def generate_input_parameters(self):
        input_ = {
            "inputdata": {
                "x": self.draw_variables(),
                "pathology": "dementia",
                "datasets": ["desd-synthdata"],
                "filters": "",
            }
        }
        return input_

    def get_input_data(self, input_):
        inputdata = input_["inputdata"]
        x = inputdata["x"]
        datasets = inputdata["datasets"]
        data = self.all_data[x + ["dataset"]]
        data = data[data.dataset.isin(datasets)]
        del data["dataset"]
        data = data.dropna()
        return data if len(data) else None

    @abstractmethod
    def compute_expected_output(self, input_data, input_parameters=None):
        pass

    def generate_test_case(self):
        while True:
            input_ = self.generate_input_parameters()
            if input_ not in self._seen_inputs:
                self._seen_inputs.append(input_)
            else:
                continue
            input_data = self.get_input_data(input_)
            if input_data is not None:
                break
        output = self.compute_expected_output(input_data)
        return {"input": input_, "output": output}

    def generate_test_cases(self):
        test_cases = [self.generate_test_case() for _ in range(self.num_test_cases)]
        return {"test_cases": test_cases}

    def write_test_cases(self):
        test_cases = self.generate_test_cases()
        with open(self.expected_path, "w") as file:
            json.dump(test_cases, file, indent=4)
