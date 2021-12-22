import random
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from os import path


def create_random_subset(variables_list, all_test_var_list):
    # variables_list = [
    #     "av45",
    #     "fdg",
    #     "pib",
    #     "brainstem",
    #     "tiv",
    #     "_3rdventricle",
    #     "_4thventricle",
    #     "csfglobal",
    #     "leftinflatvent",
    #     "leftlateralventricle",
    #     "rightinflatvent",
    #     "rightlateralventricle",
    #     "cerebellarvermallobulesiv",
    #     "cerebellarvermallobulesviiix",
    #     "cerebellarvermallobulesvivii",
    #     "leftcerebellumexterior",
    #     "rightcerebellumexterior",
    #     "leftamygdala",
    #     "rightamygdala",
    #     "leftaccumbensarea",
    #     "leftbasalforebrain",
    #     "leftcaudate",
    #     "leftpallidum",
    #     "leftputamen",
    #     "rightaccumbensarea",
    #     "rightbasalforebrain",
    #     "rightcaudate",
    #     "rightpallidum",
    #     "rightputamen",
    #     "leftventraldc",
    #     "rightventraldc",
    #     "leftaorganteriororbitalgyrus",
    #     "leftcocentraloperculum",
    #     "leftfofrontaloperculum",
    #     "leftfrpfrontalpole",
    #     "leftgregyrusrectus",
    #     "leftlorglateralorbitalgyrus",
    #     "leftmfcmedialfrontalcortex",
    #     "leftmfgmiddlefrontalgyrus",
    #     "leftmorgmedialorbitalgyrus",
    #     "leftmprgprecentralgyrusmedialsegment",
    #     "leftmsfgsuperiorfrontalgyrusmedialsegment",
    #     "leftopifgopercularpartoftheinferiorfrontalgyrus",
    #     "leftorifgorbitalpartoftheinferiorfrontalgyrus",
    #     "leftpoparietaloperculum",
    #     "leftporgposteriororbitalgyrus",
    #     "leftprgprecentralgyrus",
    #     "leftscasubcallosalarea",
    #     "leftsfgsuperiorfrontalgyrus",
    #     "leftsmcsupplementarymotorcortex",
    #     "lefttrifgtriangularpartoftheinferiorfrontalgyrus",
    #     "rightaorganteriororbitalgyrus",
    #     "rightcocentraloperculum",
    #     "rightfofrontaloperculum",
    #     "rightfrpfrontalpole",
    #     "rightgregyrusrectus",
    #     "rightlorglateralorbitalgyrus",
    #     "rightmfcmedialfrontalcortex",
    #     "rightmfgmiddlefrontalgyrus",
    #     "rightmorgmedialorbitalgyrus",
    #     "rightmprgprecentralgyrusmedialsegment",
    #     "rightmsfgsuperiorfrontalgyrusmedialsegment",
    #     "rightopifgopercularpartoftheinferiorfrontalgyrus",
    #     "rightorifgorbitalpartoftheinferiorfrontalgyrus",
    #     "rightpoparietaloperculum",
    #     "rightporgposteriororbitalgyrus",
    #     "rightprgprecentralgyrus",
    #     "rightscasubcallosalarea",
    #     "rightsfgsuperiorfrontalgyrus",
    #     "rightsmcsupplementarymotorcortex",
    #     "righttrifgtriangularpartoftheinferiorfrontalgyrus",
    #     "leftainsanteriorinsula",
    #     "leftpinsposteriorinsula",
    #     "rightainsanteriorinsula",
    #     "rightpinsposteriorinsula",
    #     "leftacgganteriorcingulategyrus",
    #     "leftententorhinalarea",
    #     "lefthippocampus",
    #     "leftmcggmiddlecingulategyrus",
    #     "leftpcggposteriorcingulategyrus",
    #     "leftphgparahippocampalgyrus",
    #     "leftthalamusproper",
    #     "rightacgganteriorcingulategyrus",
    #     "rightententorhinalarea",
    #     "righthippocampus",
    #     "rightmcggmiddlecingulategyrus",
    #     "rightpcggposteriorcingulategyrus",
    #     "rightphgparahippocampalgyrus",
    #     "rightthalamusproper",
    #     "leftcalccalcarinecortex",
    #     "leftcuncuneus",
    #     "leftioginferioroccipitalgyrus",
    #     "leftliglingualgyrus",
    #     "leftmogmiddleoccipitalgyrus",
    #     "leftocpoccipitalpole",
    #     "leftofugoccipitalfusiformgyrus",
    #     "leftsogsuperioroccipitalgyrus",
    #     "rightcalccalcarinecortex",
    #     "rightcuncuneus",
    #     "rightioginferioroccipitalgyrus",
    #     "rightliglingualgyrus",
    #     "rightmogmiddleoccipitalgyrus",
    #     "rightocpoccipitalpole",
    #     "rightofugoccipitalfusiformgyrus",
    #     "rightsogsuperioroccipitalgyrus",
    #     "leftangangulargyrus",
    #     "leftmpogpostcentralgyrusmedialsegment",
    #     "leftpcuprecuneus",
    #     "leftpogpostcentralgyrus",
    #     "leftsmgsupramarginalgyrus",
    #     "leftsplsuperiorparietallobule",
    #     "rightangangulargyrus",
    #     "rightmpogpostcentralgyrusmedialsegment",
    #     "rightpcuprecuneus",
    #     "rightpogpostcentralgyrus",
    #     "rightsmgsupramarginalgyrus",
    #     "rightsplsuperiorparietallobule",
    #     "leftfugfusiformgyrus",
    #     "leftitginferiortemporalgyrus",
    #     "leftmtgmiddletemporalgyrus",
    #     "leftppplanumpolare",
    #     "leftptplanumtemporale",
    #     "leftstgsuperiortemporalgyrus",
    #     "lefttmptemporalpole",
    #     "leftttgtransversetemporalgyrus",
    #     "rightfugfusiformgyrus",
    #     "rightitginferiortemporalgyrus",
    #     "rightmtgmiddletemporalgyrus",
    #     "rightppplanumpolare",
    #     "rightptplanumtemporale",
    #     "rightstgsuperiortemporalgyrus",
    #     "righttmptemporalpole",
    #     "rightttgtransversetemporalgyrus",
    #     "leftcerebellumwhitematter",
    #     "leftcerebralwhitematter",
    #     "opticchiasm",
    #     "rightcerebellumwhitematter",
    #     "rightcerebralwhitematter",
    #     "alzheimerbroadcategory_bin",
    #     "minimentalstate",
    #     "montrealcognitiveassessment",
    #     "updrstotal",
    #     "subjectage",
    #     "subjectageyears",
    #     "ab1_42",
    #     "ab1_40",
    #     "t_tau",
    #     "p_tau"
    # ]

    # gets the length of the variable list
    var_num = len(variables_list)
    # max number of variables
    max_range = int(var_num / 10)
    # decides on the number of variables that will be used in request
    rand = random.randint(1, max_range)
    # creates random indices
    randlist = random.sample(range(1, var_num), rand)
    # pick values based on indices
    test_var_list = [variables_list[i] for i in randlist]
    if check_var_list(all_test_var_list, test_var_list, variables_list):
        all_test_var_list.append(test_var_list)

    json_input = {
        "input": {
            "inputdata": {
                "x": [],
                "pathology": "dementia",
                "datasets": ["desd-synthdata"],
                "filters": "",
            }
        }
    }

    return json_input, all_test_var_list, test_var_list


def check_var_list(all_test_var_list, test_subset, variables_list):
    if all_test_var_list:
        for i in all_test_var_list:
            if set(i) == set(test_subset):
                create_random_subset(variables_list, all_test_var_list)
    return True


def create_json_file(path=None):
    expected_filename = "tests/algorithms/expected/pca_expected.json"
    # read csv
    df_desd = pd.read_csv("tests/demo_data/dementia/desd-synthdata.csv")
    # get only numeric columns
    df_numeric = df_desd.select_dtypes(include=[np.number])
    # get all column names of dataset
    variables_list = list(df_numeric.columns)

    json_file = {"test_cases": []}

    all_test_vars = []
    for i in range(1, 100):
        json_input, all_test_vars, iteration_subset_vars = create_random_subset(
            variables_list, all_test_vars
        )
        ds = pd.DataFrame(df_numeric[iteration_subset_vars])
        # Find the columns where each value is null
        empty_cols = [col for col in ds.columns if ds[col].isnull().all()]
        # Drop these columns from the dataframe
        ds.drop(empty_cols, axis=1, inplace=True)

        # drop full na columns
        # dropna_cols = ds.dropna(axis=1, how='all')
        # drop na rows
        cols_final = ds.dropna()
        col_names = list(cols_final.columns)
        if not col_names:
            return
        output = get_expected(cols_final, col_names)
        json_input["input"]["inputdata"]["x"] = col_names
        json_input["output"] = output
        json_file.get("test_cases").append(json_input)

    # Check if file exists
    # if path.isfile(expected_filename) is True:
    #     print("File exists")
    #     return

    with open(expected_filename, "w") as fp:
        json.dump(json_file, fp, ensure_ascii=False, indent=4)


def get_expected(df, test_subset):
    # get subset of columns for the iteration
    iter_cols = df[test_subset]
    n_obs = len(iter_cols)
    if n_obs == 0 or iter_cols.shape[0] < iter_cols.shape[1]:
        return None
    iter_cols -= np.mean(iter_cols)
    iter_cols /= np.std(iter_cols)
    pca = PCA()
    pca.fit(iter_cols)
    json_output = {
        "n_obs": n_obs,
        "eigen_vals": pca.explained_variance_.tolist(),
        "eigen_vecs": pca.components_.tolist(),
    }

    return json_output


create_json_file()
