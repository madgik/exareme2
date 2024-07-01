import json
import os
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
import pymonetdb
import requests
from flwr.common.logger import FLOWER_LOGGER
from pydantic import BaseModel
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

# Constants for project directories and environment configurations
CONTROLLER_IP = os.getenv("CONTROLLER_IP", "127.0.0.1")
CONTROLLER_PORT = os.getenv("CONTROLLER_PORT", 5000)
RESULT_URL = f"http://{CONTROLLER_IP}:{CONTROLLER_PORT}/flower/result"
INPUT_URL = f"http://{CONTROLLER_IP}:{CONTROLLER_PORT}/flower/input"
CDES_URL = f"http://{CONTROLLER_IP}:{CONTROLLER_PORT}/cdes_metadata"
HEADERS = {"Content-type": "application/json", "Accept": "text/plain"}


class Inputdata(BaseModel):
    data_model: str
    datasets: List[str]
    filters: Optional[dict]
    y: Optional[List[str]]
    x: Optional[List[str]]


def fetch_data(data_model, datasets, from_db=False) -> pd.DataFrame:
    return (
        _fetch_data_from_db(data_model, datasets)
        if from_db
        else _fetch_data_from_csv(data_model, datasets)
    )


def _fetch_data_from_db(data_model, datasets) -> pd.DataFrame:
    query = f'SELECT * FROM "{data_model}"."primary_data"'
    conn = pymonetdb.connect(
        hostname=os.getenv("MONETDB_IP"),
        port=int(os.getenv("MONETDB_PORT")),
        username=os.getenv("MONETDB_USERNAME"),
        password=os.getenv("MONETDB_PASSWORD"),
        database=os.getenv("MONETDB_DB"),
    )
    df = pd.read_sql(query, conn)
    conn.close()
    df = df[df["dataset"].isin(datasets)]
    return df


def _fetch_data_from_csv(data_model, datasets) -> pd.DataFrame:
    data_folder = Path(f"{os.getenv('DATA_PATH')}/{data_model.split(':')[0]}_v_0_1")
    dataframes = [
        pd.read_csv(data_folder / f"{dataset}.csv")
        for dataset in datasets
        if (data_folder / f"{dataset}.csv").exists()
    ]
    return pd.concat(dataframes, ignore_index=True)


def preprocess_data(inputdata, full_data):
    # Ensure x and y are specified and correct
    if not inputdata.x or not inputdata.y:
        raise ValueError("Input features 'x' and labels 'y' must be specified")

    # Select features and target based on inputdata configuration
    features = full_data[inputdata.x]  # This should be a DataFrame
    target = full_data[inputdata.y].values.ravel()  # Flatten the array if it's 2D

    # Impute missing values for features
    imputer = SimpleImputer(strategy="most_frequent")
    features_imputed = imputer.fit_transform(features)

    # Encode target variable
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(get_enumerations(inputdata.data_model, inputdata.y[0]))
    y_train = label_encoder.transform(target)

    return features_imputed, y_train


def error_handling(error):
    error_msg = {"error": str(error)}
    FLOWER_LOGGER.error(
        f"Error will try to save error message: {error_msg}! Running: {RESULT_URL}..."
    )
    requests.post(RESULT_URL, data=json.dumps(error_msg), headers=HEADERS)


def post_result(result: dict) -> None:
    FLOWER_LOGGER.debug(f"Posting result at: {RESULT_URL} ...")
    response = requests.post(RESULT_URL, data=json.dumps(result), headers=HEADERS)
    if response.status_code != 200:
        error_handling(response.text)


def get_input() -> Inputdata:
    FLOWER_LOGGER.debug(f"Getting inputdata from: {INPUT_URL} ...")
    response = requests.get(INPUT_URL)
    if response.status_code != 200:
        error_handling(response.text)
    input_data = Inputdata.parse_raw(response.text)
    return input_data


def get_enumerations(data_model: str, variable_name: str) -> list:
    try:
        FLOWER_LOGGER.debug(f"Getting enumerations from: {CDES_URL} ...")
        response = requests.get(CDES_URL)
        if response.status_code != 200:
            error_handling(response.text)
        cdes_metadata = response.json()
        if data_model not in cdes_metadata:
            raise KeyError(f"'{data_model}' key not found in cdes_metadata")

        if variable_name not in cdes_metadata[data_model]:
            raise KeyError(f"'{variable_name}' key not found in {data_model}")

        enumerations = cdes_metadata[data_model][variable_name].get("enumerations")
        if enumerations is not None:
            return [code for code, label in enumerations.items()]
        else:
            raise KeyError(f"'enumerations' key not found in {variable_name}")
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        error_handling(str(e))
