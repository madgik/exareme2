import json
import os
import time
from math import log2
from math import pow
from typing import List
from typing import Optional

import flwr as fl
import pandas as pd
import requests
from flwr.common.logger import FLOWER_LOGGER
from pydantic import BaseModel
from sklearn import preprocessing

from exareme2.algorithms.flower.df_filter import apply_filter

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
    validation_datasets: List[str]
    filters: Optional[dict]
    y: Optional[List[str]]
    x: Optional[List[str]]


def apply_inputdata(df: pd.DataFrame, inputdata: Inputdata) -> pd.DataFrame:
    if inputdata.filters:
        df = apply_filter(df, inputdata.filters)
    df = df[df["dataset"].isin(inputdata.datasets + inputdata.validation_datasets)]
    columns = inputdata.x + inputdata.y
    df = df[columns]
    df = df.dropna(subset=columns)
    return df


def fetch_data(inputdata) -> pd.DataFrame:
    dataframes = [
        pd.read_csv(f"{os.getenv('DATA_PATH')}{csv_path}")
        for csv_path in os.getenv("CSV_PATHS").split(",")
    ]
    df = pd.concat(dataframes, ignore_index=True)
    return apply_inputdata(df, inputdata)


from sklearn import preprocessing


def preprocess_data(inputdata, full_data):
    # Ensure x and y are specified and correct
    if not inputdata.x or not inputdata.y:
        raise ValueError("Input features 'x' and labels 'y' must be specified")

    # Select features (X) and target (y) based on inputdata configuration
    features = full_data[inputdata.x]  # X: Independent variables
    target = full_data[inputdata.y].values.ravel()  # y: Dependent variable, flattened

    # Encode target (y) variable if categorical
    label_encoder_y = preprocessing.LabelEncoder()
    label_encoder_y.fit(get_enumerations(inputdata.data_model, inputdata.y[0]))
    y_train = label_encoder_y.transform(target)

    # Encode features (x) if they are categorical
    for col in features.columns:
        if features[col].dtype == "object" or isinstance(
            features[col].dtype, pd.CategoricalDtype
        ):
            label_encoder_x = preprocessing.LabelEncoder()
            features[col] = label_encoder_x.fit_transform(features[col])

    return features, y_train


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


def connect_with_retries(client, client_name):
    """
    Attempts to connect the client to the Flower server with retries.

    Args:
        client: The client instance to connect.
        client_name: The name of the client (for logging purposes).
    """
    try:
        fl.client.start_client(
            server_address=os.environ["SERVER_ADDRESS"], client=client.to_client()
        )
        FLOWER_LOGGER.debug(f"{client_name} - Connection successful.")
    except Exception:
        try:
            time.sleep(1)
            fl.client.start_client(
                server_address=os.environ["SERVER_ADDRESS"], client=client.to_client()
            )
            FLOWER_LOGGER.debug(f"{client_name} - Connection successful.")
        except Exception as e:
            FLOWER_LOGGER.error(
                f"{client_name} - Could not establish connection to the server."
            )
            raise e
