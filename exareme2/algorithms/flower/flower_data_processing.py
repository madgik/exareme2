import json
import os
from pathlib import Path
from typing import List
from typing import Optional

import pandas as pd
import pymonetdb
import requests
from pydantic import BaseModel
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

# Constants for project directories and environment configurations
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Inputdata(BaseModel):
    data_model: str
    datasets: List[str]
    filters: Optional[dict]
    y: Optional[List[str]]
    x: Optional[List[str]]

    def fetch_data(self, from_db=False) -> pd.DataFrame:
        return self._fetch_from_db() if from_db else self._load_from_csv()

    def _fetch_from_db(self) -> pd.DataFrame:
        query = f'SELECT * FROM "{self.data_model}"."primary_data"'
        with pymonetdb.connect(
            {
                "hostname": os.getenv("MONETDB_IP"),
                "port": int(os.getenv("MONETDB_PORT")),
                "username": os.getenv("MONETDB_USERNAME"),
                "password": os.getenv("MONETDB_PASSWORD"),
                "database": os.getenv("MONETDB_DB"),
            }
        ) as conn:
            df = pd.read_sql(query, conn)
        df = df[df["dataset"].isin(self.datasets)]
        return df

    def _load_from_csv(self) -> pd.DataFrame:
        data_folder = (
            PROJECT_ROOT
            / "tests"
            / "test_data"
            / f"{self.data_model.split(':')[0]}_v_0_1"
        )
        dataframes = [
            pd.read_csv(data_folder / f"{dataset}.csv")
            for dataset in self.datasets
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


def post_result(result: dict) -> None:
    url = "http://127.0.0.1:5000/flower/result"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    requests.post(url, data=json.dumps(result), headers=headers)


def get_input() -> Inputdata:
    response = requests.get("http://127.0.0.1:5000/flower/input")
    return Inputdata.parse_raw(response.text)


def get_enumerations(data_model, variable_name):
    response = requests.get("http://127.0.0.1:5000/cdes_metadata")
    cdes_metadata = json.loads(response.text)
    enumerations = cdes_metadata[data_model][variable_name]["enumerations"]
    return [code for code, label in enumerations.items()]
