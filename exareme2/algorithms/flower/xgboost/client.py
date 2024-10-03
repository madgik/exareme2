import os
import time
import warnings
from logging import INFO
from math import log2

import flwr as fl
import xgboost as xgb
from flwr.common import Code
from flwr.common import EvaluateIns
from flwr.common import EvaluateRes
from flwr.common import FitIns
from flwr.common import FitRes
from flwr.common import GetParametersIns
from flwr.common import GetParametersRes
from flwr.common import Parameters
from flwr.common import Status
from flwr.common.logger import FLOWER_LOGGER
from flwr.common.logger import log

from exareme2.algorithms.flower.inputdata_preprocessing import fetch_data
from exareme2.algorithms.flower.inputdata_preprocessing import get_input
from exareme2.algorithms.flower.inputdata_preprocessing import preprocess_data

warnings.filterwarnings("ignore", category=UserWarning)


def transform_dataset_to_dmatrix(x, y) -> xgb.core.DMatrix:
    new_data = xgb.DMatrix(x, label=y)
    return new_data


# Hyper-parameters for xgboost training
num_local_round = 1
params = {
    "objective": "binary:logistic",
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


# Define Flower client
class XgbClient(fl.client.Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val):
        self.bst = None
        self.config = None

        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix

        self.num_train = num_train
        self.num_val = num_val

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        # Update trees based on local training data.
        for i in range(num_local_round):
            self.bst.update(train_dmatrix, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_local_round : self.bst.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
            FLOWER_LOGGER.info("Start training at round 1")
            bst = xgb.train(
                params,
                train_dmatrix,
                num_boost_round=num_local_round,
                evals=[(valid_dmatrix, "validate"), (train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self._local_boost()

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": auc},
        )


# Start Flower client
# fl.client.start_client(server_address="127.0.0.1:8080", client=XgbClient().to_client())

if __name__ == "__main__":
    inputdata = get_input()
    full_data = fetch_data(inputdata)
    X_train, y_train = preprocess_data(inputdata, full_data)
    # hard coded for now, later we can split X_train and y_train
    X_valid, y_valid = X_train, y_train

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    train_dmatrix = transform_dataset_to_dmatrix(X_train, y=y_train)
    valid_dmatrix = transform_dataset_to_dmatrix(X_valid, y=y_valid)

    num_train = X_train.shape[0]
    num_val = X_valid.shape[0]

    client = XgbClient(train_dmatrix, valid_dmatrix, num_train, num_val)

    attempts = 0
    max_attempts = int(log2(int(os.environ["TIMEOUT"])))
    while True:
        try:
            fl.client.start_client(
                server_address=os.environ["SERVER_ADDRESS"], client=client.to_client()
            )
            FLOWER_LOGGER.debug("Connection successful on attempt")
            break
        except Exception as e:
            FLOWER_LOGGER.warning(
                f"Connection with the server failed. Attempt {attempts + 1} failed: {e}"
            )
            time.sleep(pow(2, attempts))
            attempts += 1
            if attempts >= max_attempts:
                FLOWER_LOGGER.error("Could not establish connection to the server.")
                raise e
