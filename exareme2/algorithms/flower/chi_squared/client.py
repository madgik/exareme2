import warnings

import flwr as fl
import numpy as np
import pandas as pd
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
from scipy.stats import chi2_contingency

from exareme2.algorithms.flower.inputdata_preprocessing import connect_with_retries
from exareme2.algorithms.flower.inputdata_preprocessing import fetch_data
from exareme2.algorithms.flower.inputdata_preprocessing import get_input
from exareme2.algorithms.flower.inputdata_preprocessing import preprocess_data

warnings.filterwarnings("ignore", category=UserWarning)


# Define Flower client
class ChiSquaredClient(fl.client.Client):
    def __init__(self, X_train, y_train, num_train):
        self.X_train = X_train
        self.y_train = y_train
        self.num_train = num_train

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # Return empty parameters, as this is a non-parameterized algorithm
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        # No training is performed in chi-squared, so just return the current state
        return FitRes(
            status=Status(code=Code.OK, message="No training needed"),
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        try:
            # Check if X_train and y_train are not empty
            if self.X_train.size == 0 or self.y_train.size == 0:
                raise ValueError("Training data is empty")

            # Perform chi-squared test
            contingency_table = pd.crosstab(
                self.y_train, self.X_train[:, 0]
            )  # Adjust if necessary
            chi2, p_value, _, _ = chi2_contingency(contingency_table)

            return EvaluateRes(
                status=Status(code=Code.OK, message="Chi-squared test completed"),
                loss=0.0,  # Define a custom loss if needed
                num_examples=self.num_train,
                metrics={"p_value": p_value},
            )
        except Exception as e:
            FLOWER_LOGGER.error(f"Error during evaluation: {str(e)}")
            return EvaluateRes(
                status=Status(
                    code=Code.INTERNAL, message=str(e)
                ),  # Use Code.INTERNAL or another valid error code
                loss=1.0,  # Return high loss on failure
                num_examples=self.num_train,
                metrics={},
            )


if __name__ == "__main__":
    inputdata = get_input()
    full_data = fetch_data(inputdata)
    X_train, y_train = preprocess_data(inputdata, full_data)

    # Reformat X_train and y_train
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    num_train = X_train.shape[0]

    # Start the Flower client
    client = ChiSquaredClient(X_train, y_train, num_train)
    connect_with_retries(client, "ChiSquaredClient")
