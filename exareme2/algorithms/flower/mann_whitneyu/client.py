import warnings

import flwr as fl
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
from scipy.stats import mannwhitneyu

from exareme2.algorithms.flower.inputdata_preprocessing import connect_with_retries
from exareme2.algorithms.flower.inputdata_preprocessing import fetch_data
from exareme2.algorithms.flower.inputdata_preprocessing import get_input
from exareme2.algorithms.flower.inputdata_preprocessing import preprocess_data

warnings.filterwarnings("ignore", category=UserWarning)


# Define Flower client
class MannWhitneyClient(fl.client.Client):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        local_model_bytes = bytes()

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(
                tensor_type="", tensors=[local_model_bytes]
            ),  # Empty model
            num_examples=len(self.y_train),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        group_1 = self.X_train[self.y_train == 1]
        group_0 = self.X_train[self.y_train == 0]

        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(group_1, group_0, alternative="two-sided")

        # Convert the p_value to a valid type (float)
        p_value = float(p_value)

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=0.0,
            num_examples=len(self.y_train),
            metrics={"p_value": p_value},
        )


if __name__ == "__main__":
    inputdata = get_input()
    full_data = fetch_data(inputdata)
    X_train, y_train = preprocess_data(inputdata, full_data)

    client = MannWhitneyClient(X_train, y_train)
    connect_with_retries(client, "MannWhitneyClient")
