from typing import List
from typing import TypeVar

# import numpy
from pydantic import BaseModel

from exareme2 import DType
from exareme2.algorithms.exareme2.algorithm import Algorithm
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader
from exareme2.algorithms.exareme2.helpers import get_transfer_data
from exareme2.algorithms.exareme2.udfgen import DEFERRED
from exareme2.algorithms.exareme2.udfgen import literal
from exareme2.algorithms.exareme2.udfgen import merge_transfer
from exareme2.algorithms.exareme2.udfgen import relation
from exareme2.algorithms.exareme2.udfgen import secure_transfer
from exareme2.algorithms.exareme2.udfgen import state
from exareme2.algorithms.exareme2.udfgen import transfer
from exareme2.algorithms.exareme2.udfgen import udf
from exareme2.worker_communication import BadUserInput

ALGORITHM_NAME = "pca_with_transformation"


class PCADataLoader(AlgorithmDataLoader, algname=ALGORITHM_NAME):
    def get_variable_groups(self):
        return [self._variables.y]


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]


class PCAAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, data, metadata):
        local_run = self.engine.run_udf_on_local_workers
        global_run = self.engine.run_udf_on_global_worker

        [X_relation] = data

        if "data_transformation" in self.algorithm_parameters:
            output_schema = [("row_id", DType.INT)]
            output_schema += [(colname, DType.FLOAT) for colname in X_relation.columns]

            data_transformation = self.algorithm_parameters["data_transformation"]
            # Handle log and exp transformations
            try:
                if any(trans in data_transformation for trans in ["log", "exp"]):
                    local_step_for_data_processing = local_run(
                        func=local_data_processing,
                        keyword_args={
                            "data": X_relation,
                            "data_transformation_dict": {
                                k: v
                                for k, v in data_transformation.items()
                                if k in ["log", "exp"]
                            },
                        },
                        output_schema=output_schema,
                        share_to_global=[False],
                    )
                    X_relation = local_step_for_data_processing
            except Exception as ex:
                if "Log transformation cannot be applied to non-positive values in column." in str(
                    ex
                ) or "Unknown transformation" in str(
                    ex
                ):
                    raise BadUserInput(str(ex))
                raise ex

            # Handle standardize and center transformations
            try:
                if any(
                    trans in data_transformation for trans in ["standardize", "center"]
                ):
                    if "log" in data_transformation or "exp" in data_transformation:
                        # Use the already transformed data
                        local_step_for_data_processing = X_relation

                    local_transfers_stats = local_run(
                        func=local_stats,
                        keyword_args={"x": X_relation},
                        share_to_global=[True],
                    )
                    self.global_stats_state, global_stats_transfer = global_run(
                        func=global_stats,
                        keyword_args=dict(local_transfers=local_transfers_stats),
                        share_to_locals=[True, True],
                    )
                    transformed_data = local_run(
                        func=local_transform,
                        keyword_args=dict(
                            x=X_relation, global_transfer=global_stats_transfer
                        ),
                        output_schema=output_schema,
                        share_to_global=[False],
                    )

                    X_relation = transformed_data
            except Exception as ex:
                if "Unknown transformation" in str(
                    ex
                ) or "Standardization cannot be applied to column" in str(ex):
                    raise BadUserInput(str(ex))
                raise ex

        local_transfers = local_run(
            func=local1,
            keyword_args={"x": X_relation},
            share_to_global=[True],
        )
        global_state, global_transfer = global_run(
            func=global1,
            keyword_args=dict(local_transfers=local_transfers),
            share_to_locals=[False, True],
        )
        local_transfers = local_run(
            func=local2,
            keyword_args=dict(x=X_relation, global_transfer=global_transfer),
            share_to_global=[True],
        )
        result = global_run(
            func=global2,
            keyword_args=dict(local_transfers=local_transfers, prev_state=global_state),
        )
        result = get_transfer_data(result)
        n_obs = result["n_obs"]
        eigenvalues = result["eigenvalues"]
        eigenvectors = result["eigenvectors"]

        result = PCAResult(
            title="Eigenvalues and Eigenvectors",
            n_obs=n_obs,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        return result


S = TypeVar("S")


@udf(x=relation(schema=S), return_type=[secure_transfer(sum_op=True)])
def local1(x):
    n_obs = len(x)
    sx = numpy.einsum("ij->j", x)
    sxx = numpy.einsum("ij,ij->j", x, x)

    transfer_ = {}
    transfer_["n_obs"] = {"data": n_obs, "operation": "sum", "type": "int"}
    transfer_["sx"] = {"data": sx.tolist(), "operation": "sum", "type": "float"}
    transfer_["sxx"] = {"data": sxx.tolist(), "operation": "sum", "type": "float"}
    return transfer_


@udf(local_transfers=secure_transfer(sum_op=True), return_type=[state(), transfer()])
def global1(local_transfers):
    import numpy as np

    n_obs = local_transfers["n_obs"]
    sx = numpy.array(local_transfers["sx"])
    sxx = numpy.array(local_transfers["sxx"])

    means = sx / n_obs
    sigmas = ((sxx - n_obs * means**2) / (n_obs - 1)) ** 0.5

    state_ = dict(n_obs=n_obs)
    transfer_ = dict(means=means.tolist(), sigmas=sigmas.tolist())
    return state_, transfer_


@udf(
    x=relation(schema=S),
    global_transfer=transfer(),
    return_type=[secure_transfer(sum_op=True)],
)
def local2(x, global_transfer):
    means = numpy.array(global_transfer["means"])
    sigmas = numpy.array(global_transfer["sigmas"])

    x = x.values
    out = numpy.empty(x.shape)

    numpy.subtract(x, means, out=out)
    numpy.divide(out, sigmas, out=out)
    gramian = numpy.einsum("ji,jk->ik", out, out)

    transfer_ = {
        "gramian": {
            "data": gramian.tolist(),
            "operation": "sum",
            "type": "float",
        }
    }
    return transfer_


@udf(
    local_transfers=secure_transfer(sum_op=True),
    prev_state=state(),
    return_type=[transfer()],
)
def global2(local_transfers, prev_state):
    gramian = numpy.array(local_transfers["gramian"])

    # Ensure n_obs is greater than 1 to avoid division by zero
    n_obs = prev_state["n_obs"]

    covariance = gramian / (n_obs - 1)

    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors.T

    transfer_ = dict(
        n_obs=n_obs,
        eigenvalues=eigenvalues.tolist(),
        eigenvectors=eigenvectors.tolist(),
    )
    return transfer_


@udf(
    data=relation(schema=S),
    data_transformation_dict=literal(),
    return_type=relation(schema=DEFERRED),
)
def local_data_processing(data, data_transformation_dict):
    """
    Function to normalize a skewed distribution.

    :param data: the actual data passed to the algorithm
    :param data_transformation_dict: the dict passed to the algorithm indicating which variables need to change with which method
    :return: data columns transformed with an error message column if applicable
    """
    import numpy as np
    import pandas as pd

    for transformation, variables in data_transformation_dict.items():
        if transformation == "log":
            for variable in variables:
                if (data[variable] <= 0).any():
                    raise ValueError(
                        f"Log transformation cannot be applied to non-positive values in column '{variable}'."
                    )
                data[variable] = np.log(data[variable])
        elif transformation == "exp":
            for variable in variables:
                data[variable] = np.exp(data[variable])
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

    data_res = pd.DataFrame(data=data, index=data.index, columns=data.columns)

    return data_res


@udf(x=relation(schema=S), return_type=[secure_transfer(sum_op=True)])
def local_stats(x):
    n_obs = len(x)
    sx = numpy.einsum("ij->j", x)
    sxx = numpy.einsum("ij,ij->j", x, x)

    transfer_ = {}
    transfer_["n_obs"] = {"data": n_obs, "operation": "sum", "type": "int"}
    transfer_["sx"] = {"data": sx.tolist(), "operation": "sum", "type": "float"}
    transfer_["sxx"] = {"data": sxx.tolist(), "operation": "sum", "type": "float"}
    return transfer_


@udf(local_transfers=secure_transfer(sum_op=True), return_type=[state(), transfer()])
def global_stats(local_transfers):
    import numpy as np

    n_obs = local_transfers["n_obs"]
    sx = numpy.array(local_transfers["sx"])
    sxx = numpy.array(local_transfers["sxx"])

    means = sx / n_obs
    sigmas = ((sxx - n_obs * means**2) / (n_obs - 1)) ** 0.5

    state_ = dict(n_obs=n_obs)
    transfer_ = dict(means=means.tolist(), sigmas=sigmas.tolist())
    return state_, transfer_


@udf(
    x=relation(schema=S),
    global_transfer=transfer(),
    return_type=relation(schema=DEFERRED),
)
def local_transform(x, global_transfer):
    import numpy as np
    import pandas as pd

    # Extract means and sigmas from global_transfer
    means = np.array(global_transfer.get("means", []))
    sigmas = np.array(global_transfer.get("sigmas", []))

    # Convert input relation x to a Pandas DataFrame if it's not already
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)

    # Apply centering and standardization
    if len(means) > 0 and len(sigmas) > 0:
        for col_name in x.columns:
            if col_name in global_transfer.get("center", []):
                x[col_name] -= means[x.columns.get_loc(col_name)]
            if col_name in global_transfer.get("standardize", []):
                x[col_name] = (
                    x[col_name] - means[x.columns.get_loc(col_name)]
                ) / sigmas[x.columns.get_loc(col_name)]

    return x
