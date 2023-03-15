import json
from typing import List
from typing import TypeVar

import numpy
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics.pairwise import euclidean_distances

from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.algorithm_specification import InputDataSpecification
from mipengine.algorithm_specification import InputDataSpecifications
from mipengine.algorithm_specification import InputDataStatType
from mipengine.algorithm_specification import InputDataType
from mipengine.algorithm_specification import ParameterSpecification
from mipengine.algorithm_specification import ParameterType
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.helpers import get_transfer_data
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr

# from mipengine.udfgen import scalar
# from mipengine.udfgen import TensorBinaryOp
# from mipengine.udfgen import TensorUnaryOp
from mipengine.udfgen import literal
from mipengine.udfgen import merge_tensor
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import secure_transfer
from mipengine.udfgen import state
from mipengine.udfgen import tensor
from mipengine.udfgen import transfer
from mipengine.udfgen import udf

T = TypeVar("T")
S = TypeVar("S")


class KmeansResult(BaseModel):
    title: str
    centers: List[List[float]]


class KMeansAlgorithm(Algorithm, algname="kmeans"):
    @classmethod
    def get_specification(cls):
        return AlgorithmSpecification(
            name=cls.algname,
            desc="K-Means",
            label="K-Means",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="Features",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    notblank=True,
                    multiple=True,
                ),
            ),
            parameters={
                "k": ParameterSpecification(
                    label="k",
                    desc="k",
                    types=[ParameterType.INT],
                    notblank=True,
                    multiple=False,
                    default=4,
                    min=1,
                    max=100,
                ),
                "maxiter": ParameterSpecification(
                    label="maxiter",
                    desc="maxiter",
                    types=[ParameterType.INT],
                    notblank=True,
                    multiple=False,
                    default=1,
                    min=1,
                    max=100,
                ),
                "tol": ParameterSpecification(
                    label="tol",
                    desc="tol",
                    types=[ParameterType.REAL],
                    notblank=True,
                    multiple=False,
                    default=0.0001,
                    min=0.0,
                    max=1.0,
                ),
            },
        )

    def get_variable_groups(self):
        # return [self.executor.y_variables]
        return [self.variables.y]

    def run(self, engine):
        local_run = engine.run_udf_on_local_nodes
        global_run = engine.run_udf_on_global_node

        [X_relation] = engine.data_model_views

        n_clusters = self.algorithm_parameters["k"]
        tol = self.algorithm_parameters["tol"]
        maxiter = self.algorithm_parameters["maxiter"]

        X_not_null = local_run(
            func=relation_to_matrix,
            positional_args=[X_relation],
        )

        min_max_transfer = local_run(
            func=init_centers_local2,
            positional_args=[X_not_null],
            share_to_global=[True],
        )

        global_state, global_result = global_run(
            func=init_centers_global2,
            positional_args=[min_max_transfer, n_clusters],
            share_to_locals=[False, True],
        )

        curr_iter = 0
        centers_to_compute = global_result
        init_centers = json.loads(centers_to_compute.get_table_data()[0][0])["centers"]

        # init_centers = get_transfer_data(centers_to_compute)['centers']
        init_centers_array = numpy.array(init_centers)
        init_centers_list = init_centers_array.tolist()
        while True:
            label_state = local_run(
                func=compute_cluster_labels,
                positional_args=[X_not_null, centers_to_compute],
                share_to_global=[False],
            )

            metrics_local = local_run(
                func=compute_metrics,
                positional_args=[X_not_null, label_state, n_clusters],
                share_to_global=[True],
            )

            new_centers = global_run(
                func=compute_centers_from_metrics,
                positional_args=[metrics_local, min_max_transfer, n_clusters],
                share_to_locals=[True],
            )

            curr_iter += 1

            old_centers = json.loads(centers_to_compute.get_table_data()[0][0])[
                "centers"
            ]
            # old_centers = get_transfer_data(centers_to_compute)['centers']
            old_centers_array = numpy.array(old_centers)

            print(new_centers.get_table_data())
            new_centers_obj = json.loads(new_centers.get_table_data()[0][0])["centers"]
            # new_centers_obj = get_transfer_data(new_centers)['centers']
            new_centers_array = numpy.array(new_centers_obj)

            diff = numpy.linalg.norm(new_centers_array - old_centers_array, "fro")

            if (curr_iter >= maxiter) or (diff <= tol):
                ret_obj = KmeansResult(
                    title="K-Means Centers",
                    centers=new_centers_array.tolist(),
                )
                print("finished after " + str(curr_iter))
                return ret_obj

            else:
                centers_to_compute = new_centers
        return ret_obj


@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel


@udf(a=tensor(T, 2), return_type=tensor(T, 2))
def remove_nulls(a):
    a_sel = a[~numpy.isnan(a).any(axis=1)]
    return a_sel


@udf(X=tensor(T, 2), n_clusters=literal(), return_type=[transfer()])
def init_centers_local(X, n_clusters):
    from sklearn.utils import check_random_state

    seed = 123
    n_samples = X.shape[1]
    random_state = check_random_state(seed)
    seeds = random_state.permutation(n_samples)[:n_clusters]
    centers = X[seeds]
    # np.random.rand(n_samples,n_clusters)
    transfer_ = {"centers": centers.tolist()}
    return transfer_


@udf(
    X=tensor(T, 2), return_type=[secure_transfer(sum_op=True, min_op=True, max_op=True)]
)
def init_centers_local2(X):
    import numpy

    min_vals = numpy.nanmin(X, axis=0)

    max_vals = numpy.nanmax(X, axis=0)

    secure_transfer_ = {
        "min": {"data": min_vals.tolist(), "operation": "min", "type": "float"},
        "max": {"data": max_vals.tolist(), "operation": "max", "type": "float"},
    }

    return secure_transfer_


@udf(
    min_max_transfer=secure_transfer(sum_op=True, min_op=True, max_op=True),
    n_clusters=literal(),
    return_type=[state(), transfer()],
)
def init_centers_global2(min_max_transfer, n_clusters):
    import numpy

    min_vals = min_max_transfer["min"]
    max_vals = min_max_transfer["max"]

    min_array = numpy.array(min_vals)
    max_array = numpy.array(max_vals)

    random_state = numpy.random.RandomState(seed=123)

    centers_global = random_state.uniform(
        low=min_array, high=max_array, size=(n_clusters, min_array.shape[0])
    )

    transfer_ = {"centers": centers_global.tolist()}
    state_ = {"centers": centers_global.tolist()}
    return state_, transfer_


@udf(
    centers_transfer=merge_transfer(),
    n_clusters=literal(),
    return_type=[state(), transfer()],
)
def init_centers_global(centers_transfer, n_clusters):
    centers_all = []
    for curr_transfer in centers_transfer:
        centers_all.append(curr_transfer["centers"])
    centers_merged = numpy.vstack(centers_all)
    centers_global = centers_merged[:n_clusters]

    transfer_ = {"centers": centers_global.tolist()}
    state_ = {"centers": centers_global.tolist()}
    return state_, transfer_


@udf(X=tensor(dtype=T, ndims=2), global_transfer=transfer(), return_type=state())
def compute_cluster_labels(X, global_transfer):
    from sklearn.metrics.pairwise import euclidean_distances

    centers = numpy.array(global_transfer["centers"])
    distances = euclidean_distances(X, centers)

    labels = numpy.argmin(distances, axis=1)

    return_labels = {"labels": labels.tolist()}

    return return_labels


@udf(
    X=tensor(dtype=T, ndims=2),
    label_state=state(),
    n_clusters=literal(),
    return_type=secure_transfer(sum_op=True, min_op=True, max_op=True),
)
def compute_metrics(X, label_state, n_clusters):
    labels = numpy.array(label_state["labels"])
    sum_list = []
    count_list = []
    for i in range(n_clusters):
        relevant_features = numpy.where(labels == i)
        X_clust = X[relevant_features, :]
        X_clust = X_clust[0, :, :]
        X_sum = numpy.sum(X_clust, axis=0)
        X_count = X_clust.shape[0]
        # raise ValueError(str(relevant_features)+''+str(labels.shape)+' '+str(X.shape)+' '+str(X_sum.shape)+' '+str(X_clust.shape))
        # metrics[i] = {'X_sum': X_sum.tolist(),'X_count':X_count}
        sum_list.append(X_sum.tolist())
        count_list.append(X_count)

    secure_transfer_ = {}
    secure_transfer_["sum_list"] = {
        "data": sum_list,
        "operation": "sum",
        "type": "float",
    }
    secure_transfer_["count_list"] = {
        "data": count_list,
        "operation": "sum",
        "type": "int",
    }
    return secure_transfer_


@udf(
    transfers=secure_transfer(sum_op=True, min_op=True, max_op=True),
    min_max_transfer=secure_transfer(sum_op=True, min_op=True, max_op=True),
    n_clusters=literal(),
    return_type=transfer(),
)
def compute_centers_from_metrics(transfers, min_max_transfer, n_clusters):
    centers = []
    # raise ValueError(transfers)
    sum_list_sum = transfers["sum_list"]
    count_list_sum = transfers["count_list"]

    min_array = min_max_transfer["min"]
    max_array = min_max_transfer["max"]

    sum_array = numpy.array(sum_list_sum)
    count_array = numpy.array(count_list_sum)

    generate_random = False
    generate_uniform_clusters = True

    n_dim = sum_array.shape[1]
    for i in range(n_clusters):
        curr_sum = numpy.zeros((1, n_dim))
        curr_count = 0
        curr_sum += sum_array[i, :]
        curr_count += count_array[i]

        if curr_count != 0:
            final_i = curr_sum / curr_count
        else:
            if generate_random:
                min_array2 = numpy.array(min_array)
                max_array2 = numpy.array(max_array)
                if generate_uniform_clusters:
                    final_i = numpy.random.uniform(
                        low=0.0, high=n_clusters + 1, size=(1, n_dim)
                    )
                elif generate_uniform:
                    final_i = numpy.random.uniform(
                        low=min_array2, high=max_array, size=(1, n_dim)
                    )
                else:
                    for i in range(n_dim):
                        min_value = min_array2[i]
                        max_value = max_array2[i]
                        curr_value = (
                            min_value + (max_value - min_value) * numpy.random.randn()
                        )
                        final_i[0][i] = curr_value

            else:
                final_i = numpy.zeros((1, n_dim))

        # final_i = curr_sum / curr_count
        centers.append(final_i)
    centers_array = numpy.vstack(centers)
    # raise ValueError(centers_array.shape)
    ret_val = centers_array.tolist()
    ret_val2 = {"centers": ret_val}
    return ret_val2
