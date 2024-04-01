from exareme2.algorithms.exareme2.udfgen import literal
from exareme2.algorithms.exareme2.udfgen import secure_transfer
from exareme2.algorithms.exareme2.udfgen import transfer
from exareme2.algorithms.exareme2.udfgen import udf


@udf(
    params=secure_transfer(sum_op=True),
    num_local_workers=literal(),
    return_type=[transfer()],
)
def fed_average(params, num_local_workers):
    """
    Computes federation average for a set of model parameters

    Parameters
    ----------
    params : dict
        Model parameters to be averaged. Should be a dictionary where each value
        is a list representation of an n-dimensional numpy array.
    num_local_workers : int
        Number of local workers involved in the experiment.
    """
    import numpy as np

    averaged_params = {
        key: (np.array(val) / num_local_workers).tolist() for key, val in params.items()
    }
    return averaged_params
