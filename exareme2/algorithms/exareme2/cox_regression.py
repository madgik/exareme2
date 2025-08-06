def _compute_time_bins(self, times):
    """
    Compute the time bins for the stacking.
    This is done by computing the local maximum event time to the server.
    The server returns the time bin division to the clients.

    :param times: The times at the local clients
    :return: The global maximum event time
    """
    # Share local max time with server
    local_max_time = times.max(initial=0)
    #send local max
    # Receive and return time bins from server
    #receive time bins from server

    return 0


def run_client(
    covariates,
    times,
    events,
):
    """
    Perform the learning process.

    :param covariates: The covariates of the patients. Can have multiple columns.
    :param times: The failure/censoring times.
    :param events: The event indicators. Should contain boolean values.
    :return: The resulting model.
    """
    # Set default value for time bins
    logger.info("Computing time bins..")
    time_bins = _compute_time_bins(times)
    n_covariates_ = covariates.shape[1]

    # Stack the data
    logger.info("Stacking the data..")
    _stacked_data_,_stacked_target_ = stack(
        covariates=covariates,
        times=times,
        events=events,
        time_bins=time_bins,
    )
    logger.info("Finished stacking, starting Logistic Regression")

    # Perform the logistic regression
    logger.info("Running Logistic Regression..")
    _model_ = log_reg_solver.run(
        _stacked_data_.astype(np.float64),_stacked_target_
    )

    return _model_

def _split_time_bins(global_max_time):
    """
    Compute time bins evenly given a max event time.

    :param global_max_time: The global max event time
    :return: The time bins
    """
    return np.linspace(0, global_max_time + 1, self.n_time_bins)

def _compute_global_max_time():
    """
    Receive local maximum event times and distribute global maximum event time.
    """
    local_max_times = 0
    return float(max(local_max_times))
def _compute_time_bins(time_bins):
    """
    Compute time bins, based on the input of the clients.

    :param time_bins: Optional parameter specifying the time bins.
        If None, the bins will be spaced according to the _split_time_bins function.
    :return: The time bins
    :raises ValueError: If time bins are smaller than maximum time.
    """
    global_max_time = _compute_global_max_time()

    return _split_time_bins(global_max_time)



def stack(
    covariates: CovariatesType,
    times: TimesType,
    events: EventsType,
    time_bins: TimeBinType | None = None,
) -> tuple[DataType, TargetType]:
    """This function stacks a dataset as described in https://arxiv.org/pdf/2107.13480.pdf.
    The input is in the form of separate numpy arrays. So for patient 1, we have its covariates in covariates[0],
    its failure time in times[0], and its event indicator in failed[0]. All these arrays should have the same length.
    Time bins allow for discretized stacking, where there is one stacked block for each time bin.

    :param covariates: The covariates of the patients. Can have multiple columns.
    :param times: The start time of the time intervals.
    :param events: The event indicators. Should contain boolean values.
    :param time_bins: If provided, a discrete stacker is used. The parameter should contain the starting times of each
        time interval. E.g. [0, 200, 400, 600] denotes time intervals 0-200, 200-400, and 400-600. Time bins are closed
        on the left and open on the right. That is, 200-400, includes 200, but not 400. It's first value must
        be zero, and its largest value must be bigger than the biggest failure/censoring time value.
    :returns: The stacked data set in the form a multidimensional array containing the input data and a vector
      containing the target data.
    """

    # Initialize default parameters
    ids = np.arange(len(covariates))
    start_times = np.zeros(len(covariates))
    end_times = times

    # Return time varying stacked data set
    return stack_time_varying(
        ids, covariates, start_times, end_times, events, time_bins
    )


def stack_time_varying(
    ids: IdsType,
    covariates: CovariatesType,
    start_times: TimesType,
    end_times: TimesType,
    events: EventsType,
    time_bins: TimesType | None = None,
) -> tuple[DataType, TargetType]:
    """This function stacks a time varying dataset as described in https://arxiv.org/pdf/2107.13480.pdf.
    The input is in the form of separate numpy arrays. So for the first interval of the first patient, we have
    its patient id in ids[0], its covariates in covariates[0], the interval times in start_times[0] and
    end_times[0] and its event indicator in failed[0]. All these arrays should have the same length.
    Time bins allow for discretized stacking, where there is one stacked block for each time bin. When using
    discretization, the situation at the beginning of a time interval decides the covariates for that time bin.
    E.g. if a covariate changes within a time interval, this is recorded at the start of the next interval.
    Hence, if a covariate changes twice during an interval, the intermediate value will be lost.
    Patients censored within a time bin will be recorded as having survived the bin.
    When a patient fails in a time interval, it is recorded as having failed at the end of the time interval.

    :param ids: The patient ids. Can be used to specify time-varying covariates. The id is unique per patient and a
        patient can have multiple rows. However, a patient id can have only one failure.
    :param covariates: The covariates of the patients. Can have multiple columns.
    :param start_times: The start time of the time intervals.
    :param end_times: The end times of the time intervals.
    :param events: The event indicators. Should contain boolean values.
    :param time_bins: If provided, a discrete stacker is used. The parameter should contain the starting times of each
        time interval. E.g. [0, 200, 400, 600] denotes time intervals 0-200, 200-400, and 400-600. It's first value must
        be zero, and its largest value must be bigger than the biggest failure/censoring time value.
    :returns: The stacked data set in the form a multidimensional array containing the input data and a vector
      containing the target data.
    """

    # Validate the input - raises a ValueError in case of problems
    _validate_input(covariates, start_times, end_times, events, ids, time_bins)

    # Normalize the data, filling `ids` and sorting all arrays by time
    _normalize_data(covariates, start_times, end_times, events, ids)

    # Set defaults for optional parameter time bins.
    # If not set, set time bins at the failure times in the data.
    if time_bins is None:
        time_bins = np.append(
            np.unique(end_times[np.where(events > 0)]), end_times[-1] + 1
        )

    # Indices for the `times` array at which the time bins start
    bin_start_indices = np.array(
        [np.searchsorted(end_times, bin_start) for bin_start in time_bins]
    )

    return _stack_using_bins(covariates, events, ids, bin_start_indices)


def _validate_input(
    covariates,
    start_times,
    end_times,
    events,
    ids,
    time_bins,
):
    """Validates that the inputs to `stack` are of the correct shape,
    and that the time bins start at 0, are increasing, and contain all
    time measurements in the data.

    :param covariates: The covariates of the patients. Can have multiple columns.
    :param start_times: The start times of the time intervals.
    :param end_times: The end times of the time intervals.
    :param events: The event indicators. Should contain boolean values.
    :param ids: The patient ids. Can be used to specify time-varying covariates.
        The id is unique per patient and a patient can have multiple rows.
        However, a patient id can have only one failure.
    :param time_bins: If provided, a discrete stacker is used.
        The parameter should contain the starting/ending times of each bin.

    :raises ValueError: if inconsistencies are found, raises a ValueError explaining the problem.
    """

    n_samples = covariates.shape[0]
    if (
        start_times.shape != (n_samples,)
        or end_times.shape != (n_samples,)
        or events.shape != (n_samples,)
        or ids.shape != (n_samples,)
    ):
        raise ValueError("Not all parameters have the same lengths.")

    if np.min(end_times) <= 0:
        raise ValueError("All failure times must be greater than 0.")
    if np.min(start_times) < 0:
        raise ValueError("All start times must be positive.")
    if not (start_times < end_times).all():
        raise ValueError("End times should be strictly greater than the start times.")

    if time_bins is not None:
        if any(time_bins[i] >= time_bins[i + 1] for i in range(len(time_bins[:-1]))):
            raise ValueError("Time bins must be a strictly increasing sequence.")
        if time_bins[-1] < np.max(end_times):
            raise ValueError("The maximum time value is larger than the last time bin.")


def _normalize_data(
    covariates,
    start_times,
    end_times,
    events,
    ids,
):
    """Sort the data by event time, and generate IDs if necessary

    :param covariates: The covariates of the patients. Can have multiple columns.
    :param start_times: The start times of the time intervals.
    :param end_times: The end times of the time intervals.
    :param events: The event indicators.
    :param ids: The patient ids. Can be used to specify time-varying covariates.
    """

    # Sort the data by time
    permutation = end_times.argsort()
    np.copyto(covariates, covariates[permutation])
    np.copyto(start_times, start_times[permutation])
    np.copyto(end_times, end_times[permutation])
    np.copyto(events, events[permutation])
    np.copyto(ids, ids[permutation])


def _stack_using_bins(
    covariates,
    events,
    ids,
    bin_start_indices,
):
    """Uses the prepared bins to stack the data.
    All data should be sorted based on the failure times of the corresponding patients.

    :param covariates: The covariates of the patients.
    :param events: The event indicators.
    :param ids: The patient ids. Can be used to specify time-varying covariates.
    :param bin_start_indices: Indices into the arrays at which a new time bin starts.
    :return: Stacked data set matrix and corresponding failure vector, to be used as input
        to train a classifier.
    """

    # Number of patients still active (uncensored and unfailed) in each time bin
    risk_sets_size = np.array(
        [
            len(np.unique(ids[start_idx:], return_index=True)[1])
            for start_idx in bin_start_indices
        ]
    )

    # Store number of covariates and time bins
    n_covariates = covariates.shape[1]
    n_time_bins = len(bin_start_indices) - 1

    # Allocate memory for stacked dataset
    size_of_stacked_dataset = np.sum(risk_sets_size)
    stacked = np.zeros((size_of_stacked_dataset, n_covariates + n_time_bins))
    target = np.zeros(size_of_stacked_dataset)

    # Fill the matrix time bin after time bin
    offset = 0
    for bin_idx, bin_start_idx in enumerate(bin_start_indices[:-1]):
        # Compute risk set
        risk_set_ids, risk_set_indices = np.unique(
            ids[bin_start_idx:], return_index=True
        )
        risk_set_size = risk_sets_size[bin_idx]
        # Fill in the covariates
        stacked[offset : offset + risk_set_size, :n_covariates] = covariates[
            risk_set_indices + bin_start_idx
        ]
        # Fill in the risk set indicators
        stacked[offset : offset + risk_set_size, n_covariates + bin_idx] = 1
        # Update the target matrix (failed for events in time bin, leave 0 for all others)
        bin_end_idx = bin_start_indices[bin_idx + 1]
        failed_ids = ids[np.where(events[bin_start_idx:bin_end_idx])[0] + bin_start_idx]
        failed_in_bin_indices = np.where(np.isin(risk_set_ids, failed_ids))[0] + offset
        target[failed_in_bin_indices] = 1
        # Keep track of where we are in updating the stacked matrix and target vector
        offset = offset + risk_set_size
    return stacked, target.astype(bool)
