import numpy
import scipy.special as special
import scipy.stats as st


def kmeans(agg_client, x, n_clusters, tol=1e-4, maxiter=100, random_state=123):
    """
    Distributed K-means clustering using secure aggregation via agg_client.

    Parameters
    ----------
    agg_client
        Aggregation client providing sum/min/max over lists of floats.
    x
        Local data, 2D array-like (n_samples_local, n_features).
    n_clusters
        Number of clusters (k).
    tol
        Convergence tolerance on the Frobenius norm of center difference.
    maxiter
        Maximum number of iterations.
    random_state
        Seed for deterministic center initialization.

    Returns
    -------
    dict with keys:
        n_obs: int
            Total number of observations across all workers.
        centers: List[List[float]]
            Final cluster centers (k x n_features).
    """
    import numpy

    # Convert to numpy array
    X = numpy.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_local, n_features = X.shape

    # Global number of observations
    total_n_obs = int(agg_client.sum([float(n_local)])[0])

    # If there is no data at all, return empty centers
    if total_n_obs == 0:
        return dict(n_obs=0, centers=[])

    # ------------------------------------------------------------------
    # Global initialization of centers using global min/max per feature
    # ------------------------------------------------------------------
    if n_local > 0:
        local_min = numpy.nanmin(X, axis=0)
        local_max = numpy.nanmax(X, axis=0)
    else:
        # This worker has no rows but we still need to participate
        local_min = numpy.full((n_features,), numpy.inf, dtype=float)
        local_max = numpy.full((n_features,), -numpy.inf, dtype=float)

    global_min = numpy.asarray(agg_client.min(local_min.tolist()), dtype=float)
    global_max = numpy.asarray(agg_client.max(local_max.tolist()), dtype=float)

    rng = numpy.random.RandomState(seed=random_state)
    centers = rng.uniform(
        low=global_min,
        high=global_max,
        size=(int(n_clusters), n_features),
    )

    # ------------------------------------------------------------------
    # Lloyd's algorithm with distributed aggregation of sums/counts
    # ------------------------------------------------------------------
    for _ in range(int(maxiter)):
        # If this worker has no data, it contributes zero sums/counts
        if n_local > 0:
            # Compute squared distances to centers
            # X: (n_local, n_features)
            # centers: (k, n_features)
            # diff: (n_local, k, n_features)
            diff = X[:, numpy.newaxis, :] - centers[numpy.newaxis, :, :]
            dists_sq = numpy.einsum("ijk,ijk->ij", diff, diff)
            labels = numpy.argmin(dists_sq, axis=1)

            # Local sums and counts per cluster
            sum_local = numpy.zeros((n_clusters, n_features), dtype=float)
            count_local = numpy.zeros((n_clusters,), dtype=float)

            for k in range(n_clusters):
                mask = labels == k
                if numpy.any(mask):
                    sum_local[k] = X[mask].sum(axis=0)
                    count_local[k] = float(mask.sum())
        else:
            sum_local = numpy.zeros((n_clusters, n_features), dtype=float)
            count_local = numpy.zeros((n_clusters,), dtype=float)

        # Aggregate sums and counts across workers
        sum_flat_global = agg_client.sum(sum_local.ravel().tolist())
        count_global = numpy.asarray(agg_client.sum(count_local.tolist()), dtype=float)

        sum_global = numpy.asarray(sum_flat_global, dtype=float).reshape(
            (n_clusters, n_features)
        )

        # Update centers; keep old center if a cluster has no points
        new_centers = centers.copy()
        for k in range(n_clusters):
            if count_global[k] > 0.0:
                new_centers[k] = sum_global[k] / count_global[k]

        # Check convergence (Frobenius norm)
        diff_norm = numpy.linalg.norm(new_centers - centers, ord="fro")
        centers = new_centers
        if diff_norm <= tol:
            break

    return dict(
        n_obs=int(total_n_obs),
        centers=centers.tolist(),
    )


def pca(agg_client, x):

    n_obs = len(x)
    sx = numpy.einsum("ij->j", x)
    sxx = numpy.einsum("ij,ij->j", x, x)

    total_n_obs = agg_client.sum([float(n_obs)])[0]
    total_sx = numpy.array(agg_client.sum(sx.tolist()), dtype=float)
    total_sxx = numpy.array(agg_client.sum(sxx.tolist()), dtype=float)

    means = total_sx / total_n_obs
    variances = (total_sxx - total_n_obs * means**2) / (total_n_obs - 1)
    variances = numpy.maximum(variances, 0.0)
    sigmas = numpy.sqrt(variances)
    zero_sigma = sigmas == 0
    if numpy.any(zero_sigma):
        sigmas = sigmas.copy()
        sigmas[zero_sigma] = 1.0
    out = numpy.empty(x.shape)

    numpy.subtract(x, means, out=out)
    numpy.divide(out, sigmas, out=out)
    gramian = numpy.einsum("ji,jk->ik", out, out)
    total_gramian = numpy.array(agg_client.sum(gramian.tolist()), dtype=float)
    covariance = total_gramian / (total_n_obs - 1)

    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors.T
    return dict(
        n_obs=int(total_n_obs),
        eigenvalues=eigenvalues.real.tolist(),
        eigenvectors=eigenvectors.real.tolist(),
    )


def pearson_correlation(agg_client, x, y, alpha):
    n_obs = len(y)

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)

    sx = numpy.einsum("ij->j", x)
    sy = numpy.einsum("ij->j", y)
    sxx = numpy.einsum("ij,ij->j", x, x)
    syy = numpy.einsum("ij,ij->j", y, y)
    sxy = numpy.einsum("ji,jk->ki", x, y)

    total_n_obs = agg_client.sum([float(n_obs)])[0]
    total_sx = numpy.array(agg_client.sum(sx.tolist()), dtype=float)
    total_sy = numpy.array(agg_client.sum(sy.tolist()), dtype=float)
    total_sxx = numpy.array(agg_client.sum(sxx.tolist()), dtype=float)
    total_syy = numpy.array(agg_client.sum(syy.tolist()), dtype=float)
    total_sxy = numpy.array(agg_client.sum(sxy.tolist()), dtype=float)

    df = total_n_obs - 2
    if total_n_obs == 0:
        raise ValueError("Cannot compute Pearson correlation on empty data.")

    if df <= 0:
        raise ValueError("Not enough observations to compute Pearson correlation.")

    d = (
        numpy.sqrt(total_n_obs * total_sxx - total_sx * total_sx)
        * numpy.sqrt(total_n_obs * total_syy - total_sy * total_sy)[:, numpy.newaxis]
    )
    correlations = (total_n_obs * total_sxy - total_sx * total_sy[:, numpy.newaxis]) / d
    correlations[d == 0] = 0
    correlations = correlations.clip(-1, 1)
    t_squared = correlations**2 * (df / ((1.0 - correlations) * (1.0 + correlations)))
    p_values = special.betainc(
        0.5 * df, 0.5, numpy.fmin(numpy.asarray(df / (df + t_squared)), 1.0)
    )
    p_values[abs(correlations) == 1] = 0
    r_z = numpy.arctanh(correlations)
    se = 1 / numpy.sqrt(total_n_obs - 3)
    z = st.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    ci_lo, ci_hi = numpy.tanh((lo_z, hi_z))

    return dict(
        n_obs=int(total_n_obs),
        correlations=correlations.tolist(),
        p_values=p_values.tolist(),
        ci_lo=ci_lo.tolist(),
        ci_hi=ci_hi.tolist(),
    )


def ttest_one_sample(agg_client, sample, *, mu: float, alpha: float, alternative: str):
    sample = numpy.asarray(sample, dtype=float).reshape(-1)
    n_obs = sample.size

    sum_x = sample.sum()
    sqrd_x = numpy.dot(sample, sample)
    diff_x = sum_x - n_obs * mu
    diff_sqrd_x = sqrd_x - 2 * mu * sum_x + n_obs * mu**2

    total_n_obs = agg_client.sum([float(n_obs)])[0]
    total_sum_x = agg_client.sum([float(sum_x)])[0]
    total_sqrd_x = agg_client.sum([float(sqrd_x)])[0]
    total_diff_x = agg_client.sum([float(diff_x)])[0]
    total_diff_sqrd_x = agg_client.sum([float(diff_sqrd_x)])[0]

    if total_n_obs <= 1:
        raise ValueError("Not enough observations for one-sample t-test.")

    smpl_mean = total_sum_x / total_n_obs
    sd = numpy.sqrt(
        (total_diff_sqrd_x - (total_diff_x**2 / total_n_obs)) / (total_n_obs - 1)
    )
    sed = sd / numpy.sqrt(total_n_obs)
    t_stat = (smpl_mean - mu) / sed
    df = total_n_obs - 1

    ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=smpl_mean, scale=sed)

    if alternative == "greater":
        p_value = 1.0 - st.t.cdf(t_stat, df)
        ci_upper = "Infinity"
    elif alternative == "less":
        p_value = 1.0 - st.t.cdf(-t_stat, df)
        ci_lower = "-Infinity"
    else:
        p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

    cohens_d = -(smpl_mean - mu) / sd

    return dict(
        n_obs=int(total_n_obs),
        t_stat=t_stat,
        df=int(df),
        std=sd,
        p_value=p_value,
        mean_diff=smpl_mean,
        se_diff=sed,
        ci_upper=ci_upper,
        ci_lower=ci_lower,
        cohens_d=cohens_d,
    )


def ttest_paired(
    agg_client,
    sample_x,
    sample_y,
    *,
    alpha: float,
    alternative: str,
):
    sample_x = numpy.asarray(sample_x, dtype=float).reshape(-1)
    sample_y = numpy.asarray(sample_y, dtype=float).reshape(-1)
    if sample_x.shape != sample_y.shape:
        raise ValueError("Paired samples must have the same length.")

    n_obs = sample_x.size
    diff = sample_x - sample_y

    sum_x = sample_x.sum()
    sum_y = sample_y.sum()
    diff_sum = diff.sum()
    diff_sq_sum = numpy.dot(diff, diff)
    x_sq_sum = numpy.dot(sample_x, sample_x)
    y_sq_sum = numpy.dot(sample_y, sample_y)

    totals = agg_client.sum(
        [float(n_obs), float(sum_x), float(sum_y), float(diff_sum), float(diff_sq_sum)]
    )
    total_n_obs, total_sum_x, total_sum_y, total_diff_sum, total_diff_sq_sum = totals
    total_x_sq = agg_client.sum([float(x_sq_sum)])[0]
    total_y_sq = agg_client.sum([float(y_sq_sum)])[0]

    if total_n_obs <= 1:
        raise ValueError("Not enough observations for paired t-test.")

    mean_x = total_sum_x / total_n_obs
    mean_y = total_sum_y / total_n_obs

    sd_x = numpy.sqrt(
        (total_x_sq - 2 * mean_x * total_sum_x + (mean_x**2) * total_n_obs)
        / (total_n_obs - 1)
    )
    sd_y = numpy.sqrt(
        (total_y_sq - 2 * mean_y * total_sum_y + (mean_y**2) * total_n_obs)
        / (total_n_obs - 1)
    )
    sd_diff = numpy.sqrt(
        (total_diff_sq_sum - (total_diff_sum**2 / total_n_obs)) / (total_n_obs - 1)
    )
    sed = sd_diff / numpy.sqrt(total_n_obs)
    t_stat = (mean_x - mean_y) / sed
    df = total_n_obs - 1

    sample_mean = total_diff_sum / total_n_obs
    ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=sample_mean, scale=sed)

    if alternative == "greater":
        p_value = 1.0 - st.t.cdf(t_stat, df)
        ci_upper = "Infinity"
    elif alternative == "less":
        p_value = 1.0 - st.t.cdf(-t_stat, df)
        ci_lower = "-Infinity"
    else:
        p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

    cohens_d = (mean_x - mean_y) / numpy.sqrt((sd_x**2 + sd_y**2) / 2)

    return dict(
        t_stat=t_stat,
        df=int(df),
        p_value=p_value,
        mean_diff=sample_mean,
        se_diff=sed,
        ci_upper=ci_upper,
        ci_lower=ci_lower,
        cohens_d=cohens_d,
    )


def ttest_independent(
    agg_client,
    sample_a,
    sample_b,
    *,
    alpha: float,
    alternative: str,
):
    sample_a = numpy.asarray(sample_a, dtype=float).reshape(-1)
    sample_b = numpy.asarray(sample_b, dtype=float).reshape(-1)

    n_a = sample_a.size
    n_b = sample_b.size

    sums = agg_client.sum(
        [
            float(n_a),
            float(n_b),
            float(sample_a.sum()),
            float(sample_b.sum()),
        ]
    )
    n_a_total, n_b_total, sum_a_total, sum_b_total = sums

    sq_sums = agg_client.sum(
        [float(numpy.dot(sample_a, sample_a)), float(numpy.dot(sample_b, sample_b))]
    )
    sq_sum_a, sq_sum_b = sq_sums

    if n_a_total < 1:
        raise ValueError("Group A has no data.")
    if n_b_total < 1:
        raise ValueError("Group B has no data.")
    if n_a_total + n_b_total <= 2:
        raise ValueError("Not enough observations for independent t-test.")

    mean_a = sum_a_total / n_a_total
    mean_b = sum_b_total / n_b_total

    sd_a = numpy.sqrt(
        (sq_sum_a - 2 * mean_a * sum_a_total + (mean_a**2) * n_a_total)
        / (n_a_total - 1)
    )
    sd_b = numpy.sqrt(
        (sq_sum_b - 2 * mean_b * sum_b_total + (mean_b**2) * n_b_total)
        / (n_b_total - 1)
    )

    sed_a = sd_a / numpy.sqrt(n_a_total)
    sed_b = sd_b / numpy.sqrt(n_b_total)
    sed = numpy.sqrt(sed_a**2 + sed_b**2)
    t_stat = (mean_a - mean_b) / sed
    df = n_a_total + n_b_total - 2
    diff_mean = mean_a - mean_b

    ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=diff_mean, scale=sed)

    if alternative == "greater":
        p_value = 1.0 - st.t.cdf(t_stat, df)
        ci_upper = "Infinity"
    elif alternative == "less":
        p_value = 1.0 - st.t.cdf(-t_stat, df)
        ci_lower = "-Infinity"
    else:
        p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

    pooled_var = ((n_a_total - 1) * sd_a**2 + (n_b_total - 1) * sd_b**2) / (
        n_a_total + n_b_total - 2
    )
    cohens_d = diff_mean / numpy.sqrt(pooled_var)

    return dict(
        t_stat=t_stat,
        df=int(df),
        p_value=p_value,
        mean_diff=diff_mean,
        se_diff=sed,
        ci_upper=ci_upper,
        ci_lower=ci_lower,
        cohens_d=cohens_d,
    )


def roc_curve_binary(y_true, y_score):
    """
    Compute ROC curve points (FPR, TPR) for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels {0, 1}.
    y_score : array-like of shape (n_samples,)
        Predicted probabilities for positive class.

    Returns
    -------
    dict with keys:
        "tpr" : List[float]
        "fpr" : List[float]

    Notes
    -----
    - Identical to sklearn.metrics.roc_curve(..., drop_intermediate=False)
    - Does NOT perform secure aggregation — operates on already-local arrays.
    """
    y_true = numpy.asarray(y_true).astype(int)
    y_score = numpy.asarray(y_score).astype(float)

    # Sort by descending score
    desc_idx = numpy.argsort(-y_score)
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    # Count positives/negatives
    P = numpy.sum(y_true == 1)
    N = numpy.sum(y_true == 0)

    if P == 0 or N == 0:
        # Degenerate case: only one class present
        return {"tpr": [0.0, 1.0], "fpr": [0.0, 1.0]}

    # True positives & false positives cumulative
    tps = numpy.cumsum(y_true == 1)
    fps = numpy.cumsum(y_true == 0)

    # Threshold changes
    # Find indices where the score changes
    distinct_idx = numpy.where(numpy.diff(y_score))[0]
    # Always include last index
    threshold_idxs = numpy.r_[distinct_idx, y_true.size - 1]

    # Compute TPR, FPR at each threshold
    tpr = tps[threshold_idxs] / P
    fpr = fps[threshold_idxs] / N

    # prepend (0,0) to match sklearn behavior
    tpr = numpy.r_[0.0, tpr]
    fpr = numpy.r_[0.0, fpr]

    return {
        "tpr": tpr.tolist(),
        "fpr": fpr.tolist(),
    }
