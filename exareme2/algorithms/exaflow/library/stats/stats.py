import numpy
import scipy.special as special
import scipy.stats as st


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

    ci_lower, ci_upper = st.t.interval(alpha=1 - alpha, df=df, loc=smpl_mean, scale=sed)

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
    ci_lower, ci_upper = st.t.interval(
        alpha=1 - alpha, df=df, loc=sample_mean, scale=sed
    )

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

    ci_lower, ci_upper = st.t.interval(alpha=1 - alpha, df=df, loc=diff_mean, scale=sed)

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
