import numpy


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
