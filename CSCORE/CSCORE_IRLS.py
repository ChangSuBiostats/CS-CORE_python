'''
==========
Iteratively reweighted least squares (IRLS) procedure in CS-CORE
==========
'''

import numpy as np
import scipy.stats as stats
from scipy.sparse import issparse

def CSCORE_IRLS(X, seq_depth, post_process=True):
    """
    This function implements the iteratively reweighted least squares algorithm of CS-CORE.

    Parameters
    ----------
    X: scipy.sparse._csr.csr_matrix or numpy.ndarray
        Raw UMI count matrix. n cells by p genes.
    seq_depth: 1-dimensional Numpy array, length n
        Sum of UMI counts across all genes for each cell.
    post_process: logical
        Whether to post-process co-expression estimates to be within [-1,1].
    max_j: integer
        Maximum number of iterations in IRLS.

    Returns
    -------
    est: p by p Numpy array
        Estimates of co-expression networks among p genes. Each entry saves the correlation between two genes.
    p_value: p by p Numpy array
        p values against H_0: two gene expressions are independent. Please refer to the paper for more details.
    test_stat:
        Test statistics against H_0: two gene expressions are independent. Please refer to the paper for more details.

    References
    ----------
    Su, C., Xu, Z., Shan, X. et al. Cell-type-specific co-expression inference from single cell RNA-sequencing data.
    Nat Commun 14, 4846 (2023). https://doi.org/10.1038/s41467-023-40503-7
    """
    n_cell = X.shape[0]
    n_gene = X.shape[1]
    seq_depth_sq = np.power(seq_depth, 2)
    seq_2 = np.sum(seq_depth_sq)
    seq_4 = np.sum(np.power(seq_depth_sq, 2))
    if issparse(X):
        mu = X.transpose().dot(seq_depth) / seq_2
        # The rest of the computation will use the centered X, which is no longer sparse
        X = X.toarray()
    elif isinstance(X, np.ndarray):
        mu = np.dot(seq_depth, X) / seq_2
    else:
        raise ValueError("Unsupported type for X: \n\
        Matrix X is neither a scipy csr_matrix nor a numpy ndarray. Please reformat the input X.")
    M = np.outer(seq_depth, mu)
    X_centered = X - M
    sigma2 = np.dot(seq_depth_sq, (np.power(X_centered, 2) - M)) / seq_4
    theta = np.power(mu, 2) / sigma2
    j = 0
    delta = np.Inf

    # IRLS for estimating mu and sigma_jj
    while delta > 0.05 and j <= 10:
        theta_previous = theta
        theta_median = np.median(theta[theta > 0])
        theta[theta < 0] = np.Inf
        w = M + np.outer(seq_depth_sq, np.power(mu, 2) / theta_median)
        w[w <= 0] = 1
        mu = np.dot(seq_depth, X / w) / np.dot(seq_depth_sq, 1 / w)
        M = np.outer(seq_depth, mu)
        X_centered = X - M
        h = np.power(np.power(M, 2) / theta_median + M, 2)
        h[h <= 0] = 1
        sigma2 = np.dot(seq_depth_sq, (np.power(X_centered, 2) - M) / h) / np.dot(np.power(seq_depth_sq, 2), 1 / h)
        theta = np.power(mu, 2) / sigma2
        j = j + 1
        theta_subset = np.logical_and(theta_previous > 0, theta > 0)
        delta = np.max(np.abs(np.log(theta[theta_subset]) - np.log(theta_previous[theta_subset])))

    if j == 10 and delta > 0.05:
        print('IRLS failed to converge after 10 iterations. Please check your data.')
    else:
        print('IRLS converged after %i iterations.' % j)

    # Weighted least squares for estimating sigma_jj'
    theta_median = np.median(theta[theta > 0])
    theta[theta < 0] = np.Inf
    w = M + np.outer(seq_depth_sq, np.power(mu, 2) / theta_median)
    w[w <= 0] = 1

    X_weighted = X_centered / w
    num = np.einsum("i,ij->ij", seq_depth_sq, X_weighted).T @ X_weighted
    seq_depth_sq_weighted = np.einsum("i,ij->ij", seq_depth_sq, 1 / w)
    deno = seq_depth_sq_weighted.T @ seq_depth_sq_weighted
    covar = num / deno

    # Evaluate test statistics and p values
    Sigma = M + np.outer(seq_depth_sq, sigma2)
    X_weighted = X_centered / Sigma
    num = np.einsum("i,ij->ij", seq_depth_sq, X_weighted).T @ X_weighted
    seq_depth_sq_weighted = np.einsum("i,ij->ij", seq_depth_sq, 1 / Sigma)
    deno = seq_depth_sq_weighted.T @ seq_depth_sq_weighted
    test_stat = num / np.sqrt(deno)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(test_stat)))

    # Evaluate co-expression estimates
    neg_inds = sigma2 < 0
    sigma2[neg_inds] = 1
    sigma = np.sqrt(sigma2)
    est = covar / np.outer(sigma, sigma)
    est[neg_inds, neg_inds] = np.Inf

    # Post-process the co-expression estimates
    if post_process:
        est = post_process_est(est)

    return est, p_value, test_stat


def post_process_est(est):
    p = est.shape[0]
    # Post-process CS-CORE estimates
    neg_gene_inds = (np.diag(est) == np.Inf)
    if np.any(neg_gene_inds):
        print(
            '%i among %i genes have negative variance estimates. Their co-expressions with other genes were set to 0.' %
            (np.sum(neg_gene_inds), p))
    # Negative variances suggest insufficient biological variation,
    # and also lack of correlation
    est[neg_gene_inds, :] = 0
    est[:, neg_gene_inds] = 0
    # Set all diagonal values to 1
    np.fill_diagonal(est, 1)
    # Gene pairs with out-of-bound estimates
    print(f"{np.mean(np.triu(est) > 1) * 100 * 2:.4f}% co-expression estimates were greater than 1 and were set to 1.")
    print(f"{np.mean(np.triu(est) < -1) * 100 * 2:.4f}% co-expression estimates were greater than 1 and were set to 1.")
    est[est > 1] = 1
    est[est < -1] = -1
    return est


def CSCORE(adata, gene_index, seq_depth = None):
    """
    This function implements CS-CORE for inferring cell-type-specific co-expression networks with scanpy object

    Parameters
    ----------
    adata: AnnData
        Single cell data object. The raw UMI count data are stored as adata.raw.X, which is required for
    gene_index: 1-dimensional Numpy array, length p
        Integer indexes for the genes of interest, whose co-expression networks will be estimated in this function.

    Returns
    -------
    est: p by p Numpy array
        Estimates of co-expression networks among p genes. Each entry saves the correlation between two genes.
    p_value: p by p Numpy array
        p values against H_0: two gene expressions are independent. Please refer to the paper for more details.
    test_stat:
        Test statistics against H_0: two gene expressions are independent. Please refer to the paper for more details.

    References
    ----------
    Su, C., Xu, Z., Shan, X. et al. Cell-type-specific co-expression inference from single cell RNA-sequencing data.
    Nat Commun 14, 4846 (2023). https://doi.org/10.1038/s41467-023-40503-7
    """
    # check the size of gene set
    if len(gene_index) > 10000:
        print('We suggest running CS-CORE for only the highly expressed genes in cell types.\n\
        Lowly expressed genes are too sparse for co-expression estimation.')
    # evaluate sequencing depths
    if seq_depth is None:
        seq_depth = adata.raw.X.sum(axis = 1).A1
    else:
        if len(seq_depth) != adata.raw.X.shape[0]:
            raise ValueError('The length of sequencing depths does not match the dimension of the count matrix.')
    # extract raw UMI count data for running CS-CORE
    if adata.raw is None:
        raise TypeError('Raw UMI counts cannot be found in adata.raw.X.')
    elif not hasattr(adata.raw, "X"):
        raise TypeError('Raw UMI counts cannot be found in adata.raw.X.')

    res = CSCORE_IRLS(adata.raw.X[:, gene_index],
                      seq_depth,
                      post_process=True)
    return res
