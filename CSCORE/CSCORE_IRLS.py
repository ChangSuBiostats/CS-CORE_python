'''
==========
Iteratively reweighted least squares (IRLS) procedure in CS-CORE
==========
'''

import numpy as np
import scipy.stats as stats


def CSCORE_IRLS(X, seq_depth, post_process=True):
    n_cell = X.shape[0]
    n_gene = X.shape[1]
    seq_depth_sq = np.power(seq_depth, 2)
    seq_2 = np.sum(seq_depth_sq)
    seq_4 = np.sum(np.power(seq_depth_sq, 2))
    mu = np.dot(seq_depth, X) / seq_2
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
