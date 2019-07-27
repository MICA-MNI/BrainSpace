"""
Kernels.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from .utils import dominant_set


def compute_affinity(x, kernel=None, sparsity=.1, gamma=None):
    """Compute affinity matrix.

    Parameters
    ----------
    x : 2D ndarray, shape = (n_samples, n_feat)
        Input matrix.
    kernel : {'pearson', 'spearman', 'cosine', 'normalized_angle', 'gaussian'}
        or None, optional.
        Kernel function. If None, only sparisify. Default is None.
    sparsity : float, optional
        Only keep top ``n_feat*sparsity`` elements for each row. Zero-out
        the rest. Default is 0.1.
    gamma : float or None, optional
        Inverse kernel width. Only used if ``kernel`` == 'gaussian'.
        If None, ``gamma=1/n_feat``. Default is None.

    Returns
    -------
    affinity : 2D ndarray, shape = (n_samples, n_samples)
        Affinity matrix.
    """

    if sparsity:
        x = dominant_set(x, k=sparsity, is_thresh=False, as_sparse=False)

    if kernel is None:
        return x

    if kernel in {'pearson', 'spearman'}:
        if kernel == 'spearman':
            x = np.apply_along_axis(rankdata, 1, x)
        a = np.corrcoef(x)

    elif kernel in {'cosine', 'normalized_angle'}:
        a = cosine_similarity(x)
        np.fill_diagonal(a, 1)
        if kernel == 'normalized_angle':
            a = 1 - np.arccos(a, a)/np.pi

    elif kernel == 'gaussian':
        if gamma is None:
            gamma = 1 / x.shape[1]
        a = rbf_kernel(x, gamma=gamma)

    else:
        raise ValueError("Unknown kernel '{0}'.".format(kernel))

    return a
