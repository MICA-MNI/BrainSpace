"""
Kernels.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import warnings

import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

from .utils import dominant_set


def compute_affinity(x, kernel=None, sparsity=.9, gamma=None):
    """Compute affinity matrix.

    Parameters
    ----------
    x : ndarray, shape = (n_samples, n_feat)
        Input matrix.
    kernel : str or None, optional
        Kernel function. If None, only sparsify. Default is None.
        Valid options:

        - If 'pearson', use Pearson's correlation coefficient.
        - If 'spearman', use Spearman's rank correlation coefficient.
        - If 'cosine', use cosine similarity.
        - If 'normalized_angle': use normalized angle between two vectors. This
          option is based on cosine similarity but provides similarities bounded
          between 0 and 1.
        - If 'gaussian', use Gaussian kernel or RBF.

    sparsity : float or None, optional
        Proportion of smallest elements to zero-out for each row.
        If None, do not sparsify. Default is 0.9.
    gamma : float or None, optional
        Inverse kernel width. Only used if ``kernel == 'gaussian'``.
        If None, ``gamma = 1./n_feat``. Default is None.

    Returns
    -------
    affinity : ndarray, shape = (n_samples, n_samples)
        Affinity matrix.
    """

    if sparsity is not None:
        x = dominant_set(x, k=1-sparsity, is_thresh=False, as_sparse=False)

    if kernel is None:
        mask_neg = x < 0
        if mask_neg.any():
            x[mask_neg] = 0
            warnings.warn('The matrix contains negative values and will '
                          'be zeroed-out.')
        return x

    if kernel in {'pearson', 'spearman'}:
        if kernel == 'spearman':
            x = np.apply_along_axis(rankdata, 1, x)
        a = np.corrcoef(x)

    elif kernel in {'cosine', 'normalized_angle'}:
        # a = cosine_similarity(x)
        # np.fill_diagonal(a, 1)
        a = 1 - squareform(pdist(x, metric='cosine'))
        if kernel == 'normalized_angle':
            a = 1 - np.arccos(a, a)/np.pi

    elif kernel == 'gaussian':
        if gamma is None:
            gamma = 1 / x.shape[1]
        a = rbf_kernel(x, gamma=gamma)

    else:
        raise ValueError("Unknown kernel '{0}'.".format(kernel))

    mask_neg = a < 0
    if mask_neg.any():
        a[mask_neg] = 0
        warnings.warn('The matrix contains negative values and will '
                      'be zeroed-out.')

    return a
