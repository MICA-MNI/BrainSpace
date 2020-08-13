"""
Embedding alignment using procrustes analysis.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np

from sklearn.base import BaseEstimator


def procrustes(source, target, center=False, scale=False):
    """Align `source` to `target` using procrustes analysis.

    Parameters
    ----------
    source : 2D ndarray, shape = (n_samples, n_feat)
        Source dataset.
    target : 2D ndarray, shape = (n_samples, n_feat)
        Target dataset.
    center : bool, optional
        Center data before alignment. Default is False.
    scale : bool, optional
        Remove scale before alignment. Default is False.

    Returns
    -------
    aligned : 2D ndarray, shape = (n_samples, n_feat)
        Source dataset aligned to target dataset.
    """

    # Translate to origin
    if center:
        ms = source.mean(axis=0)
        mt = target.mean(axis=0)

        source = source - ms
        target = target - mt

    # Remove scale
    if scale:
        ns = np.linalg.norm(source)
        nt = np.linalg.norm(target)
        source /= ns
        target /= nt

    # orthogonal transformation: rotation + reflection
    u, w, vt = np.linalg.svd(target.T.dot(source).T)
    t = u.dot(vt)

    # Recover target scale
    if scale:
        t *= w.sum() * nt

    aligned = source.dot(t)
    if center:
        aligned += mt
    return aligned


# Generalized procrustes analysis
def procrustes_alignment(data, reference=None, n_iter=10, tol=1e-5,
                         return_reference=False, verbose=False):
    """Iterative alignment using generalized procrustes analysis.

    Parameters
    ----------
    data :  list of ndarray, shape = (n_samples, n_feat)
        List of datasets to align.
    reference : ndarray, shape = (n_samples, n_feat), optional
        Dataset to use as reference in the first iteration. If None, the first
        dataset in `data` is used as reference. Default is None.
    n_iter : int, optional
        Number of iterations. Default is 10.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-5.
    return_reference : bool, optional
        Whether to return the reference dataset built in the last iteration.
        Default is False.
    verbose : bool, optional
        Verbosity. Default is False.

    Returns
    -------
    aligned : list of ndarray, shape = (n_samples, n_feat)
        Aligned datsets.
    mean_dataset : ndarray, shape = (n_samples, n_feat)
        Reference dataset built in the last iteration. Only if
        ``return_reference == True``.
    """

    if n_iter <= 0:
        raise ValueError('A positive number of iterations is required.')

    if reference is None:
        # Use the first item to build the initial reference
        aligned = [data[0]] + [procrustes(d, data[0]) for d in data[1:]]
        reference = np.mean(aligned, axis=0)
    else:
        aligned = [None] * len(data)
        reference = reference.copy()

    dist = np.inf
    for i in range(n_iter):
        # Align to reference
        aligned = [procrustes(d, reference) for d in data]

        # Compute new mean
        new_reference = np.mean(aligned, axis=0)

        # Compute distance
        new_dist = np.square(reference - new_reference).sum()

        # Update reference
        reference = new_reference

        if verbose:
            print('Iteration {0:>3}: {1:.6f}'.format(i, new_dist))

        if dist != np.inf and np.abs(new_dist - dist) < tol:
            break

        dist = new_dist

    return (aligned, reference) if return_reference else aligned


class ProcrustesAlignment(BaseEstimator):
    """Iterative alignment using generalized procrustes analysis.

    Parameters
    ----------
    n_iter : int, optional
        Number of iterations. Default is 10.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-5.
    verbose : bool, optional
        Verbosity. Default is False.

    Attributes
    -------
    aligned_ : list of ndarray, shape = (n_samples, n_feat)
        Aligned datsets.
    mean_ : ndarray, shape = (n_samples, n_feat)
        Reference dataset built in the last iteration.
    """

    def __init__(self, n_iter=10, tol=1e-5, verbose=False):
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, data, reference=None):
        """Align data.

        Parameters
        ----------
        data :  list of ndarrays, shape = (n_samples, n_feat)
            List of datasets to align.
        reference : ndarray, shape = (n_samples, n_feat), optional
            Dataset to use as reference in the first iteration. If None, the
            first dataset in `data` is used as reference. Default is None.

        Returns
        -------
        self : object
            Returns self.
        """

        self.aligned_, self.mean_ = \
            procrustes_alignment(data, reference=reference, tol=self.tol,
                                 n_iter=self.n_iter, return_reference=True,
                                 verbose=self.verbose)
        return self
