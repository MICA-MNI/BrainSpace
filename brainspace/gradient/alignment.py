"""
Embedding alignment using procrustes analysis.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np

from sklearn.base import BaseEstimator


def procrustes(source, target, center=False, scale=False,
               return_transform=False):
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
    return_transform : bool, optional
        If True, also return the rotation matrix `t` such that
        ``aligned == source @ t`` (plus optional translation when
        ``center=True``). Default is False.

    Returns
    -------
    aligned : 2D ndarray, shape = (n_samples, n_feat)
        Source dataset aligned to target dataset.
    t : 2D ndarray, shape = (n_feat, n_feat)
        Rotation matrix. Returned only if ``return_transform == True``.
    """

    # Translate to origin
    if center:
        ms = source.mean(axis=0)
        mt = target.mean(axis=0)

        source = source - ms
        target = target - mt
    elif scale:
        source = source.copy()
        target = target.copy()

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
    if return_transform:
        return aligned, t
    return aligned


def aligned_lambdas(lambdas, transform):
    """Approximate eigenvalue correspondence after Procrustes alignment.

    After Procrustes alignment ``A = source @ transform``, each aligned
    gradient is a linear combination of the original gradients, so
    eigenvalues no longer correspond to aligned gradients one-to-one
    (issue #94).

    For each aligned gradient (column ``j`` of ``A``), the dominant
    original gradient is ``i = argmax_i |transform[i, j]|``, and the
    returned proxy lambda is ``lambdas[i]``. This matches the
    approximation suggested by @OualidBenkarim in #94.

    Parameters
    ----------
    lambdas : 1D ndarray, shape = (n_components,)
        Eigenvalues of the unaligned gradients.
    transform : 2D ndarray, shape = (n_components, n_components)
        Procrustes rotation matrix that maps source to aligned, i.e.
        ``aligned == source @ transform``. Available from
        :func:`procrustes` with ``return_transform=True`` or from
        :class:`.ProcrustesAlignment`'s ``transforms_`` attribute.

    Returns
    -------
    proxy : 1D ndarray, shape = (n_components,)
        ``lambdas`` reordered (possibly with repeats) so that
        ``proxy[j]`` is a heuristic eigenvalue for the j-th aligned
        gradient.
    """

    lambdas = np.asarray(lambdas)
    transform = np.asarray(transform)
    idx = np.abs(transform).argmax(axis=0)
    return lambdas[idx]


# Generalized procrustes analysis
def procrustes_alignment(data, reference=None, center=False, scale=False,
                         n_iter=10, tol=1e-5, return_reference=False,
                         return_transforms=False, verbose=False):
    """Iterative alignment using generalized procrustes analysis.

    Parameters
    ----------
    data :  list of ndarray, shape = (n_samples, n_feat)
        List of datasets to align.
    reference : ndarray, shape = (n_samples, n_feat), optional
        Dataset to use as reference in the first iteration. If None, the first
        dataset in `data` is used as reference. Default is None.
    center : bool, optional
        Center data before alignment. Default is False.
    scale : bool, optional
        Remove scale before alignment. Default is False.
    n_iter : int, optional
        Number of iterations. Default is 10.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-5.
    return_reference : bool, optional
        Whether to return the reference dataset built in the last iteration.
        Default is False.
    return_transforms : bool, optional
        Whether to also return the per-dataset rotation matrices from the
        final iteration. Default is False.
    verbose : bool, optional
        Verbosity. Default is False.

    Returns
    -------
    aligned : list of ndarray, shape = (n_samples, n_feat)
        Aligned datsets.
    mean_dataset : ndarray, shape = (n_samples, n_feat)
        Reference dataset built in the last iteration. Only if
        ``return_reference == True``.
    transforms : list of ndarray, shape = (n_feat, n_feat)
        Final-iteration rotation matrix for each dataset, such that
        ``aligned[i] == data[i] @ transforms[i]`` (plus translation when
        ``center=True``). Only if ``return_transforms == True``.
    """

    if n_iter <= 0:
        raise ValueError('A positive number of iterations is required.')

    transforms = [None] * len(data)
    if reference is None:
        # Use the first item to build the initial reference
        aligned = [data[0]]
        transforms[0] = np.eye(data[0].shape[1])
        for j, d in enumerate(data[1:], start=1):
            a, t = procrustes(d, data[0], center=center, scale=scale,
                              return_transform=True)
            aligned.append(a)
            transforms[j] = t
        reference = np.mean(aligned, axis=0)
    else:
        aligned = [None] * len(data)
        reference = reference.copy()

    dist = np.inf
    for i in range(n_iter):
        # Align to reference
        out = [procrustes(d, reference, center=center, scale=scale,
                          return_transform=True) for d in data]
        aligned = [a for a, _ in out]
        transforms = [t for _, t in out]

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

    result = (aligned,)
    if return_reference:
        result += (reference,)
    if return_transforms:
        result += (transforms,)
    if len(result) == 1:
        return result[0]
    return result


class ProcrustesAlignment(BaseEstimator):
    """Iterative alignment using generalized procrustes analysis.

    Parameters
    ----------
    center : bool, optional
        Center data before alignment. Default is False.
    scale : bool, optional
        Remove scale before alignment. Default is False.
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
    transforms_ : list of ndarray, shape = (n_feat, n_feat)
        Final-iteration rotation matrices.
        ``aligned_[i] == data[i] @ transforms_[i]``.
    """

    def __init__(self, center=False, scale=False, n_iter=10, tol=1e-5,
                 verbose=False):
        self.center = center
        self.scale = scale
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

        self.aligned_, self.mean_, self.transforms_ = \
            procrustes_alignment(data, reference=reference, center=self.center,
                                 scale=self.scale, tol=self.tol,
                                 n_iter=self.n_iter, return_reference=True,
                                 return_transforms=True,
                                 verbose=self.verbose)
        return self
