import os

import numpy as np

from sklearn.base import BaseEstimator

from .alignment import ProcrustesAlignment, aligned_lambdas as _aligned_lambdas
from .kernels import compute_affinity
from .embedding import PCAMaps, LaplacianEigenmaps, DiffusionMaps


def _is_path_like(x):
    """Return True if `x` looks like a filesystem path to a stored array."""
    return isinstance(x, (str, bytes, os.PathLike))


def _devectorize(v, discard_diagonal=False):
    """Reconstruct a symmetric square matrix from its lower-triangular vector.

    Mirrors the layout produced by :func:`nilearn.connectome.sym_matrix_to_vec`,
    which is the format ``nilearn.connectome.ConnectivityMeasure(vectorize=True)``
    returns.

    Parameters
    ----------
    v : ndarray, shape (n*(n+1)/2,) or (n*(n-1)/2,)
        Lower-triangular entries of a symmetric matrix, in row-major order.
    discard_diagonal : bool, optional
        Whether the diagonal was discarded when vectorizing. Default is False.

    Returns
    -------
    m : ndarray, shape (n, n)
        Reconstructed symmetric matrix. The diagonal is filled with zeros if
        ``discard_diagonal`` is True.
    """
    v = np.asarray(v)
    if v.ndim != 1:
        raise ValueError('Vectorized input must be 1D; got shape {}.'
                         .format(v.shape))

    size = v.shape[0]
    if discard_diagonal:
        # size = n*(n-1)/2
        n = int(round((1 + np.sqrt(1 + 8 * size)) / 2))
        if n * (n - 1) // 2 != size:
            raise ValueError('Vector length {} is not consistent with a '
                             'lower-triangular matrix without diagonal.'
                             .format(size))
    else:
        # size = n*(n+1)/2
        n = int(round((-1 + np.sqrt(1 + 8 * size)) / 2))
        if n * (n + 1) // 2 != size:
            raise ValueError('Vector length {} is not consistent with a '
                             'lower-triangular matrix with diagonal.'
                             .format(size))

    m = np.zeros((n, n), dtype=v.dtype)
    rows, cols = np.tril_indices(n, k=-1 if discard_diagonal else 0)
    m[rows, cols] = v
    m[cols, rows] = v
    return m


def _load_matrix(x, vectorized=False, discard_diagonal=False):
    """Load `x` if path-like; otherwise return as-is. Devectorize if requested."""
    if _is_path_like(x):
        x = np.load(os.fspath(x))
    if vectorized:
        x = _devectorize(x, discard_diagonal=discard_diagonal)
    return x


def _fit_one(x, app, kernel, n_components, random_state, gamma=None,
             sparsity=0.9, vectorized=False, discard_diagonal=False, **kwargs):
    """Compute gradients of `x`.

    Parameters
    ----------
    x : ndarray or path-like, shape = (n_samples, n_feat)
        Input matrix, or a path to a ``.npy`` file containing one. Path-like
        inputs are loaded lazily with :func:`numpy.load`.
    vectorized : bool, optional
        If True, treat the loaded array as the lower-triangular vector of a
        symmetric matrix (as produced by
        ``nilearn.connectome.ConnectivityMeasure(vectorize=True)``) and
        reconstruct the square matrix before computing the affinity.
        Default is False.
    discard_diagonal : bool, optional
        Only used when ``vectorized`` is True. Whether the diagonal was
        discarded when vectorizing. Default is False.
    app : {'dm', 'le', 'pca'} or object
        Embedding approach. If object. it can be an instance of PCAMaps,
        LaplacianEigenmaps or DiffusionMaps.
    kernel : {'pearson', 'spearman', 'cosine', 'normalized_angle', 'gaussian'}
        or None, optional.
        Kernel function to build the affinity matrix.
    n_components : int
        Number of components.
    random_state : int or None, optional
        Random state. Default is None.
    gamma : float or None, optional
        Inverse kernel width. Only used if ``kernel`` == 'gaussian'.
        If None, ``gamma=1/n_feat``. Default is None.
    sparsity : float, optional
        Proportion of the smallest elements to zero-out for each row.
        Default is 0.9.
    kwargs : kwds, optional
        Additional keyword parameters passed to the embedding approach.

    Returns
    -------
    lambdas_ : ndarray, shape (n_components,)
        Eigenvalues.
    gradients_ : ndarray, shape (n_samples, n_components)
        Gradients (i.e., eigenvectors).
    """

    x = _load_matrix(x, vectorized=vectorized,
                     discard_diagonal=discard_diagonal)

    a = compute_affinity(x, kernel=kernel, sparsity=sparsity, gamma=gamma)

    if np.isnan(a).any() or np.isinf(a).any():
        raise ValueError('Affinity matrix contains NaN or Inf values. Common '
                         'causes of this include NaNs/Infs or rows of zeros '
                         'in the input matrix.')

    kwds_emb = {'n_components': n_components, 'random_state': random_state}
    kwds_emb.update(kwargs)

    if isinstance(app, str):
        if app == 'pca':
            app = PCAMaps(**kwds_emb)
        elif app == 'le':
            app = LaplacianEigenmaps(**kwds_emb)
        else:
            app = DiffusionMaps(**kwds_emb)
    else:
        app.set_params(**kwds_emb)
    app.fit(a)
    return app.lambdas_, app.maps_


class GradientMaps(BaseEstimator):
    """Gradient maps.

    Parameters
    ----------
    n_components : int, optional
        Number of gradients. Default is 10.
    approach : {'dm', 'le', 'pca'} or object, optional
        Embedding approach. Default is 'dm'. It can be a string or instance:

        - 'dm' or :class:`.DiffusionMaps`: embedding using diffusion maps.
        - 'le' or :class:`.LaplacianEigenmaps`: embedding using Laplacian
          eigenmaps.
        - 'pca' or :class:`.PCAMaps`: embedding using PCA.

    kernel : str, callable or None, optional
        Kernel function to build the affinity matrix. Possible options:
        {'pearson', 'spearman', 'cosine', 'normalized_angle', 'gaussian'}.
        If callable, must receive a 2D array and return a 2D square array.
        If None, use input matrix. Default is 'normalized_angle'.
    alignment : {'procrustes', 'joint'}, object or None
        Alignment approach. Only used when two or more datasets are provided.
        If None, no alignment is performed. If `object`, it accepts an instance
        of :class:`.ProcrustesAlignment`. Default is None.

        - If 'procrustes', datasets are aligned using generalized procrustes
          analysis.
        - If 'joint', datasets are embedded simultaneously based on a joint
          affinity matrix built from the individual datasets. This option is
          only available for 'dm' and 'le' approaches.

    random_state : int or None, optional
        Random state. Default is None.

    Attributes
    ----------
    lambdas_ : ndarray or list of arrays, shape = (n_components,)
        Eigenvalues for each datatset.
    gradients_ : ndarray or list of arrays, shape = (n_samples, n_components)
        Gradients (i.e., eigenvectors).
    aligned_ : None or list of arrays, shape = (n_samples, n_components)
        Aligned gradients. None if ``alignment is None`` or only one dataset
        is used.
    aligned_lambdas_ : None or list of arrays, shape = (n_components,)
        Heuristic eigenvalues for the aligned gradients (issue #94). For
        each aligned gradient, the dominant original gradient is
        identified by the largest absolute value in the corresponding
        column of the Procrustes rotation matrix; the proxy lambda is
        the original lambda at that index. ``None`` if no Procrustes
        alignment was performed.
    """

    def __init__(self, n_components=10, approach='dm', kernel='normalized_angle',
                 alignment=None, random_state=None):
        self.n_components = n_components
        self.approach = approach
        self.kernel = kernel
        self.alignment = alignment
        self.random_state = random_state

        self.gradients_ = None
        self.lambdas_ = None
        self.aligned_ = None
        self.aligned_lambdas_ = None

    def fit(self, x, gamma=None, sparsity=0.9, n_iter=10, reference=None,
            vectorized=False, discard_diagonal=False, **kwargs):
        """Compute gradients and alignment.

        Parameters
        ----------
        x : ndarray, path-like, or list of arrays/path-likes, shape = (n_samples, n_feat)
            Input matrix or list of matrices. Each entry can be either a
            NumPy array or a path-like object pointing to a ``.npy`` file.
            Path-like inputs are loaded lazily, one at a time, so large
            matrices need not all be held in memory simultaneously.
            If a single matrix is provided and ``reference`` is not None,
            ``n_iter`` is set to 1 (fixed reference alignment).
        gamma : float or None, optional
            Inverse kernel width. Only used if ``kernel == 'gaussian'``.
            If None, ``gamma=1/n_feat``. Default is None.
        sparsity : float, optional
            Proportion of the smallest elements to zero-out for each row.
            Default is 0.9.
        n_iter : int, optional
            Number of iterations for procrustes alignment. Default is 10.
        reference : ndarray, shape = (n_samples, n_feat), optional
            Initial reference for procrustes alignments. Only used when
            ``alignment == 'procrustes'``. Default is None.
            If provided, it is used as the reference for the first iteration.
            If ``n_iter > 1``, the reference is updated at each iteration
            (Generalized Procrustes Analysis).
        vectorized : bool, optional
            If True, treat each input as the lower-triangular vector of a
            symmetric connectivity matrix (the format produced by
            ``nilearn.connectome.ConnectivityMeasure(vectorize=True)``) and
            reconstruct the square matrix before fitting. Halves the disk
            footprint when combined with path-like inputs. Default is False.
        discard_diagonal : bool, optional
            Only used when ``vectorized`` is True. Whether the diagonal was
            discarded when vectorizing. Default is False.
        kwargs : kwds, optional
            Additional keyword parameters passed to the embedding approach.

        Returns
        -------
        self : object
            Returns self.
        """

        align_single = False
        is_listlike = isinstance(x, (list, tuple))
        if self.alignment is not None and self.alignment != 'joint' and \
                not is_listlike and reference is not None:
            x = [x]
            is_listlike = True
            n_iter = 1
            align_single = True

        if not is_listlike:
            self.lambdas_, self.gradients_ = \
                _fit_one(x, self.approach, self.kernel, self.n_components,
                         self.random_state, gamma=gamma, sparsity=sparsity,
                         vectorized=vectorized,
                         discard_diagonal=discard_diagonal, **kwargs)
            self.aligned_ = None

            return self

        # Multiple datasets
        n = len(x)
        lam, grad = [None] * n, [None] * n
        if self.alignment == 'joint':
            if n < 2:
                raise ValueError('Joint alignment requires >=2 datasets.')
            loaded = [_load_matrix(x1, vectorized=vectorized,
                                   discard_diagonal=discard_diagonal)
                      for x1 in x]
            self.fit(np.vstack(loaded), gamma=gamma, sparsity=sparsity,
                     **kwargs)

            s = np.cumsum([0] + [m.shape[0] for m in loaded])
            for i in range(n):
                a, b = s[i], s[i+1]
                lam[i], grad[i] = self.lambdas_[a:b], self.gradients_[a:b]

            self.lambdas_ = lam
            self.aligned_ = self.gradients_ = grad

        else:
            for i, x1 in enumerate(x):
                self.fit(x1, gamma=gamma, sparsity=sparsity,
                         vectorized=vectorized,
                         discard_diagonal=discard_diagonal, **kwargs)
                lam[i], grad[i] = self.lambdas_, self.gradients_
            self.lambdas_, self.gradients_ = lam, grad

            if self.alignment == 'procrustes':
                pa = ProcrustesAlignment(n_iter=n_iter)
                pa.fit(self.gradients_, reference=reference)
                self.aligned_ = pa.aligned_
                self.aligned_lambdas_ = [
                    _aligned_lambdas(la, t)
                    for la, t in zip(self.lambdas_, pa.transforms_)
                ]

            elif isinstance(self.alignment, ProcrustesAlignment):
                self.alignment.set_params(n_iter=n_iter)
                self.alignment.fit(self.gradients_, reference=reference)
                self.aligned_ = self.alignment.aligned_
                self.aligned_lambdas_ = [
                    _aligned_lambdas(la, t)
                    for la, t in zip(self.lambdas_,
                                     self.alignment.transforms_)
                ]

            else:
                self.aligned_ = None
                self.aligned_lambdas_ = None

        if align_single:
            self.gradients_ = self.gradients_[0]
            self.lambdas_ = self.lambdas_[0]
            self.aligned_ = self.aligned_[0]
            if self.aligned_lambdas_ is not None:
                self.aligned_lambdas_ = self.aligned_lambdas_[0]

        return self


