import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator

from .alignment import ProcrustesAlignment
from .kernels import compute_affinity
from .embedding import PCAMaps, LaplacianEigenmaps, DiffusionMaps


def _fit_one(x, app, kernel, n_components, random_state, gamma=None,
             sparsity=0.9, **kwargs):
    """Compute gradients of `x`.

    Parameters
    ----------
    x : ndarray, shape = (n_samples, n_feat)
        Input matrix.
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
        Proportion of smallest elements to zero-out for each row.
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

    a = compute_affinity(x, kernel=kernel, sparsity=sparsity, gamma=gamma)

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
        If None, use input matrix. Default is None.
    alignment : {'procrustes', 'joint'}, object or None
        Alignment approach. Only used when two or more datasets are provided.
        If None, no alignment is peformed. If `object`, it accepts an instance
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
    """

    def __init__(self, n_components=10, approach='dm', kernel=None,
                 alignment=None, random_state=None):
        self.n_components = n_components
        self.approach = approach
        self.kernel = kernel
        self.alignment = alignment
        self.random_state = random_state

        self.gradients_ = None
        self.lambdas_ = None
        self.aligned_ = None

    def fit(self, x, gamma=None, sparsity=0.9, n_iter=10, reference=None,
            **kwargs):
        """Compute gradients and alignment.

        Parameters
        ----------
        x : ndarray or list of arrays, shape = (n_samples, n_feat)
            Input matrix or list of matrices.
        gamma : float or None, optional
            Inverse kernel width. Only used if ``kernel == 'gaussian'``.
            If None, ``gamma=1/n_feat``. Default is None.
        sparsity : float, optional
            Proportion of smallest elements to zero-out for each row.
            Default is 0.9.
        n_iter : int, optional
            Number of iterations for procrustes alignment. Default is 10.
        reference : ndarray, shape = (n_samples, n_feat), optional
            Initial reference for procrustes alignments. Only used when
            ``alignment == 'procrustes'``. Default is None.
        kwargs : kwds, optional
            Additional keyword parameters passed to the embedding approach.

        Returns
        -------
        self : object
            Returns self.
        """

        align_single = False
        if self.alignment is not None and self.alignment != 'joint' and \
                not isinstance(x, list) and reference is not None:
            x = [x]
            n_iter = 1
            align_single = True

        if isinstance(x, np.ndarray):  # or sp.issparse(x):
            self.lambdas_, self.gradients_ = \
                _fit_one(x, self.approach, self.kernel, self.n_components,
                         self.random_state, gamma=gamma, sparsity=sparsity,
                         **kwargs)
            self.aligned_ = None

            return self

        # Multiple datasets
        n = len(x)
        lam, grad = [None] * n, [None] * n
        if self.alignment == 'joint':
            if n < 2:
                raise ValueError('Joint alignment requires >=2 datasets.')
            self.fit(np.vstack(x), gamma=gamma, sparsity=sparsity, **kwargs)

            s = np.cumsum([0] + [x1.shape[0] for x1 in x])
            for i, x1 in enumerate(x):
                a, b = s[i], s[i+1]
                lam[i], grad[i] = self.lambdas_[a:b], self.gradients_[a:b]

            self.lambdas_ = lam
            self.aligned_ = self.gradients_ = grad

        else:
            for i, x1 in enumerate(x):
                self.fit(x1, gamma=gamma, sparsity=sparsity, **kwargs)
                lam[i], grad[i] = self.lambdas_, self.gradients_
            self.lambdas_, self.gradients_ = lam, grad

            if self.alignment == 'procrustes':
                pa = ProcrustesAlignment(n_iter=n_iter)
                pa.fit(self.gradients_, reference=reference)
                self.aligned_ = pa.aligned_

            elif isinstance(self.alignment, ProcrustesAlignment):
                self.alignment.set_params(n_iter=n_iter)
                self.alignment.fit(self.gradients_, reference=reference)
                self.aligned_ = self.alignment.aligned_

            else:
                self.aligned_ = None

        if align_single:
            self.gradients_ = self.gradients_[0]
            self.lambdas_ = self.lambdas_[0]
            self.aligned_ = self.aligned_[0]

        return self


