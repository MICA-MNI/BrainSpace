import numpy as np

from .alignment import ProcrustesAlignment
from .kernels import compute_affinity
from .embedding import PCAMaps, LaplacianEigenmaps, DiffusionMaps


def _fit_one(x, app, kernel, n_components, random_state, gamma=None,
             sparsity=0.9, **kwargs):
    """Compute gradients of `x`.

    Parameters
    ----------
    x : 2D ndarray, shape = (n_samples, n_feat)
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
        Only keep top ``n_feat*sparsity`` elements for each row. Zero-out
        the rest. Default is 0.1.
    kwargs : kwds, optional
        Additional keyword parameters passed to the embedding approach.

    Returns
    -------
    lambdas_ : 1D ndarray, shape (n_components,)
        Eigenvalues in descending order.
    gradients_ : 2D ndarray, shape (n_samples, n_components)
        Gradients (i.e., eigenvectors) in same order.
    """

    a = compute_affinity(x, kernel=kernel, sparsity=1-sparsity, gamma=gamma)
    print('Sparsity:', np.count_nonzero(a)/a.size)

    kwds_emb = {'n_components': n_components, 'random_state': random_state}
    kwds_emb.update(kwargs)

    if isinstance(app, str):
        if app == 'pca':
            app = PCAMaps(**kwds_emb)
        elif app == 'le':
            app = LaplacianEigenmaps(**kwds_emb)
        else:
            app = DiffusionMaps(**kwds_emb)

    app.fit(a)
    return app.lambdas_, app.maps_


class GradientMaps(object):
    """Gradient maps.

    Parameters
    ----------
    n_gradients : int, optional
        Number of gradients. Default is 2.
    approach : {'dm', 'le', 'pca'} or object
        Embedding approach. If object. it can be an instance of PCAMaps,
        LaplacianEigenmaps or DiffusionMaps.
    kernel : {'pearson', 'spearman', 'cosine', 'normalized_angle', 'gaussian'}
        or None, optional.
        Kernel function to build the affinity matrix.
    align : {'procrustes', 'manifold'}, object or None
        Alignment approach. Only used when two or more datasets are provided.
        If None, no alignment is peformed. If object, an instance of
        ProcrustesAlignment. Default is None.
    random_state : int or None, optional
        Random state. Default is None.

    Attributes
    ----------
    lambdas_ : 1D ndarray or list of arrays, shape = (n_gradients,)
        Eigenvalues in descending order for each datatset.
    gradients_ : 2D ndarray or list of arrays, shape = (n_samples, n_gradients)
        Gradients in same order.
    aligned_ : 2D ndarray or list of arrays, shape = (n_samples, n_gradients)
        Aligned gradients in same order.
    """

    def __init__(self, n_gradients=2, approach=None, kernel=None, align=None,
                 random_state=None):
        self.n_gradients = n_gradients
        self.approach = approach
        self.kernel = kernel
        self.align = align  # manifold, procrustes or None
        self.random_state = random_state

        self.gradients_ = None
        self.lambdas_ = None
        self.aligned_ = None

    def fit(self, x, gamma=None, sparsity=0.9, n_iter=10, **kwargs):
        """Compute gradients and alignment.

        Parameters
        ----------
        x : 2D ndarray or list of arrays, shape = (n_samples, n_feat)
            Input matrix or list of matrices.
        gamma : float or None, optional
            Inverse kernel width. Only used if ``kernel`` == 'gaussian'.
            If None, ``gamma=1/n_feat``. Default is None.
        sparsity : float, optional
            Only keep top ``n_feat*sparsity`` elements for each row. Zero-out
            the rest. Default is 0.1.
        n_iter : int, optional
            Number of iterations for procrustes alignment. Default is 10.
        kwargs : kwds, optional
            Additional keyword parameters passed to the embedding approach.

        Returns
        -------
        self : object
            Returns self.
        """

        if isinstance(x, np.ndarray):
            self.lambdas_, self.gradients_ = \
                _fit_one(x, self.approach, self.kernel, self.n_gradients,
                         self.random_state, gamma=gamma, sparsity=sparsity,
                         **kwargs)
            self.aligned_ = self.gradients_

            return self

        # Multiple datasets
        n = len(x)
        lam, grad = [None] * n, [None] * n
        if self.align == 'manifold':
            self.fit(np.vstack(x), gamma=gamma, sparsity=sparsity, **kwargs)

            s = np.cumsum([0] + [x1.shape[0] for x1 in x])
            for i, x1 in enumerate(x):
                a, b = s[i], s[i+1]
                lam, grad[i] = self.lambdas_[a:b], self.gradients_[a:b]

            self.lambdas_ = lam
            self.aligned_ = self.gradients_ = grad

        else:
            for i, x1 in enumerate(x):
                self.fit(x1, gamma=gamma, sparsity=sparsity, **kwargs)
                lam[i], grad[i] = self.lambdas_, self.gradients_
            self.lambdas_, self.gradients_ = lam, grad

            if self.align == 'procrustes':
                pa = ProcrustesAlignment(n_iter=n_iter).fit(self.gradients_)
                self.aligned_ = pa.aligned_
            elif isinstance(self.align, ProcrustesAlignment):
                self.aligned_ = self.align.fit(self.gradients_).aligned_
            else:
                self.aligned_ = None

        return self


