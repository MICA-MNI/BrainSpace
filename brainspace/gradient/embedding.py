"""
Embedding approaches.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse as ssp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from scipy.sparse.csgraph import connected_components

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

from .utils import is_symmetric, make_symmetric


def _graph_is_connected(graph):
    return connected_components(graph)[0] == 1


def diffusion_mapping(adj, n_components=10, alpha=0.5, diffusion_time=0,
                      random_state=None):
    """Compute diffusion map of affinity matrix.

    Parameters
    ----------
    adj : ndarray or sparse matrix, shape = (n, n)
        Affinity matrix.
    n_components : int or None, optional
        Number of eigenvectors. If None, selection of `n_components` is based
        on 95% drop-off in eigenvalues. When `n_components` is None,
        the maximum number of eigenvectors is restricted to
        ``n_components <= sqrt(n)``. Default is 10.
    alpha : float, optional
        Anisotropic diffusion parameter, ``0 <= alpha <= 1``. Default is 0.5.
    diffusion_time : int, optional
        Diffusion time or scale. If ``diffusion_time == 0`` use multi-scale
        diffusion maps. Default is 0.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    v : ndarray, shape (n, n_components)
        Eigenvectors of the affinity matrix in same order.
    w : ndarray, shape (n_components,)
        Eigenvalues of the affinity matrix in descending order.

    References
    ----------
    * Coifman, R.R.; S. Lafon. (2006). "Diffusion maps". Applied and
      Computational Harmonic Analysis 21: 5-30. doi:10.1016/j.acha.2006.04.006
    * Joseph W.R., Peter E.F., Ann B.L., Chad M.S. Accurate parameter
      estimation for star formation history in galaxies using SDSS spectra.
    """

    rs = check_random_state(random_state)
    use_sparse = ssp.issparse(adj)

    # Make symmetric
    if not is_symmetric(adj, tol=1E-10):
        warnings.warn('Affinity is not symmetric. Making symmetric.')
        adj = make_symmetric(adj, check=False, copy=True, sparse_format='coo')
    else:  # Copy anyways because we will be working on the matrix
        adj = adj.tocoo(copy=True) if use_sparse else adj.copy()

    # Check connected
    if not _graph_is_connected(adj):
        warnings.warn('Graph is not fully connected.')

    ###########################################################
    # Step 2
    ###########################################################
    # When α=0, you get back the diffusion map based on the random walk-style
    # diffusion operator (and Laplacian Eigenmaps). For α=1, the diffusion
    # operator approximates the Laplace-Beltrami operator and for α=0.5,
    # you get Fokker-Planck diffusion. The anisotropic diffusion
    # parameter: \alpha \in \[0, 1\]
    # W(α) = D^{−1/\alpha} W D^{−1/\alpha}
    if alpha > 0:
        if use_sparse:
            d = np.power(adj.sum(axis=1).A1, -alpha)
            adj.data *= d[adj.row]
            adj.data *= d[adj.col]
        else:
            d = adj.sum(axis=1, keepdims=True)
            d = np.power(d, -alpha)
            adj *= d.T
            adj *= d

    ###########################################################
    # Step 3
    ###########################################################
    # Diffusion operator
    # P(α) = D(α)^{−1}W(α)
    if use_sparse:
        d_alpha = np.power(adj.sum(axis=1).A1, -1)
        adj.data *= d_alpha[adj.row]
    else:
        adj *= np.power(adj.sum(axis=1, keepdims=True), -1)

    ###########################################################
    # Step 4
    ###########################################################
    if n_components is None:
        n_components = max(2, int(np.sqrt(adj.shape[0])))
        auto_n_comp = True
    else:
        auto_n_comp = False

    # For repeatability of results
    v0 = rs.uniform(-1, 1, adj.shape[0])

    # Find largest eigenvalues and eigenvectors
    w, v = eigsh(adj, k=n_components + 1, which='LM', tol=0, v0=v0)

    # Sort descending
    w, v = w[::-1], v[:, ::-1]

    ###########################################################
    # Step 5
    ###########################################################
    # Force first eigenvector to be all ones.
    v /= v[:, [0]]

    # Largest eigenvalue should be equal to one too
    w /= w[0]

    # Discard first (largest) eigenvalue and eigenvector
    w, v = w[1:], v[:, 1:]

    if diffusion_time <= 0:
        # use multi-scale diffusion map, ref [4]
        # considers all scales: t=1,2,3,...
        w /= (1 - w)
    else:
        # Raise eigenvalues to the power of diffusion time
        w **= diffusion_time

    if auto_n_comp:
        # Choose n_comp to coincide with a 95 % drop-off
        # in the eigenvalue multipliers, ref [4]
        lambda_ratio = w / w[0]

        # If all eigenvalues larger than 0.05, select all
        # (i.e., sqrt(adj.shape[0]))
        threshold = max(0.05, lambda_ratio[-1])
        n_components = np.argmin(lambda_ratio > threshold)

        w = w[:n_components]
        v = v[:, :n_components]

    # Rescale eigenvectors with eigenvalues
    v *= w[None, :]

    # Consistent sign (s.t. largest value of element eigenvector is pos)
    v *= np.sign(v[np.abs(v).argmax(axis=0), range(v.shape[1])])
    return v, w


def laplacian_eigenmaps(adj, n_components=10, norm_laplacian=True,
                        random_state=None):
    """Compute embedding using Laplacian eigenmaps.

    Adapted from Scikit-learn to also provide eigenvalues.

    Parameters
    ----------
    adj : 2D ndarray or sparse matrix
        Affinity matrix.
    n_components : int, optional
        Number of eigenvectors. Default is 10.
    norm_laplacian : bool, optional
        If True use normalized Laplacian. Default is True.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    v : 2D ndarray, shape (n, n_components)
        Eigenvectors of the affinity matrix in same order. Where `n` is
        the number of rows of the affinity matrix.
    w : 1D ndarray, shape (n_components,)
        Eigenvalues of the affinity matrix in ascending order.

    References
    ----------
    * Belkin, M. and Niyogi, P. (2003). Laplacian Eigenmaps for
      dimensionality reduction and data representation.
      Neural Computation 15(6): 1373-96. doi:10.1162/089976603321780317

    """

    rs = check_random_state(random_state)

    # Make symmetric
    if not is_symmetric(adj, tol=1E-10):
        warnings.warn('Affinity is not symmetric. Making symmetric.')
        adj = make_symmetric(adj, check=False)

    # Check connected
    if not _graph_is_connected(adj):
        warnings.warn('Graph is not fully connected.')

    lap, dd = laplacian(adj, normed=norm_laplacian, return_diag=True)
    if norm_laplacian:
        if ssp.issparse(lap):
            lap.setdiag(1)
        else:
            np.fill_diagonal(lap, 1)

    lap *= -1
    v0 = rs.uniform(-1, 1, lap.shape[0])
    w, v = eigsh(lap, k=n_components + 1, sigma=1, which='LM', tol=0, v0=v0)

    # Sort descending and change sign of eigenvalues
    w, v = -w[::-1], v[:, ::-1]

    if norm_laplacian:
        v /= dd[:, None]

    # Drop smallest
    w, v = w[1:], v[:, 1:]

    # Consistent sign (s.t. largest value of element eigenvector is pos)
    v *= np.sign(v[np.abs(v).argmax(axis=0), range(v.shape[1])])
    return v, w


class Embedding(BaseEstimator, metaclass=ABCMeta):
    """Base class for embedding approaches.

    Defines fit_transform method.

    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.maps_ = None
        self.lambdas_ = None

    @abstractmethod
    def fit(self, x):
        pass

    def fit_transform(self, x):
        """Compute embedding for `x`.

        Parameters
        ----------
        x : ndarray, shape = (n, n)
            Input matrix.

        Returns
        -------
        embedding : ndarray, shape(n, n_components)
            Embedded data.

        """

        return self.fit(x).maps_


class DiffusionMaps(Embedding):
    """Diffusion maps.

    Parameters
    ----------
    n_components : int or None, optional
        Number of eigenvectors. Default is 10.
    alpha : float, optional
        Anisotropic diffusion parameter, ``0 <= alpha <= 1``. Default is 0.5.
    diffusion_time : int, optional
        Diffusion time or scale.  If ``diffusion_time == 0`` use multi-scale
        diffusion maps. Default is 0.
    random_state : int or None, optional
        Random state. Default is None.

    Attributes
    ----------
    lambdas_ : 1D ndarray, shape (n_components,)
        Eigenvalues of the affinity matrix in descending order.
    maps_ : 2D ndarray, shape (n, n_components)
        Eigenvectors of the affinity matrix in same order. Where `n` is
        the number of rows of the affinity matrix.

    See Also
    --------
    :class:`.LaplacianEigenmaps`
    :class:`.PCAMaps`

    References
    ----------
    * Coifman, R.R.; S. Lafon. (2006). "Diffusion maps". Applied and
      Computational Harmonic Analysis 21: 5-30. doi:10.1016/j.acha.2006.04.006
    * Joseph W.R., Peter E.F., Ann B.L., Chad M.S. Accurate parameter
      estimation for star formation history in galaxies using SDSS spectra.

    """

    def __init__(self, n_components=10, alpha=0.5, diffusion_time=1,
                 random_state=None):
        super().__init__(n_components=n_components)
        self.alpha = alpha
        self.diffusion_time = diffusion_time
        self.random_state = random_state

    def fit(self, affinity):
        """ Compute the diffusion maps.

        Parameters
        ----------
        affinity : ndarray or sparse matrix, shape = (n, n)
            Affinity matrix.

        Returns
        -------
        self : object
            Returns self.

        """

        self.maps_, self.lambdas_ = \
            diffusion_mapping(affinity, n_components=self.n_components,
                              alpha=self.alpha,
                              diffusion_time=self.diffusion_time,
                              random_state=self.random_state)

        return self


class LaplacianEigenmaps(Embedding):
    """Laplacian eigenmaps.

    Parameters
    ----------
    n_components : int or None, optional
        Number of eigenvectors. Default is 10.
    norm_laplacian : bool, optional
        If True, use normalized Laplacian. Default is True.
    random_state : int or None, optional
        Random state. Default is None.

    Attributes
    ----------
    lambdas_ : ndarray, shape (n_components,)
        Eigenvalues of the affinity matrix in ascending order.
    maps_ : ndarray, shape (n, n_components)
        Eigenvectors of the affinity matrix in same order. Where `n` is
        the number of rows of the affinity matrix.

    See Also
    --------
    :class:`.DiffusionMaps`
    :class:`.PCAMaps`

    """

    def __init__(self, n_components=10, norm_laplacian=True, random_state=None):
        super().__init__(n_components=n_components)
        self.norm_laplacian = norm_laplacian
        self.random_state = random_state

    def fit(self, affinity):
        """ Compute the Laplacian maps.

        Parameters
        ----------
        affinity : ndarray or sparse matrix, shape = (n, n)
            Affinity matrix.

        Returns
        -------
        self : object
            Returns self.

        """

        self.maps_, self.lambdas_ = \
            laplacian_eigenmaps(affinity, n_components=self.n_components,
                                norm_laplacian=self.norm_laplacian,
                                random_state=self.random_state)

        return self


class PCAMaps(Embedding):
    """Principal component analysis.

    Parameters
    ----------
    n_components : int or None, optional
        Number of principal components. Default is 10.
    random_state : int, RandomState instance or None, optional
         Random state. Default is None.

    Attributes
    ----------
    lambdas_ :ndarray, shape (n_components,)
        Explained variance for first principal components in descending order.
    maps_ : ndarray, shape (n_samples, n_components)
        Projection of input data onto the principal components.

    See Also
    --------
    :class:`.DiffusionMaps`
    :class:`.LaplacianEigenmaps`

    """
    def __init__(self, n_components=10, random_state=None):
        super().__init__(n_components=n_components)
        self.random_state = random_state

    def fit(self, x):
        """ Compute PCA.

        Parameters
        ----------
        x : ndarray, shape(n_samples, n_feat)
            Input matrix.

        Returns
        -------
        self : object
            Returns self.

        """

        pca = PCA(n_components=self.n_components,
                  random_state=self.random_state)
        self.maps_ = pca.fit_transform(x)
        self.lambdas_ = pca.explained_variance_

        return self
