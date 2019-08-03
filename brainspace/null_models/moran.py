"""
Implementation of Moran spectral randomization.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
import scipy.sparse as ssp
from scipy.spatial.distance import cdist

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator

from ..mesh import mesh_elements as me
from ..gradient.utils import is_symmetric, make_symmetric


def compute_mem(a, spectrum='nonzero', tol=1e-10):
    """ Compute Moran eigenvectors map.

    Parameters
    ----------
    a : 2D ndarray or sparse matrix, shape = (n_vertices, n_vertices)
        Spatial weight matrix.
    spectrum : {'all', 'nonzero'}, optional
        Eigenvalues/vectors to select. If 'all', recover all eigenvectors
        except the smallest one. Otherwise, select all except non-zero
        eigenvectors. Default is 'nonzero'.
    tol : float, optional
        Minimum value for an eigenvalue to be considered non-zero.
        Default is 1e-10.

    Returns
    -------
    w : 1D ndarray, shape (n_components,)
        Eigenvalues in descending order. With ``n_components = n_vertices - 1``
        if ``spectrum == 'all'`` and ``n_components = n_vertices - n_zero`` if
        ``spectrum == 'nonzero'``, and `n_zero` is number of zero eigenvalues.
    v : 2D ndarray, shape (n_vertices, n_components)
        Eigenvectors of the weight matrix in same order.

    References
    ----------
    * Wagner H.H. and Dray S. (2015). Generating spatially constrained
      null models for irregularly spaced data using Moran spectral
      randomization methods. Methods in Ecology and Evolution, 6(10):1169-78.

    """

    if spectrum not in ['all', 'nonzero']:
        raise ValueError("Unknown autocor '{0}'.".format(spectrum))

    if not is_symmetric(a):
        a = make_symmetric(a, check=False, sparse_format='coo')

    # Doubly centering weight matrix
    if ssp.issparse(a):
        m = a.mean(axis=0)
        ac = a.mean() - m - m.T

        if not ssp.isspmatrix_coo(a):
            a_format = a.format
            a = a.tocoo(copy=False)
            row, col = a.row, a.col
            a = getattr(a, 'too' + a_format)(copy=False)
        else:
            row, col = a.row, a.col
        ac[row, col] += a.data

    else:
        m = a.mean(axis=0, keepdims=True)
        ac = a.mean() - m - m.T
        ac += a

    w, v = np.linalg.eigh(ac)
    w, v = w[::-1], v[:, ::-1]

    # Remove zero eigen-value/vector
    w_abs = np.abs(w)
    mask_zero = w_abs < tol
    n_zero = np.count_nonzero(mask_zero)

    if n_zero == 0:
        raise ValueError('Weight matrix has no zero eigenvalue.')

    # Multiple zero eigenvalues
    if spectrum == 'all':
        if n_zero > 1:
            n = a.shape[0]
            wz = np.hstack([v[:, mask_zero], np.ones((n, 1))])
            q, _ = np.linalg.qr(wz)
            v[:, mask_zero] = q[:, :-1]
            idx_zero = mask_zero.argmax()
        else:
            idx_zero = w_abs.argmin()

        w[idx_zero:-1] = w[idx_zero+1:]
        v[:, idx_zero:-1] = v[:, idx_zero + 1:]
        w = w[:-1]
        v = v[:, :-1]

    else:  # only nonzero
        mask_nonzero = ~mask_zero
        w = w[mask_nonzero]
        v = v[:, mask_nonzero]

    return w, v


def spectral_randomization(x, mem, n_rep=100, method='singleton', joint=False,
                           random_state=None):
    """ Generate random samples from `x` based on Moran spectral randomization.

    Parameters
    ----------
    x : 1D or 2D ndarray, shape = (n_vertices,) or (n_vertices, n_feat)
        Array of variables arranged in columns, where `n_feat` is the number
        of variables.
    mem : 2D ndarray, shape = (n_vertices, nv)
        Moran eigenvectors map, where `nv` is the number of eigenvectors
        arranged in columns.
    n_rep : int, optional
        Number of random samples. Default is 100.
    method : {'singleton, 'pair'}, optional
        Procedure to generate the random samples. Default is 'singleton'.
    joint : boolean, optional
        If True variables are randomized jointly. Otherwise, each variable is
        randomized separately. Default is False.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    output : ndarray, shape = (n_rep, n_feat, n_vertices)
        Random samples. If ``n_feat == 1``, shape = (n_rep, n_vertices).

    References
    ----------
    * Wagner H.H. and Dray S. (2015). Generating spatially constrained
      null models for irregularly spaced data using Moran spectral
      randomization methods. Methods in Ecology and Evolution, 6(10):1169-78.

    """

    if x.ndim == 1:
        x = np.atleast_2d(x).T

    method = method.lower()
    if method not in ['singleton', 'pair']:
        raise ValueError("Unknown method '{0}'".format(method))

    rs = check_random_state(random_state)

    n_comp = mem.shape[1]
    n_rows = x.shape[0]
    n_cols = 1 if joint else x.shape[1]

    rxv = 1 - cdist(x.T, mem.T, 'correlation').T
    if method == 'singleton':
        rxv2 = rxv * rs.choice([-1., 1.], size=(n_rep, n_comp, n_cols))

    else:  # pair
        n_pairs = n_comp // 2
        n_top = 2 * n_pairs
        is_odd = n_top != n_comp

        rsq = rxv ** 2
        rxv2 = np.empty((n_rep,) + rxv.shape)
        for i in range(n_rep):
            p = rs.permutation(n_comp)
            # ia, ib = p[:n_top:2], p[1:n_top:2]
            ia, ib = p[:n_pairs], p[n_pairs:n_top]

            if is_odd:  # singleton method for last item
                rxv2[i, p[-1]] = rxv[p[-1]] * rs.choice([-1, 1], size=n_cols)

            phi = rs.uniform(0, 2 * np.pi, size=(n_pairs, n_cols))
            if joint:
                phi = phi + np.arctan2(rxv[ia], rxv[ib])
            rxv2[i, ia] = rxv2[i, ib] = np.sqrt(rsq[ia] + rsq[ib])
            rxv2[i, ia] *= np.cos(phi)
            rxv2[i, ib] *= np.sin(phi)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=1)
    sim = x_mean + (mem @ rxv2) * (np.sqrt(n_rows - 1) * x_std)
    sim = sim.swapaxes(1, 2)

    return sim.squeeze()


class MoranSpectralRandomization(BaseEstimator):
    """ Moran spectral randomization.

    Parameters
    ----------
    method : {'singleton, 'pair'}, optional
        Procedure to generate the random samples. Default is 'singleton'.
    spectrum : {'all', 'nonzero'}, optional
        Eigenvalues/vectors to select. If 'all', recover all eigenvectors
        except one. Otherwise, select all except non-zero eigenvectors.
        Default is 'nonzero'.
    joint : boolean, optional
        If True variables are randomized jointly. Otherwise, each variable is
        randomized separately. Default is False.
    n_rep : int, optional
        Number of randomizations. Default is 100.
    n_ring : int, optional
        Neighborhood size to build the weight matrix. Only used if user provides
        a surface mesh. Default is 1.
    tol : float, optional
        Minimum value for an eigenvalue to be considered non-zero.
        Default is 1e-10.
    random_state : int or None, optional
        Random state. Default is None.

    Attributes
    ----------
    mev_ : 1D ndarray, shape (n_components,)
        Eigenvalues of the weight matrix in descending order.
    mem_ : 2D ndarray, shape (n_vertices, n_components)
        Eigenvectors of the weight matrix in same order.

    See Also
    --------
    :class:`.SpinRandomization`

    """

    def __init__(self, method='singleton', spectrum='nonzero', joint=False,
                 n_rep=100, n_ring=1, tol=1e-10, random_state=None):

        self.method = method
        self.spectrum = spectrum
        self.joint = joint
        self.n_rep = n_rep
        self.n_ring = n_ring
        self.tol = tol
        self.random_state = random_state

    def fit(self, w):
        """ Compute Moran eigenvectors map.

        Parameters
        ----------
        w : 2D ndarray or BSPolyData
            Spatial weight matrix or surface. If surface, the weight matrix is
            built based on the inverse geodesic distance between each vertex and
            the vertices in its `n_ring`.

        Returns
        -------
        self : object
            Returns self.

        """

        # If surface is provided instead of affinity
        if not isinstance(w, np.ndarray):
            w = me.get_ring_distance(w, n_ring=self.n_ring, metric='geodesic')
            w.data **= -1  # inverse of distance
            # s /= np.nansum(s, axis=1, keepdims=True)  # normalize rows
            # s = s.tocoo(copy=False)

        self.mev_, self.mem_ = compute_mem(w, spectrum=self.spectrum,
                                           tol=self.tol)
        return self

    def randomize(self, x):
        """ Generate random samples from `x`.

        Parameters
        ----------
        x : 1D or 2D ndarray, shape = (n_vertices,) or (n_vertices, n_feat)
            Array of variables arranged in columns, where `n_feat` is the number
            of variables.

        Returns
        -------
        output : ndarray, shape = (n_rep, n_feat, n_vertices)
            Random samples. If ``n_feat == 1``, shape = (n_rep, n_vertices).

        """

        rand = spectral_randomization(x, self.mem_, n_rep=self.n_rep,
                                      method=self.method, joint=self.joint,
                                      random_state=self.random_state)
        return rand

