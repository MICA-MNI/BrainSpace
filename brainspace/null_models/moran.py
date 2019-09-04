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


def compute_mem(w, n_ring=1, spectrum='nonzero', tol=1e-10):
    """ Compute Moran eigenvectors map.

    Parameters
    ----------
    w : BSPolyData, ndarray or sparse matrix, shape = (n_vertices, n_vertices)
        Spatial weight matrix or surface. If surface, the weight matrix is
        built based on the inverse geodesic distance between each vertex
        and the vertices in its `n_ring`.
    n_ring : int, optional
        Neighborhood size to build the weight matrix. Only used if user
        provides a surface mesh. Default is 1.
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
    mem : 2D ndarray, shape (n_vertices, n_components)
        Eigenvectors of the weight matrix in same order.

    See Also
    --------
    :func:`.moran_randomization`
    :class:`.MoranRandomization`

    References
    ----------
    * Wagner H.H. and Dray S. (2015). Generating spatially constrained
      null models for irregularly spaced data using Moran spectral
      randomization methods. Methods in Ecology and Evolution, 6(10):1169-78.

    """

    if spectrum not in ['all', 'nonzero']:
        raise ValueError("Unknown autocor '{0}'.".format(spectrum))

    # If surface is provided instead of affinity
    if not (isinstance(w, np.ndarray) or ssp.issparse(w)):
        w = me.get_ring_distance(w, n_ring=n_ring, metric='geodesic')
        w.data **= -1  # inverse of distance
        # w /= np.nansum(w, axis=1, keepdims=True)  # normalize rows

    if not is_symmetric(w):
        w = make_symmetric(w, check=False, sparse_format='coo')

    # Doubly centering weight matrix
    if ssp.issparse(w):
        m = w.mean(axis=0).A
        wc = w.mean() - m - m.T

        if not ssp.isspmatrix_coo(w):
            w_format = w.format
            w = w.tocoo(copy=False)
            row, col = w.row, w.col
            w = getattr(w, 'to' + w_format)(copy=False)
        else:
            row, col = w.row, w.col
        wc[row, col] += w.data

    else:
        m = w.mean(axis=0, keepdims=True)
        wc = w.mean() - m - m.T
        wc += w

    # when using float64, eigh is unstable for sparse matrices
    ev, mem = np.linalg.eigh(wc.astype(np.float32))
    ev, mem = ev[::-1], mem[:, ::-1]

    # Remove zero eigen-value/vector
    ev_abs = np.abs(ev)
    mask_zero = ev_abs < tol
    n_zero = np.count_nonzero(mask_zero)

    if n_zero == 0:
        raise ValueError('Weight matrix has no zero eigenvalue.')

    # Multiple zero eigenvalues
    if spectrum == 'all':
        if n_zero > 1:
            n = w.shape[0]
            memz = np.hstack([mem[:, mask_zero], np.ones((n, 1))])
            q, _ = np.linalg.qr(memz)
            mem[:, mask_zero] = q[:, :-1]
            idx_zero = mask_zero.argmax()
        else:
            idx_zero = ev_abs.argmin()

        ev[idx_zero:-1] = ev[idx_zero+1:]
        mem[:, idx_zero:-1] = mem[:, idx_zero + 1:]
        ev = ev[:-1]
        mem = mem[:, :-1]

    else:  # only nonzero
        mask_nonzero = ~mask_zero
        ev = ev[mask_nonzero]
        mem = mem[:, mask_nonzero]

    return mem, ev


def moran_randomization(x, mem, n_rep=100, procedure='singleton', joint=False,
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
    procedure : {'singleton, 'pair'}, optional
        Procedure to generate the random samples. Default is 'singleton'.
    joint : boolean, optional
        If True variables are randomized jointly. Otherwise, each variable is
        randomized separately. Default is False.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    output : ndarray, shape = (n_rep, n_vertices, n_feat)
        Random samples. If ``n_feat == 1``, shape = (n_rep, n_vertices).

    See Also
    --------
    :func:`.compute_mem`
    :class:`.MoranRandomization`

    References
    ----------
    * Wagner H.H. and Dray S. (2015). Generating spatially constrained
      null models for irregularly spaced data using Moran spectral
      randomization methods. Methods in Ecology and Evolution, 6(10):1169-78.

    """

    if x.ndim == 1:
        x = np.atleast_2d(x).T

    procedure = procedure.lower()
    if procedure not in ['singleton', 'pair']:
        raise ValueError("Unknown procedure '{0}'".format(procedure))

    rs = check_random_state(random_state)

    n_comp = mem.shape[1]
    n_rows = x.shape[0]
    n_cols = 1 if joint else x.shape[1]

    rxv = 1 - cdist(x.T, mem.T, 'correlation').T
    if procedure == 'singleton':
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

    return sim.squeeze()


class MoranRandomization(BaseEstimator):
    """ Moran spectral randomization.

    Parameters
    ----------
    procedure : {'singleton, 'pair'}, optional
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
    :class:`.SpinPermutations`

    """

    def __init__(self, procedure='singleton', spectrum='nonzero', joint=False,
                 n_rep=100, n_ring=1, tol=1e-10, random_state=None):

        self.procedure = procedure
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
        w : BSPolyData, ndarray or sparse matrix, shape = (n_verts, n_verts)
            Spatial weight matrix or surface. If surface, the weight matrix is
            built based on the inverse geodesic distance between each vertex
            and the vertices in its `n_ring`.

        Returns
        -------
        self : object
            Returns self.

        """

        self.mem_, self.mev_ = compute_mem(w, spectrum=self.spectrum,
                                           tol=self.tol)
        return self

    def randomize(self, x):
        """ Generate random samples from `x`.

        Parameters
        ----------
        x : 1D or 2D ndarray, shape = (n_verts,) or (n_verts, n_feat)
            Array of variables arranged in columns, where `n_feat` is the
            number of variables.

        Returns
        -------
        output : ndarray, shape = (n_rep, n_verts, n_feat)
            Random samples. If ``n_feat == 1``, shape = (n_rep, n_verts).

        """

        rand = moran_randomization(x, self.mem_, n_rep=self.n_rep,
                                   procedure=self.procedure, joint=self.joint,
                                   random_state=self.random_state)
        return rand
