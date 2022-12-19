"""
Utility functions for affinity/similarity matrices.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
from scipy import sparse as ssp


def is_symmetric(x, tol=1E-10):
    """Check if input is symmetric.

    Parameters
    ----------
    x : 2D ndarray or sparse matrix
        Input data.
    tol : float, optional
        Maximum allowed tolerance for equivalence. Default is 1e-10.

    Returns
    -------
    is_symm : bool
        True if `x` is symmetric. False, otherwise.

    Raises
    ------
    ValueError
        If `x` is not square.

    """

    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError('Array is not square.')

    if ssp.issparse(x):
        if x.format not in ['csr', 'csc', 'coo']:
            x = x.tocoo(copy=False)
        dif = x - x.T
        return np.all(np.abs(dif.data) < tol)

    return np.allclose(x, x.T, atol=tol)


def make_symmetric(x, check=True, tol=1E-10, copy=True, sparse_format=None):
    """Make array symmetric.

    Parameters
    ----------
    x : 2D ndarray or sparse matrix
        Input data.
    check : bool, optional
        If True, check if already symmetry first. Default is True.
    tol : float, optional
        Maximum allowed tolerance for equivalence. Default is 1e-10.
    copy : bool, optional
        If True, return a copy. Otherwise, work on `x`.
        If already symmetric, returns original array.
    sparse_format : {'coo', 'csr', 'csc', ...}, optional
        Format of output symmetric matrix. Only used if `x` is sparse.
        Default is None, uses original format.

    Returns
    -------
    sym : 2D ndarray or sparse matrix.
        Symmetrized version of `x`. Return `x` it is already
        symmetric.

    Raises
    ------
    ValueError
        If `x` is not square.

    """

    if not check or not is_symmetric(x, tol=tol):
        if copy:
            xs = .5 * (x + x.T)
            if ssp.issparse(x):
                if sparse_format is None:
                    sparse_format = x.format
                conversion = 'to' + sparse_format
                return getattr(xs, conversion)(copy=False)
            return xs
        else:
            x += x.T
            if ssp.issparse(x):
                x.data *= .5
            else:
                x *= .5
    return x


def _dominant_set_sparse(s, k, is_thresh=False, norm=False):
    """Compute dominant set for a sparse matrix."""
    if is_thresh:
        mask = s > k
        idx, data = np.where(mask), s[mask]
        s = ssp.coo_matrix((data, idx), shape=s.shape)

    else:  # keep top k
        nr, nc = s.shape
        idx = np.argpartition(s, nc - k, axis=1)
        col = idx[:, -k:].ravel()  # idx largest
        row = np.broadcast_to(np.arange(nr)[:, None], (nr, k)).ravel()
        data = s[row, col].ravel()
        s = ssp.coo_matrix((data, (row, col)), shape=s.shape)

    if norm:
        s.data /= s.sum(axis=1).A1[s.row]

    return s.tocsr(copy=False)


def _dominant_set_dense(s, k, is_thresh=False, norm=False, copy=True):
    """Compute dominant set for a dense matrix."""

    if is_thresh:
        s = s.copy() if copy else s
        s[s <= k] = 0

    else:  # keep top k
        nr, nc = s.shape
        idx = np.argpartition(s, nc - k, axis=1)
        row = np.arange(nr)[:, None]
        if copy:
            col = idx[:, -k:]  # idx largest
            data = s[row, col]
            s = np.zeros_like(s)
            s[row, col] = data
        else:
            col = idx[:, :-k]  # idx smallest
            s[row, col] = 0

    if norm:
        s /= np.nansum(s, axis=1, keepdims=True)

    return s


def dominant_set(s, k, is_thresh=False, norm=False, copy=True, as_sparse=True):
    """Keep the largest elements for each row. Zero-out the rest.

    Parameters
    ----------
    s : 2D ndarray
        Similarity/affinity matrix.
    k :  int or float
        If int, keep top `k` elements for each row. If float, keep top `100*k`
        percent of elements. When float, must be in range (0, 1).
    is_thresh : bool, optional
        If True, `k` is used as threshold. Keep elements greater than `k`.
        Default is False.
    norm : bool, optional
        If True, normalize rows. Default is False.
    copy : bool, optional
        If True, make a copy of the input array. Otherwise, work on original
        array. Default is True.
    as_sparse : bool, optional
        If True, return a sparse matrix. Otherwise, return the same type of the
        input array. Default is True.

    Returns
    -------
    output : 2D ndarray or sparse matrix
        Dominant set.

    """

    if not is_thresh:
        nr, nc = s.shape
        if isinstance(k, float):
            if not 0 < k < 1:
                raise ValueError('When \'k\' is float, it must be 0<k<1.')
            k = int(nc * k)

        if k <= 0:
            raise ValueError('Cannot select 0 elements.')

    if as_sparse:
        return _dominant_set_sparse(s, k, is_thresh=is_thresh, norm=norm)

    return _dominant_set_dense(s, k, is_thresh=is_thresh, norm=norm, copy=copy)


def ravel_symmetric(x, with_diagonal=False):
    """Return the flattened upper triangular part of a symmetric matrix.

    Parameters
    ----------
    x : 2D ndarray or sparse matrix, shape=(n, n)
        Input array.
    with_diagonal : bool, optional
        If True, also return diagonal elements. Default is False.

    Returns
    -------
    output : 1D ndarray, shape (n_feat,)
        The flattened upper triangular part of `x`. If with_diagonal
        is True, ``n_feat = n * (n + 1) / 2`` and
        ``n_feat = n * (n - 1) / 2`` otherwise.

    """

    n = x.shape[0]
    k = 0 if with_diagonal else -1
    mask_lt = np.tri(n, k=k, dtype=np.bool_)

    if ssp.issparse(x) and not ssp.isspmatrix_csc(x):
        x = x.tocsc(copy=False)

    return x[mask_lt.T]


def unravel_symmetric(x, size, as_sparse=False, part='both', fmt='csr'):
    """Build symmetric matrix from array with upper triangular elements.

    Parameters
    ----------
    x : 1D ndarray
        Input data with elements to go in the upper triangular part.
    size : int
        Number of rows/columns of matrix.
    as_sparse : bool, optional
        Return a sparse matrix. Default is False.
    part: {'both', 'upper', 'lower'}, optional
        Build matrix with elements if both or just on triangular part.
        Default is both.
    fmt: str, optional
        Format of sparse matrix. Only used if ``as_sparse=True``.
        Default is 'csr'.

    Returns
    -------
    sym : 2D ndarray or sparse matrix, shape = (size, size)
        Array with the lower/upper or both (symmetric) triangular parts
        built from `x`.

    """

    k = 1
    if (size * (size + 1) // 2) == x.size:
        k = 0
    elif (size * (size - 1) // 2) != x.size:
        raise ValueError('Cannot unravel data. Wrong size.')

    shape = (size, size)
    if as_sparse:
        mask = x != 0
        x = x[mask]

        idx = np.triu_indices(size, k=k)
        idx = [idx1[mask] for idx1 in idx]
        if part == 'lower':
            idx = idx[::-1]
        elif part == 'both':
            idx = np.concatenate(idx), np.concatenate(idx[::-1])
            x = np.tile(x, 2)

        xs = ssp.coo_matrix((x, idx), shape=shape)
        if fmt != 'coo':
            xs = xs.asformat(fmt, copy=False)

    else:
        mask_lt = np.tri(size, k=-k, dtype=np.bool_)
        xs = np.zeros(shape, dtype=x.dtype)

        xs[mask_lt.T] = x
        if part == 'both':
            xs.T[mask_lt.T] = x
        elif part == 'lower':
            xs = xs.T

    return xs
