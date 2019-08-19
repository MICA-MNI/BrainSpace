"""
Functions on surface mesh elements.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
from scipy import sparse as sps
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra

from ..vtk_interface.decorators import wrap_input


@wrap_input(0)
def get_points(surf, mask=None):
    """Get surface points.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only get points within the mask.
        Default is None.

    Returns
    -------
    points : ndarray, shape (n_points, 3)
        Array of points.

    See Also
    --------
    :func:`get_cells`
    :func:`get_edges`

    """

    pts = surf.Points
    return pts if mask is None else pts[mask]


@wrap_input(0)
def get_cells(surf):
    """Get surface cells.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.

    Returns
    -------
    cells : ndarray, shape (n_cells, nd)
        Array of cells. The value of nd depends on the topology. If vertex
        (nd=1), line (nd=2) or poly (nd=3). Each element is a point id.

    Raises
    ------
    ValueError
        If `surf` contains different cell types.

    See Also
    --------
    :func:`get_points`
    :func:`get_edges`

    """

    if not surf.has_unique_cell_type:
        raise ValueError('Surface has different types of cells.')
    cells = surf.Polygons
    return cells.reshape(-1, cells[0] + 1)[:, 1:]


def get_extent(surf):
    """Get data extent.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.

    Returns
    -------
    extent : 1D ndarray, shape (3,)
        Extent of data.

    """

    bounds = np.array(surf.GetBounds())
    return bounds[1::2] - bounds[::2]


def get_point2cell_connectivity(surf, dtype=np.uint8):
    """Get point to cell connectivity.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    output : sparse matrix, shape (n_points, n_cells)
        The connectivity matrix. The (i,j) entry is 1 if the j-th cell
        uses the i-th point.

    Notes
    -----
    This function returns the transpose of :func:`get_cell2point_connectivity`.

    See Also
    --------
    :func:`get_cell2point_connectivity`
    :func:`get_cell_point_neighbors`
    :func:`get_cell_edge_neighbors`

    """

    cells = get_cells(surf)
    n_cells, n_pts_cell = cells.shape
    n_pts = surf.GetNumberOfPoints()

    data = np.ones(cells.size, dtype=dtype)
    idx_row = cells.ravel()
    idx_col = np.broadcast_to(np.arange(n_cells)[:, None],
                              (n_cells, n_pts_cell)).ravel()

    return sps.csr_matrix((data, (idx_row, idx_col)), shape=(n_pts, n_cells))


def get_cell2point_connectivity(surf, dtype=np.uint8):
    """Get cell to point connectivity.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    output : sparse matrix, shape (n_cells, n_points)
        The connectivity matrix. The (i,j) entry is 1 if the i-th cell
        uses the j-th point.

    See Also
    --------
    :func:`get_point2cell_connectivity`
    :func:`get_cell_point_neighbors`
    :func:`get_cell_edge_neighbors`

    Notes
    -----
    This function returns the transpose of :func:`get_point2cell_connectivity`.

    """

    return get_point2cell_connectivity(surf, dtype=dtype).T.tocsr(copy=False)


def get_cell_point_neighbors(surf, include_self=True, dtype=np.uint8):
    """Get cell connectivity based on shared points.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    include_self : bool, optional
        If True, set diagonal elements to 1. Default is True.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    output : sparse matrix, shape (n_cells, n_cells)
        The connectivity matrix. The (i,j) entry is 1 if cells i and j share
        a point.

    See Also
    --------
    :func:`get_point2cell_connectivity`
    :func:`get_cell2point_connectivity`
    :func:`get_cell_edge_neighbors`

    """

    cp = get_cell2point_connectivity(surf, dtype=np.bool)
    cp *= cp.T
    if not include_self:
        cp.setdiag(0)
        cp.eliminate_zeros()
    cp.data = cp.data.astype(dtype, copy=False)
    return cp


def get_cell_edge_neighbors(surf, include_self=True, dtype=np.uint8):
    """Get cell connectivity based on shared edges.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    include_self : bool, optional
        If True, set diagonal elements to 1. Default is True.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    output : sparse matrix, shape (n_cells, n_cells)
        The connectivity matrix. The (i,j) entry is 1 if cells i and j share
        an edge.

    See Also
    --------
    :func:`get_point2cell_connectivity`
    :func:`get_cell2point_connectivity`
    :func:`get_cell_point_neighbors`

    """

    ce = get_cell2point_connectivity(surf, dtype=np.uint8)
    ce *= ce.T
    ce.data = ce.data >= 2
    if not include_self:
        ce.setdiag(0)
    ce.eliminate_zeros()
    ce.data = ce.data.astype(dtype, copy=False)
    return ce


def get_immediate_adjacency(surf, include_self=True, mask=None, dtype=np.uint8):
    """Get immediate adjacency matrix.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    include_self : bool, optional
        If True, set diagonal elements to 1. Default is True.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    adj : sparse matrix, shape (n_points, n_points)
        Immediate adjacency matrix.

    See Also
    --------
    :func:`get_ring_adjacency`
    :func:`get_immediate_distance`
    :func:`get_ring_distance`

    Notes
    -----
    Immediate adjacency: set to one all entries of points that
    share and edge with current point.
    """

    adj = get_point2cell_connectivity(surf, dtype=np.bool)
    if mask is not None:
        adj = adj[mask]
    adj *= adj.T
    if not include_self:
        adj.setdiag(0)
        adj.eliminate_zeros()
    adj.data = adj.data.astype(dtype, copy=False)
    return adj


def get_ring_adjacency(surf, n_ring=1, include_self=True, mask=None,
                       dtype=np.uint8):
    """Get adjacency in the neighborhood of each point.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    n_ring : int, optional
        Size of neighborhood. Default is 1.
    include_self : bool, optional
        If True, set diagonal elements to 1. Otherwise, the diagonal
        is set to 0. Default is True.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    adj : sparse matrix, shape (n_points, n_points)
        Adjacency matrix in `n_ring` ring.

    See Also
    --------
    :func:`get_immediate_adjacency`
    :func:`get_immediate_distance`
    :func:`get_ring_distance`

    """

    if n_ring == 1:
        return get_immediate_adjacency(surf, include_self=include_self,
                                       mask=mask, dtype=dtype)

    adj = get_immediate_adjacency(surf, include_self=False, mask=mask,
                                  dtype=np.bool)
    adj **= n_ring
    if not include_self:
        adj.setdiag(0)
        adj.eliminate_zeros()
    adj.data = adj.data.astype(dtype, copy=False)
    return adj


def get_edges(surf, mask=None):
    """Get surface edges.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.

    Returns
    -------
    edges : ndarray, shape (n_edges, 2)
        Array of edges. Each element is a point id.

    See Also
    --------
    :func:`get_edge_length`
    :func:`get_points`
    :func:`get_cells`

    """

    adj = get_immediate_adjacency(surf, include_self=False, mask=mask,
                                  dtype=np.bool)
    adj_ud = sps.triu(adj, k=1, format='coo')
    edges = np.column_stack([adj_ud.row, adj_ud.col])
    return edges


def get_edge_length(surf, metric='euclidean', mask=None):
    """Get surface edge lengths.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    metric : {'euclidean', 'sqeuclidean'}, optional
        Distance metric. Default is 'euclidean'.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.

    Returns
    -------
    edges : ndarray, shape (n_edges, 2)
        Array of edges. Each element is a point id.

    See Also
    --------
    :func:`get_edges`
    :func:`get_immediate_distance`

    """

    points = get_points(surf, mask=mask)
    edges = get_edges(surf, mask=mask)

    dif = points[edges[:, 0]] - points[edges[:, 1]]
    d = np.einsum('ij,ij->i', dif, dif)
    if metric == 'euclidean':
        d **= .5
    return d


def get_border_cells(surf):
    """Get cells in boundary.

    Cells in boundary have one boundary edge.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.

    Returns
    -------
    edges : 1D ndarray
        Array of cells in border.

    See Also
    --------
    :func:`get_edges`
    :func:`get_immediate_distance`

    """

    ce = get_cell_edge_neighbors(surf, include_self=False)
    return np.where(ce.getnnz(axis=1) < 3)[0]


def get_border_edges(surf):
    """Get edges in border.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.

    Returns
    -------
    edges : 2D ndarray, shape = (n_edges, 2)
        Array of edges in border. Each element is a point id.

    See Also
    --------
    :func:`get_edges`
    :func:`get_immediate_distance`

    """

    ce = get_cell_edge_neighbors(surf, include_self=False)
    return np.where(ce.getnnz(axis=1) < 3)[0]


def get_immediate_distance(surf, metric='euclidean', mask=None,
                           dtype=np.float32):
    """Get immediate distance matrix.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.
    metric : {'euclidean', 'sqeuclidean'}, optional
        Distance metric. Default is 'euclidean'.
    dtype : dtype, optional
        Data type. Default is float32.

    Returns
    -------
    dist : sparse matrix, shape (n_points, n_points)
        Immediate distance matrix.

    See Also
    --------
    :func:`get_immediate_adjacency`
    :func:`get_ring_adjacency`
    :func:`get_ring_distance`

    Notes
    -----
    Immediate distance: Euclidean distance with all points that
    share and edge with current point.

    """

    points = get_points(surf, mask=mask)
    n_pts = points.shape[0]
    edges = get_edges(surf, mask=mask)

    dif = points[edges[:, 0]] - points[edges[:, 1]]
    d = np.einsum('ij,ij->i', dif, dif)
    if metric == 'euclidean':
        d **= .5

    d = np.broadcast_to(d[:, None], (d.size, 2)).ravel()

    e1, e2 = edges.ravel(), edges[:, ::-1].ravel()
    dist = sps.csr_matrix((d, (e1, e2)), shape=(n_pts, n_pts), dtype=dtype)

    return dist


def get_ring_distance(surf, n_ring=1, metric='geodesic', mask=None,
                      dtype=np.float32):
    """Get distance matrix in the neighborhood of each point.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    n_ring : int, optional
        Size of neighborhood. Default is 1.
    metric : {'euclidean', 'sqeuclidean', 'geodesic'}, optional
        Distance metric. Default is 'geodesic'.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.
    dtype : dtype, optional
        Data type. Default is np.float32.

    Returns
    -------
    dist : sparse matrix, shape (n_points, n_points)
        Distance matrix in `n_ring` ring..

    See Also
    --------
    :func:`get_immediate_adjacency`
    :func:`get_ring_adjacency`
    :func:`get_immediate_distance`

    Notes
    -----
    Distance is only computed for points in the ring of current point.
    When using geodesic, shortest paths are restricted to points within
    the ring.

    """

    if n_ring == 1:
        return get_immediate_distance(surf, mask=mask, dtype=dtype)

    # Distance only restricted to ring
    # Geodesic distance is computed for each point based only on the points
    # in its ring
    if metric == 'geodesic':
        d = get_ring_adjacency(surf, n_ring=n_ring, mask=mask,
                               include_self=True, dtype=dtype)
    else:
        d = get_ring_adjacency(surf, n_ring=n_ring, mask=mask,
                               include_self=False, dtype=dtype)

    n_pts = surf.GetNumberOfPoints() if mask is None else np.count_nonzero(mask)
    if metric == 'geodesic':
        imm_dist = get_immediate_distance(surf, mask=mask, dtype=dtype)
        for i in range(n_pts):
            idx = d[i].indices
            idx_pnt = np.argmax(idx == i)
            d.data[d.indptr[i]:d.indptr[i+1]] = \
                dijkstra(csgraph=imm_dist[idx][:, idx], indices=idx_pnt)

    elif metric in ['euclidean', 'sqeuclidean']:
        points = get_points(surf, mask=mask)
        for i in range(n_pts):
            idx = d[i].indices
            d.data[d.indptr[i]:d.indptr[i+1]] = \
                cdist(points[i:i+1], points[idx], metric=metric)

    else:
        raise ValueError('Unknown metric \'{0}\'. Possible metrics: '
                         '{{\'euclidean\', \'sqeuclidean\', \'geodesic\'}}.'.
                         format(metric))

    return d
