"""
Functions on surface mesh elements.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
import scipy.sparse as ssp
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra

import vtk

from ..vtk_interface import wrap_vtk, serial_connect
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

    return surf.Points if mask is None else surf.Points[mask]


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

    return surf.GetCells2D()


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


@wrap_input(0)
def get_point2cell_connectivity(surf, mask=None, dtype=np.uint8):
    """Get point to cell connectivity.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only get points within the mask.
        Default is None.
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
    :func:`get_cell_neighbors`

    """

    cells = surf.GetCells2D()

    data = np.ones(cells.size, dtype=dtype)
    row = cells.ravel()
    col = np.repeat(np.arange(surf.n_cells), cells.shape[1])
    shape = (surf.n_points, surf.n_cells)

    pc = ssp.csr_matrix((data, (row, col)), shape=shape)
    return pc if mask is None else pc[mask]


@wrap_input(0)
def get_cell2point_connectivity(surf, mask=None, dtype=np.uint8):
    """Get cell to point connectivity.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only get points within the mask.
        Default is None.
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
    :func:`get_cell_neighbors`

    Notes
    -----
    This function returns the transpose of :func:`get_point2cell_connectivity`.

    """

    pc = get_point2cell_connectivity(surf, mask=mask, dtype=dtype)
    return pc.T.tocsr(copy=False)


def get_cell_neighbors(surf, include_self=True, with_edge=True,
                       dtype=np.uint8):
    """Get cell connectivity based on shared edges.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    include_self : bool, optional
        If True, set diagonal elements to 1. Default is True.
    with_edge : bool, optional
        If True, neighboring cells are based on shared edges. Otherwise,
        cells must share, at least, one point. Default is True.
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

    """

    if with_edge:
        ce = get_cell2point_connectivity(surf, dtype=np.uint8)
        ce *= ce.T
        ce.data = ce.data > 1
        if not include_self:
            ce.setdiag(0)
        ce.eliminate_zeros()

    else:
        ce = get_cell2point_connectivity(surf, dtype=np.bool)
        ce *= ce.T
        if not include_self:
            ce.setdiag(0)
            ce.eliminate_zeros()

    ce.data = ce.data.astype(dtype, copy=False)
    return ce


def get_immediate_adjacency(surf, include_self=True, mask=None,
                            dtype=np.uint8):
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

    adj = get_point2cell_connectivity(surf, mask=mask, dtype=np.bool)
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

    adj = get_immediate_adjacency(surf, include_self=True, mask=mask,
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
    adj.sort_indices()
    adj_ud = ssp.triu(adj, k=1, format='coo')
    edges = np.column_stack([adj_ud.row, adj_ud.col])
    return edges


@wrap_input(0)
def get_point2edge_connectivity(surf, mask=None, dtype=np.uint8):
    """Get point to edge connectivity.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    output : sparse matrix, shape (n_points, n_edges)
        The connectivity matrix. The (i,j) entry is 1 if the j-th edge
        uses the i-th point.

    Notes
    -----
    Edges are sorted by point ids, such as edge 0 is the one connecting the
    points with the smallest ids.
    This function returns the transpose of :func:`get_edge2point_connectivity`.

    See Also
    --------
    :func:`get_edge2point_connectivity`
    :func:`get_edges`

    """

    edges = get_edges(surf, mask=mask)
    n_pts = surf.n_points if mask is None else np.count_nonzero(mask)

    data = np.ones(edges.size, dtype=dtype)
    row = edges.ravel()
    col = np.repeat(np.arange(edges.shape[0]), 2)
    shape = (n_pts, edges.shape[0])

    return ssp.csr_matrix((data, (row, col)), shape=shape)


@wrap_input(0)
def get_edge2point_connectivity(surf, mask=None, dtype=np.uint8):
    """Get edge to point connectivity.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    output : sparse matrix, shape (n_edges, n_points)
        The connectivity matrix. The (i,j) entry is 1 if the i-th edge
        uses the j-th point.

    Notes
    -----
    Edges are sorted by point ids, such as edge 0 is the one connecting the
    points with the smallest ids.
    This function returns the transpose of :func:`get_point2edge_connectivity`.

    See Also
    --------
    :func:`get_point2edge_connectivity`
    :func:`get_edges`

    """

    pe = get_point2edge_connectivity(surf, mask=mask, dtype=dtype)
    return pe.T.tocsr(copy=False)


@wrap_input(0)
def get_edge2cell_connectivity(surf, mask=None, dtype=np.uint8):
    """Get edge to cell connectivity.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    output : sparse matrix, shape (n_edges, n_cells)
        The connectivity matrix. The (i,j) entry is 1 if the j-th cell
        uses the i-th edge.

    Notes
    -----
    Edges are sorted by point ids, such as edge 0 is the one connecting the
    points with the smallest ids.
    This function returns the transpose of :func:`get_cell2edge_connectivity`.

    See Also
    --------
    :func:`get_cell2edge_connectivity`
    :func:`get_edges`

    """

    ec = get_edge2point_connectivity(surf, mask=mask, dtype=np.uint8)
    ec *= get_point2cell_connectivity(surf, mask=mask, dtype=np.uint8)
    ec.data = ec.data == 2
    ec.eliminate_zeros()
    ec.data = ec.data.astype(dtype, copy=False)
    return ec


@wrap_input(0)
def get_cell2edge_connectivity(surf, mask=None, dtype=np.uint8):
    """Get cell to edge connectivity.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    mask : 1D ndarray, optional
        Binary mask. If specified, only use points within the mask.
        Default is None.
    dtype : dtype, optional
        Data type. Default is uint8.

    Returns
    -------
    output : sparse matrix, shape (n_cells, n_edges)
        The connectivity matrix. The (i,j) entry is 1 if the i-th cell
        uses the j-th edge.

    Notes
    -----
    Edges are sorted by point ids, such as edge 0 is the one connecting the
    points with the smallest ids.
    This function returns the transpose of :func:`get_edge2cell_connectivity`.

    See Also
    --------
    :func:`get_edge2cell_connectivity`
    :func:`get_edges`

    """

    ec = get_edge2cell_connectivity(surf, mask=mask, dtype=dtype)
    return ec.T.tocsr(copy=False)


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


@wrap_input(0)
def _get_boundary(surf):
    """ Get boundary as polyData of lines.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData

    Returns
    -------
    surf_boundary : BSPolyData
        PolyData with cells as boundary edges.
    boundary_points : 1D ndarray
        Array of point ids in the boundary.
    """

    an = surf.append_array(np.arange(surf.n_points))
    fe = wrap_vtk(vtk.vtkFeatureEdges, boundaryEdges=True, manifoldEdges=False,
                  nonManifoldEdges=False, featureEdges=False)
    bs = serial_connect(surf, fe)
    surf.remove_array(an)
    return bs, bs.get_array(an, at='p')


def get_boundary_points(surf):
    """Get points in boundary.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.

    Returns
    -------
    boundary_points : ndarray, shape (n_points, 2)
        Array of boundary point ids.

    See Also
    --------
    :func:`get_boundary_edges`
    :func:`get_boundary_cells`

    """

    _, bp = _get_boundary(surf)
    return np.sort(bp)


def get_boundary_edges(surf):
    """Get edges in boundary.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.

    Returns
    -------
    boundary_edges : ndarray, shape (n_edges, 2)
        Array of boundary edges. Each element is a point id.

    See Also
    --------
    :func:`get_boundary_points`
    :func:`get_boundary_cells`
    :func:`get_edges`

    """

    bs, bp = _get_boundary(surf)
    if bs.n_cells == 0:
        return np.array([])
    be = bp[bs.GetCells2D()]
    return np.sort(be, axis=1)


def get_boundary_cells(surf, with_edge=True):
    """Get cells in boundary.

    Parameters
    ----------
    surf : vtkDataSet or BSDataSet
        Input surface.
    with_edge : bool, optional
        If True, boundary cells need to have, at least, one boundary edge.
        Otherwise, boundary cells have, at least, one boundary point.
        Default is True.

    Returns
    -------
    cells : 1D ndarray
        Array of boundary cells.

    See Also
    --------
    :func:`get_boundary_points`
    :func:`get_boundary_edges`

    """

    ce = get_cell_neighbors(surf, include_self=False, with_edge=True)
    mask = ce.getnnz(axis=1) < 3
    if not with_edge:
        mask |= ce[mask].getnnz(axis=0) > 0
    return np.argwhere(mask).squeeze()


def get_immediate_distance(surf, metric='euclidean', mask=None,
                           dtype=np.float):
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
        Data type. Default is float.

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
    edges = get_edges(surf, mask=mask)

    dif = points[edges[:, 0]] - points[edges[:, 1]]
    dist = np.einsum('ij,ij->i', dif, dif)
    if metric == 'euclidean':
        dist **= .5

    data = np.repeat(dist, 2).ravel()
    row, col = edges.ravel(), edges[:, ::-1].ravel()
    shape = (points.shape[0], points.shape[0])

    return ssp.csr_matrix((data, (row, col)), shape=shape, dtype=dtype)


def get_ring_distance(surf, n_ring=1, metric='geodesic', mask=None,
                      dtype=np.float):
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
        Data type. Default is np.float.

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

    if metric == 'geodesic':
        imm_dist = get_immediate_distance(surf, mask=mask, dtype=dtype)

        # Faster
        d = get_ring_adjacency(surf, n_ring=n_ring, mask=mask,
                               include_self=True, dtype=dtype)
        for i in range(imm_dist.shape[0]):
            idx = d[i].indices
            idx_pnt = np.argmax(idx == i)
            d.data[d.indptr[i]:d.indptr[i + 1]] = \
                dijkstra(csgraph=imm_dist[idx][:, idx], indices=idx_pnt)

        d.data[np.isinf(d.data)] = 0
        d.eliminate_zeros()

        # Slower
        # d = get_ring_adjacency(surf, n_ring=n_ring, mask=mask,
        #                        include_self=False)
        # d = d.multiply(dijkstra(imm_dist)).astype(dtype)
        # d.data[np.isinf(d.data)] = 0

    elif metric in ['euclidean', 'sqeuclidean']:
        d = get_ring_adjacency(surf, n_ring=n_ring, mask=mask,
                               include_self=False, dtype=dtype)
        points = get_points(surf, mask=mask)
        for i in range(points.shape[0]):
            idx = d[i].indices
            d.data[d.indptr[i]:d.indptr[i+1]] = \
                cdist(points[i:i+1], points[idx], metric=metric)

    else:
        raise ValueError('Unknown metric \'{0}\'. Possible metrics: '
                         '{{\'euclidean\', \'sqeuclidean\', \'geodesic\'}}.'.
                         format(metric))

    return d
