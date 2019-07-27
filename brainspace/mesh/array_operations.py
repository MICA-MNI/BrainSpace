"""
Functions on PointData and CellData.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
from scipy.stats import mode
from scipy.sparse.csgraph import laplacian

from vtkmodules.vtkFiltersVerdictPython import vtkCellSizeFilter
from vtkmodules.vtkFiltersGeneralPython import vtkCellCenters
from vtkmodules.vtkFiltersCorePython import vtkPolyDataConnectivityFilter

from . import mesh_elements as me
from ..utils.parcellation import map_to_mask, reduce_by_labels
from ..vtk_interface.pipeline import serial_connect
from ..vtk_interface.decorators import append_vtk


@append_vtk(to='cell')
def compute_cell_area(surf, append=False, array_name='cell_area'):
    """Compute cell area.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    append : bool, optional
        If True, append array to cell data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's cell data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if `append` is False. Otherwise, return input surface
        with the new array.

    """

    alg = vtkCellSizeFilter()
    alg.SetComputeArea(True)
    alg.SetAreaArrayName('cell_area')
    alg.SetComputeVolume(False)
    alg.SetComputeLength(False)
    alg.SetComputeVertexCount(False)
    alg.ComputeSumOff()
    return serial_connect(surf, alg).CellData['cell_area']


@append_vtk(to='cell')
def compute_cell_center(surf, append=False, array_name='cell_center'):
    """Compute center of cells (parametric center).

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    append : bool, optional
        If True, append array to cell data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's cell data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if `append` is False. Otherwise, return input surface
        with the new array.

    """

    return serial_connect(surf, vtkCellCenters()).Points


@append_vtk(to='point')
def get_n_adjacent_cells(surf, append=False, array_name='point_ncells'):
    """Compute number of adjacent cells for each point.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    append : bool, optional
        If True, append array to cell data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's cell data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if `append` is False. Otherwise, return input surface
        with the new array.

    """

    return me.get_point2cell_connectivity(surf).getnnz(axis=1)


@append_vtk(to='point')
def map_celldata_to_pointdata(surf, cell_data, red_func='mean',
                              dtype=None, append=False, array_name=None):
    """Map cell data to point data.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    cell_data : str, 1D ndarray
        Array with cell data. If str, it is in `surf.CellData[cell_data]`.
        If ndarray, use this array as cell data.
    red_func : {'sum', 'mean', 'mode', 'one_third', 'min', 'max'} or callable,
        optional.
        Function used to compute point data from data of neighboring
        cells. Default is 'mean'.
    dtype : dtype, optional
        Dtype of new array. Default is None.
    append: bool, optional
        If True, append array to point data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's point data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if `append` is False. Otherwise, return input surface
        with the new array.

    """

    if red_func not in ['sum', 'mean', 'mode', 'one_third', 'min', 'max'] and \
            not callable(red_func):
        ValueError('Unknown reduction function \'{0}\'.'.format(red_func))

    if isinstance(cell_data, str):
        cell_data = surf.get_array(name=cell_data, at='c')

    pc = me.get_point2cell_connectivity(surf)
    if isinstance(red_func, str) and red_func != 'mode':

        if red_func in ['sum', 'mean', 'one_third']:
            pd = pc * cell_data
            if red_func == 'mean':
                nnz_row = pc.getnnz(axis=1)
                nnz_row[nnz_row == 0] = 1  # Avoid NaN
                pd = pd / nnz_row
            elif red_func == 'one_third':
                pd = pd / 3
        else:
            pd1 = pc.multiply(cell_data)
            if red_func == 'max':
                pd = np.maximum.reduceat(pd1.data, pc.indptr[:-1])
            else:  # min
                pd = np.minimum.reduceat(pd1.data, pc.indptr[:-1])
            pd[np.diff(pc.indptr) == 0] = 0

        return pd if dtype is None else pd.astype(dtype)

    if dtype is None:
        dtype = cell_data.dtype if red_func == 'mode' else np.float32

    if red_func == 'mode':
        def mode_func(x):
            return mode(x)[0]
        red_func = mode_func

    pd = np.zeros(surf.n_points, dtype=dtype)

    pd1 = pc.multiply(cell_data)
    for i in range(pd.size):
        data_row = pd1.data[pc.indptr[i]:pc.indptr[i + 1]]
        if data_row.size > 0:
            pd[i] = red_func(data_row)

    return pd


@append_vtk(to='cell')
def map_pointdata_to_celldata(surf, point_data, red_func='mean',
                              dtype=None, append=False, array_name=None):
    """Map point data to cell data.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    point_data : str, 1D ndarray
        Array with point data. If str, it is in `surf.PointData[cell_data]`.
        If ndarray, use this array as point data.
    red_func : {'sum', 'mean', 'mode', 'min', 'max'} or callable, optional
        Function used to compute data of each cell from data of its points.
        Default is 'mean'.
    dtype : dtype, optional
        Dtype of new array. Default is None.
    append: bool, optional
        If True, append array to cell data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's cell data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, VTKObjectWrapper or ndarray
        Return ndarray if `append` is False. Otherwise, return input surface
        with the new array.

    """

    if red_func not in ['sum', 'mean', 'mode', 'min', 'max'] and \
            not callable(red_func):
        ValueError('Unknown reduction function \'{0}\'.'.format(red_func))

    if isinstance(point_data, str):
        point_data = surf.get_array(name=point_data, at='p')

    cp = me.get_cell2point_connectivity(surf)
    if isinstance(red_func, str) and red_func != 'mode':
        if red_func in ['sum', 'mean']:
            cd = cp * point_data
            if red_func == 'mean':
                nnz_row = cp.getnnz(axis=1)
                nnz_row[nnz_row == 0] = 1  # Avoid NaN
                cd = cd / nnz_row
        else:
            pd1 = cp.multiply(point_data)
            if red_func == 'max':
                cd = np.maximum.reduceat(pd1.data, cp.indptr[:-1])
            else:  # min
                cd = np.minimum.reduceat(pd1.data, cp.indptr[:-1])
            cd[np.diff(cp.indptr) == 0] = 0

        return cd if dtype is None else cd.astype(dtype)

    if dtype is None:
        dtype = point_data.dtype if red_func == 'mode' else np.float32

    if red_func == 'mode':
        def mode_func(x):
            return mode(x)[0]
        red_func = mode_func

    cd = np.zeros(surf.GetNumberOfCells(), dtype=dtype)
    pd1 = cp.multiply(point_data)
    for i in range(cd.size):
        data_row = pd1.data[cp.indptr[i]:cp.indptr[i + 1]]
        if data_row.size > 0:
            cd[i] = red_func(data_row)

    return cd


@append_vtk(to='point')
def compute_point_area(surf, cell_area=None, area_as='one_third',
                       append=False, array_name='point_area'):
    """Compute point area (as one third of areas of adjacent cells).

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    cell_area : str, 1D ndarray or None, optional
        Array with cell areas. If str, it is in `surf.CellData[cell_area]`.
        If ndarray, use this array. If None, compute cell areas.
        Default is None.
    area_as : {'one_third', 'sum', 'mean'}, optional
        Compute point area as 'one_third', 'sum' or 'mean' of adjacent cells.
        Default is 'one_third'.
    append : bool, optional
        If True, append array to point data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's point data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, VTKObjectWrapper or ndarray
        Return ndarray if `append` is False. Otherwise, return input surface
        with the new array.

    """

    if cell_area is None:
        cell_area = compute_cell_area(surf)
    elif isinstance(cell_area, str):
        cell_area = surf.get_array(name=cell_area, at='c')

    return map_celldata_to_pointdata(surf, cell_area, red_func=area_as,
                                     odtype=cell_area.dtype)


@append_vtk(to='point')
def get_connected_components(surf, append=False, array_name='components'):
    """Get connected components.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    append : bool, optional
        If True, append array to point data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's point data. Only used if
        ``append=True``. Default is 'components'.

    Returns
    -------
    output : vtkPolyData, VTKObjectWrapper or ndarray
        A 1D array with with different labels for each connected component.
        Return array if `append` is False. Otherwise, return input surface
        with the new array.

    """

    ccf = vtkPolyDataConnectivityFilter()
    ccf.SetExtractionModeToAllRegions()
    ccf.ColorRegionsOn()
    return serial_connect(surf, ccf).PointData['RegionId']


@append_vtk(to='point')
def get_labeling_border(surf, labeling, append=False, array_name='border'):
    """Get labeling borders.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    labeling : str, 1D ndarray
        Array with labels. If str, it is in `surf.PointData[labeling]`.
        If ndarray, use this array as the labeling.
    append : bool, optional
        If True, append array to point data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's point data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, VTKObjectWrapper or ndarray
        A 1D array with ones in the borders. Return array if `append`
        is False. Otherwise, return input surface with the new array.

    """

    edges = me.get_edges(surf)
    if isinstance(labeling, str):
        labeling = surf.get_array(name=labeling, at='p')
    edge_labels = labeling[edges]
    idx_border = np.unique(edges[edge_labels[:, 0] != edge_labels[:, 1]])
    border = np.zeros_like(labeling, dtype=np.uint8)
    border[idx_border] = 1
    return border


@append_vtk(to='point')
def get_parcellation_centroids(surf, labeling, non_centroid=0,
                               append=False, array_name='centroids'):
    """Get labeling borders.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    labeling : str, 1D ndarray
        Array with labels. If str, it is in `surf.PointData[labeling]`.
        If ndarray, use this array as the labeling.
    non_centroid : int, optional
        Label assigned to non-centroid points. Default is 0.
    append : bool, optional
        If True, append array to point data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's point data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, VTKObjectWrapper or ndarray
        A 1D array with the centroids assigned to their corresponding labels
        and the rest of points assigned `no_label`.
        Return array if ``append=False``. Otherwise, return input surface
        with the new array.

    """
    if isinstance(labeling, str):
        labeling = surf.get_array(name=labeling, at='p')

    ulab = np.unique(labeling)
    if np.isin(non_centroid, ulab, assume_unique=True):
        raise ValueError("Non-centroid label is a valid label. Please choose "
                         "another label.")

    pts = me.get_points(surf)
    centroids = reduce_by_labels(pts, labeling, axis=1, target_labels=ulab)

    centroid_labs = np.full_like(labeling, non_centroid)
    idx_pts = np.arange(labeling.size)
    for i, c in centroids:
        mask_parcel = labeling == ulab[i]
        dif = c - pts[mask_parcel]
        idx = np.einsum('ij,ij->i', dif, dif).argmin()
        idx_centroid = idx_pts[mask_parcel][idx]
        centroid_labs[idx_centroid] = ulab[i]

    return centroid_labs


@append_vtk(to='point')
def propagate_labeling(surf, labeling, no_label=np.nan, mask=None, alpha=0.99,
                       n_iter=30, tol=0.001, n_ring=1, mode='connectivity',
                       append=False, array_name='propagated'):
    """Propagate labeling on surface points.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    labeling : str, 1D ndarray
        Array with initial labels. If str, it is in `surf.PointData[labeling]`.
        If ndarray, use this array as the initial labeling.
    no_label : int or np.nan, optional
        Value for unlabeled points. Default is np.nan.
    mask : 1D ndarray, optional
        Binary mask. If specified, propagation is only performed on points
        within the mask. Default is None.
    alpha : float, optional
        Clamping factor such that ``0 < aplha < 1``. Deault is 0.99.
    n_iter : int, optional
        Maximum number of propagation iterations. Default is 30.
    tol :  float, optional
        Convergence tolerance. Default is 0.001.
    n_ring : positive int, optional
        Consider points in n_ring to label the unlabeled points.
        Default is 1.
    mode : {'connectivity', 'distance'}, optional
        Propagation based on connectivity of geodesic distance. Default is
        'connectivity'.
    append : bool, optional
        If True, append array to point data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's point data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, VTKObjectWrapper or ndarray
        A 1D array with the propagated labeling. Return array if `append`
        is False. Otherwise, return input surface with the new array.

    References
    ----------
    [1] Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency. Advances in neural information
    processing systems, 16(16), 321-328.

    """

    if isinstance(labeling, str):
        labeling = surf.get_array(name=labeling, at='p')

    if mask is not None:
        labeling = labeling[mask]

    if no_label is np.nan:
        labeled = ~np.isnan(labeling) != 0
    else:
        labeled = labeling != no_label

    ulabs, idx_lab = np.unique(labeling[labeled], return_inverse=True)

    n_labs = ulabs.size
    n_pts = labeling.size

    # Graph matrix
    if mode == 'connectivity':
        adj = me.get_ring_adjacency(surf, n_ring=n_ring, include_self=False,
                                    dtype=np.float)
    else:
        adj = me.get_ring_distance(surf, n_ring=n_ring, dtype=np.float)
        adj.data[:] = np.exp(-adj.data/n_ring**2)

    if mask is not None:
        adj = adj[mask][:, mask]

    graph_matrix = -alpha * laplacian(adj, normed=True)
    diag_mask = (graph_matrix.row == graph_matrix.col)
    graph_matrix.data[diag_mask] = 0.0

    # Label distributions and label static
    lab_dist = np.zeros((n_pts, n_labs))
    lab_dist[np.argwhere(labeled)[:, 0], idx_lab] = 1

    lab_static = lab_dist.copy()
    lab_static *= 1 - alpha

    # propagation
    lab_dist_perv = lab_dist
    for i in range(n_iter):
        lab_dist = graph_matrix.dot(lab_dist) + lab_static

        if np.linalg.norm(lab_dist - lab_dist_perv, 'fro') < tol:
            break

        lab_dist_perv = lab_dist

    # lab_dist /= lab_dist.sum(axis=1, keepdims=True)
    new_labeling = labeling.copy()
    new_labeling[~labeled] = ulabs[np.argmax(lab_dist[~labeled], axis=1)]

    if mask is not None:
        new_labeling = map_to_mask(new_labeling, mask)

    return new_labeling


def smooth_array(surf, point_data, n_iter=10, mask=None, include_self=True,
                 kernel='gaussian', sigma=1):
    """Map point data to cell data.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    point_data : str, 1D ndarray
        Array with point data. If str, it is in `surf.PointData[cell_data]`.
        If ndarray, use this array as point data.
    red_func : {'sum', 'mean', 'mode', 'min', 'max'} or callable, optional
        Function used to compute data of each cell from data of its points.
        Default is 'mean'.
    dtype : dtype, optional
        Dtype of new array. Default is None.
    append: bool, optional
        If True, append array to cell data of input surface and return surface.
        Otherwise, only return array.
    array_name : str, optional
        Array name to append to surface's cell data. Only used if
        ``append=True``.

    Returns
    -------
    output : vtkPolyData, VTKObjectWrapper or ndarray
        Return ndarray if `append` is False. Otherwise, return input surface
        with the new array.

    """

    if isinstance(point_data, str):
        pd = surf.get_array(name=point_data, at='p')

    # adj = me.get_immediate_adjacency(surf)
    # d = me.get_immediate_distance(surf)
    # d.setdiag(0)
    if kernel == 'uniform':
        w = me.get_immediate_adjacency(surf, include_self=include_self)
    elif kernel == 'gaussian':
        w = me.get_immediate_distance(surf, metric='sqeuclidean')
        # w.setdiag(0)
        w.data *= -.5 / (sigma*sigma)
        w.data[:] = np.exp(w.data)
        if include_self:
            w.setdiag(1)
    elif kernel == 'inverse_distance':
        w = me.get_immediate_distance(surf, metric='euclidean')
        w.data **= -1
        if include_self:
            w.setdiag(0)

    lam = 1
    mu = lam
    norm = w.sum(axis=1)
    alpha = np.ones(pd.shape)
    alpha[norm > 0] = 1 - lam

    beta = np.zeros(pd.shape)
    beta[norm > 0] = lam / norm[norm > 0]

    spd = pd.copy()
    for i in range(n_iter):
        spd = alpha * spd + beta * w.dot(spd)

    return spd


