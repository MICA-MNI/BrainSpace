"""
Functions on PointData and CellData.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
from scipy.stats import mode
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import laplacian

from sklearn.utils.extmath import weighted_mode

from vtk import (vtkCellSizeFilter, vtkCellCenters,
                 vtkPolyDataConnectivityFilter)

from . import mesh_elements as me
from ..utils.parcellation import map_to_mask, reduce_by_labels
from ..vtk_interface import wrap_vtk, serial_connect
from ..vtk_interface.decorators import append_vtk, wrap_input


@append_vtk(to='cell')
def compute_cell_area(surf, append=False, key='cell_area'):
    """Compute cell area.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    append : bool, optional
        If True, append array to cell data attributes of input surface
        and return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's cell data attributes. Only used if
        ``append == True``. Default is 'cell_area'.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if ``append == False``. Otherwise, return input surface
        with the new array.

    """

    alg = wrap_vtk(vtkCellSizeFilter, computeArea=True, areaArrayName=key,
                   computeVolume=False, computeLength=False, computeSum=False,
                   computeVertexCount=False)

    # alg = vtkCellSizeFilter()
    # alg.SetComputeArea(True)
    # alg.SetAreaArrayName(key)
    # alg.SetComputeVolume(False)
    # alg.SetComputeLength(False)
    # alg.SetComputeVertexCount(False)
    # alg.ComputeSumOff()
    return serial_connect(surf, alg).CellData[key]


@append_vtk(to='cell')
def compute_cell_center(surf, append=False, key='cell_center'):
    """Compute center of cells (parametric center).

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    append : bool, optional
        If True, append array to cell data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's cell data attributes. Only used if
        ``append == True``. Default is 'cell_center'.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if ``append == False``. Otherwise, return input surface
        with the new array.

    """

    return serial_connect(surf, vtkCellCenters()).Points


@append_vtk(to='point')
def get_n_adjacent_cells(surf, append=False, key='point_ncells'):
    """Compute number of adjacent cells for each point.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    append : bool, optional
        If True, append array to cell data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's point data attributes. Only used if
        ``append == True``. Default is 'point_ncells'.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if ``append == False``. Otherwise, return input surface
        with the new array.

    """

    return me.get_point2cell_connectivity(surf).getnnz(axis=1)


@append_vtk(to='point')
def map_celldata_to_pointdata(surf, cell_data, red_func='mean',
                              dtype=None, append=False, key=None):
    """Map cell data to point data.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    cell_data : str, 1D ndarray
        Array with cell data. If str, it must be in cell data attributes
        of `surf`.
    red_func : str or callable, optional.
        Function used to compute point data from data of neighboring
        cells. If str, options are {'sum', 'mean', 'mode', 'one_third', 'min',
        'max'}. Default is 'mean'.
    dtype : dtype, optional
        Data type of new array. If None, use the same data type of cell data
        array. Default is None.
    append: bool, optional
        If True, append array to point data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's point data attributes. Only used if
        ``append == True``. Default is None.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if ``append == False``. Otherwise, return input surface
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
                              dtype=None, append=False, key=None):
    """Map point data to cell data.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    point_data : str, 1D ndarray
        Array with point data. If str, it is in the point data attributes
        of `surf`. If ndarray, use this array as point data.
    red_func : {'sum', 'mean', 'mode', 'min', 'max'} or callable, optional
        Function used to compute data of each cell from data of its points.
        Default is 'mean'.
    dtype : dtype, optional
        Data type of new array. If None, use the same data type of point data
        array. Default is None.
    append: bool, optional
        If True, append array to cell data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's cell data attributes. Only used if
        ``append == True``. Default is None.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        Return ndarray if ``append == False``. Otherwise, return input surface
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
                       append=False, key='point_area'):
    """Compute point area from its adjacent cells.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    cell_area : str, 1D ndarray or None, optional
        Array with cell areas. If str, it must be in the cell data attributes
        of `surf`. If None, cell areas are computed first.
        Default is None.
    area_as : {'one_third', 'sum', 'mean'}, optional
        Compute point area as 'one_third', 'sum' or 'mean' of adjacent cells.
        Default is 'one_third'.
    append : bool, optional
        If True, append array to point data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's point data attributes. Only used if
        ``append == True``. Default is 'point_area'.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        1D array with point area. Return ndarray if ``append == False``.
        Otherwise, return input surface with the new array.

    """

    if cell_area is None:
        cell_area = compute_cell_area(surf)
    elif isinstance(cell_area, str):
        cell_area = surf.get_array(name=cell_area, at='c')

    return map_celldata_to_pointdata(surf, cell_area, red_func=area_as,
                                     dtype=cell_area.dtype)


@append_vtk(to='point')
def get_connected_components(surf, append=False, key='components'):
    """Get connected components.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    append : bool, optional
        If True, append array to point data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's point data attributes. Only used if
        ``append == True``. Default is 'components'.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        1D array with different labels for each connected component.
        Return ndarray if ``append == False``. Otherwise, return input surface
        with the new array.

    """

    alg = wrap_vtk(vtkPolyDataConnectivityFilter, extractionMode='AllRegions',
                   colorRegions=True)

    # alg = vtkPolyDataConnectivityFilter()
    # alg.SetExtractionModeToAllRegions()
    # alg.ColorRegionsOn()
    return serial_connect(surf, alg).PointData['RegionId']


@append_vtk(to='point')
def get_labeling_border(surf, labeling, append=False, key='border'):
    """Get labeling borders.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    labeling : str, 1D ndarray
        Array with labels. If str, it must be in the point data
        attributes of `surf`.
    append : bool, optional
        If True, append array to point data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's point data attributes. Only used if
        ``append == True``. Default is 'border'.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        A 1D array with ones in the borders. Return array if
        ``append == False``. Otherwise, return input surface with the
        new array.

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
def get_parcellation_centroids(surf, labeling, non_centroid=0, mask=None,
                               append=False, key='centroids'):
    """Compute parcels centroids.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    labeling : str, 1D ndarray
        Array with labels. If str, it must be in the point data
        attributes of `surf`. If ndarray, use this array as the labeling.
    non_centroid : int, optional
        Label assigned to non-centroid points. Default is 0.
    mask : 1D ndarray, optional
        Binary mask. If specified, only consider points within the mask.
        Default is None.
    append : bool, optional
        If True, append array to point data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's point data attributes. Only used if
        ``append == True``. Default is 'centroids'.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        A 1D array with the centroids assigned to their corresponding labels
        and the rest of points assigned `non_centroid`. Return array if
        ``append == False``. Otherwise, return input surface with the
        new array.

    """
    if isinstance(labeling, str):
        labeling = surf.get_array(name=labeling, at='p')

    if mask is not None:
        labeling = labeling[mask]

    ulab = np.unique(labeling)
    if np.isin(non_centroid, ulab, assume_unique=True):
        raise ValueError("Non-centroid label is a valid label. Please choose "
                         "another label.")

    pts = me.get_points(surf)
    if mask is not None:
        pts = pts[mask]

    centroids = reduce_by_labels(pts, labeling, axis=1, target_labels=ulab)

    centroid_labs = np.full_like(labeling, non_centroid)
    idx_pts = np.arange(labeling.size)
    for i, c in enumerate(centroids):
        mask_parcel = labeling == ulab[i]
        dif = c - pts[mask_parcel]
        idx = np.einsum('ij,ij->i', dif, dif).argmin()
        idx_centroid = idx_pts[mask_parcel][idx]
        centroid_labs[idx_centroid] = ulab[i]

    if mask is not None:
        centroid_labs = map_to_mask(centroid_labs, mask=mask,
                                    fill=non_centroid)

    return centroid_labs


@append_vtk(to='point')
def propagate_labeling(surf, labeling, no_label=np.nan, mask=None, alpha=0.99,
                       n_iter=30, tol=0.001, n_ring=1, mode='connectivity',
                       append=False, key='propagated'):
    """Propagate labeling on surface points.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    labeling : str, 1D ndarray
        Array with initial labels. If str, it must be in the point data
        attributes of `surf`. If ndarray, use this array as the initial
        labeling.
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
        Consider points in the n-th ring to label the unlabeled points.
        Default is 1.
    mode : {'connectivity', 'distance'}, optional
        Propagation based on connectivity or geodesic distance. Default is
        'connectivity'.
    append : bool, optional
        If True, append array to point data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's point data attributes. Only used if
        ``append == True``. Default is 'propagated'.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        A 1D array with the propagated labeling. Return array if
        ``append == False``. Otherwise, return input surface with the
        new array.

    References
    ----------
    * Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
      Learning with local and global consistency. Advances in neural
      information processing systems, 16(16), 321-328.

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


@append_vtk(to='point')
def smooth_array(surf, point_data, n_iter=5, mask=None, kernel='gaussian',
                 relax=0.2, sigma=None, append=False, key=None):
    """Propagate labeling on surface points.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    point_data : str, 1D ndarray
        Input array to smooth. If str, it must be in the point data
        attributes of `surf`. If ndarray, use this array.
    n_iter : int, optional
        Number of smoothing iterations. Default is 5.
    mask : 1D ndarray, optional
        Binary mask. If specified, smoothing is only performed on points
        within the mask. Default is None.
    kernel : {'uniform', 'gaussian', 'inverse_distance'}, optional
        Smoothing kernel. Default is 'gaussian'.
    relax : float, optional
        Relaxation factor, contribution of neighboring points such that
        ``0 < relax < 1``. Default is 0.2.
    sigma :  float, optional
        Gaussian kernel width. If None, use standard deviation of egde lengths.
        Default is None.
    append : bool, optional
        If True, append array to point data attributes of input surface and
        return surface. Otherwise, only return array. Default is False.
    key : str, optional
        Array name to append to surface's point data attributes. Only used if
        ``append == True``. Default is None.

    Returns
    -------
    output : vtkPolyData, BSPolyData or ndarray
        A 1D array with the smoothed data. Return array if
        ``append == False``. Otherwise, return input surface with the
        new array.

    """

    if relax <= 0 or relax >= 1:
        raise ValueError('Relaxation factor must be between 0 and 1.')

    if isinstance(point_data, str):
        point_data = surf.get_array(name=point_data, at='p')

    if mask is not None:
        pd = point_data[mask]
    else:
        pd = point_data

    if kernel == 'uniform':
        w = me.get_immediate_adjacency(surf, include_self=False, mask=mask,
                                       dtype=np.float)
    elif kernel == 'gaussian':
        w = me.get_immediate_distance(surf, metric='sqeuclidean', mask=mask)
        if sigma is None:
            # sigma = w.data.mean() + 3 * w.data.std()
            sigma = w.data.std()
        w.data *= -.5 / (sigma*sigma)
        np.exp(w.data, w.data)
    elif kernel == 'inverse_distance':
        w = me.get_immediate_distance(surf, metric='euclidean', mask=mask)
        w.data **= -1
    else:
        raise ValueError("Unknown kernel: {0}".format(kernel))

    w = w.tocoo(copy=False)
    ws = w.sum(axis=1).A1
    w.data *= relax/ws[w.row]

    retain = np.ones(pd.shape)
    retain[ws > 0] -= relax

    if np.issubdtype(pd.dtype, np.floating):
        spd = pd.copy()
    else:
        spd = pd.astype(np.float)

    for i in range(n_iter):
        wp = w.dot(spd)
        spd *= retain
        spd += wp

    if mask is not None:
        spd = map_to_mask(spd, mask=mask)
        spd[mask] = point_data[mask]

    return spd


@wrap_input(0, 1)
def resample_pointdata(source_surf, target_surf, source_name, ops='mean',
                       append=False, key=None):
    """Resample point data in source to target surface.

    Parameters
    ----------
    source_surf : vtkPolyData or BSPolyData
        Source surface.
    target_surf : vtkPolyData or BSPolyData
        Target surface.
    source_name : str or list of str
        Point data in source surface to resample.
    ops : {'mean', 'weighted_mean', 'mode', 'weighted_mode'}, optional
        How is data resampled. Default is 'mean'.
    append: bool, optional
        If True, append array to point data attributes of target surface and
        return surface. Otherwise, only return resampled arrays.
        Default is False.
    key : str or list of str, optional
        Array names to append to target's point data attributes. Only used if
        ``append == True``. If None, use names in `source_name`.
        Default is None.

    Returns
    -------
    output : vtkPolyData, BSPolyData or list of ndarray
        Resampled point data. Return ndarray or list of ndarray if
        ``append == False``. Otherwise, return target surface with the
        new arrays.

    Notes
    -----
    This function is meant for the same source and target surfaces but with
    different number of points. For other types of resampling, see
    vtkResampleWithDataSet.

    """

    if not isinstance(source_name, list):
        source_name = [source_name]
        is_list = False
    else:
        is_list = True

    if not isinstance(ops, list):
        ops = [ops] * len(source_name)

    if key is None:
        key = source_name

    cell_centers = compute_cell_center(source_surf)
    cells = me.get_cells(source_surf)

    tree = cKDTree(cell_centers, leafsize=20, compact_nodes=False,
                   copy_data=False, balanced_tree=False)
    _, idx_cell = tree.query(target_surf.Points, k=1, eps=0, n_jobs=1)

    closest_cells = cells[idx_cell]
    if np.any([op1 in ['weighted_mean', 'weighted_mode'] for op1 in ops]):
        dist_to_cell_points = np.sum((target_surf.Points[:, None] -
                                      source_surf.Points[closest_cells])**2,
                                     axis=-1)
        dist_to_cell_points **= .5
        dist_to_cell_points += np.finfo(np.float).eps
        weights = 1 / dist_to_cell_points

    resampled = [None] * len(source_name)
    for i, fn in enumerate(source_name):
        candidate_feat = source_surf.get_array(fn, at='p')[closest_cells]
        if ops[i] == 'mean':
            feat = np.mean(candidate_feat, axis=1)
        elif ops[i] == 'weighted_mean':
            feat = np.average(candidate_feat, weights=weights, axis=1)
        elif ops[i] == 'mode':
            feat = mode(candidate_feat, axis=1)[0].squeeze()
            feat = feat.astype(candidate_feat.dtype)
        elif ops[i] == 'weighted_mode':
            feat = weighted_mode(candidate_feat, weights, axis=1)[0].squeeze()
            feat = feat.astype(candidate_feat.dtype)
        else:
            raise ValueError('Unknown op: {0}'.format(ops[i]))

        resampled[i] = feat

    if append:
        for i, feat in enumerate(resampled):
            target_surf.append_array(feat, name=key[i], at='p')
        return target_surf
    return resampled if is_list else resampled[0]
