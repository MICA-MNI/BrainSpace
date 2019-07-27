"""
Basic functions on surface meshes.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mode

from sklearn.utils.extmath import weighted_mode

from vtkmodules.vtkCommonDataModelPython import vtkDataObject
from vtkmodules.vtkFiltersCorePython import vtkThreshold
from vtkmodules.vtkFiltersGeometryPython import vtkGeometryFilter

from . import mesh_elements as me, array_operations as aop
from ..vtk_interface.pipeline import serial_connect
from ..vtk_interface.wrappers import wrap_vtk
from ..vtk_interface.decorators import wrap_input


ASSOC_CELLS = vtkDataObject.FIELD_ASSOCIATION_CELLS
ASSOC_POINTS = vtkDataObject.FIELD_ASSOCIATION_POINTS


@wrap_input(only_args=0)
def _surface_selection(surf, array_name, low=-np.inf, upp=np.inf,
                       use_cell=False, keep=True):
    """Selection of points or cells meeting some thresholding criteria.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    array_name : str or ndarray
        Array used to perform selection.
    low : float or -np.inf
        Lower threshold. Default is -np.inf.
    upp : float or np.inf
        Upper threshold. Default is +np.inf.
    use_cell : bool, optional
        If True, apply selection to cells. Otherwise, use points.
        Default is False.
    keep : bool, optional
        If True, elements within the thresholds (inclusive) are kept.
        Otherwise, are discarded. Default is True.

    Returns
    -------
    surf_selected : BSPolyData
        Surface after thresholding.

    """

    if low > upp:
        raise ValueError('Threshold limits are not valid: {0} -- {1}'.
                         format(low, upp))

    at = 'c' if use_cell else 'p'
    if isinstance(array_name, np.ndarray):
        drop_array = True
        array = array_name
        array_name = surf.append_array(array, at=at)
    else:
        drop_array = False
        array = surf.get_array(name=array_name, at=at, return_name=False)

    if array.ndim > 1:
        raise ValueError('Arrays has more than one dimension.')

    if low == -np.inf:
        low = array.min()
    if upp == np.inf:
        upp = array.max()

    tf = wrap_vtk(vtkThreshold(), invert=not keep)
    tf.ThresholdBetween(low, upp)
    if use_cell:
        tf.SetInputArrayToProcess(0, 0, 0, ASSOC_CELLS, array_name)
    else:
        tf.SetInputArrayToProcess(0, 0, 0, ASSOC_POINTS, array_name)

    gf = wrap_vtk(vtkGeometryFilter(), merging=False)
    surf_sel = serial_connect(surf, tf, gf)

    # Check results
    mask = np.logical_and(array >= low, array <= upp)
    if keep:
        n_expected = np.count_nonzero(mask)
    else:
        n_expected = np.count_nonzero(~mask)

    n_sel = surf_sel.n_cells() if use_cell else surf_sel.n_points
    if n_expected != n_sel:
        element = 'cells' if use_cell else 'points'
        warnings.warn('The number of selected {0} is different than expected. '
                      'This may be due to the topology after after selection: '
                      'expected={1}, selected={2}.'.
                      format(element, n_expected, n_sel))

    if drop_array:
        surf.remove_array(name=array_name, at=at)
        surf_sel.remove_array(name=array_name, at=at)

    return surf_sel


@wrap_input(only_args=0)
def _surface_mask(surf, mask, use_cell=False):
    """Selection fo points or cells meeting some criteria.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    mask : str or ndarray
        Binary boolean or integer array. Zero or False elements are
        discarded.
    use_cell : bool, optional
        If True, apply selection to cells. Otherwise, use points.
        Default is False.

    Returns
    -------
    surf_masked : BSPolyData
        PolyData after masking.

    """

    if isinstance(mask, np.ndarray):
        if np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(np.uint8)
    else:
        mask = surf.get_array(name=mask, at='c' if use_cell else 'p')

    if np.any(np.unique(mask) > 1):
        raise ValueError('Cannot work with non-binary mask.')

    return _surface_selection(surf, mask, low=1, upp=1, use_cell=use_cell,
                              keep=True)


def drop_points(surf, array_name, low=-np.inf, upp=np.inf):
    """Remove surface points whose values fall within the threshold.

    Cells corresponding to these points are also removed.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    array_name : str or ndarray
        Array used to perform selection. If str, it must be an
        array in PointData of the surface.
    low : float or -np.inf
        Lower threshold. Default is -np.inf.
    upp : float or np.inf
        Upper threshold. Default is np.inf.

    Returns
    -------
    surf_selected : vtkPolyData or VTKObjectWrapper
        PolyData after thresholding.

    """

    return _surface_selection(surf, array_name, low=low, upp=upp, keep=False)


def drop_cells(surf, array_name, low=-np.inf, upp=np.inf):
    """Remove surface cells whose values fall within the threshold.

    Points corresponding to these cells are also removed.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    array_name : str or ndarray
        Array used to perform selection. If str, it must be an
        array in CellData of the surface.
    low : float or -np.inf
        Lower threshold. Default is -np.inf.
    upp : float or np.inf
        Upper threshold. Default is np.inf.

    Returns
    -------
    surf_selected : vtkPolyData or VTKObjectWrapper
        PolyData after thresholding.

    """

    return _surface_selection(surf, array_name, low=low, upp=upp, use_cell=True,
                              keep=False)


def select_points(surf, array_name, low=-np.inf, upp=np.inf):
    """Select surface points whose values fall within the threshold.

    Cells corresponding to these points are also kept.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    array_name : str or ndarray
        Array used to perform selection. If str, it must be an
        array in PointData of the surface.
    low : float or -np.inf
        Lower threshold. Default is -np.inf.
    upp : float or np.inf
        Upper threshold. Default is np.inf.

    Returns
    -------
    surf_selected : vtkPolyData or VTKObjectWrapper
        PolyData after selection.

    """

    return _surface_selection(surf, array_name, low=low, upp=upp, keep=True)


def select_cells(surf, array_name, low=-np.inf, upp=np.inf):
    """Select surface cells whose values fall within the threshold.

    Points corresponding to these cells are also kept.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    array_name : str or ndarray
        Array used to perform selection. If str, it must be an
        array in CellData of the surface.
    low : float or -np.inf
        Lower threshold. Default is -np.inf.
    upp : float or np.inf
        Upper threshold. Default is np.inf.

    Returns
    -------
    surf_selected : vtkPolyData or VTKObjectWrapper
        PolyData after selection.

    """

    return _surface_selection(surf, array_name, low=low, upp=upp, use_cell=True,
                              keep=True)


def mask_points(surf, mask):
    """Mask surface points.

    Cells corresponding to these points are also kept.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    mask : ndarray
        Binary boolean array. Zero elements are discarded.

    Returns
    -------
    surf_masked : vtkPolyData or VTKObjectWrapper
        PolyData after masking.

    """

    return _surface_mask(surf, mask)


def mask_cells(surf, mask):
    """Mask surface cells.

    Points corresponding to these cells are also kept.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    mask : ndarray
        Binary boolean array. Zero elements are discarded.

    Returns
    -------
    surf_masked : vtkPolyData or VTKObjectWrapper
        PolyData after masking.

    """

    return _surface_mask(surf, mask, use_cell=True)


@wrap_input(only_args=[0, 1])
def project_pointdata_onto_surface(source_surf, target_surf, source_names, ops,
                                   target_names=None):

    if not isinstance(source_names, list):
        source_names = [source_names]

    if not isinstance(ops, list):
        ops = [ops] * len(source_names)

    if target_names is None:
        target_names = source_names

    cell_centers = aop.compute_cell_center(source_surf)
    cells = me.get_cells(source_surf)

    tree = cKDTree(cell_centers, leafsize=20, compact_nodes=False,
                   copy_data=False, balanced_tree=False)
    _, idx_cell = tree.query(target_surf.Points, k=1, eps=0, n_jobs=1)

    if np.any([op1 in ['weighted_mean', 'weighted_mode'] for op1 in ops]):
        closest_cells = cells[idx_cell]
        dist_to_cell_points = np.sum((target_surf.Points[:, None] -
                                      source_surf.Points[closest_cells])**2,
                                     axis=-1)
        dist_to_cell_points **= .5
        dist_to_cell_points += np.finfo(np.float).eps
        weights = 1 / dist_to_cell_points

    if target_names is None:
        target_names = source_names

    for i, fn in enumerate(source_names):
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

        target_surf.append_array(feat, name=target_names[i], at='p')

    return target_surf
