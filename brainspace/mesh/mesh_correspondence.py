"""
Basic utility functions for vtk.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
from scipy.spatial import cKDTree

from .mesh_elements import get_points
from .array_operations import compute_cell_center


def _find_correspondence(surf, ref_surf, eps=0, n_jobs=1, use_cell=False):

    if use_cell:
        points = compute_cell_center(surf)
        ref_points = compute_cell_center(ref_surf)
    else:
        points = get_points(surf)
        ref_points = get_points(ref_surf)

    tree = cKDTree(ref_points, leafsize=20, compact_nodes=False,
                   copy_data=False, balanced_tree=False)
    d, idx = tree.query(points, k=1, eps=0, n_jobs=n_jobs,
                        distance_upper_bound=eps+np.finfo(np.float).eps)

    if np.isinf(d).any():
        raise ValueError('Cannot find correspondences. Try increasing '
                         'tolerance.')

    return idx


def find_point_correspondence(surf, ref_surf, eps=0, n_jobs=1):
    """For each point in the input surface find its corresponding point
    in the reference surface.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    ref_surf : vtkPolyData or VTKObjectWrapper
        Reference surface.
    eps : non-negative float, optional
        Correspondence tolerance. If ``eps=0``, find exact
        correspondences. Default is 0.
    n_jobs : int, optional
        Number of parallel jobs. Default is 1.

    Returns
    -------
    correspondence : ndarray, shape (n_points,)
        Array of correspondences (indices) with `n_points` elements,
        where `n_points` is the number of points of the input
        surface `surf`. Each entry indexes its corresponding
        point in the reference surface `ref_surf`.

    """

    return _find_correspondence(surf, ref_surf, eps=eps, n_jobs=n_jobs,
                                use_cell=False)


def find_cell_correspondence(surf, ref_surf, eps=0, n_jobs=1):
    """For each cell in the input surface find its corresponding cell
    in the reference surface.

    Parameters
    ----------
    surf : vtkPolyData or VTKObjectWrapper
        Input surface.
    ref_surf : vtkPolyData or VTKObjectWrapper
        Reference surface.
    eps : non-negative float, optional
        Correspondence tolerance. If ``eps=0``, find exact
        correspondences. Default is 0.
    n_jobs : int, optional
        Number of parallel jobs. Default is 1.

    Returns
    -------
    correspondence : ndarray, shape (n_cells,)
        Array of correspondences (indices) with `n_cells` elements,
        where `n_cells` is the number of cells of the input
        surface `surf`. Each entry indexes its corresponding
        cell in the reference surface `ref_surf`.

    """

    return _find_correspondence(surf, ref_surf, eps=eps, n_jobs=n_jobs,
                                use_cell=True)
