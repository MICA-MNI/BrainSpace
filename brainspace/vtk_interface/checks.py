"""
VTK basic checks.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np

from vtk.util.vtkConstants import (VTK_VERTEX, VTK_LINE, VTK_TRIANGLE,
                                   VTK_QUAD)
from vtk import vtkCellTypes


def get_cell_types(surf):
    """Get cell types of `surf`.

    Parameters
    ----------
    surf : BSDataSet
        Input data.

    Returns
    -------
    cell_types : 1D ndarray
        Array of cell types.
    """
    lid = vtkCellTypes()
    surf.GetCellTypes(lid)
    types = [lid.GetCellType(i) for i in range(lid.GetNumberOfTypes())]
    return np.asarray(types)


def get_number_of_cell_types(surf):
    """Get number of cell types of `surf`.

    Parameters
    ----------
    surf : BSDataSet
        Input data.

    Returns
    -------
    int
        Number of cell types.
    """
    lid = vtkCellTypes()
    surf.GetCellTypes(lid)
    return lid.GetNumberOfTypes()


def has_unique_cell_type(surf):
    """Check if `surf` has a unique cell type.

    Parameters
    ----------
    surf : BSDataSet
        Input data.

    Returns
    -------
    bool
        True if `surf` has a unique cell type. False, otherwise.
    """
    return get_number_of_cell_types(surf) == 1


def has_only_triangle(surf):
    """Check if `surf` has only triangles.

    Parameters
    ----------
    surf : BSDataSet
        Input data.

    Returns
    -------
    bool
        True if `surf` has only triangles. False, otherwise.
    """
    ct = get_cell_types(surf)
    if ct.size != 1:
        return False
    return ct[0] == VTK_TRIANGLE


def has_only_quad(surf):
    """Check if `surf` has only quads.

    Parameters
    ----------
    surf : BSDataSet
        Input data.

    Returns
    -------
    bool
        True if `surf` has only quads. False, otherwise.
    """
    ct = get_cell_types(surf)
    if ct.size != 1:
        return False
    return ct[0] == VTK_QUAD


def has_only_line(surf):
    """Check if `surf` has only lines.

    Parameters
    ----------
    surf : BSDataSet
        Input data.

    Returns
    -------
    bool
        True if `surf` has only lines. False, otherwise.
    """
    ct = get_cell_types(surf)
    if ct.size != 1:
        return False
    return ct[0] == VTK_LINE


def has_only_vertex(surf):
    """Check if `surf` has only vertex cells.

    Parameters
    ----------
    surf : BSDataSet
        Input data.

    Returns
    -------
    bool
        True if `surf` has only vertex cells. False, otherwise.
    """
    ct = get_cell_types(surf)
    if ct.size != 1:
        return False
    return ct[0] == VTK_VERTEX
