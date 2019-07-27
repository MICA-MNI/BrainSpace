"""
VTK basic checks.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np

from vtkmodules.util.vtkConstants import VTK_TRIANGLE, VTK_LINE, VTK_VERTEX
from vtkmodules.vtkCommonDataModelPython import vtkCellTypes


def get_cell_types(surf):
    lid = vtkCellTypes()
    surf.GetCellTypes(lid)
    types = [lid.GetCellType(i) for i in range(lid.GetNumberOfTypes())]
    return np.asarray(types)


def get_number_of_cell_types(surf):
    lid = vtkCellTypes()
    surf.GetCellTypes(lid)
    return lid.GetNumberOfTypes()


def has_unique_cell_type(surf):
    return get_cell_types(surf).size == 1


def has_only_triangle(surf):
    ct = get_cell_types(surf)
    if ct.size != 1:
        return False
    return ct[0] == VTK_TRIANGLE


def has_only_line(surf):
    ct = get_cell_types(surf)
    if ct.size != 1:
        return False
    return ct[0] == VTK_LINE


def has_only_vertex(surf):
    ct = get_cell_types(surf)
    if ct.size != 1:
        return False
    return ct[0] == VTK_VERTEX
