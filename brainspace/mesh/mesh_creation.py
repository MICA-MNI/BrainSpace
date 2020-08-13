"""
Functions for surface creation.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from vtk import (vtkPolyData, vtkCellArray, vtkTriangleFilter,
                 vtkVertexGlyphFilter)

from .mesh_elements import get_edges
from ..vtk_interface.wrappers import BSPolyData
from ..vtk_interface.decorators import wrap_input
from ..vtk_interface.pipeline import serial_connect


def build_polydata(points, cells=None):
    """Build surface (PolyData) from points and cells.

    Parameters
    ----------
    points : ndarray, shape = (n_points, 3)
        Array of points.
    cells : ndarray, shape = (n_cells, nd), optional
        Array of cells. Cells can be vertex (nd=1), line (nd=2) or
        triangle (nd=3). Default is None (no topology information).

    Returns
    -------
    output : BSPolyData
        Returns surface (PolyData).

    See Also
    --------
    :func:`to_vertex`
    :func:`to_lines`

    Notes
    -----
    Point ids within cells must start from 0 (first point) and contain all
    points.

    """

    s = BSPolyData(points=points)
    if cells is not None:
        n_cells, n_points_cell = cells.shape
        if n_points_cell == 1:
            s.SetVerts(cells)
        elif n_points_cell == 2:
            s.SetLines(cells)
        else:
            s.SetPolys(cells)

    # Triangulation needed
    return serial_connect(s, vtkTriangleFilter())


def to_vertex(surf):
    """Convert all cells in PolyData to vertex cells.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.

    Returns
    -------
    output : BSPolyData
        PolyData with vertex points.

    See Also
    --------
    :func:`to_lines`
    :func:`build_polydata`

    """

    return serial_connect(surf, vtkVertexGlyphFilter())


@wrap_input(0)
def to_lines(surf):
    """Convert all cells in PolyData to lines.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.

    Returns
    -------
    output : BSPolyData
        PolyData with lines.

    See Also
    --------
    :func:`to_vertex`
    :func:`build_polydata`
    """

    edges = get_edges(surf)
    s = build_polydata(surf.Points, cells=edges)

    for k in surf.point_keys:
        s.append_array(surf.PointData[k], name=k, at='p')

    return s

