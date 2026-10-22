"""Contour overlays for VTK surface plots."""

# License: BSD 3 clause

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def _contour_polydata(surf, values):
    poly = vtk.vtkPolyData()
    poly.DeepCopy(getattr(surf, 'VTKObject', surf))
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != poly.GetNumberOfPoints():
        raise ValueError('Contour must have one value per surface vertex.')
    if not np.isfinite(values).all():
        raise ValueError('Contour values must be finite.')
    scalars = numpy_to_vtk(values, deep=True)
    scalars.SetName('_brainspace_contour')
    poly.GetPointData().SetScalars(scalars)
    return poly


def _check_style(color, alpha, line_width=1, isovalue=0.5):
    color = np.asarray(color, dtype=float)
    if color.shape != (3,) or not np.isfinite(color).all() or \
            np.any((color < 0) | (color > 1)):
        raise ValueError('Contour colors must be RGB triples in [0, 1].')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Contour opacity must be in [0, 1].')
    if not np.isfinite(line_width) or line_width <= 0:
        raise ValueError('Contour line width must be positive.')
    if not np.isfinite(isovalue):
        raise ValueError('Contour isovalue must be finite.')


def _make_actor(poly, color, alpha, isovalue, line_width=None):
    if line_width is None:
        filt = vtk.vtkClipPolyData()
    else:
        filt = vtk.vtkContourFilter()
    filt.SetInputData(poly)
    if line_width is None:
        filt.SetValue(isovalue)
    else:
        filt.SetValue(0, isovalue)
    filt.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(filt.GetOutputPort())
    mapper.ScalarVisibilityOff()
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    if line_width is None:
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
    else:
        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-2, -2)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(alpha)
    if line_width is not None:
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().LightingOff()
    return actor


def make_contour_actor(surf, contour, color=(0, 0, 0), line_width=3,
                       alpha=1.0, isovalue=0.5):
    """Create a contour line actor without changing the source surface.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Surface to contour.
    contour : array-like, shape = (n_points,)
        Finite scalar values, one per vertex.
    color : tuple, optional
        Line color as an RGB triple in [0, 1].
    line_width : float, optional
        Positive line width in pixels. Default is 3.
    alpha : float, optional
        Line opacity in [0, 1]. Default is 1.
    isovalue : float, optional
        Scalar level at which to draw the contour. Default is 0.5.

    Returns
    -------
    vtkActor
        Actor in the input surface's coordinates.
    """
    _check_style(color, alpha, line_width, isovalue)
    return _make_actor(_contour_polydata(surf, contour), color, alpha,
                       isovalue, line_width)


def add_contours_to_plotter(plotter, surf_lh=None, surf_rh=None,
                            contour_map=None, lwd=3, lcol=(0, 0, 0),
                            lalpha=1.0, col=(0, 0, 0), alpha=0,
                            isovalue=0.5):
    """Add contour overlays to the surface entries of a built plotter.

    Parameters
    ----------
    plotter : Plotter
        Plotter returned by ``build_plotter`` or ``plot_hemispheres``.
    surf_lh, surf_rh : BSPolyData, optional
        Left and right surfaces when passing ``contour_map`` explicitly.
        Alternatively, pass contour data as the second argument alone.
    contour_map : array-like, str or dict, optional
        Combined left/right vertex values, a point-data array name, or a
        dictionary mapping surface keys to per-vertex values or array names.
    lwd : float, optional
        Positive contour line width in pixels. Default is 3.
    lcol, col : tuple, optional
        RGB line and fill colors in [0, 1]. Default is black.
    lalpha, alpha : float, optional
        Line and fill opacity. Defaults are 1 and 0 (no fill).
    isovalue : float, optional
        Contour level; fill covers values above this level. Default is 0.5.

    Returns
    -------
    list of vtkActor
        Added actors, including optional fill actors. Their transforms match
        the base surface in every view, including custom actor transforms.
    """
    _check_style(lcol, lalpha, lwd, isovalue)
    _check_style(col, alpha, isovalue=isovalue)
    entries = getattr(plotter, '_surface_entries', None)
    if entries is None:
        raise ValueError('Contour overlays require a plotter from build_plotter.')
    if contour_map is None and surf_rh is None:
        contour_map = surf_lh
    if contour_map is None:
        raise ValueError('Contour values are required.')
    surfaces = {key: surf for key, surf, _, _ in entries}
    if not isinstance(contour_map, (str, dict)):
        if set(surfaces) != {'lh', 'rh'}:
            raise ValueError('Use a contour dictionary for non-hemisphere layouts.')
        n_lh = surfaces['lh'].n_points
        n_rh = surfaces['rh'].n_points
        values = np.asarray(contour_map)
        if values.ndim != 1 or values.size != n_lh + n_rh:
            raise ValueError('Contour must have n_lh + n_rh vertex values.')
        contour_map = {'lh': values[:n_lh], 'rh': values[n_lh:]}
    # Validate and copy all inputs before adding any actors.
    polys = {}
    for key, surf in surfaces.items():
        values = contour_map if isinstance(contour_map, str) else contour_map.get(key)
        if values is None:
            continue
        if isinstance(values, str):
            values = surf.PointData[values]
        polys[key] = _contour_polydata(surf, values)
    actors = []
    for key, _, renderer, base_actor in entries:
        if key not in polys:
            continue
        new_actors = []
        if alpha > 0:
            new_actors.append(_make_actor(polys[key], col, alpha, isovalue))
        new_actors.append(_make_actor(polys[key], lcol, lalpha, isovalue, lwd))
        for actor in new_actors:
            transform = vtk.vtkMatrix4x4()
            transform.DeepCopy(base_actor.VTKObject.GetMatrix())
            actor.SetUserMatrix(transform)
            renderer.AddActor(actor)
            actors.append(actor)
    return actors
