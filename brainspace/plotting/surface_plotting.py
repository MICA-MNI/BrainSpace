"""
Surface plotting functions.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt

from vtkmodules.vtkFiltersGeneralPython import vtkTransformFilter
from vtkmodules.vtkCommonTransformsPython import vtkTransform

from .base import Plotter
from .colormaps import colormaps

from ..vtk_interface.wrappers import wrap_vtk
from ..vtk_interface.decorators import wrap_input
from ..vtk_interface.pipeline import serial_connect


orientations = {'lateral': (0, -90, -90),
                'medial': (0, 90, 90),
                'ventral': (0, 180, 0),
                'dorsal': (0, 0, 0)}


def plot_surf(surfs, layout, array_name=None, view=None, share=None,
              nan_color=(0, 0, 0, 1), cmap_name='viridis', color=(0, 0, 0.5),
              size=(400, 400), interactive=True, embed_nb=False):
    """Plot surfaces arranged according to the `layout`.

    Parameters
    ----------
    surfs : dict[str, BSPolyData]
        Dictionary of surfaces.
    layout : ndarray, shape = (n_rows, n_cols)
        Array of surface keys in `surfs`. Specifies how window is arranged.
    array_name : ndarray, optional
        Names of point data array to plot for each layout entry.
        Default is None.
    view : ndarray, optional
        View for each each layout entry. Possible views are {'lateral',
        'medial', 'ventral', 'dorsal'}. If None, use default view.
        Default is None.
    share : {'row', 'col', 'both'} or None, optional
        If ``share == 'row'``, point data for surfaces in the same row share
        same data range. If ``share == 'col'``, the same but for columns.
        If ``share == 'both'``, all data shares same range. Default is None.
    nan_color : tuple
        Color for nan values. Default is (0, 0, 0, 1).
    cmap_name : str, optional
        Color map name (from matplotlib). Default is 'viridis'.
    color : tuple
        Default color if `array_name` is not provided. Default is (0, 0, 0.5).
    size : tuple, optional
        Window size. Default is (400, 400).
    interactive : bool, optional
        Whether to enable interaction. Default is True.
    embed_nb : bool, optional
        Whether to embed figure in notebook. Only used if running in a
        notebook. Default is False.

    Returns
    -------
    figure : Ipython Image or panel or None
        Figure to plot. None if using vtk for rendering (i.e.,
        ``embed_nb == False``).

    See Also
    --------
    :func:`plot_hemispheres`

    Notes
    -----
    Shapes of `array_name` and `view` must be the equal or broadcastable to
    the shape of `layout`.
    """

    layout = np.atleast_2d(layout)
    array_name = np.broadcast_to(array_name, layout.shape)
    view = np.broadcast_to(view, layout.shape)
    cmap_name = np.broadcast_to(cmap_name, layout.shape)

    if color is None:
        color = (1, 1, 1)

    nrow, ncol = layout.shape

    # Compute data ranges
    n_vals = np.full_like(layout, 256, dtype=np.uint)
    min_rg = np.full_like(layout, np.nan, dtype=np.float)
    max_rg = np.full_like(layout, np.nan, dtype=np.float)
    is_discrete = np.zeros_like(layout, dtype=np.bool)
    vals = np.full_like(layout, None, dtype=np.object)
    for i in range(layout.size):
        s = surfs[layout.flat[i]]
        if s is None:
            continue

        if array_name.flat[i] not in s.PointData.keys():
            continue
        array = s.PointData[array_name.flat[i]]

        if not np.issubdtype(array.dtype, np.floating):
            is_discrete.flat[i] = True
            vals.flat[i] = np.unique(array)
            n_vals.flat[i] = vals.flat[i].size

        min_rg.flat[i] = np.nanmin(array)
        max_rg.flat[i] = np.nanmax(array)

    # Build lookup tables
    if share in ['both', 'b']:
        min_rg[:] = np.nanmin(min_rg)
        max_rg[:] = np.nanmax(max_rg)

        # Assume everything is discrete
        if is_discrete.all():
            v = np.concatenate([v for v in vals.ravel() if v is not None])
            # print(np.concatenate(vals.ravel()))
            print(np.unique(v))
            print(n_vals.shape)
            n_vals[:] = np.unique(v).size

    elif share in ['row', 'r']:
        min_rg[:] = np.nanmin(min_rg, axis=1, keepdims=True)
        max_rg[:] = np.nanmax(max_rg, axis=1, keepdims=True)
        is_discrete_row = is_discrete.all(axis=1)
        for i, dr in enumerate(is_discrete_row):
            if dr:
                v = np.concatenate([v for v in vals[i] if v is not None])
                n_vals[i, :] = np.unique(v).size

    elif share in ['col', 'c']:
        min_rg[:] = np.nanmin(min_rg, axis=0, keepdims=True)
        max_rg[:] = np.nanmax(max_rg, axis=0, keepdims=True)
        is_discrete_col = is_discrete.all(axis=0)
        for i, dc in enumerate(is_discrete_col):
            if dc:
                v = np.concatenate([v for v in vals[:, i] if v is not None])
                n_vals[i, :] = np.unique(v).size

    p = Plotter(n_rows=nrow, n_cols=ncol, try_qt=False, size=size)
    for k in range(layout.size):
        ren1 = p.AddRenderer(row=k // ncol, col=k % ncol, background=(1, 1, 1))
        s = surfs[layout.flat[k]]
        if s is None:
            continue

        ac1 = ren1.AddActor(color=color)
        if view.flat[k] is not None:
            ac1.orientation = orientations[view.flat[k]]

        # Only interpolate if floating
        interpolate = not is_discrete.flat[k]
        m1 = ac1.SetMapper(InputDataObject=s, ColorMode='MapScalars',
                           ScalarMode='UsePointFieldData',
                           InterpolateScalarsBeforeMapping=interpolate,
                           UseLookupTableScalarRange=True)

        if array_name.flat[k] is None:
            m1.ScalarVisibility = False
        else:
            m1.ArrayName = array_name.flat[k]

        # Set lookuptable
        if cmap_name.flat[k] is not None:
            if cmap_name.flat[k] in colormaps:
                table = colormaps[cmap_name.flat[k]]
            else:
                cmap = plt.get_cmap(cmap_name.flat[k])
                table = cmap(np.linspace(0, 1, n_vals.flat[k])) * 255
                table = table.astype(np.uint8)
            lut1 = m1.SetLookupTable(NumberOfTableValues=n_vals.flat[k],
                                     Range=(min_rg.flat[k], max_rg.flat[k]),
                                     Table=table)
            if nan_color is not None:
                lut1.NanColor = nan_color
            lut1.Build()

        ren1.ResetCamera()
        ren1.GetActiveCamera().Zoom(1.2)

    return p.show(interactive=interactive, embed_nb=embed_nb)


@wrap_input(only_args=[0, 1])
def plot_hemispheres(surf_lh, surf_rh, array_name=None, nan_color=(0, 0, 0, 1),
                     cmap_name='viridis', color=(0, 0, 0.5), size=(400, 400),
                     interactive=True, embed_nb=False):
    """Plot left and right hemispheres in lateral and medial views.

    Parameters
    ----------
    surf_lh : vtkPolyData or BSPolyData
        Left hemisphere.
    surf_rh : vtkPolyData or BSPolyData
        Right hemisphere.
    array_name : str or list of str, optional
        Name of point data array to plot. Default is None.
    nan_color : tuple
        Color for nan values. Default is (0, 0, 0, 1).
    cmap_name : str, optional
        Color map name (from matplotlib). Default is 'viridis'.
    color : tuple
        Default color if `array_name` is not provided. Default is (0, 0, 0.5).
    size : tuple, optional
        Window size. Default is (400, 400).
    interactive : bool, optional
        Whether to enable interaction. Default is True.
    embed_nb : bool, optional
        Whether to embed figure in notebook. Only used if running in a
        notebook. Default is False.


    Returns
    -------
    figure : Ipython Image or panel or None
        Figure to plot. None if using vtk for rendering (i.e.,
        ``embed_nb == False``).

    See Also
    --------
    :func:`plot_surf`

    """

    surfs = {'lh': surf_lh, 'rh': surf_rh}
    layout = ['lh', 'lh', 'rh', 'rh']
    view = ['medial', 'lateral', 'medial', 'lateral']

    if isinstance(array_name, list):
        layout = [layout] * len(array_name)
        array_name2 = []
        n_pts_lh = surf_lh.n_points
        for an in array_name:
            if isinstance(an, np.ndarray):
                name = surf_lh.append_array(an[:n_pts_lh], at='p')
                surf_rh.append_array(an[n_pts_lh:], name=name, at='p')
                array_name2.append(name)
            else:
                array_name2.append(an)
        array_name = np.asarray(array_name2)[:, None]
    elif isinstance(array_name, np.ndarray):
        n_pts_lh = surf_lh.n_points
        array_name = surf_lh.append_array(array_name[:n_pts_lh], at='p')
        surf_rh.append_array(array_name[n_pts_lh:], name=array_name, at='p')

    # print(array_name, 2)

    return plot_surf(surfs, layout, array_name=array_name, nan_color=nan_color,
                     view=view, cmap_name=cmap_name, color=color, size=size,
                     interactive=interactive, embed_nb=embed_nb)
