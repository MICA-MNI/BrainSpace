"""
Surface plotting functions.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause

import matplotlib.pyplot as plt

import numpy as np

import vtk

from .base import Plotter
from .colormaps import colormaps

from ..vtk_interface import wrap_vtk
from ..vtk_interface.decorators import wrap_input


orientations = {'lateral': (0, -90, -90),
                'medial': (0, 90, 90),
                'ventral': (0, 180, 0),
                'dorsal': (0, 0, 0)}


def get_colorbar(lut, is_discrete=False):

    fmt = '%-#6.3g' if is_discrete else '%-#6.2e'

    cb = wrap_vtk(vtk.vtkScalarBarActor, lookuptable=lut, numberOfLabels=2,
                  height=0.5, position=(0.08, 0.25), width=.8, barRatio=.27,
                  # labelFormat=fmt,
                  UnconstrainedFontSize=True)

    tp = wrap_vtk(cb.labelTextProperty)
    tp.setVTK(color=(0, 0, 0), italic=False, shadow=False,
              bold=True, fontFamily='Arial', fontSize=16)

    return cb


def get_actor_text(name):
    ta = wrap_vtk(vtk.vtkTextActor)
    ta.positionCoordinate.SetCoordinateSystemToNormalizedViewport()

    ta.setVTK(input=name, textScaleMode='viewport', orientation=90,
              position=(0.5, 0.5))
    tp = wrap_vtk(ta.textProperty)
    tp.setVTK(color=(0, 0, 0), italic=False, shadow=False,
              bold=True, fontFamily='Arial', fontSize=40,
              Verticaljustification='centered', justification='centered')
    return ta.VTKObject


def plot_surf(surfs, layout, array_name=None, view=None, share=None,
              color_bar=False, label_text=None, nan_color=(0, 0, 0, 1),
              cmap='viridis', color=(0, 0, 0.5), size=(400, 400),
              interactive=True, embed_nb=False, **kwargs):
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
    color_bar : bool, optional
        Plot color bar for each array (row). Default is False.
    label_text : list of str, optional
        Label text for each array (row). Default is None.
    nan_color : tuple
        Color for nan values. Default is (0, 0, 0, 1).
    cmap : str, optional
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
    kwargs : keyword-valued args
            Additional arguments passed to the plotter.

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
    cmap = np.broadcast_to(cmap, layout.shape)

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

    if not np.all([a is None for a in array_name.flat]):
        # Build lookup tables
        if share in ['both', 'b']:
            min_rg[:] = np.nanmin(min_rg)
            max_rg[:] = np.nanmax(max_rg)

            # Assume everything is discrete
            if is_discrete.all():
                v = [v for v in vals.ravel() if v is not None]
                n_vals[:] = np.unique(v).size

        elif share in ['row', 'r']:
            min_rg[:] = np.nanmin(min_rg, axis=1, keepdims=True)
            max_rg[:] = np.nanmax(max_rg, axis=1, keepdims=True)
            is_discrete_row = is_discrete.all(axis=1)
            for i, dr in enumerate(is_discrete_row):
                if dr:
                    v = [v for v in vals[i] if v is not None]
                    n_vals[i, :] = np.unique(v).size

        elif share in ['col', 'c']:
            min_rg[:] = np.nanmin(min_rg, axis=0, keepdims=True)
            max_rg[:] = np.nanmax(max_rg, axis=0, keepdims=True)
            is_discrete_col = is_discrete.all(axis=0)
            for i, dc in enumerate(is_discrete_col):
                if dc:
                    v = [v for v in vals[:, i] if v is not None]
                    n_vals[i, :] = np.unique(v).size

    add_text = label_text is not None
    grow, gcol = nrow, ncol
    if color_bar or add_text:
        pad0 = 0.05 if add_text else 0
        pad1 = 0.11 if color_bar else 0

        if share in ['c', 'col']:
            ly = 1 - (pad0 + pad1)
            dy = ly / nrow
            grow = np.arange(pad0, ly, dy)
            if color_bar:
                grow = np.concatenate([grow, grow[-1:] + dy])
            if pad0 > 0:
                grow = np.concatenate([[0], grow])
            grow = np.concatenate([grow, [1]])
        elif share in ['r', 'row']:
            lx = 1 - (pad0 + pad1)
            dx = lx / ncol
            gcol = np.arange(pad0, lx, dx)
            if color_bar:
                gcol = np.concatenate([gcol, gcol[-1:]+dx])
            if pad0 > 0:
                gcol = np.concatenate([[0], gcol])
            gcol = np.concatenate([gcol, [1]])

    kwargs.update({'n_rows': grow, 'n_cols': gcol, 'try_qt': False,
                   'size': size})
    p = Plotter(**kwargs)

    bg = (1, 1, 1)
    for k in range(layout.size):
        irow, icol = k // ncol, k % ncol
        if add_text:
            icol += 1

        ren1 = p.AddRenderer(row=irow, col=icol, background=bg)
        s = surfs[layout.flat[k]]
        if s is None:
            continue

        ac1 = ren1.AddActor(color=color, specular=0.1, specularPower=1,
                            diffuse=1, ambient=0.05)

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
        if cmap.flat[k] is not None:
            if cmap.flat[k] in colormaps:
                table = colormaps[cmap.flat[k]]
            else:
                cm = plt.get_cmap(cmap.flat[k])
                table = cm(np.linspace(0, 1, n_vals.flat[k])) * 255
                table = table.astype(np.uint8)
            lut1 = m1.SetLookupTable(NumberOfTableValues=n_vals.flat[k],
                                     Range=(min_rg.flat[k], max_rg.flat[k]),
                                     Table=table)
            if nan_color is not None:
                lut1.NanColor = nan_color
            lut1.Build()

        if share in ['r', 'row'] and icol == ncol - 1:
            pad = 0
            if add_text:
                ren2 = p.AddRenderer(row=irow, col=0, background=bg)
                ren2.AddActor2D(get_actor_text(label_text[irow]))
                pad = 1
            if color_bar:
                ren2 = p.AddRenderer(row=irow, col=ncol + pad, background=bg)
                cb = get_colorbar(m1.lookupTable.VTKObject,
                                  is_discrete=is_discrete.flat[k])
                ren2.AddActor2D(cb.VTKObject)

        if share in ['c', 'col'] and irow == nrow - 1:
            if label_text is not None or color_bar:
                raise NotImplementedError

        ren1.ResetCamera()
        # ren1.GetActiveCamera().Zoom(1.1)
        ren1.GetActiveCamera().Zoom(1.2)

        # Fix conte69:
        # if icol in np.array([0, 3]) + add_text:
        #     ren1.GetActiveCamera().Zoom(1.19)
        # elif icol in np.array([1, 2]) + add_text:
        #     ren1.GetActiveCamera().Zoom(1.1)

    return p.show(interactive=interactive, embed_nb=embed_nb)


@wrap_input(0, 1)
def plot_hemispheres(surf_lh, surf_rh, array_name=None, color_bar=False,
                     label_text=None, cmap='viridis', color=(0, 0, 0.5),
                     nan_color=(0, 0, 0, 1), size=(800, 150), interactive=True,
                     embed_nb=False, **kwargs):
    """Plot left and right hemispheres in lateral and medial views.

    Parameters
    ----------
    surf_lh : vtkPolyData or BSPolyData
        Left hemisphere.
    surf_rh : vtkPolyData or BSPolyData
        Right hemisphere.
    array_name : str, list of str, ndarray or list of ndarray, optional
        Name of point data array to plot. If ndarray, the array is split for
        the left and right hemispheres. If list, plot one row per array.
        If None, defaults to 'color'. Default is None.
    color_bar : bool, optional
        Plot color bar for each array (row). Default is False.
    label_text : list of str, optional
        Label text for each array (row). Default is None.
    nan_color : tuple
        Color for nan values. Default is (0, 0, 0, 1).
    cmap : str, optional
        Color map name (from matplotlib). Default is 'viridis'.
    color : tuple
        Default color if `array_name` is not provided. Default is (0, 0, 0.5).
    size : tuple, optional
        Window size. Default is (800, 200).
    interactive : bool, optional
        Whether to enable interaction. Default is True.
    embed_nb : bool, optional
        Whether to embed figure in notebook. Only used if running in a
        notebook. Default is False.
    kwargs : keyword-valued args
        Additional arguments passed to the plotter.


    Returns
    -------
    figure : Ipython Image or None
        Figure to plot. None if using vtk for rendering (i.e.,
        ``embed_nb == False``).

    See Also
    --------
    :func:`plot_surf`

    """

    surfs = {'lh': surf_lh, 'rh': surf_rh}
    layout = ['lh', 'lh', 'rh', 'rh']
    view = ['medial', 'lateral', 'medial', 'lateral']

    if isinstance(array_name, np.ndarray) and array_name.ndim == 2:
        array_name = [a for a in array_name]

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
        array_name2 = surf_lh.append_array(array_name[:n_pts_lh], at='p')
        surf_rh.append_array(array_name[n_pts_lh:], name=array_name2, at='p')
        array_name = array_name2

    if isinstance(cmap, list):
        cmap = np.asarray(cmap)[:, None]

    return plot_surf(surfs, layout, array_name=array_name, nan_color=nan_color,
                     view=view, cmap=cmap,  color_bar=color_bar,
                     label_text=label_text, color=color, size=size, share='r',
                     interactive=interactive, embed_nb=embed_nb, **kwargs)
