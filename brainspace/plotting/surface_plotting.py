"""
Surface plotting functions.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import itertools
from collections import namedtuple

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

Entry = namedtuple('Entry', ['label', 'orient', 'row', 'col'])


def add_colorbar(ren, lut, is_discrete=False):

    cb = ren.AddScalarBarActor(lookuptable=lut, numberOfLabels=2, height=0.5,
                               position=(0.08, 0.25), width=.8, barRatio=.27,
                               unconstrainedFontSize=True)
    cb.labelTextProperty.setVTK(color=(0, 0, 0), italic=False, shadow=False,
                                bold=True, fontFamily='Arial', fontSize=16)
    return cb


def add_text_actor(ren, name):
    ta = ren.AddTextActor(input=name, textScaleMode='viewport', orientation=90,
                          position=(0.5, 0.5))
    ta.positionCoordinate.coordinateSystem = 'NormalizedViewport'
    ta.textProperty.setVTK(color=(0, 0, 0), italic=False, shadow=False,
                           bold=True, fontFamily='Arial', fontSize=40,
                           verticaljustification='centered',
                           justification='centered')
    return ta


def _compute_range(surfs, layout, array_name, color_range, share=None,
                   nvals=256):

    if share not in [None, 'row', 'r', 'col', 'c', 'both', 'b']:
        raise ValueError("Unknown share=%s" % share)

    # Compute data ranges
    n_vals = np.full_like(layout, nvals, dtype=np.uint)
    min_rg = np.full_like(layout, np.nan, dtype=np.float)
    max_rg = np.full_like(layout, np.nan, dtype=np.float)
    is_discrete = np.zeros_like(layout, dtype=np.bool)

    vals = np.full_like(layout, np.nan, dtype=np.object)
    for i in range(layout.size):
        s = surfs[layout.flat[i]]
        if s is None or array_name.flat[i] not in s.point_keys:
            continue

        x = s.PointData[array_name.flat[i]]
        if not np.issubdtype(x.dtype, np.floating):
            is_discrete.flat[i] = True
            vals.flat[i] = np.unique(x)
            n_vals.flat[i] = vals.flat[i].size

        min_rg.flat[i] = np.nanmin(x)
        max_rg.flat[i] = np.nanmax(x)

    if share and not np.all([a is None for a in array_name.ravel()]):
        # Build lookup tables
        if share in ['both', 'b']:
            min_rg[:] = np.nanmin(min_rg)
            max_rg[:] = np.nanmax(max_rg)

            # Assume everything is discrete
            if is_discrete.all():
                v = [v for v in vals.ravel() if v != np.nan]
                n_vals[:] = np.unique(v).size

        elif share in ['row', 'r']:
            min_rg[:] = np.nanmin(min_rg, axis=1, keepdims=True)
            max_rg[:] = np.nanmax(max_rg, axis=1, keepdims=True)
            is_discrete_row = is_discrete.all(axis=1)
            for i, dr in enumerate(is_discrete_row):
                if dr:
                    v = [v for v in vals[i] if v != np.nan]
                    n_vals[i, :] = np.unique(v).size

        elif share in ['col', 'c']:
            min_rg[:] = np.nanmin(min_rg, axis=0, keepdims=True)
            max_rg[:] = np.nanmax(max_rg, axis=0, keepdims=True)
            is_discrete_col = is_discrete.all(axis=0)
            for i, dc in enumerate(is_discrete_col):
                if dc:
                    v = [v for v in vals[:, i] if v != np.nan]
                    n_vals[i, :] = np.unique(v).size

    return min_rg, max_rg, n_vals, is_discrete


def _gen_entries(idx, shift, n_entries, pos, labs):
    n = len(labs)
    if n_entries % n != 0:
        raise ValueError('Labels not compatible with number of columns')

    st = n_entries // n
    res = [labs, [pos] * n, [idx] * n,
           [(i, i + st) for i in range(shift, shift + n_entries, st)]]

    if pos in ['left', 'right']:
        res[2:] = res[3:1:-1]

    return list(map(lambda x: Entry(*x), zip(*res)))


def _gen_grid(nrow, ncol, lab_text, cbar, share, size_bar=0.11, size_lab=0.05):
    lpos = ['top', 'bottom', 'left', 'right']
    ridx, cidx = list(range(nrow)), list(range(ncol))

    def _extend_index(pos, lab):
        nonlocal cidx, ridx
        if pos in ['left', 'right']:
            cidx.insert(0 if pos == 'left' else len(cidx), lab)
        else:
            ridx.insert(0 if pos == 'top' else len(ridx), lab)

    # Color bar
    if cbar is not None and share:
        _extend_index(cbar, 'cb')

    # Label text
    for pos in lab_text.keys():
        _extend_index(pos, pos)

    # generate grid
    grid = [np.zeros_like(idx, dtype=float) for idx in [ridx, cidx]]
    for i, (idx, g, n) in enumerate(zip([ridx, cidx], grid, [nrow, ncol])):
        np.place(g, np.isin(idx, lpos), size_lab)
        np.place(g, np.isin(idx, ['cb']), size_bar)
        g[g == 0] = (1 - g.sum()) / n
        grid[i] = np.insert(np.cumsum(g), 0, 0)

    # generate entries
    rshift = min([i for i, v in enumerate(ridx) if isinstance(v, int)])
    cshift = min([i for i, v in enumerate(cidx) if isinstance(v, int)])
    entries = []
    for idx, ne, shift in zip([ridx, cidx], [ncol, nrow], [cshift, rshift]):
        for i, k in enumerate(idx):
            if k == 'cb':
                if share in ['both', 'b']:
                    entries += _gen_entries(i, shift, ne, cbar,
                                            [(rshift, cshift)])
                else:
                    entries += _gen_entries(i, shift, ne, cbar,
                                            [(rshift, cshift)] * ne)
            elif isinstance(k, str):
                entries += _gen_entries(i, shift, ne, k, lab_text[k])

    return grid, ridx, cidx, entries


def plot_surf(surfs, layout, array_name=None, view=None, share=None,
              color_bar=False, label_text=None, nan_color=(0, 0, 0, 1),
              cmap='viridis', color=(0, 0, 0.5), size=(400, 400),
              interactive=True, embed_nb=False, color_range=None,
              scale=None, transparent_bg=True, as_mpl=False, screenshot=False,
              filename=None, **kwargs):
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

    # Check color bar
    if color_bar is True:
        color_bar = 'right'
    elif color_bar is False:
        color_bar = None

    if color_bar in ['left', 'right'] and share in ['c', 'col']:
        raise ValueError("Incompatible color_bar=%s and "
                         "share=%s" % (color_bar, share))

    if color_bar in ['top', 'bottom'] and share in ['r', 'row']:
        raise ValueError("Incompatible color_bar=%s and "
                         "share=%s" % (color_bar, share))

    # Check label text
    if label_text is None:
        label_text = {}
    elif isinstance(label_text, (list, np.ndarray)):
        label_text = {'top': label_text}

    if color is None:
        color = (1, 1, 1)

    bg = (1, 1, 1)

    layout = np.atleast_2d(layout)
    array_name = np.broadcast_to(array_name, layout.shape)
    view = np.broadcast_to(view, layout.shape)
    cmap = np.broadcast_to(cmap, layout.shape)

    min_rg, max_rg, n_vals, is_discrete = \
        _compute_range(surfs, layout, array_name, color_range, share=share,
                       nvals=256)

    nrow, ncol = layout.shape
    grid, ridx, cidx, entries = _gen_grid(nrow, ncol, label_text, color_bar,
                                          share)
    grow, gcol = grid
    print(grow)
    print(gcol)
    print(ridx)
    print(cidx)
    kwargs.update({'n_rows': grow, 'n_cols': gcol, 'try_qt': False,
                   'size': size})
    if screenshot or as_mpl:
        kwargs.update({'offscreen': True})
    p = Plotter(**kwargs)

    for irow, icol in itertools.product(range(len(ridx)), range(len(cidx))):
        i, j = ridx[irow], cidx[icol]

        # plot color bar, label_text of white ren
        if isinstance(i, str) or isinstance(j, str):
            if isinstance(i, str) and isinstance(j, str):
                ren1 = p.AddRenderer(row=irow, col=icol, background=bg)
            continue

        ren1 = p.AddRenderer(row=irow, col=icol, background=bg)
        s = surfs[layout[i, j]]
        if s is None:
            continue

        ac1 = ren1.AddActor(color=color, specular=0.1, specularPower=1,
                            diffuse=1, ambient=0.05)
        #
        if view[i, j] is not None:
            ac1.orientation = orientations[view[i, j]]

        # Only interpolate if floating
        interpolate = not is_discrete[i, j]
        m1 = ac1.SetMapper(InputDataObject=s, ColorMode='MapScalars',
                           ScalarMode='UsePointFieldData',
                           InterpolateScalarsBeforeMapping=interpolate,
                           UseLookupTableScalarRange=True)

        if array_name[i, j] is None:
            m1.ScalarVisibility = False
        else:
            m1.ArrayName = array_name[i, j]

        # Set lookuptable
        if cmap[i, j] is not None:
            if cmap[i, j] in colormaps:
                table = colormaps[cmap[i, j]]
            else:
                cm = plt.get_cmap(cmap[i, j])
                table = cm(np.linspace(0, 1, n_vals[i, j])) * 255
                table = table.astype(np.uint8)
            lut1 = m1.SetLookupTable(NumberOfTableValues=n_vals[i, j],
                                     Range=(min_rg[i, j], max_rg[i, j]),
                                     # Range=(-0.5, 2),

                                     Table=table)
            if nan_color is not None:
                lut1.NanColor = nan_color
            lut1.Build()

        ren1.ResetCamera()
        # ren1.GetActiveCamera().Zoom(1.1)
        ren1.GetActiveCamera().Zoom(1.2)

        # Fix conte69:
        # if icol in np.array([0, 3]) + add_text:
        #     ren1.GetActiveCamera().Zoom(1.19)
        # elif icol in np.array([1, 2]) + add_text:
        #     ren1.GetActiveCamera().Zoom(1.1)

    print(p.populated)
    for e in entries:
        ren1 = p.AddRenderer(row=e.row, col=e.col, background=bg)
        if isinstance(e.label, str):
            add_text_actor(ren1, e.label)
        else:  # color bar
            ren_lut = p.renderers[p.populated[e.label]]
            lut = ren_lut.actors.lastActor.mapper.lookupTable
            add_colorbar(ren1, lut.VTKObject)
        print(e)

    if screenshot:
        p.show(interactive=interactive, embed_nb=embed_nb, scale=scale,
               transparent_bg=transparent_bg, as_mpl=as_mpl)
        return p.screenshot(filename=filename, scale=scale,
                            transparent_bg=transparent_bg)
    return p.show(interactive=interactive, embed_nb=embed_nb, scale=scale,
                  transparent_bg=transparent_bg, as_mpl=as_mpl)


@wrap_input(0, 1)
def plot_hemispheres(surf_lh, surf_rh, array_name=None, color_bar=False,
                     label_text=None, cmap='viridis', color=(0, 0, 0.5),
                     nan_color=(0, 0, 0, 1), size=(800, 150), interactive=True,
                     embed_nb=False,
                     scale=None, transparent_bg=True, as_mpl=False,
                     screenshot=False, filename=None,
                     **kwargs):
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
                     interactive=interactive, embed_nb=embed_nb,
                     scale=scale,
                     transparent_bg=transparent_bg, as_mpl=as_mpl,
                     filename=filename, screenshot=screenshot,
                     **kwargs)
