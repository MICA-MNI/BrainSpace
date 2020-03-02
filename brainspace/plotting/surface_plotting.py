"""
Surface plotting functions.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from itertools import product as iter_prod

import matplotlib.pyplot as plt
import numpy as np

from .base import Plotter
from .colormaps import colormaps
from . import defaults_plotting as dp
from .utils import _broadcast, _expand_arg, _grep_args, _gen_grid, _get_ranges

from ..vtk_interface.decorators import wrap_input


orientations = {'medial': (0, -90, -90),
                'lateral': (0, 90, 90),
                'ventral': (0, 180, 0),
                'dorsal': (0, 0, 0)}


def _add_colorbar(ren, lut, location, **cb_kwds):

    kwds = dp.scalarBarActor_kwds.copy()
    kwds = {k.lower(): v for k, v in kwds.items()}

    orientation = 'vertical'
    if location in {'top', 'bottom'}:
        orientation = 'horizontal'
        kwds['width'], kwds['height'] = kwds['height'], kwds['width']

    if lut.GetIndexedLookup():
        if location == 'left':
            kwds['position'] = (.32, 0.25)
        elif location == 'right':
            kwds['position'] = (-.32, 0.25)
        elif location == 'bottom':
            kwds['position'] = (0.25, 0.73)
        else:
            kwds['position'] = (0.25, -.43)
    elif location in {'top', 'bottom'}:
        kwds['position'] = kwds['position'][::-1]

    text_pos = 'precedeScalarBar'
    if lut.GetIndexedLookup():
        if location in {'left', 'bottom'}:
            text_pos = 'succeedScalarBar'
    elif location in {'right', 'top'}:
        text_pos = 'succeedScalarBar'

    for k, v in cb_kwds.items():
        if isinstance(kwds.get(k, None), dict):
            kwds[k].update(v)
        else:
            kwds[k] = v

    kwds.update({'lookuptable': lut, 'orientation': orientation,
                 'textPosition': text_pos})

    return ren.AddScalarBarActor(**kwds)


def _add_text(ren, text, location, **lt_kwds):
    orientation = 0
    if location == 'left':
        orientation = 90
    elif location == 'right':
        orientation = -90

    kwds = dp.textActor_kwds.copy()
    kwds = {k.lower(): v for k, v in kwds.items()}
    for k, v in lt_kwds.items():
        if isinstance(kwds.get(k, None), dict):
            kwds[k].update(v)
        else:
            kwds[k] = v

    kwds.update({'input': text, 'orientation': orientation})
    return ren.AddTextActor(**kwds)


def build_plotter(surfs, layout, array_name=None, view=None, color_bar=None,
                  color_range=None, share=False, label_text=None,
                  cmap='viridis', nan_color=(0, 0, 0, 1), zoom=1,
                  background=(1, 1, 1), size=(400, 400), **kwargs):
    """Build plotter arranged according to the `layout`.

    Parameters
    ----------
    surfs : dict[str, BSPolyData]
        Dictionary of surfaces.
    layout : array-like, shape = (n_rows, n_cols)
        Array of surface keys in `surfs`. Specifies how window is arranged.
    array_name : array-like, optional
        Names of point data array to plot for each layout entry.
        Use a tuple with multiple array names to plot multiple arrays
        (overlays) per layout entry. Default is None.
    view : array-like, optional
        View for each each layout entry. Possible views are {'lateral',
        'medial', 'ventral', 'dorsal'}. If None, use default view.
        Default is None.
    color_bar : {'left', 'right', 'top', 'bottom'} or None, optional
        Location where color bars are rendered. If None, color bars are not
        included. Default is None.
    color_range : {'sym'}, tuple or sequence.
        Range for each array name. If 'sym', uses a symmetric range. Only used
        if array has positive and negative values. Default is None.
    share : {'row', 'col', 'both'} or bool, optional
        If ``share == 'row'``, point data for surfaces in the same row share
        same data range. If ``share == 'col'``, the same but for columns.
        If ``share == 'both'``, all data shares same range. If True, similar
        to ``share == 'both'``. Default is False.
    label_text : dict[str, array-like], optional
        Label text for column/row. Possible keys are {'left', 'right',
        'top', 'bottom'}, which indicate the location. Default is None.
    cmap : str or sequence of str, optional
        Color map name (from matplotlib) for each array name.
        Default is 'viridis'.
    nan_color : tuple
        Color for nan values. Default is (0, 0, 0, 1).
    zoom : float or sequence of float, optional
        Zoom applied to the surfaces in each layout entry.
    background : tuple
        Background color. Default is (1, 1, 1).
    size : tuple, optional
        Window size. Default is (400, 400).
    kwargs : keyword-valued args
        Additional arguments passed to the renderers, actors, mapper, color_bar
        or plotter. Keywords starting with:

        - 'renderer__' are passed to the renderers.
        - 'actor__' are passed to the actors.
        - 'mapper__' are passed to the mappers.
        - 'cb__' are passed to color bar actors.
        - 'text__' are passed to color text actors.

        The rest of keywords are passed to the plotter.

    Returns
    -------
    plotter : Plotter
        An instance of Plotter.

    See Also
    --------
    :func:`plot_surf`
    :func:`plot_hemispheres`

    Notes
    -----
    If sequences, shapes of `array_name`, `view` and `zoom` must be equal
    or broadcastable to the shape of `layout`. Renderer keywords must also
    be broadcastable to the shape of `layout`.

    If sequences, shapes of `cmap` and `cbar_range` must be equal or
    broadcastable to the shape of `array_name`, including the number of array
    names per entry. Actor and mapper keywords must also be broadcastable to
    the shape of `array_name`.

    """

    # Layout
    for k in np.unique(layout):
        if k not in surfs and k is not None:
            raise ValueError("Key '%s' is not in 'surfs'" % k)

    # Share
    if share is True:
        share = 'b'
    elif share is None or share is False:
        share = None
    elif share in {'row', 'r', 'col', 'c', 'both', 'b'}:
        share = share[0]
    else:
        raise ValueError("Unknown share=%s" % share)

    # Color bar
    if color_bar is True:
        color_bar = 'right'
    elif color_bar is None or color_bar is False:
        color_bar = None
    elif color_bar not in {'left', 'right', 'top', 'bottom'}:
        raise ValueError("Unknown color_bar=%s" % color_bar)

    if share == 'c' and color_bar in {'left', 'right'}:
        raise ValueError("Incompatible color_bar=%s and "
                         "share=%s" % (color_bar, share))

    if share == 'r' and color_bar in {'top', 'bottom'}:
        raise ValueError("Incompatible color_bar=%s and "
                         "share=%s" % (color_bar, share))

    layout = np.atleast_2d(layout)
    nrow, ncol = shape = layout.shape

    view = _broadcast(view, 'view', shape)
    zoom = _broadcast(zoom, 'zoom', shape)

    array_name = _expand_arg(array_name, 'array_name', shape)
    cmap = _expand_arg(cmap, 'cmap', shape, ref=array_name)
    color_range = _expand_arg(color_range, 'cbar_range', shape, ref=array_name)

    ren_kwds = _grep_args('renderer', kwargs, shape=shape)
    actor_kwds = _grep_args('actor', kwargs, shape=shape, ref=array_name)
    mapper_kwds = _grep_args('mapper', kwargs, shape=shape, ref=array_name)
    cb_kwds = _grep_args('cb', kwargs)
    text_kwds = _grep_args('text', kwargs)
    # lut_kwds = _grep_args('lut', kwargs)

    # Label text
    if label_text is None:
        label_text = {}
    elif isinstance(label_text, (list, np.ndarray)):
        label_text = {'left': label_text}

    # Array ranges
    specs = _get_ranges(layout, surfs, array_name, share, color_range)

    # Grid
    grid_row, grid_col, ridx, cidx, entries = \
        _gen_grid(nrow, ncol, label_text, color_bar, share)

    kwargs.update({'nrow': grid_row, 'ncol': grid_col, 'size': size})
    p = Plotter(**kwargs)

    for iren, jren in iter_prod(range(len(ridx)), range(len(cidx))):
        i, j = ridx[iren], cidx[jren]

        kwds = dp.renderer_kwds.copy()
        kwds.update({'row': iren, 'col': jren, 'background': background})

        # Renderers for empty entries
        if isinstance(i, str) or isinstance(j, str):
            if isinstance(i, str) and isinstance(j, str):
                p.AddRenderer(**kwds)
            continue

        kwds.update({k: v[i, j] for k, v in ren_kwds.items()})
        kwds['background'] = background  # just in case
        ren = p.AddRenderer(**kwds)

        if layout[i, j] is None:
            continue

        s = surfs[layout[i, j]]
        for ia, name in enumerate(array_name[i, j]):
            if name is False or name is None:
                continue

            sp = specs[ia, i, j]

            # Actor
            actor = dp.actor_kwds.copy()
            actor.update({k: v[i, j][ia] for k, v in actor_kwds.items()})
            if view[i, j] is not None:
                actor['orientation'] = orientations[view[i, j]]

            # Mapper
            mapper = dp.mapper_kwds.copy()
            mapper['scalarVisibility'] = name is not True
            mapper['interpolateScalarsBeforeMapping'] = not sp['disc']
            mapper.update({k: v[i, j][ia] for k, v in mapper_kwds.items()})
            mapper['inputDataObject'] = s
            if name is not True:
                mapper['arrayName'] = name

            # Lut
            lut = dp.lookuptable_kwds.copy()
            lut['numberOfTableValues'] = sp['nval']
            lut['range'] = (sp['min'], sp['max'])

            cm = cmap[i, j][ia]
            if cm is not None:
                if cm in colormaps:
                    table = colormaps[cm]
                else:
                    cm = plt.get_cmap(cm)
                    nvals = lut['numberOfTableValues']
                    table = cm(np.linspace(0, 1, nvals)) * 255
                    table = table.astype(np.uint8)

                lut['table'] = table
            if nan_color:
                lut['nanColor'] = nan_color

            # Do not support indexed lut for now
            # if sp['disc']:
                # lut['IndexedLookup'] = True
                # color_idx = sp['val']
                # lut['annotations'] = (color_idx, color_idx.astype(str))
                # cb_kwds['labelFormat'] = '%-4.0f'

            mapper['lookuptable'] = lut

            ren.AddActor(**actor, mapper=mapper)

        ren.ResetCamera()
        ren.activeCamera.parallelProjection = True
        ren.activeCamera.Zoom(zoom[i, j])

    # Plot renderers for color bar, text
    for e in entries:
        kwds = dp.renderer_kwds.copy()
        kwds.update({'row': e.row, 'col': e.col, 'background': background})
        ren1 = p.AddRenderer(**kwds)
        if isinstance(e.label, str):
            _add_text(ren1, e.label, e.loc, **text_kwds)
        else:  # color bar
            ren_lut = p.renderers[p.populated[e.label]][-1]
            lut = ren_lut.actors.lastActor.mapper.lookupTable
            _add_colorbar(ren1, lut.VTKObject, e.loc, **cb_kwds)

    return p


def plot_surf(surfs, layout, array_name=None, view=None, color_bar=None,
              color_range=None, share=False, label_text=None, cmap='viridis',
              nan_color=(0, 0, 0, 1), zoom=1, background=(1, 1, 1),
              size=(400, 400), embed_nb=False, interactive=True, scale=(1, 1),
              transparent_bg=True, screenshot=False, filename=None,
              return_plotter=False, **kwargs):

    """Plot surfaces arranged according to the `layout`.

    Parameters
    ----------
    surfs : dict[str, BSPolyData]
        Dictionary of surfaces.
    layout : array-like, shape = (n_rows, n_cols)
        Array of surface keys in `surfs`. Specifies how window is arranged.
    array_name : array-like, optional
        Names of point data array to plot for each layout entry.
        Use a tuple with multiple array names to plot multiple arrays
        (overlays) per layout entry. Default is None.
    view : array-like, optional
        View for each each layout entry. Possible views are {'lateral',
        'medial', 'ventral', 'dorsal'}. If None, use default view.
        Default is None.
    color_bar : {'left', 'right', 'top', 'bottom'} or None, optional
        Location where color bars are rendered. If None, color bars are not
        included. Default is None.
    color_range : {'sym'}, tuple or sequence.
        Range for each array name. If 'sym', uses a symmetric range. Only used
        if array has positive and negative values. Default is None.
    share : {'row', 'col', 'both'} or bool, optional
        If ``share == 'row'``, point data for surfaces in the same row share
        same data range. If ``share == 'col'``, the same but for columns.
        If ``share == 'both'``, all data shares same range. If True, similar
        to ``share == 'both'``. Default is False.
    label_text : dict[str, array-like], optional
        Label text for column/row. Possible keys are {'left', 'right',
        'top', 'bottom'}, which indicate the location. Default is None.
    cmap : str or sequence of str, optional
        Color map name (from matplotlib) for each array name.
        Default is 'viridis'.
    nan_color : tuple
        Color for nan values. Default is (0, 0, 0, 1).
    zoom : float or sequence of float, optional
        Zoom applied to the surfaces in each layout entry.
    background : tuple
        Background color. Default is (1, 1, 1).
    size : tuple, optional
        Window size. Default is (400, 400).
    interactive : bool, optional
        Whether to enable interaction. Default is True.
    embed_nb : bool, optional
        Whether to embed figure in notebook. Only used if running in a
        notebook. Default is False.
    screenshot : bool, optional
        Take a screenshot instead of rendering. Default is False.
    filename : str, optional
        Filename to save the screenshot. Default is None.
    transparent_bg : bool, optional
        Whether to us a transparent background. Only used if
        ``screenshot==True``. Default is False.
    scale : tuple, optional
        Scale (magnification). Only used if ``screenshot==True``.
        Default is None.
    kwargs : keyword-valued args
        Additional arguments passed to the renderers, actors, mapper or
        plotter. Keywords starting with:

        - 'renderer__' are passed to the renderers.
        - 'actor__' are passed to the actors.
        - 'mapper__' are passed to the mappers.

        The rest of keywords are passed to the plotter.

    Returns
    -------
    figure : Ipython Image or panel or None
        Figure to plot. None if using vtk for rendering (i.e.,
        ``embed_nb == False``).

    See Also
    --------
    :func:`build_plotter`
    :func:`plot_hemispheres`

    Notes
    -----
    If sequences, shapes of `array_name`, `view` and `zoom` must be equal
    or broadcastable to the shape of `layout`. Renderer keywords must also
    be broadcastable to the shape of `layout`.

    If sequences, shapes of `cmap` and `cbar_range` must be equal or
    broadcastable to the shape of `array_name`, including the number of array
    names per entry. Actor and mapper keywords must also be broadcastable to
    the shape of `array_name`.

    """

    if screenshot and filename is None:
        raise ValueError('Filename is required.')

    if screenshot or embed_nb:
        kwargs.update({'offscreen': True})

    p = build_plotter(surfs, layout, array_name=array_name, view=view,
                      color_bar=color_bar, color_range=color_range,
                      share=share, label_text=label_text, cmap=cmap,
                      nan_color=nan_color, zoom=zoom, background=background,
                      size=size, **kwargs)
    if return_plotter:
        return p
    if screenshot:
        return p.screenshot(filename, transparent_bg=transparent_bg,
                            scale=scale)

    return p.show(embed_nb=embed_nb, interactive=interactive, scale=scale,
                  transparent_bg=transparent_bg)


@wrap_input(0, 1)
def plot_hemispheres(surf_lh, surf_rh, array_name=None, color_bar=False,
                     color_range=None, label_text=None,
                     cmap='viridis', nan_color=(0, 0, 0, 1), zoom=1,
                     background=(1, 1, 1), size=(400, 400), interactive=True,
                     embed_nb=False, screenshot=False, filename=None,
                     scale=(1, 1), transparent_bg=True, **kwargs):
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
        Default is None.
    color_bar : bool, optional
        Plot color bar for each array (row). Default is False.
    color_range : {'sym'}, tuple or sequence.
        Range for each array name. If 'sym', uses a symmetric range. Only used
        if array has positive and negative values. Default is None.
    label_text : dict[str, array-like], optional
        Label text for column/row. Possible keys are {'left', 'right',
        'top', 'bottom'}, which indicate the location. Default is None.
    nan_color : tuple
        Color for nan values. Default is (0, 0, 0, 1).
    zoom : float or sequence of float, optional
        Zoom applied to the surfaces in each layout entry.
    background : tuple
        Background color. Default is (1, 1, 1).
    cmap : str, optional
        Color map name (from matplotlib). Default is 'viridis'.
    size : tuple, optional
        Window size. Default is (800, 200).
    interactive : bool, optional
        Whether to enable interaction. Default is True.
    embed_nb : bool, optional
        Whether to embed figure in notebook. Only used if running in a
        notebook. Default is False.
    screenshot : bool, optional
        Take a screenshot instead of rendering. Default is False.
    filename : str, optional
        Filename to save the screenshot. Default is None.
    transparent_bg : bool, optional
        Whether to us a transparent background. Only used if
        ``screenshot==True``. Default is False.
    scale : tuple, optional
        Scale (magnification). Only used if ``screenshot==True``.
        Default is None.
    kwargs : keyword-valued args
        Additional arguments passed to the plotter.


    Returns
    -------
    figure : Ipython Image or None
        Figure to plot. None if using vtk for rendering (i.e.,
        ``embed_nb == False``).

    See Also
    --------
    :func:`build_plotter`
    :func:`plot_surf`

    """

    if color_bar is True:
        color_bar = 'right'

    surfs = {'lh': surf_lh, 'rh': surf_rh}
    layout = ['lh', 'lh', 'rh', 'rh']
    view = ['lateral', 'medial', 'lateral', 'medial']

    if isinstance(array_name, np.ndarray):
        if array_name.ndim == 2:
            array_name = [a for a in array_name]
        elif array_name.ndim == 1:
            array_name = [array_name]

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

    if isinstance(cmap, list):
        cmap = np.asarray(cmap)[:, None]

    kwds = {'view': view, 'share': 'r'}
    kwds.update(kwargs)
    return plot_surf(surfs, layout, array_name=array_name, color_bar=color_bar,
                     color_range=color_range, label_text=label_text, cmap=cmap,
                     nan_color=nan_color, zoom=zoom, background=background,
                     size=size, interactive=interactive, embed_nb=embed_nb,
                     screenshot=screenshot, filename=filename, scale=scale,
                     transparent_bg=transparent_bg, **kwds)
