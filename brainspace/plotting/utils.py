"""
Utility functions for plotting.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from itertools import product as iter_prod
import collections
from collections import namedtuple

import numpy as np


class PTuple(collections.abc.Sequence):
    def __init__(self, *args):
        self.args = args

    def __getitem__(self, i):
        return self.args[i]

    def __len__(self):
        return len(self.args)

    def __repr__(self):
        return self.args.__repr__()


def _broadcast(arg, name, shape):
    """Broadcast arg to shape.

    Parameters
    ----------
    arg : sequence
        Sequence of elements to broadcast.
    name : str
        Argument name.
    shape : tuple
        The shape of the desired array.

    Returns
    -------
    broadcast : ndarray
        An array with the new shape.

    Raises
    ------
    ValueError
        If `arg` is not compatible with the new shape.
    """

    nrow, ncol = shape

    arg_np = np.empty(shape, dtype=np.object_)
    if not isinstance(arg, (list, np.ndarray)):
        arg_np[:] = [[arg] * ncol] * nrow
        return arg_np

    arg = [list(a) if isinstance(a, np.ndarray) else a for a in arg]
    if all([not isinstance(a, list) for a in arg]) and len(arg) == ncol:
        arg_np[:] = [arg] * nrow
        return arg_np

    if len(arg) == nrow:
        for i, a in enumerate(arg):
            if not isinstance(a, list):
                a = [a]
            if len(a) == 1:
                arg_np[i] = a * ncol
            elif len(a) == ncol:
                arg_np[i] = a
            else:
                raise ValueError("Cannot broadcast '{0}' to shape {1}. Row "
                                 "length mismatch.".format(name, shape))
        return arg_np

    raise ValueError("Cannot broadcast '{0}' with shape {1} to shape {2}"
                     "".format(name, (len(arg),), shape))


def _expand_arg(arg, name, shape, ref=None):
    """Broadcast arg and expand each entry according to ref.

    Parameters
    ----------
    arg : sequence
        Sequence of elements to expand.
    name : str
        Argument name.
    shape : tuple
        The shape of the desired array.
    ref : array of tuples, optional
        Reference array of tuples. If None, only broadcast is used.

    Returns
    -------
    expand : ndarray of tuples
        An array with the new shape and with each entry having the same length
        as its corresponding entry in the reference array.

    """
    arg = _broadcast(arg, name, shape)
    if ref is None:
        for i, a in np.ndenumerate(arg):
            arg[i] = a if isinstance(a, PTuple) else PTuple(a)
        return arg

    for i, a in np.ndenumerate(arg):
        nr = len(ref[i])
        if not isinstance(a, PTuple):
            arg[i] = PTuple(*([a] * nr))
        elif nr > len(a) == 1:
            arg[i] = PTuple(*([a[0]] * nr))
        elif nr != len(a):
            raise ValueError("Cannot parse '%s'. Length mismatch." % name)
    return arg


def _grep_args(name, kwds, shape=None, ref=None):
    """Broadcast/expand each value in `kwds` if its key starts with `name__`.

    Parameters
    ----------
    name : str
        Key prefix.
    kwds : dict
        Dictionary.
    shape : tuple, default
        The shape of the desired arrays. Default is None.
        If None, no broadcasting.
    ref : array of tuples, optional
        Reference array of tuples. If None, only broadcast is used.

    Returns
    -------
    expand : dict
        Dictionary with expanded arrays.

    Notes
    -----
    Entries are removed from `kwds`.
    """

    d = {k.split('__')[1].lower(): k for k in kwds
         if k.lower().startswith(name + '__')}
    for k, korg in d.items():
        if shape is None:
            d[k] = kwds.pop(korg)
        elif ref is None:
            d[k] = _broadcast(kwds.pop(korg), korg, shape)
        else:
            d[k] = _expand_arg(kwds.pop(korg), korg, shape, ref=ref)
    return d


#############################################################
# Grid entry bounds
#############################################################
def _gen_entries(loc, idx, labs):
    """Generate an entry for each text label.

    Parameters
    ----------
    loc : {'left', 'right', 'top', 'bottom'}
        Location.
    idx : ndarray, shape = (n, 2)
        Array of indices.
    labs : sequence
        Labels.

    Returns
    -------
    Entries: list of tuples
        List of entries.
    """

    Entry = namedtuple('Entry', ['row', 'col', 'loc', 'label'])
    n = len(labs)
    st = idx.shape[0] // n

    ent = [*idx[:n].T, [loc] * n, labs]
    k = np.argmax(idx[-1] - idx[0])
    ent[k] = [(i, i + st) for i in range(idx[0, k], idx[-1, k] + 1, st)]
    return list(map(lambda x: Entry(*x), zip(*ent)))


def _gen_grid(nrow, ncol, lab_text, cbar, share, size_bar=0.11, size_lab=0.05):
    """ Generate grid for vtk window.

    Parameters
    ----------
    nrow : int
        Number of rows.
    ncol : int
        Number of columns.
    lab_text : dict
        Dictionary of label texts.
    cbar : {'left', 'right', 'top', 'bottom'} or None
    share : bool or {'b', 'r', 'c'}
        If colorbars are shared.
    size_bar : float
        Percentage of the vtk window use by the color bar
    size_lab : float
        Percentage of the vtk window use by the text labels.

    Returns
    -------
    grid_row : ndarray
        Row bounds for vtk renderes.
    grid_col : ndarray
        Column bounds for vtk renderes.
    row_idx : list
        list of row indices.
    col_idx : list
        list of column indices.
    entries : list of tuples
        Entries for color bar and text labels.
    """

    locs = ['top', 'bottom', 'left', 'right', 'cb']
    ridx, cidx = list(range(nrow)), list(range(ncol))

    def _extend_index(loct, lab):
        nonlocal cidx, ridx
        ix = cidx if loct in {'left', 'right'} else ridx
        pos = 0 if loct in {'left', 'top'} else len(ix)
        ix.insert(pos, lab)

    if cbar is not None and share:
        _extend_index(cbar, 'cb')

    for loc in lab_text.keys():
        _extend_index(loc, loc)

    # generate entries
    specs = np.zeros((2, len(ridx), len(cidx)), dtype=np.object_)
    specs[0].T[:], specs[1] = ridx, cidx

    sel = (~np.isin(specs, locs)).any(axis=0)
    s0, s1 = ridx.index(0), cidx.index(0)

    entries = []
    for loc in set(ridx + cidx).intersection(locs):
        idx = np.argwhere((specs == loc) & sel)[:, 1:]
        if loc == 'cb':
            labs = [(s0, s1)]  # lut location
            if share == 'r':
                labs = [(s0 + i, s1) for i in range(nrow)]
            elif share == 'c':
                labs = [(s0, s1 + i) for i in range(ncol)]
            entries += _gen_entries(cbar, idx, labs)
        else:
            if idx.shape[0] % len(lab_text[loc]) > 0:
                raise ValueError("Incompatible number of text labels: len({0})"
                                 " != {1}".format(lab_text[loc], idx.shape[0]))
            entries += _gen_entries(loc, idx, lab_text[loc])

    # generate grid
    grid = [np.zeros_like(idx, dtype=float) for idx in [ridx, cidx]]
    for i, (idx, g, n) in enumerate(zip([ridx, cidx], grid, [nrow, ncol])):
        np.place(g, [el in locs[:-1] for el in idx], size_lab)
        np.place(g, [el == 'cb' for el in idx], size_bar)
        g[g == 0] = (1 - g.sum()) / n
        grid[i] = np.insert(np.cumsum(g), 0, 0)
    grid[0] = 1 - grid[0][::-1]

    return grid[0], grid[1], ridx, cidx, entries


#############################################################
# Get array specs: min/max value, is float and n_vals
#############################################################
def _get_specs(layout, surfs, array_name, cbar_range, nvals=256):
    """Get array specifications.

    Parameters
    ----------
    layout : ndarray, shape = (n_rows, n_cols)
        Array of surface keys in `surfs`. Specifies how window is arranged.
    surfs : dict[str, BSPolyData]
        Dictionary of surfaces.
    array_name : ndarray
        Names of point data array to plot for each layout entry.
    cbar_range : {'sym'} or tuple,
        Range for each array. If 'sym', uses a symmetric range. Only used is
        array has positive and negative values.
    nvals : int, optional
        Number of lookup table values for continuous arrays.
        Default is 256.

    Returns
    -------
    specs : ndarray
        Array with specifications for each array entry.
    """

    nrow, ncol = layout.shape
    n_overlays = max([len(a) for a in array_name.ravel()])

    def _set_spec(x, rg):
        if rg is None or rg == 'sym':
            a, b = np.nanmin(x), np.nanmax(x)
            if rg == 'sym' and np.sign(a) != np.sign(b):
                b = max(np.abs(a), b)
                a = -b
            rg = (a, b)

        if np.issubdtype(x.dtype, np.floating):
            return (*rg, nvals, np.array([]), False)
        vals = np.unique(x)
        return (*rg, vals.size, vals, True)

    dt = np.dtype([('min', 'f8'), ('max', 'f8'), ('nval', 'i8'),
                   ('val', 'O'), ('disc', '?')])
    specs = np.zeros((n_overlays, nrow, ncol), dtype=dt)
    specs[:] = (np.nan, np.nan, nvals, np.array([]), False)
    map_sp = {k: {} for k in surfs.keys()}
    for idx, k in np.ndenumerate(layout):
        if k is None:
            continue
        for ia, name in enumerate(array_name[idx]):
            if name not in surfs[k].point_keys:
                continue
            if name not in map_sp[k]:
                arr = surfs[k].PointData[name]
                map_sp[k][name] = _set_spec(arr, cbar_range[idx][ia])
            specs[(ia,) + idx] = map_sp[k][name]

    return specs


def _get_ranges(layout, surfs, array_name, share, cbar_range, nvals=256):
    """Get data range for each array.

    Parameters
    ----------
    layout : ndarray, shape = (n_rows, n_cols)
        Array of surface keys in `surfs`. Specifies how window is arranged.
    surfs : dict[str, BSPolyData]
        Dictionary of surfaces.
    array_name : ndarray
        Names of point data array to plot for each layout entry.
    cbar_range : {'sym'} or tuple,
        Range for each array. If 'sym', uses a symmetric range. Only used if
        array has positive and negative values.
    nvals : int, optional
        Number of lookup table values for continuous arrays.
        Default is 256.

    Returns
    -------
    specs : ndarray
        Array with specifications for each array entry.
    """

    specs = _get_specs(layout, surfs, array_name, cbar_range, nvals=nvals)

    if share:
        n_overlays, nrow, ncol = specs.shape
        ax, idx = (1, 2), range(n_overlays)
        if share == 'r':
            ax, idx = 2, iter_prod(idx, range(nrow))
        elif share == 'c':
            ax, idx = 1, iter_prod(idx, [slice(None)], range(ncol))

        specs['min'][:] = np.nanmin(specs['min'], axis=ax, keepdims=True)
        specs['max'][:] = np.nanmax(specs['max'], axis=ax, keepdims=True)

        for i in idx:
            if specs['disc'][i].all():
                uv = np.unique(np.concatenate(specs['val'][i].ravel()))
                specs['nval'][i] = uv.size
                for j in range(specs['val'][i].size):
                    specs['val'][i].flat[j] = uv
            else:
                specs['disc'][i] = False

    return specs[['min', 'max', 'nval', 'disc', 'val']]
