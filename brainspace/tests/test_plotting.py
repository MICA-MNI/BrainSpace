""" Test plotting """


import pytest

import numpy as np

import vtk

try:
    import panel as pn
except ImportError:
    pn = None

try:
    import IPython as ipy
except ImportError:
    ipy = None

try:
    import PyQt5  # noqa: F401
    has_pyqt5 = True
except ImportError:
    has_pyqt5 = False


from brainspace.vtk_interface.pipeline import to_data
from brainspace.plotting.base import Plotter
from brainspace.plotting import build_plotter, plot_surf, plot_hemispheres


def plotter_single_renderer():
    s = to_data(vtk.vtkSphereSource())

    p = Plotter(offscreen=True)
    ren0 = p.AddRenderer(row=0, col=0)
    ac0 = ren0.AddActor()
    ac0.SetMapper(inputdata=s)
    return p


def plotter_multiple_renderers():
    s = to_data(vtk.vtkSphereSource())

    p = Plotter(nrow=1, ncol=2, offscreen=True)
    ren0 = p.AddRenderer(row=0, col=0)
    ac0 = ren0.AddActor()
    ac0.SetMapper(inputdata=s)

    ren1 = p.AddRenderer(row=0, col=1)
    ac1 = ren1.AddActor()
    ac1.SetMapper(inputdata=s)
    return p


@pytest.mark.skipif(ipy is None or pn is None, reason="Requires panel")
def test_plotter_panel():

    cls = (pn.pane.VTK, pn.pane.vtk.vtk.VTKRenderWindowSynchronized)
    p = plotter_single_renderer()
    img = p.to_panel()
    assert isinstance(img, cls)

    img = p.show(embed_nb=True, interactive=True)
    assert isinstance(img, cls)
    p.close()

    p = plotter_multiple_renderers()
    with pytest.warns(UserWarning, match=r"Support for interactive \w+"):
        img = p.to_panel()
        assert isinstance(img, ipy.display.Image)

    with pytest.warns(UserWarning, match=r"Support for interactive \w+"):
        img = p.show(embed_nb=True, interactive=True)
        assert isinstance(img, ipy.display.Image)
    p.close()


@pytest.mark.skipif(ipy is None, reason="Requires IPython")
def test_plotter_ipython():
    p = plotter_single_renderer()
    img = p.show(embed_nb=True, interactive=False)
    assert isinstance(img, ipy.display.Image)

    img = p.to_notebook()
    assert isinstance(img, ipy.display.Image)
    p.close()

    p = plotter_multiple_renderers()
    img = p.show(embed_nb=True, interactive=False)
    assert isinstance(img, ipy.display.Image)

    img = p.to_notebook()
    assert isinstance(img, ipy.display.Image)
    p.close()


def test_plotter_numpy():
    p = plotter_single_renderer()
    img = p.to_numpy()
    assert isinstance(img, np.ndarray)
    p.close()

    p = plotter_multiple_renderers()
    img = p.to_numpy()
    assert isinstance(img, np.ndarray)
    p.close()


def test_plotter_screenshot():
    import os
    root_pth = os.path.dirname(__file__)

    p = plotter_single_renderer()
    pth = p.screenshot(os.path.join(root_pth, '_test_single_screenshot.png'))
    assert os.path.exists(pth)
    os.remove(pth)
    p.close()

    p = plotter_multiple_renderers()
    pth = p.screenshot(os.path.join(root_pth, '_test_multiple_screenshot.png'))
    assert os.path.exists(pth)
    os.remove(pth)
    p.close()


def test_build_plotter():
    s1 = to_data(vtk.vtkSphereSource())
    s2 = to_data(vtk.vtkSphereSource())

    surfs = {'s1': s1, 's2': s2}
    layout = np.array([['s1', 's2'], ['s2', 's2']])
    p = build_plotter(surfs, layout, offscreen=True)
    assert isinstance(p, Plotter)


def test_plot_surf():
    s1 = to_data(vtk.vtkSphereSource())
    s2 = to_data(vtk.vtkSphereSource())

    surfs = {'s1': s1, 's2': s2}
    layout = np.array([['s1', 's2'], ['s2', 's2']])
    plot_surf(surfs, layout, offscreen=True)


def test_plot_hemispheres():
    s1 = to_data(vtk.vtkSphereSource())
    s2 = to_data(vtk.vtkSphereSource())

    plot_hemispheres(s1, s2, offscreen=True)


# ---------------------------------------------------------------------------
# Regression tests for cmap argument types (issue: ListedColormap unhashable).
#
# Earlier brainspace versions did ``if cm in colormaps:`` directly, which
# requires the cmap argument to be hashable. Passing a matplotlib
# ``ListedColormap`` / ``LinearSegmentedColormap`` instance therefore raised
# ``TypeError: unhashable type: 'ListedColormap'``. The build_plotter path
# must accept both string names and Colormap instances; these tests pin that
# contract so it can't regress again.
# ---------------------------------------------------------------------------

def _sphere_with_scalar(name='val'):
    """Sphere mesh with a per-point scalar so cmap-driven LUTs are exercised."""
    s = to_data(vtk.vtkSphereSource())
    n_pts = s.n_points
    s.append_array(np.linspace(0, 1, n_pts).astype(np.float32), name=name,
                   at='p')
    return s


def test_build_plotter_cmap_listedcolormap():
    """ListedColormap instance must not raise (regression for unhashable cmap)."""
    from matplotlib.colors import ListedColormap

    s = _sphere_with_scalar()
    surfs = {'s': s}
    layout = np.array([['s']])
    cmap = ListedColormap([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1],
                           [0, 0, 1, 1]])
    p = build_plotter(surfs, layout, array_name='val', cmap=cmap,
                      offscreen=True)
    assert isinstance(p, Plotter)
    p.close()


def test_build_plotter_cmap_linearsegmentedcolormap():
    """LinearSegmentedColormap instance must not raise either."""
    from matplotlib.colors import LinearSegmentedColormap

    s = _sphere_with_scalar()
    surfs = {'s': s}
    layout = np.array([['s']])
    cmap = LinearSegmentedColormap.from_list(
        'custom_div', ['#2c7bb6', '#ffffbf', '#d7191c'])
    p = build_plotter(surfs, layout, array_name='val', cmap=cmap,
                      offscreen=True)
    assert isinstance(p, Plotter)
    p.close()


def test_build_plotter_cmap_matplotlib_string():
    """Plain matplotlib colormap names continue to work."""
    s = _sphere_with_scalar()
    surfs = {'s': s}
    layout = np.array([['s']])
    p = build_plotter(surfs, layout, array_name='val', cmap='viridis',
                      offscreen=True)
    assert isinstance(p, Plotter)
    p.close()


def test_build_plotter_cmap_brainspace_registered():
    """Names registered in brainspace.plotting.colormaps still resolve."""
    s = _sphere_with_scalar()
    surfs = {'s': s}
    layout = np.array([['s']])
    p = build_plotter(surfs, layout, array_name='val', cmap='BuGyRd',
                      offscreen=True)
    assert isinstance(p, Plotter)
    p.close()


def test_build_plotter_cmap_mixed_across_cells():
    """Per-cell cmap list with mixed string and Colormap entries works."""
    from matplotlib.colors import ListedColormap

    s = _sphere_with_scalar()
    surfs = {'s': s}
    layout = np.array([['s', 's']])
    cmap_obj = ListedColormap([[0, 0, 0, 1], [1, 1, 1, 1]])
    p = build_plotter(surfs, layout, array_name='val',
                      cmap=['viridis', cmap_obj], offscreen=True)
    assert isinstance(p, Plotter)
    p.close()


def test_build_plotter_cmap_invalid_type_raises_clear_error():
    """Non-str / non-Colormap cmap arg fails with a useful message."""
    s = _sphere_with_scalar()
    surfs = {'s': s}
    layout = np.array([['s']])
    with pytest.raises(TypeError, match='cmap must be a string'):
        build_plotter(surfs, layout, array_name='val', cmap=42,
                      offscreen=True)


def test_plot_surf_cmap_listedcolormap_smoke():
    """End-to-end plot_surf must accept a Colormap instance (the entry point
    hippomaps.plotting.surfplot_canonical_foldunfold actually goes through)."""
    from matplotlib.colors import ListedColormap

    s = _sphere_with_scalar()
    surfs = {'s': s}
    layout = np.array([['s']])
    cmap = ListedColormap([[0, 0, 0, 1], [1, 0.5, 0, 1]])
    plot_surf(surfs, layout, array_name='val', cmap=cmap, offscreen=True)


def test_try_qt_no_longer_warns_unsupported():
    """try_qt=True must not emit the old 'Qt rendering is not supported' warning (#136).

    We pass offscreen=True so no actual Qt window is built (that requires a
    display and would crash on headless CI). The check is purely that the
    warn-and-disable shim is gone.
    """
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter('always')
        Plotter(offscreen=True, try_qt=True).close()
    assert not any(
        'Qt rendering is not supported' in str(x.message) for x in w
    )
