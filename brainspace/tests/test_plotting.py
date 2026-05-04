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
