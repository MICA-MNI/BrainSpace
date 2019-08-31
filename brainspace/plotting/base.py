"""
Plotting functionality based on VTK.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import warnings
import numpy as np
from numpy.lib.stride_tricks import as_strided

import matplotlib.pyplot as plt
from PIL import Image

from ..vtk_interface import wrap_vtk
from ..vtk_interface.decorators import wrap_input

# from vtkmodules.vtkCommonCorePython import vtkCommand
# from vtkmodules.vtkIOImagePython import vtkPNGWriter
# from vtkmodules.vtkRenderingCorePython import (vtkRenderWindowInteractor,
#                                                vtkWindowToImageFilter,
#                                                vtkRenderWindow)
# from vtkmodules import qt as vtk_qt

from vtk import (vtkCommand, vtkPNGWriter, vtkRenderWindowInteractor,
                 vtkWindowToImageFilter, vtkRenderWindow)

import vtk.qt as vtk_qt

from ..vtk_interface.wrappers import BSRenderer
from ..vtk_interface.pipeline import get_output


# for display bugs due to older intel integrated GPUs
vtk_qt.QVTKRWIBase = 'QGLWidget'


try:
    import IPython
    has_ipython = True
except ImportError:
    has_ipython = False


try:
    import panel as pn
    pn.extension('vtk')
    has_panel = True
except ImportError:
    has_panel = False


try:
    # from vtkmodules.qt.QVTKRenderWindowInteractor import \
    #     QVTKRenderWindowInteractor
    from vtk.qt.QVTKRenderWindowInteractor import \
        QVTKRenderWindowInteractor
    from PyQt5 import QtGui
    from PyQt5.QtWidgets import QVBoxLayout, QFrame, QMainWindow
    has_pyqt = True
except ImportError:
    has_pyqt = False


def in_ipython():
    is_ipy = False
    if has_ipython:
        try:
            ipy = IPython.get_ipython()
            if ipy is not None:
                is_ipy = True
        except:
            pass
    return is_ipy


def in_notebook():
    is_nb = False
    if has_ipython:
        try:
            ipy = IPython.get_ipython()
            if ipy is not None:
                is_nb = type(ipy).__module__.startswith('ipykernel.')
        except:
            pass
    return is_nb


def _create_grid(nrow, ncol):
    dx, dy = 1 / ncol, 1 / nrow
    x_min = np.tile(np.arange(0, 1, dx), nrow)
    x_max = x_min + dx
    y_min = np.repeat(np.arange(0, 1, dy), ncol)[::-1]
    y_max = y_min + dy
    g = np.column_stack([x_min, y_min, x_max, y_max])

    strides = (4*g.itemsize*ncol, 4*g.itemsize, g.itemsize)
    return as_strided(g, shape=(nrow, ncol, 4), strides=strides)


# def _capture_as_array(ren_win, scale=None, transparent_bg=True):
#     scale = (1, 1) if scale is None else scale
#     bg = 'RGBA' if transparent_bg else 'RGB'
#
#     w2if = wrap_vtk(vtkWindowToImageFilter, readFrontBuffer=False,
#                     input=ren_win, scale=scale, inputBufferType=bg)
#
#     img = get_output(w2if)
#     array = img.get_array(name='ImageScalars', at='p')
#     shape = img.dimensions[::-1][1:] + (-1,)
#     return array.reshape(shape)[::-1]
#
#     # writer = wrap_vtk(vtkPNGWriter, writeToMemory=True,
#     #                   inputConnection=w2if.outputPort)
#     # writer.Write()
#     # data = memoryview(writer.result).tobytes()
#     # from IPython.display import Image
#     # return Image(data)


class BasePlotter(object):

    @wrap_input('ren_win', 'iren')
    def __init__(self, n_rows=1, n_cols=1, offscreen=None, ren_win=None,
                 iren=None, **kwargs):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.offscreen = offscreen

        self.ren_win = ren_win
        if self.ren_win is None:
            self.ren_win = wrap_vtk(vtkRenderWindow)
        self.ren_win.setVTK(**kwargs)

        self.iren = iren
        if self.iren is None:
            self.iren = wrap_vtk(vtkRenderWindowInteractor)

        self.n_renderers = 0
        self.renderers = dict()
        self.bounds = dict()
        self.populated = -np.ones((self.n_rows, self.n_cols), dtype=np.int32)
        self.grid = _create_grid(self.n_rows, self.n_cols)

    def AddRenderer(self, row=None, col=None, renderer=None, **kwargs):

        # row/col = 1, (0, 2), (None, 2), (1, None), (None, None) or None
        # bounds in the form :xmins[i], ymins[i], xmaxs[i], ymaxs[i]
        if row is None or isinstance(row, tuple):
            row = slice(None) if row is None else slice(*row)
        else:
            row = slice(row, row + 1)
        if col is None or isinstance(col, tuple):
            col = slice(None) if col is None else slice(*col)
        else:
            col = slice(col, col + 1)

        if np.any(self.populated[row, col] != -1):
            raise ValueError('Renderer overlaps with other renderers.')

        self.populated[row, col] = self.n_renderers

        subgrid = self.grid[row, col]
        bounds = np.empty(4)
        bounds[:2] = subgrid[..., :2].min(axis=(0, 1))
        bounds[2:] = subgrid[..., 2:].max(axis=(0, 1))

        renderer = BSRenderer(vtkobject=renderer, **kwargs)
        renderer.SetViewport(*bounds)
        self.ren_win.AddRenderer(renderer.VTKObject)

        self.renderers[self.n_renderers] = renderer
        self.bounds[self.n_renderers] = bounds
        self.n_renderers += 1
        return renderer

    def __getattr__(self, name):
        """Forwards unknown attribute requests to RenderWindow."""
        return getattr(self.ren_win, name)

    def show(self, interactive=True, embed_nb=False, scale=None,
             transparent_bg=True, as_mpl=False):

        if self.n_renderers == 0:
            raise ValueError('No renderers available.')

        embed_nb = embed_nb and in_notebook()
        if embed_nb and interactive and not has_panel:
            interactive = False
            warnings.warn("Interactive requires 'panel'.")

        if embed_nb and interactive and (self.n_rows > 1 or self.n_cols > 1):
            interactive = False
            warnings.warn("For the moment, interactive support is only "
                          "provided for a single renderer: "
                          "'n_rows=1' and 'n_cols=1'")

        if embed_nb or as_mpl or self.offscreen is True:
            self.ren_win.SetOffScreenRendering(True)
        else:
            self.ren_win.SetOffScreenRendering(False)

            self.iren.SetRenderWindow(self.ren_win.VTKObject)
            self.iren.Initialize()
            if not interactive:
                self.iren.SetInteractorStyle(None)
            self.iren.AddObserver(vtkCommand.ExitEvent, self.close)

        self.ren_win.Render()

        if embed_nb and interactive:
            try:
                return self._render_panel()
            except:
                pass

        if embed_nb:
            return self._capture_image(scale=scale,
                                       transparent_bg=transparent_bg)

        if as_mpl:
            return self._plot_mpl(scale=scale, transparent_bg=transparent_bg)

        if self.offscreen is not True:
            self.iren.Start()
        return None

    def close(self, *args):
        # try:
        #     if hasattr(self, 'panel'):
        #         del self.panel
        # except:
        #     pass
        self.ren_win.Finalize()
        # self.iren.RemoveAllObservers()
        self.iren.TerminateApp()

    def _render_panel(self):
        w, h = self.ren_win.GetSize()
        self.panel = pn.pane.VTK(self.ren_win.VTKObject, width=w, height=h)
        return self.panel

    def _capture_image(self, scale=None, transparent_bg=True):
        # self.ren_win.Render()
        scale = (1, 1) if scale is None else scale
        bg = 'RGBA' if transparent_bg else 'RGB'

        w2if = wrap_vtk(vtkWindowToImageFilter, readFrontBuffer=False,
                        input=self.ren_win.VTKObject, scale=scale,
                        inputBufferType=bg)

        img = get_output(w2if)
        array = img.get_array(name='ImageScalars', at='p')
        shape = img.dimensions[::-1][1:] + (-1,)
        return Image.fromarray(array.reshape(shape)[::-1])

    def _display_notebook(self, scale=None, transparent_bg=True):
        return self._capture_image(scale=scale, transparent_bg=transparent_bg)

    def screenshot(self, filename=None, scale=None, transparent_bg=True):
        # if not self.active:
        #     raise ValueError("Cannot take screenshot. Call 'show' first.")

        img = self._capture_image(scale=scale, transparent_bg=transparent_bg)
        if filename is None:
            return img
        img.save(filename)

    def _plot_mpl(self, scale=None, transparent_bg=True):
        scale = (1, 1) if scale is None else scale
        bg = 'RGBA' if transparent_bg else 'RGB'

        w2if = wrap_vtk(vtkWindowToImageFilter, readFrontBuffer=False,
                        input=self.ren_win.VTKObject, scale=scale,
                        inputBufferType=bg)

        img = get_output(w2if)
        array = img.get_array(name='ImageScalars', at='p')
        shape = img.dimensions[::-1][1:] + (-1,)
        array = array.reshape(shape)[::-1]

        h, w = array.shape[:2]
        fig = plt.figure(figsize=(w/100, h/100), dpi=100)
        ax = fig.gca()
        ax.set_axis_off()
        ax.imshow(array, interpolation='bilinear')
        plt.show()

    def Render(self):
        self.ren_win.Render()
        # if hasattr(self, 'panel'):
        #     self.panel.param.trigger('object')


def _get_qt_app():
    app = None

    if in_ipython():
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.magic('gui qt')

        from IPython.external.qt_for_kernel import QtGui
        app = QtGui.QApplication.instance()

    if app is None:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if not app:
            app = QApplication([''])

    return app


class Plotter(BasePlotter):

    def __init__(self, n_rows=1, n_cols=1, try_qt=True, offscreen=None,
                 **kwargs):
        self.try_qt = try_qt
        self.use_qt = has_pyqt and try_qt and offscreen is not True

        # Prepare qt
        iren = None
        ren_win = None
        if self.use_qt:
            self.app = _get_qt_app()
            self.app_window = QMainWindow()

            self.frame = QFrame()
            self.frame.setFrameStyle(QFrame.NoFrame)

            self.qt_ren = QVTKRenderWindowInteractor(parent=self.frame)

            self.vlayout = QVBoxLayout()
            self.vlayout.addWidget(self.qt_ren)

            self.frame.setLayout(self.vlayout)
            self.app_window.setCentralWidget(self.frame)

            ren_win = wrap_vtk(self.qt_ren.GetRenderWindow())
            iren = ren_win.GetInteractor()

        super().__init__(n_rows=n_rows, n_cols=n_cols, ren_win=ren_win,
                         iren=iren, offscreen=offscreen, **kwargs)

        # Exit with 'q' and 'e'
        self.iren.AddObserver("KeyPressEvent", self.key_quit)

    def show(self, interactive=True, embed_nb=False, scale=None,
             transparent_bg=True, as_mpl=False):

        embed_nb = embed_nb and in_notebook()
        if embed_nb and interactive and not has_panel:
            interactive = False

        if self.use_qt and not as_mpl and not embed_nb and self.offscreen is not True:
            self.iren.Initialize()
            if not interactive:
                self.iren.SetInteractorStyle(None)
            self.app_window.show()
            self.qt_ren.show()
        else:
            return super().show(interactive=interactive, embed_nb=embed_nb,
                                scale=scale, transparent_bg=transparent_bg,
                                as_mpl=as_mpl)

    def key_quit(self, obj=None, event=None):
        try:
            key = self.iren.GetKeySym().lower()
            if key in ['q', 'e']:
                self.close()
        except:
            pass

    def close(self, *args):
        super().close()
        if self.use_qt:
            self.app_window.close()


class GridPlotter(Plotter):
    def __init__(self, n_rows=1, n_cols=1, try_qt=True, offscreen=None,
                 **kwargs):
        super().__init__(n_rows=n_rows, n_cols=n_cols, try_qt=try_qt,
                         offscreen=offscreen, **kwargs)

    def AddRenderer(self, row, col, renderer=None, **kwargs):
        if not isinstance(row, int) or not isinstance(row, int):
            raise ValueError('GridPlotter only supports one renderer '
                             'for each grid entry')
        return self.ren_win.AddRenderer(row=row, col=col, renderer=renderer,
                                        **kwargs)

    def AddRenderers(self, **kwargs):
        ren = np.empty((self.n_rows, self.n_cols), dtype=np.object)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                ren[i, j] = self.ren_win.AddRenderer(row=i, col=j, **kwargs)

        return ren
