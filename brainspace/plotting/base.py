"""
Plotting functionality based on VTK.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import warnings
import numpy as np
from numpy.lib.stride_tricks import as_strided

from vtkmodules.vtkRenderingOpenGL2Python import \
    vtkXRenderWindowInteractor as vtkRenderWindowInteractor
from vtkmodules.vtkIOImagePython import vtkPNGWriter
from vtkmodules.vtkRenderingCorePython import (vtkWindowToImageFilter,
                                               vtkRenderWindow)

from vtkmodules import qt as vtk_qt

from ..vtk_interface.base import BSVTKObjectWrapper
from ..vtk_interface.wrappers import BSRenderer
from ..vtk_interface.pipeline import serial_connect


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
    from PyQt5.QtCore import pyqtSignal
    from vtkmodules.qt.QVTKRenderWindowInteractor import \
        QVTKRenderWindowInteractor
    from PyQt5 import QtCore, QtGui
    from PyQt5.QtWidgets import QVBoxLayout, QFrame, QMainWindow
    has_pyqt = True
except ImportError:
    has_pyqt = False


def in_ipython():
    is_ipy = False
    if has_ipython:
        try:
            ipy = IPython.get_ipython()
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


def _create_grid(nrows, ncols):
    dx, dy = 1 / ncols, 1 / nrows
    x_min = np.tile(np.arange(0, 1, dx), nrows)
    x_max = x_min + dx
    y_min = np.repeat(np.arange(0, 1, dy), ncols)[::-1]
    y_max = y_min + dy
    g = np.vstack([x_min, y_min, x_max, y_max]).T.ravel()
    return as_strided(g, shape=(nrows, ncols, 4), strides=(4*8*ncols, 4*8, 8))


class BasePlotter(BSVTKObjectWrapper):

    def __init__(self, n_rows=1, n_cols=1, iren=None, vtkobject=None, **kwargs):
        if vtkobject is None:
            vtkobject = vtkRenderWindow()
        super().__init__(vtkobject=vtkobject, **kwargs)

        self.iren = vtkRenderWindowInteractor() if iren is None else iren

        self.n_rows = n_rows
        self.n_cols = n_cols

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
            row = slice(row, row+1)
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
        self.VTKObject.AddRenderer(renderer.VTKObject)

        self.renderers[self.n_renderers] = renderer
        self.bounds[self.n_renderers] = bounds
        self.n_renderers += 1
        return renderer

    def show(self, interactive=True, embed_nb=False, scale=None,
             transparent_bg=True):
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

        if embed_nb:
            self.SetOffScreenRendering(True)
        else:
            self.SetOffScreenRendering(False)
            if not interactive:
                self.iren.SetInteractorStyle(None)
            self.iren.SetRenderWindow(self.VTKObject)

        self.Render()

        if embed_nb and interactive:
            try:
                width, height = self.GetSize()
                disp = pn.pane.VTK(self.VTKObject, width=width, height=height)
                return disp
            except:
                pass

        if embed_nb:
            w2if = vtkWindowToImageFilter()
            w2if.ReadFrontBufferOff()
            w2if.SetInput(self.VTKObject)
            if scale is not None:
                w2if.SetScale(*scale)
            if transparent_bg:
                w2if.SetInputBufferTypeToRGBA()
            else:
                w2if.SetInputBufferTypeToRGB()
            w2if.Modified()
            w2if.Update()

            # Not working (tilted) when window width != height
            # import PIL.Image
            # img = lia.get_output(w2if, as_data=True)
            # shape = img.GetDimensions()[:-1] + (-1,)
            # img = img.PointData['ImageScalars'].reshape(shape)[::-1]
            # disp = IPython.display.display(PIL.Image.fromarray(img))
            # return disp

            writer = vtkPNGWriter()
            writer.SetWriteToMemory(1)
            serial_connect(w2if, writer, update=True, as_data=False)
            data = memoryview(writer.GetResult()).tobytes()
            from IPython.display import Image
            return Image(data)

        self.iren.Start()
        return None


class Plotter(object):

    def __init__(self, n_rows=1, n_cols=1, try_qt=True, **kwargs):
        self.try_qt = try_qt
        self.use_qt = has_pyqt and try_qt

        if self.use_qt:
            self.app = None

            if in_ipython():
                from IPython import get_ipython
                ipython = get_ipython()
                ipython.magic('gui qt')

                from IPython.external.qt_for_kernel import QtGui
                self.app = QtGui.QApplication.instance()

            if self.app is None:
                from PyQt5.QtWidgets import QApplication
                self.app = QApplication.instance()
                if not self.app:
                    self.app = QApplication([''])

            self.app_window = QMainWindow()

            self.frame = QFrame()
            self.frame.setFrameStyle(QFrame.NoFrame)

            self.qt_ren = QVTKRenderWindowInteractor(parent=self.frame)

            #################################
            # ################################
            rw = self.qt_ren.GetRenderWindow()
            self.iren = rw.GetInteractor()
            self.ren_win = BasePlotter(n_rows=n_rows, n_cols=n_cols,
                                       iren=self.iren, vtkobject=rw)
            self.ren_win.setVTK(**kwargs)
            #################################################################

            self.vlayout = QVBoxLayout()
            self.vlayout.addWidget(self.qt_ren)

            self.frame.setLayout(self.vlayout)
            self.app_window.setCentralWidget(self.frame)

        else:
            self.ren_win = BasePlotter(n_rows=n_rows, n_cols=n_cols, **kwargs)
            self.iren = self.ren_win.iren

            # self.iren.AddObserver("ExitEvent", self.quit) # SegFault!

        # Without this -> Qt window disappears!!
        self.iren.AddObserver("KeyPressEvent", self.key_quit)

    def __getattr__(self, name):
        try:
            return getattr(self.ren_win, name)
        except AttributeError:
            return getattr(self.iren, name)

    def show(self, interactive=True, embed_nb=False):
        embed_nb = embed_nb and in_notebook()
        if embed_nb and interactive and not has_panel:
            interactive = False

        if self.use_qt and not embed_nb:
            try:
                self.iren.Initialize()
                self.app_window.show()
                self.qt_ren.show()
            except:
                pass
        return self.ren_win.show(interactive=interactive, embed_nb=embed_nb)

    def key_quit(self, obj=None, event=None):
        try:
            key = self.iren.GetKeySym().lower()
            if key == 'q':
                self.quit()
        except:
            pass

    def quit(self, *args):

        # Close the window
        self.ren_win.Finalize()

        # Remove observers
        self.iren.RemoveAllObservers()

        # Stop the interactor
        self.iren.TerminateApp()

        # Is this even needed
        if self.use_qt:
            self.app.quit()


class GridPlotter(Plotter):
    def __init__(self, n_rows=1, n_cols=1, try_qt=True, **kwargs):
        super().__init__(n_rows=n_rows, n_cols=n_cols, try_qt=try_qt, **kwargs)

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
