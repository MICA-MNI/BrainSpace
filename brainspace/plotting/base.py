"""
Plotting functionality based on VTK.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import os
import warnings
from collections import defaultdict

import numpy as np
from numpy.lib.stride_tricks import as_strided

from vtk import vtkCommand
import vtk.qt as vtk_qt

from brainspace import OFF_SCREEN
from ..vtk_interface.pipeline import serial_connect, get_output
from ..vtk_interface.wrappers import (BSWindowToImageFilter, BSPNGWriter,
                                      BSBMPWriter, BSJPEGWriter, BSTIFFWriter,
                                      BSRenderWindow, BSRenderWindowInteractor,
                                      BSGenericRenderWindowInteractor,
                                      BSGL2PSExporter)


# for display bugs due to older intel integrated GPUs (see PyVista)
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
    from PyQt5 import QtGui
    from PyQt5.QtWidgets import QVBoxLayout, QFrame
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from .utils_qt import MainWindow
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


def _create_grid(nrow, ncol):
    """ Create bounds for vtk rendering

    Parameters
    ----------
    nrow : int or array-like
        Number of rows. If array-like, must be an array with values in
        ascending order between 0 and 1.
    ncol : int or array-like
        Number of columns. If array-like, must be an array with values in
        ascending order between 0 and 1.

    Returns
    -------
    grid: ndarray, shape = (nrow, ncol, 4)
        Grid for vtk rendering.

    Examples
    --------
    >>> _create_grid(1, 2)
    array([[[0. , 0. , 0.5, 1. ],
            [0.5, 0. , 1. , 1. ]]])
    >>> _create_grid(1, [0, .5, 1])
    array([[[0. , 0. , 0.5, 1. ],
            [0.5, 0. , 1. , 1. ]]])
    >>> _create_grid(1, [0, .5, .9])
    array([[[0. , 0. , 0.5, 1. ],
            [0.5, 0. , 0.9, 1. ]]])
    >>> _create_grid(1, [0, .5, .9, 1])
    array([[[0. , 0. , 0.5, 1. ],
            [0.5, 0. , 0.9, 1. ],
            [0.9, 0. , 1. , 1. ]]])
    >>> _create_grid(2, [.5, .6, .7])
    array([[[0.5, 0.5, 0.6, 1. ],
            [0.6, 0.5, 0.7, 1. ]],

           [[0.5, 0. , 0.6, 0.5],
            [0.6, 0. , 0.7, 0.5]]])
    """

    if not isinstance(nrow, int):
        nrow = np.atleast_1d(nrow)
        if nrow.size < 2 or np.any(np.sort(nrow) != nrow) or \
                nrow[0] < 0 or nrow[-1] > 1:
            raise ValueError('Incorrect row values.')

    if not isinstance(ncol, int):
        ncol = np.atleast_1d(ncol)
        if ncol.size < 2 or np.any(np.sort(ncol) != ncol) or \
                ncol[0] < 0 or ncol[-1] > 1:
            raise ValueError('Incorrect column values.')

    if isinstance(ncol, np.ndarray):
        x_min, x_max = ncol[:-1], ncol[1:]
        ncol = x_min.size
    else:
        dx = 1 / ncol
        x_min = np.arange(0, 1, dx)
        x_max = x_min + dx

    if isinstance(nrow, np.ndarray):
        y_min, y_max = nrow[:-1], nrow[1:]
        nrow = y_min.size
    else:
        dy = 1 / nrow
        y_min = np.arange(0, 1, dy)
        y_max = y_min + dy

    y_min = np.repeat(y_min, ncol)[::-1]
    y_max = np.repeat(y_max, ncol)[::-1]

    x_min = np.tile(x_min, nrow)
    x_max = np.tile(x_max, nrow)

    g = np.column_stack([x_min, y_min, x_max, y_max])

    strides = (4 * g.itemsize * ncol, 4 * g.itemsize, g.itemsize)
    return as_strided(g, shape=(nrow, ncol, 4), strides=strides)


class Plotter(object):

    DICT_PLOTTERS = dict()

    def __init__(self, nrow=1, ncol=1, offscreen=None, force_close=False,
                 try_qt=False, **kwargs):

        if try_qt:
            warnings.warn('Qt rendering is not supported for the moment.')
            try_qt = False

        self.grid = _create_grid(nrow, ncol)
        self.nrow, self.ncol = self.grid.shape[:2]
        self.offscreen = OFF_SCREEN if offscreen is None else offscreen
        self.force_close = force_close
        self.use_qt = has_pyqt and try_qt and not self.offscreen

        self.ren_win = BSRenderWindow(**kwargs)
        if not self.offscreen:
            if self.use_qt:
                self.iren = BSGenericRenderWindowInteractor()
            else:
                self.iren = BSRenderWindowInteractor()
            self.iren.renderWindow = self.ren_win
            self.iren_interactorStyle = self.iren.interactorStyle
            self.iren.AddObserver(vtkCommand.ExitEvent, self.quit)
        else:
            self.iren = None
            self.ren_win.offScreenRendering = True

        if self.use_qt:
            self.app = _get_qt_app()
            self.app_window = MainWindow()
            self.app_window.signal_close.connect(self.quit)

            self.frame = QFrame()
            self.frame.setFrameStyle(QFrame.NoFrame)

            self.qt_ren = QVTKRenderWindowInteractor(parent=self.frame,
                                                     iren=self.iren.VTKObject,
                                                     rw=self.ren_win.VTKObject)

            self.vlayout = QVBoxLayout()
            self.vlayout.addWidget(self.qt_ren)

            self.frame.setLayout(self.vlayout)
            self.app_window.setCentralWidget(self.frame)

        self.n_renderers = 0
        self.renderers = defaultdict(list)
        self.populated = -np.ones((self.nrow, self.ncol), dtype=np.int32)
        self.panel = None
        self._cancel_show = False
        self._rendered_once = False

        self.DICT_PLOTTERS[id(self)] = self

    @classmethod
    def close_all(cls):
        for k in list(cls.DICT_PLOTTERS.keys()):
            cls.DICT_PLOTTERS.pop(k).close()

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

        p = np.unique(self.populated[row, col])
        if p.size > 1:
            raise ValueError('Subplot overlaps with existing subplots.')
        p = p[0]

        if p == -1:
            self.populated[row, col] = p = self.n_renderers
            self.n_renderers += 1

        subgrid = self.grid[row, col]
        bounds = np.empty(4)
        bounds[:2] = subgrid[..., :2].min(axis=(0, 1))
        bounds[2:] = subgrid[..., 2:].max(axis=(0, 1))

        renderer = self.ren_win.AddRenderer(obj=renderer, **kwargs)
        renderer.SetViewport(*bounds)

        self.renderers[p].append(renderer)
        return renderer

    def __getattr__(self, name):
        """Forwards unknown attribute requests to BSRenderWindow."""
        return getattr(self.ren_win, name)

    def _check_interactive(self, embed_nb, interactive):

        if not embed_nb or not interactive:
            return interactive
        # if embed_nb and not in_notebook():
        #     raise ValueError("Cannot find notebook.")

        if not has_panel:
            warnings.warn("Interactive mode requires 'panel'. "
                          "Setting 'interactive=False'")
            return False

        if self.nrow > 1 or self.ncol > 1:
            warnings.warn("Support for interactive mode is only provided for "
                          "a single renderer: 'nrow=1' and 'ncol=1'. Setting "
                          "'interactive=False'")
            return False

        return interactive

    def show(self, embed_nb=False, interactive=True, transparent_bg=True,
             scale=(1, 1)):

        if embed_nb:
            interactive = self._check_interactive(embed_nb, interactive)
            if interactive:
                return self.to_panel(scale)
            return self.to_notebook(transparent_bg, scale)

        else:

            self._check_closed()

            if self.offscreen:
                # self._check_offscreen()
                # raise ValueError('Only offscreen rendering is available. '
                #                  'Please use offscreen=False.')
                return None

            if self._rendered_once:
                raise ValueError('Cannot render multiple times.')

            if self._cancel_show:
                raise ValueError('Cannot render after offscreen rendering.')

            self.iren.Initialize()
            if not interactive:
                self.iren.interactorStyle = None
                self.iren.AddObserver(vtkCommand.KeyPressEvent, self.key_quit)

            self.ren_win.Render()
            if self.use_qt:
                self.app_window.show()
            else:
                self.iren.Start()

            self._rendered_once = True

        return None

    def key_quit(self, obj=None, event=None):
        if self.iren.keySym.lower() in ['q', 'e']:
            self.quit()

    def close(self):
        self.ren_win.Finalize()
        del self.ren_win
        self.ren_win = None
        if self.iren:
            self.iren.TerminateApp()
            del self.iren
            self.iren = None
        if self.use_qt:
            self.app_window.close()

    def quit(self, *args):
        if self.force_close:
            self.close()
        else:
            self.ren_win.Finalize()
            if self.iren:
                self.iren.TerminateApp()
            if self.use_qt:
                self.app_window.close()

    def _check_closed(self):
        if self.ren_win is None:
            raise ValueError('This plotter has been closed.')

    def _check_offscreen(self):
        if not self.offscreen:
            self.ren_win.offScreenRendering = True
            self.ren_win.interactor = None
            self._cancel_show = True
        self.ren_win.Render()

    def to_panel(self, scale=(1, 1)):
        if not self._check_interactive(True, True):
            return self.to_notebook(scale=scale)

        self._check_closed()
        self._check_offscreen()

        w, h = np.asarray(self.ren_win.size) * scale
        w, h = int(w), int(h)
        self.panel = pn.pane.VTK(self.ren_win.VTKObject, width=w, height=h)
        return self.panel

    def _win2img(self, transparent_bg, scale):
        self._check_closed()
        self._check_offscreen()

        wf = BSWindowToImageFilter(input=self.ren_win, readFrontBuffer=False,
                                   shouldRerender=True, fixBoundary=True,
                                   scale=scale)
        wf.inputBufferType = 'RGBA' if transparent_bg else 'RGB'
        return wf

    def to_notebook(self, transparent_bg=True, scale=(1, 1)):
        # if not in_notebook():
        #     raise ValueError("Cannot find notebook.")

        wimg = self._win2img(transparent_bg, scale)
        writer = BSPNGWriter(writeToMemory=True)
        result = serial_connect(wimg, writer, as_data=False).result
        data = memoryview(result).tobytes()
        from IPython.display import Image
        return Image(data)

    def to_numpy(self, transparent_bg=True, scale=(1, 1)):
        wf = self._win2img(transparent_bg, scale)
        img = get_output(wf)
        shape = img.dimensions[::-1][1:] + (-1,)
        img = img.PointData['ImageScalars'].reshape(shape)[::-1]
        return img

    def _to_image(self, filename, transparent_bg, scale):
        pth = os.path.abspath(os.path.expanduser(filename))
        pth_no_ext, ext = os.path.splitext(filename)
        ext = ext[1:]

        fmts1 = {'bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff'}
        fmts2 = {'eps', 'pdf', 'ps', 'svg'}
        if ext in fmts1:
            wimg = self._win2img(transparent_bg, scale)
            if ext == 'bmp':
                writer = BSBMPWriter(filename=filename)
            elif ext in ['jpg', 'jpeg']:
                writer = BSJPEGWriter(filename=filename)
            elif ext == 'png':
                writer = BSPNGWriter(filename=filename)
            else:  # if ext in ['tif', 'tiff']:
                writer = BSTIFFWriter(filename=filename)

            serial_connect(wimg, writer, as_data=False)

        elif ext in fmts2:
            self._check_closed()
            self._check_offscreen()

            orig_sz = self.ren_win.size
            self.ren_win.size = np.array(scale) * orig_sz

            w = BSGL2PSExporter(input=self.ren_win, fileFormat=ext,
                                compress=False, simpleLineOffset=True,
                                filePrefix=pth_no_ext,
                                title='', write3DPropsAsRasterImage=True)
            w.UsePainterSettings()
            w.Update()

            self.ren_win.size = orig_sz

        else:
            raise ValueError("Format '%s' not supported. Supported formats "
                             "are: %s" % (ext, fmts1.union(fmts2)))

        return pth

    def screenshot(self, filename, transparent_bg=True, scale=(1, 1)):
        return self._to_image(filename, transparent_bg, scale)


class GridPlotter(Plotter):
    def __init__(self, nrow=1, ncol=1, try_qt=True, offscreen=None,
                 **kwargs):
        super().__init__(nrow=nrow, ncol=ncol, try_qt=try_qt,
                         offscreen=offscreen, **kwargs)

    def AddRenderer(self, row, col, renderer=None, **kwargs):
        if not isinstance(row, int) or not isinstance(row, int):
            raise ValueError('GridPlotter only supports one renderer '
                             'for each grid entry')
        return super().AddRenderer(row=row, col=col, renderer=renderer,
                                   **kwargs)

    def AddRenderers(self, **kwargs):
        ren = np.empty((self.nrow, self.ncol), dtype=np.object)
        for i in range(self.nrow):
            for j in range(self.ncol):
                ren[i, j] = super().AddRenderer(row=i, col=j, **kwargs)

        return ren
