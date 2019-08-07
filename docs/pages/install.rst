.. _install_page:

Installation Guide
==============================

BrainSpace is available in Python and Matlab.


Python installation
-------------------
BrainSpace works on Python 3.6+, and probably with older versions of Python 3,
although it is not tested.


Dependencies
^^^^^^^^^^^^

To use BrainSpace, the following Python packages are required:

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/scipylib/index.html>`_
* `scikit-learn <https://scikit-learn.org/stable/>`_
* `vtk <https://vtk.org/>`_
* `matplotlib <https://matplotlib.org/>`_

Matplotlib is only used for colormaps and we may remove this dependency in
future releases.


Additional dependencies
^^^^^^^^^^^^^^^^^^^^^^^
For reading/writing of Gifti surfaces BrainSpace requires nibabel. To enable
interactivity, some plotting functionality in IPython notebooks makes
use of the panel package. PyQT is another dependency for background plotting.
See `PyVista <https://docs.pyvista.org/plotting/qt_plotting.html#background-plotting>`_
for more on background plotting. The support of background rendering however
is still experimental.

* `nibabel <https://nipy.org/nibabel/index.html>`_
* `panel <https://panel.pyviz.org/>`_
* `pyqt <https://riverbankcomputing.com/software/pyqt/intro>`_

Although these dependencies are optional, we recommend installing, at least, nibabel.


Installation
^^^^^^^^^^^^
You can install the python Package using ``pip``: ::

    pip install -U brainspace


To install with ``conda``: ::

    conda install brainspace



MATLAB installation
-------------------
This toolbox has been tested with MATLAB versions R2017a and R2018b, it likely
throws errors with MATLAB R2016a or older.

To install the MATLAB toolbox simply download the package and run the following in MATLAB: ::

    addpath('/path/to/BrainSpace/matlab/')


