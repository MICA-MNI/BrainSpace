.. _install_page:

Installation Guide
==============================

BrainSpace is available in Python and MATLAB.


Python installation
-------------------

BrainSpace works on Python 3.5+, and probably with older versions of Python 3,
although it is not tested. 


Dependencies
^^^^^^^^^^^^

To use BrainSpace, the following Python packages are required:

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/scipylib/index.html>`_
* `scikit-learn <https://scikit-learn.org/stable/>`_
* `vtk <https://vtk.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `nibabel <https://nipy.org/nibabel/index.html>`_
* `nilearn <https://nilearn.github.io/>`_

Nibabel is required for reading/writing Gifti surfaces. Matplotlib is only
used for colormaps and we may remove this dependency in future releases.


Additional dependencies
^^^^^^^^^^^^^^^^^^^^^^^
To enable interactivity, some plotting functionality in IPython notebooks makes
use of the panel package. PyQT is another dependency for background plotting.
See `PyVista <https://docs.pyvista.org/plotting/qt_plotting.html#background-plotting>`_
for more on background plotting. The support of background rendering however
is still experimental.

* `panel <https://panel.pyviz.org/>`_
* `pyqt <https://riverbankcomputing.com/software/pyqt/intro>`_


Installation
^^^^^^^^^^^^

BrainSpace can be installed using ``pip``: ::

    pip install brainspace


Alternatively, you can install the package from Github as follows: ::

    git clone https://github.com/MICA-MNI/BrainSpace.git
    cd BrainSpace
    python setup.py install



MATLAB installation
-------------------

This toolbox has been tested with MATLAB versions R2018b, although we expect it
to work with versions R2018a and newer. Operating systems used during testing were OSX Mojave (10.14.6)
and Linux Xenial Xerus (16.04.6).

To install the MATLAB toolbox simply `download
<https://github.com/MICA-MNI/BrainSpace/releases>`_ and unzip the GitHub toolbox and run
the following in MATLAB: ::

    addpath(genpath('/path/to/BrainSpace/matlab/'))

If you want to load BrainSpace every time you start MATLAB, type ``edit
startup`` and append the above line to the end of this file. 

You can move the MATLAB directory to other locations. However, the example data
loader functions used in our tutorials require the MATLAB and shared directories
to both be in the same directory. 
    
If you wish to open gifti files (necessary for the tutorial) you will also need to install the `gifti library
<https://www.artefact.tk/software/matlab/gifti/>`_.
