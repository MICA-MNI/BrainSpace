.. _install_page:

Installation Guide
==============================

BrainSpace is available in Python and MATLAB.


Python installation
-------------------

BrainSpace requires Python 3.9 or newer. It is tested on Python 3.9 through
3.13 across Linux, macOS, and Windows.


Dependencies
^^^^^^^^^^^^

To use BrainSpace, the following Python packages are required:

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/scipylib/index.html>`_
* `scikit-learn <https://scikit-learn.org/stable/>`_
* `vtk <https://vtk.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `nibabel <https://nipy.org/nibabel/index.html>`_

Nibabel is required for reading/writing Gifti surfaces. Matplotlib is only
used for colormaps and we may remove this dependency in future releases.
``nilearn`` is only needed to run the tutorial notebooks; install it with
``pip install brainspace[examples]``.


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


Plotting on a remote / headless server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VTK requires a rendering context. On a server with no attached display
(SSH session, container, CI runner, JupyterHub) the default ``vtk`` wheel
will fail to render and ``plot_hemispheres`` may hang or error. Install
one of the offscreen-capable VTK builds instead:

* OSMesa (software rendering, no GPU needed): ::

    pip uninstall vtk
    pip install vtk-osmesa

* EGL (GPU-accelerated, requires an NVIDIA driver with EGL support): ::

    pip uninstall vtk
    pip install vtk-egl

If you prefer ``conda``, equivalent OSMesa builds are published on
``conda-forge``: ::

    conda install -c conda-forge mesalib
    conda install -c conda-forge "vtk=*=osmesa_*"

After installing, set ``embed_nb=True`` (Jupyter) or pass
``offscreen=True`` to the plotting functions.



MATLAB installation
-------------------

This toolbox has been tested with MATLAB versions R2018b, although we expect it
to work with versions R2018a and newer. Operating systems used during testing were OSX Mojave (10.14.6)
and Linux Xenial Xerus (16.04.6).

BrainSpace can be installed by `downloading
<https://github.com/MICA-MNI/BrainSpace/releases>`_ and unzipping the code from Github and running
the following in MATLAB: ::

    addpath(genpath('/path/to/BrainSpace/matlab/'))

If you want to load BrainSpace every time you start MATLAB, type ``edit
startup`` and append the above line to the end of this file. 

You can move the MATLAB directory to other locations. However, the example data
loader functions used in our tutorials require the MATLAB and shared directories
to both be in the same directory. 
    
If you wish to open gifti files (necessary for the tutorial) you will also need to install the `gifti library
<https://www.artefact.tk/software/matlab/gifti/>`_.
