.. _laplacian_eigenmap_matlab:

laplacian_eigenmap
==============================

Synopsis
---------

Performs the laplacian eigenmap computations (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/analysis_code/laplacian_eigenmaps.m>`_). 

Usage 
----------
::

    [gradients, lambdas] = laplacian_eigenmap(data, n_components)

- *data*: the data matrix to perform the laplacian eigenmapping on. 
- *n_components*: the number of components to return.
- *gradients*: the output gradients.
- *mapping*: structure containing the output gradients and lambdas. 

Description
--------------

Performs the laplacian eigenmap procedure. Original implemented in the `MATLAB
Toolbox for Dimensionality Reduction <https://lvdmaaten.github.io/drtoolbox/>`_ by
Laurens van der Maaten. Modified for use in the BrainSpace toolbox. 
