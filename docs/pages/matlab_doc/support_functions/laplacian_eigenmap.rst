.. _laplacian_eigenmap:

laplacian_eigenmap
==============================

Synopsis
---------

Performs the laplacian eigenmap computations (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/analysis_code/laplacian_eigenmap.m>`_). 

Usage 
----------
::

    [gradients, mapping] = laplacian_eigenmap(data, n_components)

- *data*: the data matrix to perform the laplacian eigenmapping on. 
- *n_components*: the number of components to return.
- *gradients*: the output gradients.
- *mapping*: structure containing the output gradients and lambdas. 

Description
--------------

Performs the laplacian eigenmap procedure. Original implemented in the `MATLAB
Toolbox for Dimensionality Reduction <http://homepage.tudelft.nl/19j49>`_ by
Laurens van der Maaten. Modified for use in the BrainSpace toolbox. 