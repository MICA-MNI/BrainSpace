.. _load_gradient_matlab:

=======================
load_gradient
=======================

------------------
Synopsis
------------------

Loads template gradients (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/example_data_loaders/load_gradient.m>`_). 

------------------
Usage
------------------

::

    data = load_gradient(name,number)

- *name*: The type of gradient, either 'fc' for functional connectivity or 'mpc' for microstructural profile covariance. 
- *number*: The rank of the gradient, either 1 or 2. 
- *data*: Output data. 

