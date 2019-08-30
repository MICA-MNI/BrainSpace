.. _load_template:

=======================
load_template
=======================

------------------
Synopsis
------------------

Loads template gradients (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/example_data_loaders/load_template.m>`_). 

------------------
Usage
------------------

::

    data = load_template(type,number)

- *type*: The type of gradient, either 'fc' for functional connectivity or 'mpc' for microstructural profile covariance. 
- *number*: The rank of the gradient, either 1 or 2. 
- *data*: Output data. 

