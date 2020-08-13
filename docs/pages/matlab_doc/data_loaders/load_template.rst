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

------------------
Description
------------------

Loads normative functional connectivity and microstructural profile covariance
gradients, both computed from the HCP dataset. Gradients were computed with a
cosine similarity kernel and diffusion mapping on a downsampled 5K cortical
mesh. Resulting gradients were upsampled to the 32K mesh. 
