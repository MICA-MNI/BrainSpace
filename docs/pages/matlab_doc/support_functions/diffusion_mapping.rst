.. _diffusion_mapping_matlab:

diffusion_mapping
==============================

Synopsis
---------

Performs the diffusion mapping computations (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/analysis_code/diffusion_mapping.m>`_). 

Usage 
----------
::

    [gradients,lambdas] = diffusion_mapping(data, n_components, alpha, diffusion_time, random_state);

- *data*: the data matrix to perform the diffusion mapping on. 
- *n_components*: the number of components to return.
- *alpha*: the alpha parameter.
- *diffusion_time*: the diffusion_time parameter; set to 0 for automatic estimation.
- *random_state*: Input passed to the rng() function for randomization initialization (default: no initialization). 
- *gradients*: the output gradients.
- *lambdas*: the output eigenvalues. 

Description
--------------
Performs the diffusion procedure. This implementation is based on the `mapalign package <https://github.com/satra/mapalign>`_ by Satrajid Ghosh.
