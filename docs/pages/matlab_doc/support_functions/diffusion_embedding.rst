.. _diffusion_embedding:

diffusion_embedding
==============================

Synopsis
---------

Performs the diffusion embedding computations (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/analysis_code/diffusion_embedding.m>`_). 

Usage 
----------
::

    [gradients,lambdas] = diffusion_embedding(data, n_components, alpha, diffusion_time);

- *data*: the data matrix to perform the diffusion embedding on. 
- *n_components*: the number of components to return.
- *alpha*: the alpha parameter.
- *diffusion_time*: the diffusion_time parameter; set to 0 for automatic estimation.
- *gradients*: the output gradients.
- *lambdas*: the output eigenvalues. 

Description
--------------
Performs the diffusion procedure. This implementation is based on the `mapalign package <https://github.com/satra/mapalign>`_ by Satrajid Ghosh.