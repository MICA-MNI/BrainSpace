.. _compute_mem:

==================
compute_mem
==================

------------------
Synopsis
------------------

Computes the moran eigenvectors required for Moran spectral randomization 
 (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/analysis_code/compute_mem.m>`_).

------------------
Usage
------------------

::

	MEM = compute_mem(W,varargin);

- *W*: A matrix denoting distance between features or a cortical surface. 
- *varargin*: Name-value pairs (see below). 

------------------ Description ------------------ 
The Moran eigenvectors hold information on the spatial autocorrelation of the
data. Their computation is kept separate from the randomization as these
eigenvectors can be used for any feature on the same surface.

Name-Value pairs
------------------
- *n_ring*: Only used if W is a surface. Vertices that are within `n_ring` steps of each other have their distance computed (Default: 1).
- *
