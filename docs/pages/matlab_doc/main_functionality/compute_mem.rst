.. _compute_mem_matlab:

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

------------------ 
Description 
------------------ 

The Moran eigenvectors hold information on the spatial autocorrelation of the
data. Their computation is kept separate from the randomization as these
eigenvectors can be used for any feature on the same surface. These eigenvectors
are used for Moran spectral randomization by :ref:`moran_randomization_matlab`.
See also `(Wagner and Dray, 2015)
<https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12407>`_.

If W is provided as a cortical surface, then this can either be a surface
in memory or a file readable by :ref:`read_surface_matlab`. 

Name-Value pairs
------------------
- *n_ring*: Only used if W is a surface. Vertices that are within `n_ring` steps of each other have their distance computed (Default: 1).
- *mask*: Only used if W is a surface. A n-by-1 logical denoting a mask. ``n`` denotes the number of vertices. Vertices corresponding to True values are discarded when computing the eigenvectors. You can also provide an empty logical array to discard nothing  (Default: []). 
- *spectrum*: Determines the behavior for discarding eigenvectors with eigenvalue=0. Set to 'all' for discarding only one and reorthogonalizing the remainder or 'NonZero' for discarding all zero eigenvalues (Default: 'all').  
