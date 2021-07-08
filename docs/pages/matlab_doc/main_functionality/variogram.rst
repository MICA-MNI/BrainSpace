.. _variogram_matlab:

====================
variogram
====================

------------------
Synopsis
------------------

Generates null model data correcting for spatial autocorrelation. 

------------------
Usage
------------------

::

    obj = variogram(D,varargin);
    surrogates = obj.fit(x,n);

- *D*: A symmetric distance matrix.
- *n*: Number of surrogate datasets to generate (default: 1000).
- *varargin*: See name-value pairs below. 
- *obj*: The variogram object.
- *surrogates*: Surrogate datasets.

------------------ 
Description 
------------------ 

Implementation of the variogram matching procedure as presented by  `(Burt et al., 2020)
<https://www.sciencedirect.com/science/article/pii/S1053811920305243>`_.
This class generates surrogate data by permuting the input data, smoothing it with different 
kernels, and determining which smoothed permuted map best fits the variogram of the 
empirical data. 

-----------------------
Name-Value Pairs
-----------------------

- *deltas*: Proportion of neighbours to include for smoothing (default: 0.1:0.1:0.9).
- *kernel*: Kernel with which to smooth permuted maps. Valid options are 'gaussian', 'exp' (default), 'invdist', 'uniform', and a function handle.
- *pv*: Percentile of pairwise distance distribution at which to truncate during variogram fitting (default: 25).
- *nh*: Number of uniformly spaced distances at which to compute variograms (default: 25).
- *resample*: Resample surrogate maps' values from target brain map (default: false).
- *b*:  Gaussian kernel bandwidth for variogram smoothing (default: three times the spacing between variogram x-coordinates).
- *random_state*: any valid input for the rng() function. Set to nan for no random initialization (default: nan).
- *ns*: Number of samples to use when subsampling the brainmap. Set to inf to use the entire brainmap (default: inf).
- *knn*: Number of nearest neighbours to use when smoothing the map. knn must be smaller than ns (default: 1000). 
- *num_workers*: Number of workers in the parallel pool. Requires the parallel processing toolbox. Set to 0 for no parallelization (default: 0). 

