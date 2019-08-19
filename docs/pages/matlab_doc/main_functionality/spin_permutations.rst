.. _spin_permutations:

==================
spin_permutations
==================

------------------
Synopsis
------------------

Performs a spin test to generate null data for hypothesis testing (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/analysis_code/spin_permutations.m>`_).

------------------
Usage
------------------

::

	null_data = spin_permutations(data,spheres,permutation_number,varargin);

- *data*: An n-by-m matrix of data to be randomised (single sphere) or cell array of n-by-m matrices (two spheres).  
- *spheres*: Cell array containing spheres. 
- *permutation_number*: The number of permutation to perform.
- *null_data*: The randomised data. 
- *varargin*: See name-value pairs below. 

------------------
Description
------------------

Spin test as described by `(Alexander-Bloch, 2018)
<https://www.sciencedirect.com/science/article/pii/S1053811918304968>`_. Data is
either an n-by-m matrix where n is the number of vertices and/or parcels or,
when spinning on two spheres, a two-element cell array each containing an n-by-m
matrix. Spheres is either a sphere in a format readable by
:ref:`convert_surface` or a two-element cell array containing spheres
corresponding to the respective elements in the data cell array. 

Name-Value pairs
------------------

- *'parcellation'*: a n-by-1 vector containing the parcellation scheme. Do not provide for no vertex-level analysis.

- *'type'*: program used to generate the spheres. Either 'FreeSurfer' (default), or 'CIVET'. If Freesurfer, rotations are flipped along the x-axis for the second sphere. 
