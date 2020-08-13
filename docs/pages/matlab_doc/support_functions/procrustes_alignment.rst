.. _procrustes_alignment_matlab:

procrustes_alignment
==============================

Synopsis
---------

Performs gradient alignment through iterative singular value decomposition
(`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/analysis_code/procrustes_alignment.m>`_).

Usage 
----------
::

    aligned = procrustes_alignment(gradients,varargin)

- *gradients*: cell array of gradients
- *varargin*: Name-value pairs (see below).
- *aligned*: Aligned gradients. 

Description
-------------

Gradient alignment through Procrustes analysis [see `(Langs et al., 2015)
<https://link.springer.com/chapter/10.1007/978-3-319-24571-3_38>`_]. On the
first iteration all gradients are aligned to either the first provided gradient
set or to a template (see Name-Value pairs) using singular value decomposition.
On every subsequent iteration alignment is to the mean. 

Name-Value pairs
^^^^^^^^^^^^^^^^^

- *nIterations*: Number of iterations to run (default: 100).
- *reference*: Template to align to on the first iteration (default: first provided gradient set)

