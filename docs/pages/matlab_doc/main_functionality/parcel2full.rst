.. _parcel2full_matlab:

parcel2full
==============================

Synopsis
---------

Converts data from a parcellated data matrix to full data matrix by assigning
each vertex the value of its parcel (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/surface_manipulation/parcel2full.m>`_).

Usage 
----------
::

    full_data = parcel2full(parcellated_data,parcellation);

- *parcellated_data*: an n-by-max(parcellation) data matrix.
- *parcellation*: a 1-by-m parcel vector containing natural numbers.
- *full_data*: n-by-m matrix containing the data at the vertex level.

Description 
--------------

A useful tool for quickly moving data between parcel and vertex level,
especially in combination with :ref:`full2parcel_matlab`. This is mostly used for
plotting parcellated data on the surface. 

