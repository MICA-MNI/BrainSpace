.. _full2parcel_matlab:

full2parcel
==============================

Synopsis
---------

Converts data from a full data matrix to parcellated data (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/surface_manipulation/full2parcel.m>`_).

Usage 
----------
::

    parcellated_data = full2parcel(data,parcellation);

- *data*: an n-by-m data matrix.
- *parcellation*: a 1-by-m parcel vector containing natural numbers.
- *parcellated_data*: n-by-max(parcellation) matrix containing the mean column of each parcel.

Description
--------------

A useful tool for quickly moving data between vertex and parcel level,
especially in combination with :ref:`parcel2full_matlab`. For more flexible usage, see
also :ref:`labelmean_matlab`.
