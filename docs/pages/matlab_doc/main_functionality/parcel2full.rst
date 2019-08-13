.. _parcel2full:

parcel2full
==============================

Synopsis
---------

Converts data from a parcellated data matrix to full data matrix by assigning each vertex the value of its parcel.

Usage 
----------
::

    full_data = parcel2full(parcellated_data,parcellation);

- *parcellated_data*: an n-by-max(parcellation) data matrix.
- *parcellation*: a 1-by-m parcel vector containing natural numbers.
- *full_data*: n-by-m matrix containing the data at the vertex level.

Description
--------------
A useful tool for quickly moving data between parcel and vertex level, especially in combination with :ref:`full2parcel`. This is mostly used for plotting parcellated data on the surface. 

