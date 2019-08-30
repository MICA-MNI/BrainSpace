.. _load_parcellation_matlab:

=======================
load_parcellation
=======================

------------------
Synopsis
------------------

Loads parcellation schemes (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/example_data_loaders/load_parcellation.m>`_). 

------------------
Usage
------------------

::

    labels = load_parcellation(parcellation,scale)

- *parcellation*: Name of the parcellation, either 'schaeffer' for Schaeffer (functional) parcellations or 'vosdewael' for a subparcellation of aparc. Both may be provided as a cell or string array. 
- *scale*: Scale of the parcellation. Either 100, 200, 300, or 400. Multiple may be provided as a vector.
- *labels*: Structure containing the parcellation schemes. 
