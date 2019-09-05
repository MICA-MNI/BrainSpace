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

- *parcellation*: Name of the parcellation, either 'schaefer' for Schaefer (functional) parcellations or 'vosdewael' for a subparcellation of the `Desikan-Killiany atlas`__. Both may be provided as a cell or string array. 
- *scale*: Scale of the parcellation. Either 100, 200, 300, or 400. Multiple may be provided as a vector.
- *labels*: Structure containing the parcellation schemes. 

.. _DK: https://surfer.nmr.mgh.harvard.edu/ftp/articles/desikan06-parcellation.pdf

__ DK_
