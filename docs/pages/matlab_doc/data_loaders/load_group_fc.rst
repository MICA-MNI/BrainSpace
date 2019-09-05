.. _load_group_fc_matlab:

=======================
load_group_fc
=======================

------------------
Synopsis
------------------

Loads group level functional connectivity matrices (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/example_data_loaders/load_group_fc.m>`_). 

------------------
Usage
------------------

::

    conn_matrices = load_group_fc(parcellation,scale,group)

- *parcellation*: Name of the parcellation, either 'schaefer' for Schaefer (functional) parcellations or 'vosdewael' for a subparcellation of the `Desikan-Killiany atlas`__. Both may be provided as a cell or string array. 
- *scale*: Scale of the parcellation. Either 100, 200, 300, or 400. Multiple may be provided as a vector.
- *group*: Loads data from the main group if set to 'main' (default) or the holdout group if set to 'holdout'. 
- *conn_matrices*: Structure of all requested data. 

.. _DK: https://surfer.nmr.mgh.harvard.edu/ftp/articles/desikan06-parcellation.pdf

__ DK_
