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

- *parcellation*: Name of the parcellation, either 'schaeffer' for Schaeffer (functional) parcellations or 'vosdewael' for a subparcellation of aparc. Both may be provided as a cell or string array. 
- *scale*: Scale of the parcellation. Either 100, 200, 300, or 400. Multiple may be provided as a vector.
- *group*: Loads data from the main group if set to 'main' (default) or the holdout group if set to 'holdout'. 
- *conn_matrices*: Structure of all requested data. 
