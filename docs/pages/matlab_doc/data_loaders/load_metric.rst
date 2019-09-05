.. _load_marker_matlab:

=======================
load_marker
=======================

------------------
Synopsis
------------------

Loads metric data (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/example_data_loaders/load_marker.m>`_). 

------------------
Usage
------------------

::

   [metric_lh,metric_rh] = load_marker(name)

- *name*: The type of surface. Either 'thickness' for cortical thickness, 't1wt2w' for t1w/t2w intensity or 'curvature' for curvature. 
- *metric_lh*: Data on left hemisphere.
- *metric_rh*: Data on right hemisphere. 
