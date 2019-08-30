.. _load_metric_matlab:

=======================
load_metric
=======================

------------------
Synopsis
------------------

Loads metric data (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/example_data_loaders/load_metric.m>`_). 

------------------
Usage
------------------

::

   [metric_lh,metric_rh] = load_metric(type)

- *type*: The type of surface. Either 'thickness' for cortical thickness, 't1wt2w' for t1w/t2w intensity or 'curvature' for curvature. 
- *metric_lh*: Data on left hemisphere.
- *metric_rh*: Data on right hemisphere. 
