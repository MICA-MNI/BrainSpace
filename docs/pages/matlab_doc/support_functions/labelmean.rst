.. _labelmean_matlab:

labelmean
==============================

Synopsis
---------

Takes the mean of columns with the same label (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/analysis_code/labelmean.m>`_). 

Usage 
----------
::

    label_data = labelmean(data,label,varargin);

- *data*: an n-by-m data matrix.
- *label*: a 1-by-m label vector containing natural numbers.
- *label_data*: n-by-max(label) matrix containing the mean column of each label.
- *varargin*: See below.

Description
---------------
A fast way of computing the mean for each label. Accepts the following additional arguments in varargin: 

- *sortByLabelVector*: Sorts the columns of ``label_data`` by the order of appearance in the ``label`` vector, rather than ascending order. 
- *ignoreWarning*: Does not display order of column sorting or warnings due to 0s in the label vector. 
