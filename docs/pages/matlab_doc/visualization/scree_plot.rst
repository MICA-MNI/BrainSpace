.. _scree_plot_matlab:

=======================
scree_plot
=======================

------------------
Synopsis
------------------

Produces a scree plot of the lambdas (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/plot_data/scree_plot.m>`_).


------------------
Usage
------------------

::

   handles = scree_plot(lambdas)

- *lambdas*: a vector of lambdas; can be taken from the :ref:`GradientMaps_matlab` object. 
- *handles*: a structure containing the handles of the graphics objects. 

------------------
Description
------------------

`scree_plot` plots the lambdas scaled to a sum of 1. It is a useful tool for
identifying the difference in the importance of each eigenvector (i.e. gradient)
with higher lambdas being more important in principal component analysis and
diffusion embedding, and lower ones more important in Laplacian eigenmaps.

BrainSpace only provides basic figure building functionality. For more
information on how to use MATLAB to create publication-ready figures we
recommend delving into `graphics object properties
<https://www.mathworks.com/help/matlab/graphics-object-properties.html>`_ (e.g.
`figure
<https://www.mathworks.com/help/matlab/ref/matlab.ui.figure-properties.html>`_,
`axes
<https://www.mathworks.com/help/matlab/ref/matlab.graphics.axis.axes-properties.html>`_,
`surface
<https://www.mathworks.com/help/matlab/ref/matlab.graphics.primitive.surface-properties.html>`_).
