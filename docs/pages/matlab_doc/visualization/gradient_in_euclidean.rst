.. gradient_in_euclidean:

=======================
gradient_in_euclidean
=======================

------------------
Synopsis
------------------

Plots gradient data in 3D Euclidean space (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/plot_data/gradient_in_euclidean.m>`_).


------------------
Usage
------------------

::

   handles = gradient_in_euclidean(gradients)

- *gradients*: an n-by-3 matrix of gradients (usually the first three). 
- *handles*: a structure containing the handles of the graphics objects. 

------------------
Description
------------------
Produces a 3D scatter plot of gradient data in Euclidean space with each
datapoint color coded by their location. 

BrainSpace only provides basic figure building functionality. For more
information on how to use Matlab to create publication-ready figures we
recommend delving into `graphics object properties
<https://www.mathworks.com/help/matlab/graphics-object-properties.html>`_ (e.g.
`figure
<https://www.mathworks.com/help/matlab/ref/matlab.ui.figure-properties.html>`_,
`axes
<https://www.mathworks.com/help/matlab/ref/matlab.graphics.axis.axes-properties.html>`_,
`surface
<https://www.mathworks.com/help/matlab/ref/matlab.graphics.primitive.surface-properties.html>`_).
