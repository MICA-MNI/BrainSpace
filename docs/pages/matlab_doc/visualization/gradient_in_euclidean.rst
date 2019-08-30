.. _gradient_in_euclidean_matlab:

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
   handles = gradient_in_euclidean(gradients,surface)
   handles = gradient_in_euclidean(gradients,surface,parcellation)

- *gradients*: an n-by-3 matrix of gradients (usually the first three). 
- *surface*: a surface readable by :ref:`convert_surface_matlab` or a two-element cell array containing left and right hemispheric surfaces in that order. 
- *parcellation*: an m-by-1 vector containing the parcellation scheme, where m is the number of vertices. 
- *handles*: a structure containing the handles of the graphics objects. 

------------------
Description
------------------

Produces a 3D scatter plot of gradient data in Euclidean space with each
datapoint color coded by their location. If provided a surface (and
parcellation), also produces surface plots with the colors projected back to the
surface. 

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
